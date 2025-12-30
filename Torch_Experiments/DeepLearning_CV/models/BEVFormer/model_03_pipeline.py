import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1) Config
# =========================================================
class Config:
    # BEV grid
    bev_h = 50
    bev_w = 50
    bev_z = 4  # N_ref (pillar samples)
    embed_dims = 256
    num_heads = 8
    num_cams = 6

    # point cloud range in meters: [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [-50, -50, -5, 50, 50, 3]

    # optional
    dropout = 0.1
    num_layers = 3

    # NOTE: real BEVFormer uses backbone+FPN multi-level; here we keep single-level for clarity.
    # If lidar2img projects to original image pixels, you must map to feature coords by /stride.
    feat_stride = 1  # set to backbone stride if needed


# =========================================================
# 2) Utilities
# =========================================================
def bev_positional_encoding(cfg, device, dtype):
    """
    Simple 2D sinusoidal position encoding for BEV tokens.
    Return: (Nq, C)
    """
    H, W, C = cfg.bev_h, cfg.bev_w, cfg.embed_dims
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    y = y.reshape(-1)  # (Nq,)
    x = x.reshape(-1)

    # normalize to [0,1]
    y = y / max(H - 1, 1)
    x = x / max(W - 1, 1)

    # sinusoidal
    pe = torch.zeros((H * W, C), device=device, dtype=dtype)
    div = torch.exp(torch.arange(0, C, 2, device=device, dtype=dtype) * (-math.log(10000.0) / C))
    pe[:, 0::2] = torch.sin(x.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(y.unsqueeze(1) * div)
    return pe


def meters_to_bev_pixels(cfg, dx_m, dy_m):
    """
    Convert translation in meters to BEV pixel offsets (dx_px, dy_px).
    """
    x_min, y_min, _, x_max, y_max, _ = cfg.pc_range
    meter_per_px_x = (x_max - x_min) / cfg.bev_w
    meter_per_px_y = (y_max - y_min) / cfg.bev_h
    dx_px = dx_m / meter_per_px_x
    dy_px = dy_m / meter_per_px_y
    return dx_px, dy_px


# =========================================================
# 3) Spatial Cross Attention (SCA) — query-conditioned cam×z weights + hit-view norm
#    - still 1 sample per ref point (no deformable offsets yet)
# =========================================================
class SpatialCrossAttention(nn.Module):
    """
    Inputs:
      query:     (B, Nq, C)
      img_feats: (B, Ncam, C, Hf, Wf)
      lidar2img: (B, Ncam, 4, 4)

    Output (delta):
      delta_q:   (B, Nq, C)  (to be added to query in EncoderLayer)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # predict weights over (Ncam * Z) for each query
        self.weight_proj = nn.Linear(cfg.embed_dims, cfg.num_cams * cfg.bev_z)
        self.out_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)

        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _get_reference_points(H, W, Z, B, device, dtype):
        """
        Return normalized ref points in [0,1]:
          ref: (B, Nq, Z, 3) where 3 = (x_norm, y_norm, z_norm)
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype),
            torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype),
            indexing="ij"
        )
        ref_y = (ref_y.reshape(-1)[None] / H)  # (1,Nq)
        ref_x = (ref_x.reshape(-1)[None] / W)  # (1,Nq)

        # (1,Nq,Z)
        ref_x = ref_x.unsqueeze(-1).repeat(1, 1, Z)
        ref_y = ref_y.unsqueeze(-1).repeat(1, 1, Z)

        # z_norm in (0,1)
        ref_z = torch.linspace(0.5, Z - 0.5, Z, device=device, dtype=dtype) / Z
        ref_z = ref_z.view(1, 1, Z).repeat(1, H * W, 1)

        ref = torch.stack([ref_x, ref_y, ref_z], dim=-1)  # (1,Nq,Z,3)
        return ref.repeat(B, 1, 1, 1)  # (B,Nq,Z,3)

    def _point_sampling(self, ref_norm, img_feats, lidar2img):
        """
        ref_norm:  (B, Nq, Z, 3) normalized in [0,1]
        img_feats: (B, Ncam, C, Hf, Wf)
        lidar2img: (B, Ncam, 4, 4)

        Returns:
          sampled: (B, Ncam, C, Nq, Z)
          valid:   (B, Ncam, 1, Nq, Z) boolean mask
        """
        B, Nq, Z, _ = ref_norm.shape
        B2, Ncam, C, Hf, Wf = img_feats.shape
        assert B == B2

        # --- A) denorm to world coords (meters) ---
        x_min, y_min, z_min, x_max, y_max, z_max = self.cfg.pc_range
        ref = ref_norm.clone()
        ref[..., 0] = ref[..., 0] * (x_max - x_min) + x_min
        ref[..., 1] = ref[..., 1] * (y_max - y_min) + y_min
        ref[..., 2] = ref[..., 2] * (z_max - z_min) + z_min

        # homogeneous: (x,y,z,1)
        ref = torch.cat([ref, torch.ones_like(ref[..., :1])], dim=-1)  # (B,Nq,Z,4)

        # --- B) project to each camera ---
        ref = ref.view(B, 1, Nq * Z, 4).repeat(1, Ncam, 1, 1)  # (B,Ncam,Nq*Z,4)
        l2i = lidar2img.view(B, Ncam, 1, 4, 4)                 # (B,Ncam,1,4,4)

        cam = torch.matmul(l2i, ref.unsqueeze(-1)).squeeze(-1)  # (B,Ncam,Nq*Z,4)
        eps = 1e-5
        depth = cam[..., 2:3]
        valid = depth > eps

        xy = cam[..., 0:2] / torch.clamp(depth, min=eps)       # pixel coords in feature plane assumption

        # If lidar2img gives original image pixels, map to feature coords:
        if self.cfg.feat_stride != 1:
            xy = xy / float(self.cfg.feat_stride)

        # --- C) normalize to [-1,1] for grid_sample ---
        # align_corners=False, we normalize by (W-1)/(H-1) for pixel space
        x = xy[..., 0] / max(Wf - 1, 1)
        y = xy[..., 1] / max(Hf - 1, 1)
        grid = torch.stack([(x - 0.5) * 2.0, (y - 0.5) * 2.0], dim=-1)  # (B,Ncam,Nq*Z,2)

        # in-bounds
        valid = valid & (grid[..., 0:1] > -1.0) & (grid[..., 0:1] < 1.0) & \
                      (grid[..., 1:2] > -1.0) & (grid[..., 1:2] < 1.0)

        # --- D) sample ---
        img = img_feats.view(B * Ncam, C, Hf, Wf)
        grid_rs = grid.view(B * Ncam, Nq * Z, 1, 2)
        sampled = F.grid_sample(img, grid_rs, align_corners=False)  # (B*Ncam,C,Nq*Z,1)
        sampled = sampled.view(B, Ncam, C, Nq, Z)

        valid = valid.view(B, Ncam, 1, Nq, Z)
        sampled = torch.nan_to_num(sampled) * valid
        return sampled, valid

    def forward(self, query, img_feats, lidar2img):
        B, Nq, C = query.shape
        Z = self.cfg.bev_z
        Ncam = self.cfg.num_cams

        # ref points: (B,Nq,Z,3)
        ref = self._get_reference_points(self.cfg.bev_h, self.cfg.bev_w, Z, B, query.device, query.dtype)

        sampled, valid = self._point_sampling(ref, img_feats, lidar2img)
        # sampled: (B,Ncam,C,Nq,Z), valid:(B,Ncam,1,Nq,Z)

        # query-conditioned weights over cam*Z
        w = self.weight_proj(query).view(B, Nq, Ncam, Z)  # (B,Nq,Ncam,Z)
        w = F.softmax(w.flatten(2), dim=-1).view(B, Nq, Ncam, Z)

        # apply hit-view normalization: only normalize among valid views
        # valid_camz: (B,Nq,Ncam,Z)
        valid_camz = valid[:, :, 0, :, :]          # (B, Ncam, Nq, Z)
        valid_camz = valid_camz.permute(0, 2, 1, 3) # (B, Nq, Ncam, Z)
        w = w * valid_camz
        denom = w.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        w = w / denom

        # aggregate
        # sampled -> (B,Nq,Ncam,Z,C)
        sampled = sampled.permute(0, 3, 1, 4, 2)
        out = (sampled * w.unsqueeze(-1)).sum(dim=2).sum(dim=2)  # (B,Nq,C)

        delta = self.out_proj(out)
        delta = self.dropout(delta)
        return delta


# =========================================================
# 4) Temporal Self Attention (TSA) — ego-motion warp + attention
#    Return delta only (no residual inside)
# =========================================================
class TemporalSelfAttention(nn.Module):
    """
    Inputs:
      query:    (B,Nq,C)
      prev_bev: (B,Nq,C) or None
      ego_motion: (B,4,4) transformation from t-1 to t

    Output (delta):
      delta_q:  (B,Nq,C)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(cfg.embed_dims, cfg.num_heads, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def _warp_prev_bev(self, prev_bev, ego_motion):
        """
        Warp prev_bev tokens (B,Nq,C) to current frame using ego motion on BEV plane.
        Returns aligned_prev: (B,Nq,C)
        """
        B, Nq, C = prev_bev.shape
        H, W = self.cfg.bev_h, self.cfg.bev_w
        assert Nq == H * W

        prev = prev_bev.permute(0, 2, 1).reshape(B, C, H, W)  # (B,C,H,W)

        # Extract planar translation + yaw from ego_motion
        tx = ego_motion[:, 0, 3]  # meters
        ty = ego_motion[:, 1, 3]
        yaw = torch.atan2(ego_motion[:, 1, 0], ego_motion[:, 0, 0])

        dx_px, dy_px = meters_to_bev_pixels(self.cfg, tx, ty)  # (B,)

        cos_r = torch.cos(yaw)
        sin_r = torch.sin(yaw)

        theta = torch.zeros(B, 2, 3, device=prev.device, dtype=prev.dtype)
        theta[:, 0, 0] = cos_r
        theta[:, 0, 1] = -sin_r
        theta[:, 1, 0] = sin_r
        theta[:, 1, 1] = cos_r

        # pixel -> normalized shift in [-1,1]
        theta[:, 0, 2] = dx_px * 2.0 / max(W, 1)
        theta[:, 1, 2] = dy_px * 2.0 / max(H, 1)

        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        aligned = F.grid_sample(prev, grid, align_corners=False, padding_mode="zeros")
        aligned = aligned.flatten(2).permute(0, 2, 1)  # (B,Nq,C)
        return aligned

    def forward(self, query, prev_bev, ego_motion):
        if prev_bev is None:
            # self-attn on current query only
            out, _ = self.attn(query, query, query)
            return self.dropout(out)

        aligned_prev = self._warp_prev_bev(prev_bev, ego_motion)  # (B,Nq,C)

        # memory = concat current + history (simple, effective)
        memory = torch.cat([query, aligned_prev], dim=1)  # (B,2Nq,C)
        out, _ = self.attn(query, memory, memory)         # (B,Nq,C)
        return self.dropout(out)


# =========================================================
# 5) Encoder Layer — standard Transformer residual+norm (no duplication)
# =========================================================
class BEVFormerEncoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tsa = TemporalSelfAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.embed_dims)

        self.sca = SpatialCrossAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dims, cfg.embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dims * 2, cfg.embed_dims),
            nn.Dropout(cfg.dropout),
        )
        self.norm3 = nn.LayerNorm(cfg.embed_dims)

    def forward(self, query, prev_bev, img_feats, ego_motion, lidar2img):
        # TSA
        delta = self.tsa(query, prev_bev, ego_motion)
        query = self.norm1(query + delta)

        # SCA
        delta = self.sca(query, img_feats, lidar2img)
        query = self.norm2(query + delta)

        # FFN
        delta = self.ffn(query)
        query = self.norm3(query + delta)
        return query


# =========================================================
# 6) BEVFormer Encoder-only model
# =========================================================
class BEVFormer(nn.Module):
    """
    Forward inputs:
      imgs:      (B,Ncam,3,H,W)
      prev_bev:  (B,Nq,C) or None
      ego_motion:(B,4,4)
      lidar2img: (B,Ncam,4,4)

    Output:
      bev_tokens:(B,Nq,C)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # toy backbone: single-level feature
        self.backbone = nn.Sequential(
            nn.Conv2d(3, cfg.embed_dims, kernel_size=1),
            nn.BatchNorm2d(cfg.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList([BEVFormerEncoderLayer(cfg) for _ in range(cfg.num_layers)])

        # learnable BEV content + fixed sinusoidal pos enc (common practice)
        self.bev_queries = nn.Parameter(torch.randn(cfg.bev_h * cfg.bev_w, cfg.embed_dims) * 0.02)
        self.register_buffer("_bev_pos", torch.zeros(1), persistent=False)  # placeholder

    def forward(self, imgs, prev_bev=None, ego_motion=None, lidar2img=None):
        B, Ncam, Cin, H, W = imgs.shape
        assert Ncam == self.cfg.num_cams

        # image features
        x = imgs.view(B * Ncam, Cin, H, W)
        feat = self.backbone(x)  # (B*Ncam,C,H,W)  (single-level)
        C = feat.shape[1]
        Hf, Wf = feat.shape[-2], feat.shape[-1]
        img_feats = feat.view(B, Ncam, C, Hf, Wf)  # (B,Ncam,C,Hf,Wf)

        # BEV queries + pos
        bev_pos = bev_positional_encoding(self.cfg, device=imgs.device, dtype=img_feats.dtype)  # (Nq,C)
        queries = self.bev_queries + bev_pos  # (Nq,C)
        queries = queries.unsqueeze(0).repeat(B, 1, 1)  # (B,Nq,C)

        # encoder layers
        for layer in self.layers:
            queries = layer(queries, prev_bev, img_feats, ego_motion, lidar2img)

        return queries


# =========================================================
# 7) Simple smoke test
# =========================================================
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVFormer(cfg).to(device)

    B = 1
    H_img, W_img = 128, 128
    imgs = torch.randn(B, cfg.num_cams, 3, H_img, W_img, device=device)

    # For a real pipeline, lidar2img must be valid projection matrices.
    lidar2img = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, cfg.num_cams, 1, 1)

    # ego_motion: identity means no motion between frames
    ego_motion = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

    print("Frame t-1")
    bev0 = model(imgs, prev_bev=None, ego_motion=ego_motion, lidar2img=lidar2img)
    print("bev0:", bev0.shape)  # (B,Nq,C)

    print("Frame t")
    bev1 = model(imgs, prev_bev=bev0.detach(), ego_motion=ego_motion, lidar2img=lidar2img)
    print("bev1:", bev1.shape)

    print("OK ✅")

if __name__ == "__main__":
    main()
