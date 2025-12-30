import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 0) Config
# =========================================================
class Cfg:
    # --------- data / sensor ---------
    num_cams = 6

    # camera image size (input)
    img_h = 256
    img_w = 704

    # backbone output feature size (we keep same spatial size for simplicity by stride=4)
    feat_stride = 4
    feat_h = img_h // feat_stride
    feat_w = img_w // feat_stride

    # --------- BEV grid ---------
    # [x_min, y_min, z_min, x_max, y_max, z_max] in meters
    pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    bev_h = 128
    bev_w = 128

    # meter per BEV pixel
    # (x_range / W, y_range / H) computed from pc_range + bev size

    # --------- feature dims ---------
    cam_feat_c = 64          # backbone feature channels
    cam_ctx_c = 80           # context channels after depth-net
    cam_depth_bins = 48      # D in LSS

    pts_feat_in = 4          # [x,y,z,intensity] (radar/lidar)
    pts_embed_c = 80         # point feature embed channels (pillar/voxel feature)

    bev_c = 128              # fused BEV channels

    # fusion
    fusion_mode = "gated"    # "concat" or "gated" or "xattn"

    # detection head
    num_classes = 1


# =========================================================
# 1) Camera backbone (simple conv)
#    imgs: (B, Ncam, 3, H, W) -> feats: (B, Ncam, C, Hf, Wf)
# =========================================================
class SimpleBackbone(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        # stride=4 downsample
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, cfg.cam_feat_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.cam_feat_c), nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        B, N, C, H, W = imgs.shape
        x = imgs.view(B * N, C, H, W)
        feat = self.net(x)  # (B*N, cam_feat_c, Hf, Wf)
        feat = feat.view(B, N, feat.shape[1], feat.shape[2], feat.shape[3])
        return feat


# =========================================================
# 2) LSS Depth + Context net
#    img_feats -> depth logits (B,N,D,Hf,Wf) and context (B,N,Cctx,Hf,Wf)
# =========================================================
class DepthContextNet(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        out_c = cfg.cam_depth_bins + cfg.cam_ctx_c
        self.head = nn.Sequential(
            nn.Conv2d(cfg.cam_feat_c, cfg.cam_feat_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.cam_feat_c, out_c, 1),
        )

    def forward(self, img_feats):
        # img_feats: (B,N,C,Hf,Wf)
        B, N, C, Hf, Wf = img_feats.shape
        x = img_feats.view(B * N, C, Hf, Wf)
        y = self.head(x)  # (B*N, D+Cctx, Hf, Wf)
        y = y.view(B, N, y.shape[1], Hf, Wf)
        depth_logits = y[:, :, : self.cfg.cam_depth_bins]     # (B,N,D,Hf,Wf)
        context = y[:, :, self.cfg.cam_depth_bins :]          # (B,N,Cctx,Hf,Wf)
        depth_prob = F.softmax(depth_logits, dim=2)           # softmax over D
        return depth_prob, context


# =========================================================
# 3) Camera frustum -> 3D points -> BEV pooling (scatter_add)
#    This is the real "Lift (depth bins) + Splat (to BEV)" idea.
#
# We keep it "paper-faithful":
# - create frustum points per pixel per depth bin in camera coordinates
# - transform to ego frame with cam2ego
# - optionally apply intrinsics; here we do the standard:
#   X_cam = (u - cx)/fx * Z
#   Y_cam = (v - cy)/fy * Z
#   Z_cam = depth
# - then cam2ego -> ego coords (x,y,z)
# - map (x,y) to BEV indices, scatter-add weighted context features
# =========================================================
class CameraViewTransformerLSS(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        # depth bin centers in meters (linear)
        z_min = 1.0
        z_max = 60.0
        self.register_buffer(
            "depth_values",
            torch.linspace(z_min, z_max, cfg.cam_depth_bins).view(1, 1, cfg.cam_depth_bins, 1, 1),
            persistent=False,
        )

        # after pooling, we project ctx_c -> bev_c
        self.bev_proj = nn.Sequential(
            nn.Conv2d(cfg.cam_ctx_c, cfg.bev_c, 1),
            nn.BatchNorm2d(cfg.bev_c),
            nn.ReLU(inplace=True),
        )

    def _make_pixel_grid(self, Hf, Wf, device, dtype):
        """
        pixel grid in the ORIGINAL IMAGE coordinates (not feature).
        We map feature cell center -> image pixel center by *stride.
        Returns:
          u,v: (1,1,1,Hf,Wf)
        """
        s = self.cfg.feat_stride
        # feature coords (0..Wf-1) -> image pixel centers
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * s
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * s
        v, u = torch.meshgrid(ys, xs, indexing="ij")  # (Hf,Wf)
        u = u.view(1, 1, 1, Hf, Wf)
        v = v.view(1, 1, 1, Hf, Wf)
        return u, v

    def forward(self, depth_prob, context, intrinsics, cam2ego):
        """
        depth_prob: (B,N,D,Hf,Wf)
        context:    (B,N,Cctx,Hf,Wf)
        intrinsics: (B,N,3,3)  (fx,fy,cx,cy)
        cam2ego:    (B,N,4,4)

        Returns:
          cam_bev: (B, bev_c, Hbev, Wbev)
        """
        cfg = self.cfg
        B, N, D, Hf, Wf = depth_prob.shape
        device, dtype = depth_prob.device, depth_prob.dtype

        # -------- A) build frustum 3D points in camera frame --------
        # u,v: (1,1,1,Hf,Wf)
        u, v = self._make_pixel_grid(Hf, Wf, device, dtype)
        Z = self.depth_values.to(device=device, dtype=dtype)  # (1,1,D,1,1)

        fx = intrinsics[:, :, 0, 0].view(B, N, 1, 1, 1)
        fy = intrinsics[:, :, 1, 1].view(B, N, 1, 1, 1)
        cx = intrinsics[:, :, 0, 2].view(B, N, 1, 1, 1)
        cy = intrinsics[:, :, 1, 2].view(B, N, 1, 1, 1)

        # X_cam = (u-cx)/fx * Z, Y_cam = (v-cy)/fy * Z
        Xc = (u - cx) / fx * Z
        Yc = (v - cy) / fy * Z
        Zc = Z.expand_as(Xc)

        # (B,N,D,Hf,Wf,3)
        pts_cam = torch.stack([Xc, Yc, Zc], dim=-1)

        # homogeneous (B,N,D,Hf,Wf,4)
        ones = torch.ones_like(pts_cam[..., :1])
        pts_cam_h = torch.cat([pts_cam, ones], dim=-1)

        # -------- B) cam -> ego --------
        # cam2ego: (B,N,4,4)
        T = cam2ego.view(B, N, 1, 1, 1, 4, 4)  # broadcast to D,Hf,Wf
        pts_ego = torch.matmul(T, pts_cam_h.unsqueeze(-1)).squeeze(-1)  # (B,N,D,Hf,Wf,4)
        pts_ego = pts_ego[..., :3]  # (x,y,z) in meters

        # -------- C) prepare features to splat --------
        # context: (B,N,Cctx,Hf,Wf) -> (B,N,1,Cctx,Hf,Wf)
        ctx = context.unsqueeze(2)  # (B,N,1,Cctx,Hf,Wf)
        # depth_prob: (B,N,D,Hf,Wf) -> (B,N,D,1,Hf,Wf)
        dp = depth_prob.unsqueeze(3)

        # Lift: per-depth weighted context
        # (B,N,D,Cctx,Hf,Wf)
        feat_lift = dp * ctx

        # -------- D) map ego (x,y) -> BEV indices and scatter_add --------
        x_min, y_min, _, x_max, y_max, _ = cfg.pc_range
        # meters per pixel
        mx = (x_max - x_min) / cfg.bev_w
        my = (y_max - y_min) / cfg.bev_h

        # (B,N,D,Hf,Wf)
        x = pts_ego[..., 0]
        y = pts_ego[..., 1]

        ix = torch.floor((x - x_min) / mx).long()
        iy = torch.floor((y - y_min) / my).long()

        # valid mask inside BEV
        valid = (ix >= 0) & (ix < cfg.bev_w) & (iy >= 0) & (iy < cfg.bev_h)

        # flatten everything for scatter
        # target index in flattened BEV: idx = iy*W + ix  -> shape (B, N, D, Hf, Wf)
        bev_ind = (iy * cfg.bev_w + ix)  # (B,N,D,Hf,Wf)
        bev_ind = bev_ind.clamp(min=0, max=cfg.bev_h * cfg.bev_w - 1)

        # features: (B,N,D,Cctx,Hf,Wf) -> (B, N*D*Hf*Wf, Cctx)
        feat = feat_lift.permute(0, 1, 2, 4, 5, 3).contiguous()  # (B,N,D,Hf,Wf,Cctx)
        feat = feat.view(B, -1, cfg.cam_ctx_c)

        ind = bev_ind.view(B, -1)         # (B, NDHW)
        msk = valid.view(B, -1)           # (B, NDHW)

        # output BEV context: (B, Cctx, H*W)
        bev_ctx = torch.zeros(B, cfg.cam_ctx_c, cfg.bev_h * cfg.bev_w, device=device, dtype=dtype)

        # scatter-add per batch
        for b in range(B):
            if msk[b].any():
                bev_ctx[b].index_add_(dim=1, index=ind[b][msk[b]], source=feat[b][msk[b]].t())

        # reshape to (B,Cctx,H,W)
        cam_bev_ctx = bev_ctx.view(B, cfg.cam_ctx_c, cfg.bev_h, cfg.bev_w)

        # project to bev_c
        cam_bev = self.bev_proj(cam_bev_ctx)  # (B, bev_c, H, W)
        return cam_bev


# =========================================================
# 4) Point (LiDAR/Radar) -> BEV (pillar/voxel features)
#    We implement a simple pillar feature:
#      - bin points by (x,y) into BEV grid
#      - per cell: mean of embedded point features
# =========================================================
class PointsToBEV(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        # embed point feature (x,y,z,i) -> pts_embed_c
        self.mlp = nn.Sequential(
            nn.Linear(cfg.pts_feat_in, cfg.pts_embed_c),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.pts_embed_c, cfg.pts_embed_c),
            nn.ReLU(inplace=True),
        )
        # project to bev_c
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.pts_embed_c, cfg.bev_c, 1),
            nn.BatchNorm2d(cfg.bev_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, points):
        """
        points: (B, Npts, 4) -> [x,y,z,intensity] in meters (ego frame)
        return: pts_bev: (B, bev_c, Hbev, Wbev)
        """
        cfg = self.cfg
        B, Np, Fdim = points.shape
        device, dtype = points.device, points.dtype

        x_min, y_min, _, x_max, y_max, _ = cfg.pc_range
        mx = (x_max - x_min) / cfg.bev_w
        my = (y_max - y_min) / cfg.bev_h

        x = points[..., 0]
        y = points[..., 1]

        ix = torch.floor((x - x_min) / mx).long()
        iy = torch.floor((y - y_min) / my).long()
        valid = (ix >= 0) & (ix < cfg.bev_w) & (iy >= 0) & (iy < cfg.bev_h)

        ind = (iy * cfg.bev_w + ix).clamp(0, cfg.bev_h * cfg.bev_w - 1)  # (B,Np)

        # embed each point: (B,Np,pts_embed_c)
        pts_emb = self.mlp(points)  # uses x,y,z,i (could add radar vel)

        # scatter mean: sum / count
        bev_sum = torch.zeros(B, cfg.pts_embed_c, cfg.bev_h * cfg.bev_w, device=device, dtype=dtype)
        bev_cnt = torch.zeros(B, 1, cfg.bev_h * cfg.bev_w, device=device, dtype=dtype)

        for b in range(B):
            m = valid[b]
            if m.any():
                bev_sum[b].index_add_(dim=1, index=ind[b][m], source=pts_emb[b][m].t())
                ones = torch.ones(ind[b][m].shape[0], device=device, dtype=dtype).view(1, -1)
                bev_cnt[b].index_add_(dim=1, index=ind[b][m], source=ones)

        bev_mean = bev_sum / bev_cnt.clamp(min=1.0)
        pts_bev = bev_mean.view(B, cfg.pts_embed_c, cfg.bev_h, cfg.bev_w)
        pts_bev = self.proj(pts_bev)  # (B, bev_c, H, W)
        return pts_bev


# =========================================================
# 5) Fusion: how to fuse Camera BEV and LiDAR/Radar BEV
#
# BEVFusion (feature-level BEV fusion) usually:
#   - Align BEV grids (same pc_range, same resolution)
#   - Fuse features by:
#       (a) concat + 1x1
#       (b) gated fusion (learn alpha)
#       (c) attention fusion (cross-attn in BEV)
# We'll implement (a)+(b) and provide (c) optional.
# =========================================================
class BEVFusion(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.fusion_mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(cfg.bev_c * 2, cfg.bev_c, 1),
                nn.BatchNorm2d(cfg.bev_c),
                nn.ReLU(inplace=True),
            )
        elif cfg.fusion_mode == "gated":
            # alpha = sigmoid(Conv([cam,pts])) -> (B, C, H, W)
            self.alpha = nn.Sequential(
                nn.Conv2d(cfg.bev_c * 2, cfg.bev_c, 1),
                nn.Sigmoid()
            )
            self.out = nn.Sequential(
                nn.Conv2d(cfg.bev_c, cfg.bev_c, 3, padding=1),
                nn.BatchNorm2d(cfg.bev_c),
                nn.ReLU(inplace=True),
            )
        elif cfg.fusion_mode == "xattn":
            # simple BEV cross-attention: cam as Q, pts as K,V
            self.q = nn.Conv2d(cfg.bev_c, cfg.bev_c, 1)
            self.k = nn.Conv2d(cfg.bev_c, cfg.bev_c, 1)
            self.v = nn.Conv2d(cfg.bev_c, cfg.bev_c, 1)
            self.proj = nn.Conv2d(cfg.bev_c, cfg.bev_c, 1)
        else:
            raise ValueError("fusion_mode must be concat/gated/xattn")

    def forward(self, cam_bev, pts_bev):
        """
        cam_bev: (B,C,H,W)
        pts_bev: (B,C,H,W)
        """
        if self.cfg.fusion_mode == "concat":
            x = torch.cat([cam_bev, pts_bev], dim=1)
            return self.fuse(x)

        if self.cfg.fusion_mode == "gated":
            x = torch.cat([cam_bev, pts_bev], dim=1)
            a = self.alpha(x)  # (B,C,H,W)
            # fused = a*cam + (1-a)*pts  (learn per-location reliability)
            fused = a * cam_bev + (1.0 - a) * pts_bev
            return self.out(fused)

        # xattn (toy, but explicit):
        # flatten HW tokens: Q=cam, K/V=pts
        B, C, H, W = cam_bev.shape
        q = self.q(cam_bev).flatten(2).transpose(1, 2)  # (B,HW,C)
        k = self.k(pts_bev).flatten(2)                  # (B,C,HW)
        v = self.v(pts_bev).flatten(2).transpose(1, 2)  # (B,HW,C)

        attn = torch.matmul(q, k) / math.sqrt(C)        # (B,HW,HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                     # (B,HW,C)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.proj(out)
        return out + cam_bev  # residual


# =========================================================
# 6) Detection head (simple Center-style head)
# =========================================================
class CenterHead(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        out_c = cfg.num_classes + 6
        self.net = nn.Sequential(
            nn.Conv2d(cfg.bev_c, cfg.bev_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.bev_c, out_c, 1),
        )

    def forward(self, bev):
        return self.net(bev)  # (B, 7, H, W)


# =========================================================
# 7) Full model: BEVFusion pipeline
# =========================================================
class BEVFusionModel(nn.Module):
    """
    Inputs:
      imgs:        (B,Ncam,3,H,W)
      points:      (B,Npts,4)   (ego frame)
      intrinsics:  (B,Ncam,3,3)
      cam2ego:     (B,Ncam,4,4)

    Output:
      preds: (B,7,Hbev,Wbev)
      aux:   dict of intermediate tensors
    """
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        # camera branch
        self.backbone = SimpleBackbone(cfg)
        self.depth_ctx = DepthContextNet(cfg)
        self.view_tf = CameraViewTransformerLSS(cfg)

        # points branch (lidar/radar)
        self.pts_bev = PointsToBEV(cfg)

        # fusion
        self.fusion = BEVFusion(cfg)

        # head
        self.head = CenterHead(cfg)

    def forward(self, imgs, points, intrinsics, cam2ego):
        img_feats = self.backbone(imgs)                            # (B,N,C,Hf,Wf)
        depth_prob, context = self.depth_ctx(img_feats)            # (B,N,D,Hf,Wf), (B,N,Cctx,Hf,Wf)
        cam_bev = self.view_tf(depth_prob, context, intrinsics, cam2ego)  # (B,bev_c,H,W)

        pts_bev = self.pts_bev(points)                             # (B,bev_c,H,W)

        fused = self.fusion(cam_bev, pts_bev)                      # (B,bev_c,H,W)

        preds = self.head(fused)                                   # (B,7,H,W)

        aux = {"cam_bev": cam_bev, "pts_bev": pts_bev, "fused_bev": fused}
        return preds, aux


# =========================================================
# 8) Smoke test
# =========================================================
def main():
    cfg = Cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVFusionModel(cfg).to(device)

    B = 2
    imgs = torch.randn(B, cfg.num_cams, 3, cfg.img_h, cfg.img_w, device=device)

    # points in ego frame (x,y,z,i)
    Npts = 40000
    points = torch.randn(B, Npts, 4, device=device)
    # make x,y roughly inside range
    points[..., 0] = points[..., 0].clamp(-60, 60)
    points[..., 1] = points[..., 1].clamp(-60, 60)
    points[..., 2] = points[..., 2].clamp(-5, 3)

    # intrinsics (fx,fy,cx,cy) - simple plausible values
    intrinsics = torch.eye(3, device=device).view(1,1,3,3).repeat(B, cfg.num_cams, 1, 1)
    intrinsics[:, :, 0, 0] = 800.0  # fx
    intrinsics[:, :, 1, 1] = 800.0  # fy
    intrinsics[:, :, 0, 2] = cfg.img_w / 2
    intrinsics[:, :, 1, 2] = cfg.img_h / 2

    # cam2ego (identity for demo)
    cam2ego = torch.eye(4, device=device).view(1,1,4,4).repeat(B, cfg.num_cams, 1, 1)

    preds, aux = model(imgs, points, intrinsics, cam2ego)

    print("preds:", preds.shape)
    print("cam_bev:", aux["cam_bev"].shape, "pts_bev:", aux["pts_bev"].shape, "fused:", aux["fused_bev"].shape)


if __name__ == "__main__":
    main()
