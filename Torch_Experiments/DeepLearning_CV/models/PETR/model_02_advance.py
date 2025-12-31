import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================
# 0) utils
# =========================================================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # x in (0,1) -> R
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

def normalize_pc_range(xyz: torch.Tensor, pc_range) -> torch.Tensor:
    """
    xyz: (..., 3) in world/ego coordinates
    pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    return: (..., 3) normalized to [0,1]
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mins = torch.tensor([x_min, y_min, z_min], device=xyz.device, dtype=xyz.dtype)
    maxs = torch.tensor([x_max, y_max, z_max], device=xyz.device, dtype=xyz.dtype)
    return (xyz - mins) / (maxs - mins)

# =========================================================
# 1) Config
# =========================================================
class PETRConfig:
    # cameras & image
    num_cams = 6
    img_h = 256
    img_w = 704

    # backbone/feature
    stride = 16
    embed_dim = 256

    # depth bins
    num_depth = 4
    depth_values = [10.0, 20.0, 30.0, 40.0]  # meters

    # 3D range for normalization
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # transformer
    num_queries = 900
    num_decoder_layers = 6
    num_heads = 8
    ff_dim = 1024
    dropout = 0.1

    # heads
    num_classes = 10
    box_dim = 10  # x,y,z,w,l,h,sin,cos,vx,vy


# =========================================================
# 2) Simple Backbone (single-scale)
# =========================================================
class SimpleBackbone(nn.Module):
    """
    Input: (B*N, 3, H, W)
    Output: (B*N, C, H/stride, W/stride)
    """
    def __init__(self, cfg: PETRConfig):
        super().__init__()
        C = cfg.embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # /2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # /4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# /8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, C, kernel_size=3, stride=2, padding=1), # /16
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3) PETR-like Model
# =========================================================
class PETR(nn.Module):
    def __init__(self, cfg: PETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone(cfg)

        # feature map size (assuming fixed input size)
        self.feat_h = cfg.img_h // cfg.stride
        self.feat_w = cfg.img_w // cfg.stride
        self.num_tokens_per_cam = self.feat_h * self.feat_w

        # make pixel-center grid in image coordinates (u,v)
        # u = (x + 0.5) * stride, v = (y + 0.5) * stride
        ys = torch.arange(self.feat_h, dtype=torch.float32) + 0.5
        xs = torch.arange(self.feat_w, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        uu = uu * cfg.stride
        vv = vv * cfg.stride
        # (Hf, Wf, 2) where last dim is (u,v)
        grid_uv = torch.stack([uu, vv], dim=-1)
        self.register_buffer("grid_uv", grid_uv, persistent=False)

        # depth values buffer
        depth = torch.tensor(cfg.depth_values, dtype=torch.float32).view(1, 1, 1, cfg.num_depth)  # (1,1,1,D)
        self.register_buffer("depth_values", depth, persistent=False)

        # position encoder: input 3*D -> embed_dim
        in_dim = 3 * cfg.num_depth
        self.position_encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )

        # transformer decoder (PyTorch native)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=False,  # (S,B,C)
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)

        # learnable object queries
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        # heads
        self.cls_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.reg_head = nn.Linear(cfg.embed_dim, cfg.box_dim)

        # optional: stabilize token features
        self.feat_ln = nn.LayerNorm(cfg.embed_dim)

    @torch.no_grad()
    def _check_shapes(self, imgs, rots, trans, intrins):
        assert imgs.dim() == 5, "imgs should be (B,N,3,H,W)"
        B, N, C, H, W = imgs.shape
        assert N == self.cfg.num_cams
        assert H == self.cfg.img_h and W == self.cfg.img_w
        assert rots.shape == (B, N, 3, 3)
        assert trans.shape == (B, N, 3)
        assert intrins.shape == (B, N, 3, 3)

    def backproject_to_world(self, rots, trans, intrins):
        """
        Compute world coords for each feature token and each depth bin.

        rots: (B,N,3,3) cam->world rotation  (IMPORTANT!)
        trans:(B,N,3)   cam->world translation
        intrins:(B,N,3,3)

        Returns:
          xyz_world: (B,N,HW,D,3)
        """
        B, N = trans.shape[:2]
        Hf, Wf = self.feat_h, self.feat_w
        HW = Hf * Wf
        D = self.cfg.num_depth

        # (1) grid to homogeneous pixel coordinates: (Hf,Wf,3) => (HW,3)
        uv = self.grid_uv  # (Hf,Wf,2) in image pixels
        ones = torch.ones((Hf, Wf, 1), device=uv.device, dtype=uv.dtype)
        uv1 = torch.cat([uv, ones], dim=-1).view(1, 1, HW, 3)  # (1,1,HW,3)
        uv1 = uv1.repeat(B, N, 1, 1)  # (B,N,HW,3)

        # (2) K^-1 * [u,v,1]^T  => ray direction (camera frame, depth=1)
        K_inv = torch.linalg.inv(intrins)  # (B,N,3,3)
        # (B,N,HW,3) = (B,N,3,3) @ (B,N,HW,3,1)
        ray = (K_inv.unsqueeze(2) @ uv1.unsqueeze(-1)).squeeze(-1)  # (B,N,HW,3)

        # (3) apply depth bins
        # ray: (B,N,HW,3) -> (B,N,HW,D,3)
        ray_d = ray.unsqueeze(3) * self.depth_values.unsqueeze(-1)  # depth_values: (1,1,1,D,1)

        # (4) cam->world transform
        # world = R * cam + t
        R = rots.view(B, N, 1, 1, 3, 3)
        t = trans.view(B, N, 1, 1, 3)
        xyz = (R @ ray_d.unsqueeze(-1)).squeeze(-1) + t  # (B,N,HW,D,3)
        return xyz

    def build_pos_embed(self, xyz_world):
        """
        xyz_world: (B,N,HW,D,3)
        Return: pos_embed (B,N,HW,C)
        """
        B, N, HW, D, _ = xyz_world.shape

        # normalize to [0,1] using pc_range
        xyz01 = normalize_pc_range(xyz_world, self.cfg.pc_range).clamp(0.0, 1.0)
        # inverse sigmoid to expand dynamic range
        xyz_pe = inverse_sigmoid(xyz01)

        # flatten D into channel: (B,N,HW,D,3) -> (B,N,HW,3D)
        xyz_pe = xyz_pe.permute(0, 1, 2, 4, 3).contiguous().view(B, N, HW, 3 * D)

        pos = self.position_encoder(xyz_pe)  # (B,N,HW,C)
        return pos

    def forward(self, imgs, rots, trans, intrins):
        """
        imgs:   (B,N,3,H,W)
        rots:   (B,N,3,3) cam->world rotation  (make sure!)
        trans:  (B,N,3)   cam->world translation
        intrins:(B,N,3,3)
        """
        # self._check_shapes(imgs, rots, trans, intrins)  # uncomment for debugging

        B, N, _, H, W = imgs.shape

        # (1) backbone features
        x = imgs.view(B * N, 3, H, W)
        feat = self.backbone(x)  # (B*N,C,Hf,Wf)
        _, C, Hf, Wf = feat.shape
        assert Hf == self.feat_h and Wf == self.feat_w

        # (2) flatten tokens per camera: (B,N,C,Hf,Wf) -> (B,N,HW,C)
        feat = feat.view(B, N, C, Hf, Wf)
        feat_tok = feat.flatten(3).permute(0, 1, 3, 2).contiguous()  # (B,N,HW,C)
        feat_tok = self.feat_ln(feat_tok)

        # (3) 3D coords + pos embed
        xyz_world = self.backproject_to_world(rots, trans, intrins)  # (B,N,HW,D,3)
        pos = self.build_pos_embed(xyz_world)                        # (B,N,HW,C)

        # (4) fuse: token + 3D pos
        tok = feat_tok + pos  # (B,N,HW,C)

        # (5) build transformer memory: concat cameras
        memory = tok.view(B, N * self.num_tokens_per_cam, C).permute(1, 0, 2).contiguous()
        # memory: (S_mem, B, C)

        # (6) queries
        # use query embedding as initial tgt
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Q,B,C)

        # (7) decode
        hs = self.decoder(tgt=tgt, memory=memory)  # (Q,B,C)
        hs = hs.permute(1, 0, 2).contiguous()      # (B,Q,C)

        # (8) heads
        cls = self.cls_head(hs)  # (B,Q,num_classes)
        box = self.reg_head(hs)  # (B,Q,box_dim)

        return cls, box


# =========================================================
# 4) Smoke test
# =========================================================
def main():
    cfg = PETRConfig()
    model = PETR(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    B = 1
    N = cfg.num_cams
    imgs = torch.randn(B, N, 3, cfg.img_h, cfg.img_w, device=device)

    # mock camera params (cam->world)
    rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    trans = torch.zeros(B, N, 3, device=device)

    # intrinsics
    intrins = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    with torch.no_grad():
        cls, box = model(imgs, rots, trans, intrins)

    print("imgs:", imgs.shape)
    print("cls :", cls.shape, " expected:", (B, cfg.num_queries, cfg.num_classes))
    print("box :", box.shape, " expected:", (B, cfg.num_queries, cfg.box_dim))
    print("✅ forward ok")

if __name__ == "__main__":
    main()
