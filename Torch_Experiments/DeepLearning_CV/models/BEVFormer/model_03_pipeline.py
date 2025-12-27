import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 1) 你的 Config（补几个必要项）
# =========================================================
class Config:
    bev_h = 50
    bev_w = 50
    bev_z = 4
    embed_dims = 256
    num_heads = 8
    num_cams = 6

    # [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [-50, -50, -5, 50, 50, 3]

    # detection config
    num_classes = 1
    out_stride = 1  # 这里 BEV 是直接 50x50，不再下采样


# =========================================================
# 2) 你的 SpatialCrossAttention / TemporalSelfAttention / Encoder / BEVFormer
#    （保持你发的版本，不再重复粘贴；只增加一个输出 reshape 函数）
# =========================================================

# -----------------------------
# 这里粘贴你当前的四个类：
# SpatialCrossAttention / TemporalSelfAttention / BEVFormerEncoderLayer / BEVFormer
# -----------------------------

# !!! 为了让这段代码可独立运行，我把你发的 encoder 部分也放进来（原样）：

class SpatialCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn_weights = nn.Linear(cfg.embed_dims, cfg.num_cams * cfg.bev_z)
        self.output_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)

    def get_reference_points(self, H, W, Z, bs, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = (ref_y.reshape(-1)[None] / H)
        ref_x = (ref_x.reshape(-1)[None] / W)

        ref_x = ref_x.unsqueeze(-1).repeat(1, 1, Z)
        ref_y = ref_y.unsqueeze(-1).repeat(1, 1, Z)

        ref_z = torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float32, device=device)
        ref_z = (ref_z / Z).view(1, 1, Z).repeat(1, H * W, 1)

        ref_3d = torch.stack((ref_x, ref_y, ref_z), dim=-1)
        return ref_3d.repeat(bs, 1, 1, 1)

    def point_sampling(self, reference_points, img_feats, lidar2img):
        B, Nq, Z, _ = reference_points.shape
        Ncam = img_feats.shape[1]
        C = img_feats.shape[2]
        Hf, Wf = img_feats.shape[-2], img_feats.shape[-1]

        pc = self.cfg.pc_range
        ref = reference_points.clone()
        ref[..., 0] = ref[..., 0] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1] = ref[..., 1] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2] = ref[..., 2] * (pc[5] - pc[2]) + pc[2]
        ref = torch.cat([ref, torch.ones_like(ref[..., :1])], dim=-1)  # (B,Nq,Z,4)

        ref = ref.view(B, 1, Nq * Z, 4).repeat(1, Ncam, 1, 1)
        l2i = lidar2img.view(B, Ncam, 1, 4, 4)
        cam = torch.matmul(l2i, ref.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        depth = cam[..., 2:3]
        valid = depth > eps
