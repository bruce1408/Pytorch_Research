import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    bev_z = 8          # <- 新增：BEV 高度 bins（论文实现通常不会直接丢 z）


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


class SparseDepthLabeler(nn.Module):
    """
        Project ego-frame points to each camera, build sparse depth labels on feature grid.
        Output:
        depth_labels: (B,N,Hf,Wf) with int in [0..D-1] or -1(ignore)
    """
    def __init__(self, cfg: Cfg, depth_values: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        # depth_values: (1,1,D,1,1) -> (D,)
        dv = depth_values.reshape(-1).detach().cpu()
        self.register_buffer("depth_values_1d", torch.tensor(dv), persistent=False)

        # build bin edges for bucketize (linear bins works well here)
        # edges length D+1
        if len(dv) >= 2:
            step = float(dv[1] - dv[0])
        else:
            step = 1.0
        edges = torch.tensor([dv[0] - step / 2] + [float(x + step / 2) for x in dv], dtype=torch.float32)
        self.register_buffer("bin_edges", edges, persistent=False)

    @torch.no_grad()
    def forward(self, points_ego, intrinsics, cam2ego, feat_hw):
        """
            points_ego: (B,Npts,3 or 4) [x,y,z,(i)]
            intrinsics: (B,N,3,3)
            cam2ego:    (B,N,4,4)
            feat_hw:    (Hf,Wf)
        """
        cfg = self.cfg
        B, Npts, Pdim = points_ego.shape
        Ncam = intrinsics.shape[1]
        Hf, Wf = feat_hw
        device = points_ego.device

        # init: -1 ignore
        labels = torch.full((B, Ncam, Hf, Wf), -1, device=device, dtype=torch.long)

        # ego2cam
        ego2cam = torch.inverse(cam2ego)  # (B,N,4,4)

        # points homogeneous
        pts = points_ego[..., :3]
        ones = torch.ones((B, Npts, 1), device=device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=-1)  # (B,Npts,4)

        stride = float(cfg.feat_stride)
        img_h, img_w = cfg.img_h, cfg.img_w

        # loop over cams (Ncam is small=6, loop is fine and clear)
        for n in range(Ncam):
            T = ego2cam[:, n, :, :]  # (B,4,4)

            # p_cam = T @ p_ego
            p_cam = torch.matmul(T, pts_h.transpose(1, 2)).transpose(1, 2)  # (B,Npts,4)
            x = p_cam[..., 0]
            y = p_cam[..., 1]
            z = p_cam[..., 2]

            # in front
            m_front = z > 0.1

            fx = intrinsics[:, n, 0, 0].view(B, 1)
            fy = intrinsics[:, n, 1, 1].view(B, 1)
            cx = intrinsics[:, n, 0, 2].view(B, 1)
            cy = intrinsics[:, n, 1, 2].view(B, 1)

            u = fx * (x / z.clamp(min=1e-6)) + cx
            v = fy * (y / z.clamp(min=1e-6)) + cy

            # inside image
            m_img = (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1)

            m = m_front & m_img
            if not m.any():
                continue

            # feature indices
            uf = torch.floor(u / stride).long()
            vf = torch.floor(v / stride).long()

            m_feat = (uf >= 0) & (uf < Wf) & (vf >= 0) & (vf < Hf)
            m = m & m_feat
            if not m.any():
                continue

            # For each feature cell, keep nearest depth (min z)
            # We do a "scatter min" using linear index + sort trick.
            # idx = vf*Wf + uf
            idx = (vf * Wf + uf)  # (B,Npts)

            for b in range(B):
                mb = m[b]
                if not mb.any():
                    continue

                idx_b = idx[b, mb]          # (K,)
                z_b = z[b, mb]              # (K,)

                # sort by idx then by depth
                # If we sort by (idx, z), the first occurrence per idx is nearest depth.
                order = torch.argsort(idx_b * 1e6 + z_b)  # stable enough for typical ranges
                idx_s = idx_b[order]
                z_s = z_b[order]

                # pick first for each unique idx
                first = torch.ones_like(idx_s, dtype=torch.bool)
                first[1:] = idx_s[1:] != idx_s[:-1]

                idx_u = idx_s[first]
                z_u = z_s[first]

                # map depth -> bin label
                # bucketize with edges -> [0..D-1]
                d = z_u.clamp(min=float(self.bin_edges[0].item() + 1e-3),
                              max=float(self.bin_edges[-1].item() - 1e-3))
                bin_id = torch.bucketize(d, self.bin_edges.to(device=d.device, dtype=d.dtype)) - 1
                bin_id = bin_id.clamp(0, len(self.depth_values_1d) - 1).long()

                # write to labels[b,n,:,:]
                vf_u = (idx_u // Wf).long()
                uf_u = (idx_u % Wf).long()
                labels[b, n, vf_u, uf_u] = bin_id

        return labels  # (B,N,Hf,Wf)


def depth_ce_loss(depth_logits, depth_labels, ignore_index=-1):
    """
    depth_logits: (B,N,D,Hf,Wf)
    depth_labels: (B,N,Hf,Wf) in [0..D-1] or -1
    """
    B, N, D, Hf, Wf = depth_logits.shape
    logits = depth_logits.view(B * N, D, Hf, Wf)
    labels = depth_labels.view(B * N, Hf, Wf)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)




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
        
        # (B*N, D + Cctx, Hf, Wf)
        y = self.head(x)  
        y = y.view(B, N, y.shape[1], Hf, Wf)

        # (B, N, D, Hf, Wf)
        depth_logits = y[:, :, :self.cfg.cam_depth_bins]  
        
        # (B, N, Cctx, Hf, Wf)
        context = y[:, :, self.cfg.cam_depth_bins: ]       

        depth_prob = F.softmax(depth_logits, dim=2)
        return depth_logits, depth_prob, context


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
class CameraViewTransformerLSSVoxel(nn.Module):
    """
    LSS Lift-Splat but splat into (x,y,z) voxel volume first, then collapse Z->BEV.
    """
    def __init__(self, cfg: Cfg, reduce="mean", chunk=2_000_000):
        super().__init__()
        self.cfg = cfg
        self.reduce = reduce
        self.chunk = int(chunk)

        # depth bin centers (meters) - 你也可以改成 LID/SID
        z_min = 1.0
        z_max = 60.0
        self.register_buffer(
            "depth_values",
            torch.linspace(z_min, z_max, cfg.cam_depth_bins).view(1, 1, cfg.cam_depth_bins, 1, 1),
            persistent=False,
        )

        # voxel->BEV: S2C then conv
        self.bev_proj = nn.Sequential(
            nn.Conv2d(cfg.cam_ctx_c * cfg.bev_z, cfg.bev_c, 1, bias=False),
            nn.BatchNorm2d(cfg.bev_c),
            nn.ReLU(inplace=True),
        )

    def _make_pixel_grid(self, Hf, Wf, device, dtype):
        s = self.cfg.feat_stride
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * s
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * s
        v, u = torch.meshgrid(ys, xs, indexing="ij")
        u = u.view(1, 1, 1, Hf, Wf)
        v = v.view(1, 1, 1, Hf, Wf)
        return u, v

    @staticmethod
    def _flatten_with_batch_offset(linear_idx, B, HWZ):
        # linear_idx: (B, M)
        # -> global: (B*M,)
        offset = (torch.arange(B, device=linear_idx.device).view(B, 1) * HWZ)
        return (linear_idx + offset).reshape(-1)

    def forward(self, depth_prob, context, intrinsics, cam2ego):
        """
            depth_prob: (B, N, D, Hf, Wf)
            context:    (B, N, Cctx, Hf, Wf)
            intrinsics: (B, N, 3, 3)
            cam2ego:    (B, N, 4, 4)
            
            return:
                cam_bev: (B, bev_c, Hbev, Wbev)
        """
        cfg = self.cfg
        B, N, D, Hf, Wf = depth_prob.shape
        device, dtype = depth_prob.device, depth_prob.dtype

        # ---- A) frustum points (camera) ----
        # (1, 1, 1, Hf, Wf)
        u, v = self._make_pixel_grid(Hf, Wf, device, dtype) 
        
        # (1, 1, D, 1, 1)
        Z = self.depth_values.to(device=device, dtype=dtype)  

        fx = intrinsics[:, :, 0, 0].view(B, N, 1, 1, 1)
        fy = intrinsics[:, :, 1, 1].view(B, N, 1, 1, 1)
        cx = intrinsics[:, :, 0, 2].view(B, N, 1, 1, 1)
        cy = intrinsics[:, :, 1, 2].view(B, N, 1, 1, 1)

        Xc = (u - cx) / fx * Z
        Yc = (v - cy) / fy * Z
        Zc = Z.expand_as(Xc)

        # (B, N, D, Hf, Wf, 3)
        pts_cam = torch.stack([Xc, Yc, Zc], dim=-1)              
        ones = torch.ones_like(pts_cam[..., :1])
        
        # (B, N, D, Hf, Wf, 4)
        pts_cam_h = torch.cat([pts_cam, ones], dim=-1)           
        
        # ---- B) cam -> ego ----
        T = cam2ego.view(B, N, 1, 1, 1, 4, 4)
        pts_ego = torch.matmul(T, pts_cam_h.unsqueeze(-1)).squeeze(-1)[..., :3]  # (B,N,D,Hf,Wf,3)

        # ---- C) lift features (depth-weighted context) ----
        # (B,N,D,Cctx,Hf,Wf)
        feat_lift = depth_prob.unsqueeze(3) * context.unsqueeze(2)

        # ---- D) voxelize: map (x,y,z)->(ix,iy,iz) ----
        x_min, y_min, z_min, x_max, y_max, z_max = cfg.pc_range
        mx = (x_max - x_min) / cfg.bev_w
        my = (y_max - y_min) / cfg.bev_h
        mz = (z_max - z_min) / cfg.bev_z

        x = pts_ego[..., 0]
        y = pts_ego[..., 1]
        z = pts_ego[..., 2]

        ix = torch.floor((x - x_min) / mx).long()
        iy = torch.floor((y - y_min) / my).long()
        iz = torch.floor((z - z_min) / mz).long()

        valid = (ix >= 0) & (ix < cfg.bev_w) & \
                (iy >= 0) & (iy < cfg.bev_h) & \
                (iz >= 0) & (iz < cfg.bev_z)

        # linear voxel index inside one batch volume:
        # idx = ((iz * H) + iy) * W + ix   where H=bev_h, W=bev_w
        voxel_ind = ((iz * cfg.bev_h + iy) * cfg.bev_w + ix)  # (B,N,D,Hf,Wf)

        # ---- E) flatten for scatter ----
        # features -> (B, M, Cctx)
        feat = feat_lift.permute(0, 1, 2, 4, 5, 3).contiguous()  # (B,N,D,Hf,Wf,Cctx)
        feat = feat.view(B, -1, cfg.cam_ctx_c)                   # M=N*D*Hf*Wf

        ind = voxel_ind.view(B, -1)            # (B, M)
        msk = valid.view(B, -1)                # (B, M)

        HWZ = cfg.bev_w * cfg.bev_h * cfg.bev_z

        # ---- F) scatter-add (vectorized across batch) ----
        # global index to fuse all batches in one tensor
        gind = self._flatten_with_batch_offset(ind, B, HWZ)      # (B*M,)
        gmsk = msk.reshape(-1)                                   # (B*M,)
        feat_flat = feat.reshape(-1, cfg.cam_ctx_c)              # (B*M, Cctx)

        # apply mask
        gind = gind[gmsk]
        feat_flat = feat_flat[gmsk]

        # allocate flattened voxel volume: (Cctx, B*HWZ)
        vox = torch.zeros(cfg.cam_ctx_c, B * HWZ, device=device, dtype=dtype)

        # chunk to control memory peak
        for s in range(0, gind.numel(), self.chunk):
            e = min(s + self.chunk, gind.numel())
            vox.index_add_(dim=1, index=gind[s:e], source=feat_flat[s:e].t())

        # mean reduce option (divide by counts)
        if self.reduce == "mean":
            cnt = torch.zeros(1, B * HWZ, device=device, dtype=dtype)
            ones = torch.ones((gind.numel(),), device=device, dtype=dtype)
            for s in range(0, gind.numel(), self.chunk):
                e = min(s + self.chunk, gind.numel())
                cnt.index_add_(dim=1, index=gind[s:e], source=ones[s:e].view(1, -1))
            vox = vox / cnt.clamp(min=1.0)

        # reshape back: (B, Cctx, Z, H, W)
        vox = vox.view(cfg.cam_ctx_c, B, cfg.bev_z, cfg.bev_h, cfg.bev_w).permute(1, 0, 2, 3, 4).contiguous()

        # ---- G) collapse Z -> BEV (S2C) ----
        # (B, Cctx*Z, H, W) -> bev_c
        bev_in = vox.view(B, cfg.cam_ctx_c * cfg.bev_z, cfg.bev_h, cfg.bev_w)
        cam_bev = self.bev_proj(bev_in)  # (B, bev_c, H, W)
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

        # bev网格id
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
        self.view_tf = CameraViewTransformerLSSVoxel(cfg)

        # points branch (lidar/radar)
        self.pts_bev = PointsToBEV(cfg)

        self.depth_labeler = SparseDepthLabeler(cfg, self.view_tf.depth_values)

        # 特征融合 + fusion
        self.fusion = BEVFusion(cfg)

        # head
        self.head = CenterHead(cfg)

    def forward(self, imgs, points, intrinsics, cam2ego):
        
        # (B, N, C, Hf, Wf)
        img_feats = self.backbone(imgs)                            
        
        # 获取深度估计和语义特征 (B, N, D, Hf, Wf), (B, N, Cctx, Hf, Wf)
        depth_logits, depth_prob, context = self.depth_ctx(img_feats)
        
        # (B, bev_c, H, W)
        cam_bev = self.view_tf(depth_prob, context, intrinsics, cam2ego)      

        # 激光雷达数据进行处理 -> (B,bev_c,H,W)
        pts_bev = self.pts_bev(points)                             

        # 特征融合 -> (B,bev_c,H,W)
        fused = self.fusion(cam_bev, pts_bev)                      
        
        # (B,7,H,W)
        preds = self.head(fused)                                   

        aux = {
            "cam_bev": cam_bev,
            "pts_bev": pts_bev,
            "fused_bev": fused,
            "depth_logits": depth_logits,   # (B,N,D,Hf,Wf)
        }
        
        return preds, aux


def gaussian2d(shape, sigma=1.0, device="cpu", dtype=torch.float32):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1, device=device, dtype=dtype),
        torch.arange(-n, n + 1, device=device, dtype=dtype),
        indexing="ij"
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h

def draw_gaussian(heatmap, center, radius=2, sigma=1.0):
    """
    heatmap: (H,W)
    center: (x,y)
    """
    H, W = heatmap.shape
    x, y = center
    x, y = int(x), int(y)
    if x < 0 or x >= W or y < 0 or y >= H:
        return heatmap

    diameter = 2 * radius + 1
    g = gaussian2d((diameter, diameter), sigma=sigma, device=heatmap.device, dtype=heatmap.dtype)

    left, right = min(x, radius), min(W - 1 - x, radius)
    top, bottom = min(y, radius), min(H - 1 - y, radius)

    masked_h = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
    masked_g = g[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    heatmap[y - top:y + bottom + 1, x - left:x + right + 1] = torch.maximum(masked_h, masked_g)
    return heatmap

class CenterDetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian_focal(self, pred, target):
        # pred/target: (B,1,H,W), pred already sigmoid
        pos = target.eq(1.0)
        neg = target.lt(1.0)
        neg_w = torch.pow(1 - target[neg], 4)

        loss = 0.0
        pos_loss = torch.log(pred[pos].clamp(min=1e-6)) * torch.pow(1 - pred[pos], 2)
        neg_loss = torch.log((1 - pred[neg]).clamp(min=1e-6)) * torch.pow(pred[neg], 2) * neg_w

        num_pos = pos.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    def forward(self, preds, gt_centers):
        """
        preds: (B, 7, H, W)  [hm, dx,dy,logw,logl,sin,cos]
        gt_centers: (B,2)  in BEV pixel coords (x,y)
        """
        B, C, H, W = preds.shape
        pred_hm = torch.sigmoid(preds[:, 0:1])
        pred_reg = preds[:, 1:]

        target_hm = torch.zeros((B, 1, H, W), device=preds.device, dtype=preds.dtype)
        target_reg = torch.zeros((B, 6, H, W), device=preds.device, dtype=preds.dtype)
        mask = torch.zeros((B, 1, H, W), device=preds.device, dtype=preds.dtype)

        for b in range(B):
            cx, cy = gt_centers[b]
            cx_i, cy_i = int(cx.item()), int(cy.item())
            if 0 <= cx_i < W and 0 <= cy_i < H:
                draw_gaussian(target_hm[b, 0], (cx_i, cy_i), radius=2, sigma=1.0)
                # dx,dy
                target_reg[b, 0, cy_i, cx_i] = cx - cx_i
                target_reg[b, 1, cy_i, cx_i] = cy - cy_i
                # toy size + yaw
                target_reg[b, 2, cy_i, cx_i] = math.log(4.0)
                target_reg[b, 3, cy_i, cx_i] = math.log(2.0)
                target_reg[b, 4, cy_i, cx_i] = 0.0  # sin
                target_reg[b, 5, cy_i, cx_i] = 1.0  # cos
                mask[b, 0, cy_i, cx_i] = 1.0

        loss_hm = self.gaussian_focal(pred_hm, target_hm)
        loss_reg = F.l1_loss(pred_reg * mask, target_reg * mask, reduction="sum") / (mask.sum() + 1e-6)
        return loss_hm + 2.0 * loss_reg, {"loss_hm": loss_hm.item(), "loss_reg": loss_reg.item()}


class ToyBEVFusionDataset(Dataset):
    def __init__(self, cfg: Cfg, length=200, npts=20000):
        self.cfg = cfg
        self.length = length
        self.npts = npts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cfg = self.cfg

        # imgs: (Ncam,3,H,W)
        imgs = torch.randn(cfg.num_cams, 3, cfg.img_h, cfg.img_w)

        # points: (Npts,4) ego frame
        points = torch.randn(self.npts, 4)
        points[:, 0] = points[:, 0].clamp(cfg.pc_range[0] - 10, cfg.pc_range[3] + 10)
        points[:, 1] = points[:, 1].clamp(cfg.pc_range[1] - 10, cfg.pc_range[4] + 10)
        points[:, 2] = points[:, 2].clamp(cfg.pc_range[2], cfg.pc_range[5])

        # intrinsics (simple plausible)
        intr = torch.eye(3).unsqueeze(0).repeat(cfg.num_cams, 1, 1)
        intr[:, 0, 0] = 800.0
        intr[:, 1, 1] = 800.0
        intr[:, 0, 2] = cfg.img_w / 2
        intr[:, 1, 2] = cfg.img_h / 2

        # cam2ego: identity for toy
        cam2ego = torch.eye(4).unsqueeze(0).repeat(cfg.num_cams, 1, 1)

        # GT center in BEV pixel coords (x,y)
        # random inside bev map
        gt_center = torch.tensor([
            torch.randint(0, cfg.bev_w, (1,)).float().item(),
            torch.randint(0, cfg.bev_h, (1,)).float().item(),
        ], dtype=torch.float32)

        return {
            "imgs": imgs,
            "points": points,
            "intrinsics": intr,
            "cam2ego": cam2ego,
            "gt_center": gt_center
        }

def collate_fn(batch):
    # stack into B dimension
    imgs = torch.stack([b["imgs"] for b in batch], dim=0)             # (B,Ncam,3,H,W)
    points = torch.stack([b["points"] for b in batch], dim=0)         # (B,Npts,4)
    intr = torch.stack([b["intrinsics"] for b in batch], dim=0)       # (B,Ncam,3,3)
    cam2ego = torch.stack([b["cam2ego"] for b in batch], dim=0)       # (B,Ncam,4,4)
    gt_center = torch.stack([b["gt_center"] for b in batch], dim=0)   # (B,2)
    return {"imgs": imgs, "points": points, "intrinsics": intr, "cam2ego": cam2ego, "gt_center": gt_center}


def train_one_epoch(model, loader, optimizer, det_criterion, device, w_depth=1.0):
    model.train()
    total = 0.0
    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        points = batch["points"].to(device)
        intr = batch["intrinsics"].to(device)
        cam2ego = batch["cam2ego"].to(device)
        gt_center = batch["gt_center"].to(device)

        optimizer.zero_grad(set_to_none=True)

        preds, aux = model(imgs, points, intr, cam2ego)

        # det loss
        loss_det, det_log = det_criterion(preds, gt_center)

        # ========= 这就是你问的两行：放在训练 loop 里 =========
        B, N, D, Hf, Wf = aux["depth_logits"].shape
        depth_labels = model.depth_labeler(points, intr, cam2ego, feat_hw=(Hf, Wf))
        loss_depth = depth_ce_loss(aux["depth_logits"], depth_labels)
        # =======================================================

        loss = loss_det + w_depth * loss_depth
        loss.backward()
        optimizer.step()

        total += loss.item()
        if it % 10 == 0:
            print(f"[train] it={it:03d} loss={loss.item():.4f} det={loss_det.item():.4f} depth={loss_depth.item():.4f} "
                  f"(hm={det_log['loss_hm']:.4f}, reg={det_log['loss_reg']:.4f})")
    return total / max(len(loader), 1)

@torch.no_grad()
def validate(model, loader, det_criterion, device, w_depth=1.0):
    model.eval()
    total = 0.0
    for it, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        points = batch["points"].to(device)
        intr = batch["intrinsics"].to(device)
        cam2ego = batch["cam2ego"].to(device)
        gt_center = batch["gt_center"].to(device)

        preds, aux = model(imgs, points, intr, cam2ego)
        loss_det, _ = det_criterion(preds, gt_center)

        B, N, D, Hf, Wf = aux["depth_logits"].shape
        depth_labels = model.depth_labeler(points, intr, cam2ego, feat_hw=(Hf, Wf))
        loss_depth = depth_ce_loss(aux["depth_logits"], depth_labels)

        loss = loss_det + w_depth * loss_depth
        total += loss.item()
    return total / max(len(loader), 1)


# =========================================================
# 8) Smoke test
# =========================================================
def main():
    cfg = Cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVFusionModel(cfg).to(device)

    B = 2
    imgs = torch.randn(B, cfg.num_cams, 3, cfg.img_h, cfg.img_w, device=device)

    # points in ego frame (x,y,z,i) 模拟 激光雷达数据
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



def main_train():
    cfg = Cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = BEVFusionModel(cfg).to(device)

    # attach labeler (方案A：模型里持有 labeler，但不在 forward 里算 labels)
    model.depth_labeler = SparseDepthLabeler(cfg, model.view_tf.depth_values).to(device)

    # data
    train_ds = ToyBEVFusionDataset(cfg, length=200)
    val_ds   = ToyBEVFusionDataset(cfg, length=50)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # optim / loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    det_criterion = CenterDetLoss().to(device)

    for epoch in range(3):
        tr = train_one_epoch(model, train_loader, optimizer, det_criterion, device, w_depth=1.0)
        va = validate(model, val_loader, det_criterion, device, w_depth=1.0)
        print(f"Epoch {epoch}: train={tr:.4f}, val={va:.4f}")

if __name__ == "__main__":
    main_train()


# if __name__ == "__main__":
    # main()
