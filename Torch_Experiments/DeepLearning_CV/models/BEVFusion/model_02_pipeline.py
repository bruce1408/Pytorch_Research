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
    bev_z = 8          # <- 新增：BEV 高度 bins


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
    稀疏深度标签生成器
    
    功能：
    1. 将点云（ego坐标系）投影到每个相机视角
    2. 计算每个点在特征图上的位置
    3. 将点的真实深度值转换为深度bin标签（0到D-1）
    4. 生成稀疏的深度标签图，用于监督深度估计网络
    
    输出：
    depth_labels: (B, N, Hf, Wf) 
        - B: batch size
        - N: 相机数量
        - Hf, Wf: 特征图高度和宽度
        - 值：深度bin索引 [0..D-1] 或 -1（忽略/无效位置）
    """
    def __init__(self, cfg: Cfg, depth_values: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        
        # ========== 步骤1：存储深度值 ==========
        # depth_values: (1,1,D,1,1) -> (D,) - 深度bin的中心值
        # 例如：D=48，深度范围[0.5, 80.0]米，均匀分布
        # depth_values = [0.5, 1.2, 1.9, ..., 80.0]
        dv = depth_values.reshape(-1).detach().cpu()
        self.register_buffer("depth_values_1d", dv, persistent=False)
        
        # ========== 步骤2：构建深度bin边界 ==========
        # 为了使用bucketize函数，需要构建bin的边界
        # 例如：如果depth_values = [1.0, 2.0, 3.0]
        # 则bin_edges = [0.5, 1.5, 2.5, 3.5]
        # 这样可以将深度值映射到对应的bin索引
        if len(dv) >= 2:
            step = float(dv[1] - dv[0])  # 计算bin的步长
        else:
            step = 1.0
        
        # 构建边界：每个bin的左右边界
        edges = torch.tensor(
            [dv[0] - step / 2] + [float(x + step / 2) for x in dv], 
            dtype=torch.float32
        )
        
        self.register_buffer("bin_edges", edges, persistent=False)

    @torch.no_grad()
    def forward(self, points_ego, intrinsics, cam2ego, feat_hw):
        """
        前向传播：生成深度标签
        
        Args:
            points_ego: (B, Npts, 3 or 4) 
                - 点云在ego坐标系中的坐标 [x, y, z, (intensity)]
                - 这些是LiDAR/Radar的真实点云数据
            intrinsics: (B, N, 3, 3)
                - 相机内参矩阵（每个相机一个）
            cam2ego: (B, N, 4, 4)
                - 相机到ego坐标系的变换矩阵
            feat_hw: (Hf, Wf)
                - 特征图的高度和宽度
        
        Returns:
            labels: (B, N, Hf, Wf)
                - 深度标签图，每个位置的值是深度bin索引 [0..D-1]
                - -1 表示该位置没有有效的深度标签（点云未覆盖）
        """
        cfg = self.cfg
        B, Npts, Pdim = points_ego.shape
        Ncam = intrinsics.shape[1]
        Hf, Wf = feat_hw
        device = points_ego.device
        
        # ========== 初始化：所有位置标记为无效 ==========
        labels = torch.full((B, Ncam, Hf, Wf), -1, 
                           device=device, dtype=torch.long)
        
        # ========== 坐标变换：ego -> camera ==========
        # 计算ego到相机的变换矩阵
        ego2cam = torch.inverse(cam2ego)  # (B, N, 4, 4)
        
        # 将点转换为齐次坐标
        pts = points_ego[..., :3]  # 只取x, y, z
        ones = torch.ones((B, Npts, 1), device=device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=-1)  # (B, Npts, 4)
        
        stride = float(cfg.feat_stride)  # 特征图下采样倍数（通常是4）
        img_h, img_w = cfg.img_h, cfg.img_w
        
        # ========== 遍历每个相机 ==========
        for n in range(Ncam):
            T = ego2cam[:, n, :, :]  # (B, 4, 4) - 当前相机的变换矩阵
            
            # 将点从ego坐标系变换到相机坐标系
            # p_cam = T @ p_ego
            p_cam = torch.matmul(T, pts_h.transpose(1, 2)).transpose(1, 2)  # (B, Npts, 4)
            x = p_cam[..., 0]  # 相机坐标系X
            y = p_cam[..., 1]  # 相机坐标系Y
            z = p_cam[..., 2]  # 深度（相机坐标系Z）
            
            # ========== 有效性检查1：点在相机前方 ==========
            m_front = z > 0.1  # 只保留深度>0.1米的点（避免数值不稳定）
            
            # ========== 投影到图像平面 ==========
            # 使用相机内参将3D点投影到2D图像
            fx = intrinsics[:, n, 0, 0].view(B, 1)  # 焦距x
            fy = intrinsics[:, n, 1, 1].view(B, 1)  # 焦距y
            cx = intrinsics[:, n, 0, 2].view(B, 1)  # 主点x
            cy = intrinsics[:, n, 1, 2].view(B, 1)  # 主点y
            
            # 透视投影：u = fx * (x/z) + cx, v = fy * (y/z) + cy
            u = fx * (x / z.clamp(min=1e-6)) + cx  # 像素坐标u
            v = fy * (y / z.clamp(min=1e-6)) + cy  # 像素坐标v
            
            # ========== 有效性检查2：点在图像范围内 ==========
            m_img = (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1)
            
            # 综合有效性：前方 + 图像范围内
            m = m_front & m_img
            if not m.any():
                continue  # 如果没有有效点，跳过这个相机
            
            # ========== 计算特征图坐标 ==========
            # 将像素坐标转换为特征图坐标（考虑下采样）
            uf = torch.floor(u / stride).long()  # 特征图u坐标
            vf = torch.floor(v / stride).long()  # 特征图v坐标
            
            # ========== 有效性检查3：点在特征图范围内 ==========
            m_feat = (uf >= 0) & (uf < Wf) & (vf >= 0) & (vf < Hf)
            m = m & m_feat
            if not m.any():
                continue
            
            # ========== 关键步骤：为每个特征图位置选择最近的深度 ==========
            # 策略：如果多个点投影到同一个特征图位置，选择深度最小的（最近的）
            # 这符合"最近点优先"的原则
            
            # 计算线性索引：将2D坐标转换为1D索引
            idx = (vf * Wf + uf)  # (B, Npts)
            
            # 对每个batch分别处理
            for b in range(B):
                mb = m[b]  # 当前batch的有效点掩码
                if not mb.any():
                    continue
                
                idx_b = idx[b, mb]  # (K,) - 有效点的特征图索引
                z_b = z[b, mb]      # (K,) - 有效点的深度值
                
                # ========== 排序技巧：实现"scatter min" ==========
                # 目标：对于每个唯一的特征图位置，找到深度最小的点
                # 方法：按 (索引, 深度) 排序，然后取每个索引的第一个
                
                # 排序键：idx * 1e6 + z
                # 这样先按索引排序，索引相同时按深度排序
                # 深度小的排在前面
                order = torch.argsort(idx_b * 1e6 + z_b)
                idx_s = idx_b[order]  # 排序后的索引
                z_s = z_b[order]      # 排序后的深度
                
                # ========== 提取每个唯一索引的第一个点 ==========
                # 因为已经排序，相同索引的点中，深度最小的在第一个
                first = torch.ones_like(idx_s, dtype=torch.bool)
                first[1:] = idx_s[1:] != idx_s[:-1]  # 标记每个唯一索引的第一个
                
                idx_u = idx_s[first]  # 唯一的特征图索引
                z_u = z_s[first]       # 对应的深度值（最近的）
                
                # ========== 将深度值映射到深度bin ==========
                # 使用bucketize函数将连续深度值映射到离散的bin索引
                d = z_u.clamp(
                    min=float(self.bin_edges[0].item() + 1e-3),
                    max=float(self.bin_edges[-1].item() - 1e-3)
                )
                # bucketize返回的是bin索引（1-based），需要减1转为0-based
                bin_id = torch.bucketize(d, self.bin_edges.to(device=d.device, dtype=d.dtype)) - 1
                bin_id = bin_id.clamp(0, len(self.depth_values_1d) - 1).long()
                
                # ========== 写入标签 ==========
                # 将bin索引写入对应的特征图位置
                vf_u = (idx_u // Wf).long()  # 从线性索引恢复v坐标
                uf_u = (idx_u % Wf).long()   # 从线性索引恢复u坐标
                labels[b, n, vf_u, uf_u] = bin_id
        
        return labels  # (B, N, Hf, Wf)

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
        # (B,N,D,Cctx,Hf,Wf) 视锥体特征
        feat_lift = depth_prob.unsqueeze(3) * context.unsqueeze(2)

        # ---- D) voxelize: map (x,y,z)->(ix,iy,iz) ----
        x_min, y_min, z_min, x_max, y_max, z_max = cfg.pc_range
        mx = (x_max - x_min) / cfg.bev_w
        my = (y_max - y_min) / cfg.bev_h
        mz = (z_max - z_min) / cfg.bev_z

        x = pts_ego[..., 0]
        y = pts_ego[..., 1]
        z = pts_ego[..., 2]

        # 计算 bev 网格
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
    """
    点云到BEV特征图的转换模块
    
    功能：
    将LiDAR/Radar点云数据转换为BEV（Bird's Eye View）特征图。
    这是BEVFusion中处理点云数据的核心模块。
    
    算法流程：
    1. 将每个点云的特征（x, y, z, intensity）通过MLP编码为特征向量
    2. 将3D点投影到2D BEV网格（忽略Z轴，只使用X-Y平面）
    3. 使用scatter操作将点特征聚合到对应的BEV网格单元
    4. 对每个网格单元内的点特征求平均（mean pooling）
    5. 通过卷积层将聚合后的特征投影到目标通道数
    
    输入：点云数据 (B, Npts, 4) - [x, y, z, intensity]
    输出：BEV特征图 (B, bev_c, Hbev, Wbev)
    """
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        
        # ========== 点特征编码器 (Point Feature Embedding) ==========
        # 功能：将每个点的原始特征（x, y, z, intensity）编码为高维特征向量
        # 输入：4维点特征 [x, y, z, intensity]
        # 输出：pts_embed_c 维的特征向量
        # 
        # 为什么需要编码？
        # - 原始点特征（坐标+强度）信息有限
        # - 通过MLP学习更丰富的特征表示
        # - 为后续的特征融合做准备
        self.mlp = nn.Sequential(
            nn.Linear(cfg.pts_feat_in, cfg.pts_embed_c),  # 第一层：4 -> 80
            nn.ReLU(inplace=True),                         # 激活函数
            nn.Linear(cfg.pts_embed_c, cfg.pts_embed_c),  # 第二层：80 -> 80
            nn.ReLU(inplace=True),                         # 激活函数
        )
        
        # ========== BEV特征投影层 ==========
        # 功能：将聚合后的点特征投影到目标BEV通道数
        # 输入：(B, pts_embed_c, H, W) - 聚合后的点特征
        # 输出：(B, bev_c, H, W) - 最终BEV特征图
        # 
        # 为什么需要投影？
        # - 统一特征维度，便于与相机BEV特征融合
        # - 进一步提取和压缩特征
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.pts_embed_c, cfg.bev_c, 1),  # 1x1卷积：降维/升维
            nn.BatchNorm2d(cfg.bev_c),                  # 批归一化：稳定训练
            nn.ReLU(inplace=True),                      # 激活函数
        )


    
    
    def forward(self, points):
        """
            前向传播：将点云转换为BEV特征图
            
            Args:
                points: (B, Npts, 4) 
                    - B: batch size
                    - Npts: 点云中的点数（每帧可能不同，但这里假设已padding到相同长度）
                    - 4: [x, y, z, intensity]
                        * x, y, z: 点在世界坐标系（ego frame）中的坐标（单位：米）
                        * intensity: 点云强度值（LiDAR反射强度或Radar信号强度）
            
            Returns:
                pts_bev: (B, bev_c, Hbev, Wbev)
                    - BEV特征图，每个像素代表一个BEV网格单元
                    - 特征值是该网格内所有点特征的聚合结果
        """
        cfg = self.cfg
        B, Np, Fdim = points.shape
        device, dtype = points.device, points.dtype

        # ========== 步骤1：计算BEV网格的分辨率 ==========
        # 从点云范围（pc_range）和BEV尺寸计算每个网格单元的大小
        # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        x_min, y_min, _, x_max, y_max, _ = cfg.pc_range
        
        # mx: X方向每个网格单元的大小（米）
        # 例如：x范围100米，bev_w=128 -> mx = 100/128 ≈ 0.78米/像素
        mx = (x_max - x_min) / cfg.bev_w
        
        # my: Y方向每个网格单元的大小（米）
        my = (y_max - y_min) / cfg.bev_h
        HW = cfg.bev_h * cfg.bev_w

        # ========== 步骤2：提取点的X-Y坐标 ==========
        # 注意：BEV是俯视图，只使用X-Y坐标，忽略Z轴
        x = points[..., 0]
        y = points[..., 1]
        
        # ========== 步骤3：计算每个点对应的BEV网格索引 ==========
        # 将连续的世界坐标转换为离散的网格索引
        # 公式：grid_idx = floor((coord - min) / grid_size)
        # 例如：x=0, x_min=-50, mx=0.78 -> ix = floor((0-(-50))/0.78) = floor(64.1) = 64
        ix = torch.floor((x - x_min) / mx).long()
        iy = torch.floor((y - y_min) / my).long()


        # ========== 步骤4：有效性检查 ==========
        # 过滤掉超出BEV范围的点（这些点可能是噪声或无效数据）
        valid = (ix >= 0) & (ix < cfg.bev_w) & (iy >= 0) & (iy < cfg.bev_h)

        # ========== 步骤5：计算线性索引 ==========
        # 将2D网格索引 (iy, ix) 转换为1D线性索引
        # 公式：linear_idx = iy * W + ix
        # 这样可以将2D BEV网格展平为1D数组，便于后续的scatter操作
        ind = (iy * cfg.bev_w + ix)
        
        
        # 只保留 valid 的点
        if valid.any():
            # 取有效点的 (b_idx, ind)
            b_idx = torch.arange(B, device=device).view(B, 1).expand(B, Np)
            b_idx = b_idx[valid]               # (M,)
            ind_v = ind[valid].clamp(0, HW-1)  # (M,)

            # ========== 步骤6：点特征编码 ==========
            # 将每个点的原始特征（x, y, z, intensity）通过MLP编码为高维特征，只对有效点做 MLP，避免对 padding 点白算
            # 输入：(B, Np, 4) -> 输出：(B, Np, pts_embed_c)
            # 例如：(B, 10000, 4) -> (B, 10000, 80)
            pts_v = points[valid]              # (M, 4)
            pts_emb_v = self.mlp(pts_v)        # (M, C)
            C = pts_emb_v.shape[1]

            # 全局索引：0..B*HW-1
            g = b_idx * HW + ind_v             # (M,)

            # ========== 步骤7：Scatter聚合操作 ==========
            # 将点特征聚合到对应的BEV网格单元
            # 策略：使用 mean pooling（平均值池化）
            # 实现方式：先求和，再除以计数
        
            # 初始化聚合容器
            # bev_sum: 存储每个网格单元内所有点特征的和
            bev_sum = torch.zeros((B * HW, C), device=device, dtype=pts_emb_v.dtype)
            bev_cnt = torch.zeros((B * HW,), device=device, dtype=pts_emb_v.dtype)

            bev_sum.index_add_(0, g, pts_emb_v)
            bev_cnt.index_add_(0, g, torch.ones_like(g, dtype=pts_emb_v.dtype))

            # ========== 步骤8：计算平均值 ==========
            # mean = sum / count
            # clamp(min=1.0) 防止除零（如果某个网格没有点，count=0，则结果为0）
            bev_mean = bev_sum / bev_cnt.clamp(min=1.0).unsqueeze(-1)  # (B*HW, C)
            
            # ========== 步骤9：重塑为2D BEV特征图 ==========
            # (B, pts_embed_c, H*W) -> (B, pts_embed_c, H, W)
            pts_bev = bev_mean.view(B, HW, C).permute(0, 2, 1).contiguous()
            pts_bev = pts_bev.view(B, C, cfg.bev_h, cfg.bev_w)
        
        else:    
            # 没有有效点
            C = cfg.pts_embed_c
            pts_bev = torch.zeros((B, C, cfg.bev_h, cfg.bev_w), device=device, dtype=dtype)

        # ========== 步骤10：特征投影 ==========
        # 通过卷积层将特征投影到目标通道数
        # (B, pts_embed_c, H, W) -> (B, bev_c, H, W)
        # 例如：(B, 80, 128, 128) -> (B, 128, 128, 128)
        pts_bev = self.proj(pts_bev)
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
        
        # (B,HW,C)
        q = self.q(cam_bev).flatten(2).transpose(1, 2)  
        
        # (B,C,HW)
        k = self.k(pts_bev).flatten(2)               
        
        # (B,HW,C)   
        v = self.v(pts_bev).flatten(2).transpose(1, 2)  

        # (B,HW,HW)
        attn = torch.matmul(q, k) / math.sqrt(C)        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)   
        
        # (B,HW,C)                  
        out = out.transpose(1, 2).view(B, C, H, W)
        
        out = self.proj(out)
        
        # residual
        return out + cam_bev  


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
