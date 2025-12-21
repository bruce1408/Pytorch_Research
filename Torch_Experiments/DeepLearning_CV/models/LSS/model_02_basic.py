import torch
import torch.nn as nn

class LSS_Core(nn.Module):
    def __init__(self, grid_conf, data_conf, outC):
        super().__init__()
        self.grid_conf = grid_conf  # BEV 网格配置 (xbound, ybound, zbound)
        self.data_conf = data_conf  # 图像配置 (input_size, D)
        self.outC = outC            # 输出特征维度 C
        
        # 1. 对应论文 3.1 节 & 截图1: 深度 D 和 上下文 C
        # 论文提到 D=41 (4m-45m, 间隔1m)
        self.D = data_conf['D'] 
        self.C = outC 
        
        # 模拟 backbone 输出头 (CamEncode)
        # 输出通道 = D (深度概率) + C (上下文特征)
        self.cam_encode = nn.Conv2d(512, self.D + self.C, kernel_size=1) 

        # 预计算视锥 (Frustum)，对应截图1提到的 "D·H·W 的点云"
        self.frustum = self.create_frustum()

    def create_frustum(self):
        """
        生成固定的视锥网格 (D, H, W, 3)
        3 代表 (u, v, d) -> (图像横坐标, 图像纵坐标, 深度)
        """
        # 深度网格 d
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        
        # 图像网格 u, v
        H, W = self.data_conf['img_size']
        xs = torch.linspace(0, W - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, H - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)

        # 堆叠得到初始视锥
        frustum = torch.stack((xs, ys, ds.expand(D, H, W)), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrinsics):
        """
        几何计算：将 (u,v,d) 结合内参外参，投影到车身坐标系 (x,y,z)
        """
        B, N, _ = trans.shape 
        
        # 1. 准备点云 (B, N, D, H, W, 3)
        points = self.frustum.unsqueeze(0).unsqueeze(0).expand(B, N, *self.frustum.shape)
        
        # 2. 这里的数学公式： P_ego = Rot * (K_inv * P_pixel * depth) + Trans
        # (为了代码简洁，此处省略具体的矩阵乘法实现，假设 output 已经是转换好的 xyz)
        # 实际代码会利用 intrinsics.inverse() 和 rots/trans 进行批量矩阵运算
        
        # 模拟转换后的坐标 (x, y, z)
        coords_xyz = torch.randn(B, N, self.D, *self.data_conf['img_size'], 3)
        return coords_xyz

    def get_depth_feat(self, x):
        """
        对应论文 3.1 节的核心公式 (1): c_d = alpha_d * c
        """
        B, N, C_in, H, W = x.shape
        x = x.view(B*N, C_in, H, W)
        
        # 1. 卷积输出 D+C 通道
        x = self.cam_encode(x) # (B*N, D+C, H, W)
        
        # 2. 拆分深度 (alpha) 和 上下文 (c)
        depth_logits = x[:, :self.D] # (B*N, D, H, W)
        context = x[:, self.D:]      # (B*N, C, H, W)
        
        # 3. Softmax 得到深度概率分布 alpha
        depth_probs = depth_logits.softmax(dim=1)
        
        # 4. 外积操作 (Broadcasting)
        # Context (C) * Depth (D) -> (D, C)
        context = context.unsqueeze(1)    # (B*N, 1, C, H, W)
        depth_probs = depth_probs.unsqueeze(2) # (B*N, D, 1, H, W)
        
        # 得到 3D 特征体 (B*N, D, C, H, W)
        cam_feats = context * depth_probs 

        # ==========================================
        # 修复部分 (Fix): 调整形状以匹配 voxel_pooling
        # ==========================================
        
        # 1. 把 B*N 拆开 -> (B, N, D, C, H, W)
        cam_feats = cam_feats.view(B, N, self.D, self.C, H, W)
        
        # 2. 把 C 移动到最后 (Permute) -> (B, N, D, H, W, C)
        # 必须把 C 放到最后，因为 voxel_pooling 里需要把 (B,N,D,H,W) 展平，而保持 C 不变
        cam_feats = cam_feats.permute(0, 1, 2, 4, 5, 3)
        return cam_feats.contiguous()  # <--- 加上这一句，把内存理顺
        
        # return cam_feats

    def voxel_pooling(self, geom_feats, x, y, z):
        """
        对应论文 4.2 节: 截锥池化累积和技巧 (CumSum Trick)
        该技巧避免了填充导致的内存浪费，并利用排序加速。
        """
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W

        # 1. 展平 (Flatten) 所有点
        geom_feats = geom_feats.view(Nprime, C)
        x = x.view(Nprime)
        y = y.view(Nprime)
        z = z.view(Nprime)

        # 2. 过滤掉飞出 BEV 边界的点 (Range Check)
        # 假设 grid 是 200x200
        mask = (x >= 0) & (x < 200) & (y >= 0) & (y < 200) & (z >= 0) & (z < 1)
        x, y, z = x[mask], y[mask], z[mask]
        geom_feats = geom_feats[mask]

        # 3. 计算唯一索引 (Linear Indexing)
        # 就像把二维表格拉直一样
        indices = x * 200 + y + z * (200*200) # 这里只演示 xy 平面，实际可能有 batch 偏移
        
        # --- 论文核心技巧：排序 + 累积和 (Sort + CumSum) ---
        
        # 3.1 排序 (Sorting)
        # 论文提到的 "通过按 bin id 对所有点进行排序"
        ranks = indices.argsort()
        x, y, z = x[ranks], y[ranks], z[ranks]
        geom_feats = geom_feats[ranks]
        indices = indices[ranks]

        # 3.2 累积和 (Cumulative Sum)
        # 标记边界：如果 index 变了，说明换了一个 voxel
        keep = torch.ones_like(indices, dtype=torch.bool)
        keep[:-1] = (indices[1:] != indices[:-1])
        
        # 论文提到的 "对所有特征进行累积和"
        cumsum = torch.cumsum(geom_feats, 0)
        
        # 论文提到的 "减去 bin 分区的边界处的累积和值"
        # 这样就得到了每个 voxel 内部的 Sum Pooling 结果
        cumsum = cumsum[keep]
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        # 4. 填回 BEV 网格
        final_bev = torch.zeros((1, 200, 200, C), device=x.device)
        # 这里的 indices[keep] 就是去重后的有效 voxel 坐标
        final_bev.view(-1, C)[indices[keep]] = cumsum
        
        return final_bev.permute(0, 3, 1, 2) # (B, C, H, W)

    def forward(self, x, rots, trans, intrinsics):
        # Lift
        cam_feats = self.get_depth_feat(x)
        
        # Geometry
        geom_xyz = self.get_geometry(rots, trans, intrinsics)
        
        # 离散化坐标 (量化为 Grid Index)
        geom_xyz = ((geom_xyz - (self.grid_conf['xbound'][0])) / self.grid_conf['xbound'][2]).long()
        x_idx, y_idx, z_idx = geom_xyz[..., 0], geom_xyz[..., 1], geom_xyz[..., 2]
        
        # Splat (Pooling)
        bev_feature = self.voxel_pooling(cam_feats, x_idx, y_idx, z_idx)
        
        return bev_feature
    
    
    
if __name__ == "__main__":
    print("正在初始化 LSS 模型...")

    # --- 配置参数 (模拟 NuScenes 设置) ---
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],   # X轴范围: -50m 到 50m, 间隔 0.5m -> 200个格子
        'ybound': [-50.0, 50.0, 0.5],   # Y轴范围: 同上 -> 200个格子
        'zbound': [-10.0, 10.0, 20.0],  # Z轴: 这里简化为 1 层
        'dbound': [4.0, 45.0, 1.0],     # 深度: 4m 到 45m, 间隔 1m
    }
    
    # 假设图片经过 backbone 缩小了 16 倍 (原始图片可能是 256x704)
    # 所以特征图大小设为 16x44
    data_conf = {
        'img_size': (16, 44), 
        'dbound': grid_conf['dbound'],
        'D': 41 # (45-4)/1 = 41 个深度桶
    }
    
    # 实例化模型
    out_channels = 64
    lss = LSS_Core(grid_conf, data_conf, outC=out_channels)

    # --- 生成伪造输入数据 ---
    B = 1   # Batch size
    N = 6   # 6个环视摄像头
    C_backbone = 512 # 假设 ResNet 出来的特征维度
    H_feat, W_feat = data_conf['img_size']

    # 1. 输入特征 (B, N, 512, 16, 44)
    fake_features = torch.randn(B, N, C_backbone, H_feat, W_feat)
    
    # 2. 外参/内参 (这里仅占位，实际计算在 get_geometry 里被 mock 掉了)
    fake_rots = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)
    fake_trans = torch.zeros(1, 1, 3).expand(B, N, 3)
    fake_intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)

    print(f"输入特征形状: {fake_features.shape}")
    print("开始前向传播 (Forward)...")

    # --- 运行模型 ---
    bev_map = lss(fake_features, fake_rots, fake_trans, fake_intrinsics)

    # --- 打印结果 ---
    print("-" * 30)
    print("运行成功！")
    print(f"输出 BEV 特征形状: {bev_map.shape}")
    print("-" * 30)
    
    # 简单的验证
    expected_shape = (B, out_channels, 200, 200)
    if bev_map.shape == expected_shape:
        print("✅ 形状检查通过：符合 (Batch, Channel, X_grid, Y_grid)")
        print(f"   X_grid = (50 - (-50)) / 0.5 = 200")
    else:
        print(f"❌ 形状检查失败，期望 {expected_shape}，实际 {bev_map.shape}")