import torch
import torch.nn as nn

class LSS_Core(nn.Module):
    """
    Lift-Splat-Shoot (LSS) 核心模块实现
    
    主要功能：
    1. Lift: 将 2D 图像特征 + 深度分布 转换为 3D 视锥特征 (Frustum Features)。
    2. Splat: 将 3D 特征点投影并聚合到 BEV (鸟瞰图) 网格上 (Voxel Pooling)。
    """
    def __init__(self, grid_conf, data_conf, input_channels, num_classes):
        """
        初始化参数:
        grid_conf: BEV 网格配置 (范围、分辨率)
        data_conf: 数据配置 (图像尺寸、深度范围)
        input_channels: 输入特征图的通道数 (通常来自 Backbone, 如 512)
        num_classes: 输出 BEV 特征的通道数 (通常为 64 或 num_classes)
        """
        super().__init__()
        self.grid_conf = grid_conf
        self.data_conf = data_conf
        
        # --- 1. Lift 模块配置 ---
        self.D = data_conf['D'] # 深度桶的数量 (例如 41 个, 对应 4m~45m)
        self.C = 64             # 最终生成的 3D 特征的通道数 (Context Dimension)
        
        # CamEncode 卷积层: 
        # 作用: 作为一个简单的 "Head"，将 Backbone 提取的特征转换为 "深度分布" 和 "上下文特征"。
        # 输入: (B*N, 512, H, W) -> 输出: (B*N, D + C, H, W)
        self.cam_encode = nn.Conv2d(input_channels, self.D + self.C, kernel_size=1) 

        # 预先生成视锥网格 (Frustum Grid)
        # 这是一个固定的 (D, H, W, 3) 张量，代表每个像素在不同深度下的 3D 坐标模板。
        self.frustum = self.create_frustum()

    def create_frustum(self):
        """
        生成视锥点云模板 (Frustum Point Cloud Template)
        对应论文 3.1 节提到的 "D·H·W 的点云"。
        
        frustum 张量就是一本 “花名册”。 只要你给出一个索引 [d, h, w]，
        它就查表告诉你这个体素中心在原始图像的 u 坐标是多少，v 坐标是多少，以及它代表的物理深度 d 是多少
        相机坐标系的参数化,严格说他是相机视锥体上的“离散参数空间”
        相机视锥体的离散索引空间 + 每个索引对应的几何语义注释（u, v, depth）
        
        self.frustum[40][15][43]
        tensor([43., 15., 44.])
        
        返回: 
            frustum: shape (D, H, W, 3) 
            其中最后一维 3 代表 (u, v, d) -> (图像横坐标, 图像纵坐标, 深度值)
        """
        # 1. 生成深度网格 (Depth Grid)
        # 例如: [4.0, 5.0, ..., 44.0] -> shape (D, 1, 1)
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        
        # 2. 生成图像坐标网格 (Image Coordinate Grid)
        H, W = self.data_conf['img_size']
        
        # xs: 沿着宽度的坐标 [0, 1, ..., W-1] -> shape (1, 1, W) -> expand to (D, H, W) 从0 --> 43
        xs = torch.linspace(0, W - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
        
        # ys: 沿着高度的坐标 [0, 1, ..., H-1] -> shape (1, H, 1) -> expand to (D, H, W) 从0 --> 15
        ys = torch.linspace(0, H - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)
    
        # 3. 堆叠 (Stack)
        # 将 u, v, d 堆叠在一起，形成每个点的初始参数
        frustum = torch.stack((xs, ys, ds.expand(D, H, W)), -1)
        
        # 注册为 Parameter 但不需要梯度 (因为它只是坐标模板)
        return nn.Parameter(frustum, requires_grad=False)
    
    def create_frustum_with_meshgrid(self):
        
        # 1. 准备三个维度的 1D 向量
        # Depth (D)
        d_vec = torch.arange(*self.data_conf['dbound'], dtype=torch.float)
        
        # Height (H) -> 对应 y
        h_vec = torch.linspace(0, self.data_conf['img_size'][0] - 1, self.data_conf['img_size'][0], dtype=torch.float)
        
        # Width (W)  -> 对应 x
        w_vec = torch.linspace(0, self.data_conf['img_size'][1] - 1, self.data_conf['img_size'][1], dtype=torch.float)
        
        # 2. 使用 meshgrid 生成网格
        # indexing='ij' 表示输出形状遵循输入的顺序: (D, H, W)
        # grid_d: (D, H, W)
        # grid_h: (D, H, W) -> 对应 y 坐标
        # grid_w: (D, H, W) -> 对应 x 坐标
        grid_d, grid_h, grid_w = torch.meshgrid(d_vec, h_vec, w_vec, indexing='ij')
        
        # 3. 堆叠 (Stack)
        # 原代码的顺序是 (xs, ys, ds)，即 (Width, Height, Depth)
        # 所以这里必须是 (grid_w, grid_h, grid_d)
        frustum = torch.stack((grid_w, grid_h, grid_d), dim=-1)
        
        return frustum



    def get_geometry(self, rots, trans, intrinsics):
        """
        [真实实现] 几何投影 (Geometry Projection)
        这个函数，通过对视锥frustum 存的 [u,v,d] 是 “图像参数 + 深度”（不是 xyz);
        然后通过投影，计算在自车坐标系下的3D点xyz
        
        输入:
            rots:       (B, N, 3, 3) 相机到车身的旋转矩阵
            trans:      (B, N, 3)    相机到车身的平移向量
            intrinsics: (B, N, 3, 3) 相机内参矩阵
            N 表示有 N 个视角的摄像头sensor
            
        输出:
            coords_xyz: (B, N, D, H, W, 3) 车身坐标系下的点云
            对每个 batch b、相机 n、深度 bin d、像素格点 (h,w)，
            都有一个 ego 坐标系下的 3D 点 xyz = points_ego[b,n,d,h,w]
        """
        # 1. 维度准备
        B, N, _ = trans.shape # N 表示 6个视角的摄像头sensor
        D, H, W, _ = self.frustum.shape
        
        # 将 frustum 扩展到 Batch 和 Camera 维度，并移动到正确的设备(GPU), 相机坐标系的参数化
        # frustum shape: (D, H, W, 3) -> (B, N, D, H, W, 3)
        # 内容: points[..., 0]是u, [..., 1]是v, [..., 2]是d
        points = self.frustum.to(trans.device).unsqueeze(0).unsqueeze(0).expand(B, N, D, H, W, 3)

        # 2. 提取像素坐标和深度
        # 深度 d: (B, N, D, H, W)
        depth = points[..., 2] 
        
        # 构造齐次像素坐标 [u, v, 1]
        # shape: (B, N, D, H, W, 3)
        points_uv1 = torch.stack([
            points[..., 0], 
            points[..., 1], 
            torch.ones_like(depth)
        ], dim=-1)

        # 3. 展平空间维度以进行批量矩阵乘法
        # 为了高效计算，我们将 D, H, W 展平为 NumPoints
        NumPoints = D * H * W
        
        # (B, N, D, H, W, 3) -> (B, N, NumPoints, 3) -> (B, N, 3, NumPoints)
        # 这里转置是为了符合矩阵乘法 (3x3) @ (3xN) 的格式
        points_uv1_flat = points_uv1.view(B, N, NumPoints, 3).permute(0, 1, 3, 2)
        
        # 深度也展平: (B, N, 1, NumPoints) 用于广播乘法
        depth_flat = depth.view(B, N, 1, NumPoints)

        # =================================================
        # 阶段 1: 像素坐标 -> 相机坐标 (Unprojection)
        # 公式: P_cam = d * K_inv * P_pix
        # =================================================
        
        # 计算内参的逆矩阵: (B, N, 3, 3)
        intrinsics_inv = torch.inverse(intrinsics)
        
        # 矩阵乘法: K_inv @ P_uv1
        # (B, N, 3, 3) @ (B, N, 3, NumPoints) -> (B, N, 3, NumPoints)
        points_cam = torch.matmul(intrinsics_inv, points_uv1_flat)
        
        # 乘以深度可以转换为 --> 相机坐标系 (B, N, 3, NumPoints)
        # 相机坐标系 (camera frame) 下的 3D 点云（但还是平铺形状）
        points_cam = points_cam * depth_flat


        # =================================================
        # 阶段 2: 相机坐标 -> 车身坐标 (Extrinsic)
        # 公式: P_ego = Rot * P_cam + Trans
        # =================================================
        
        # 旋转: Rot @ P_cam
        # (B, N, 3, 3) @ (B, N, 3, NumPoints) -> (B, N, 3, NumPoints)
        points_ego = torch.matmul(rots, points_cam)
        
        # 平移: + Trans, 这里得到的是转化为 --> 自车坐标系
        # trans: (B, N, 3) -> (B, N, 3, 1) 以便广播相加
        points_ego = points_ego + trans.view(B, N, 3, 1)

        # 4. 恢复形状
        # (B, N, 3, NumPoints) -> (B, N, 3, D, H, W)
        points_ego = points_ego.view(B, N, 3, D, H, W)
        
        # 调整最后维度的顺序: (B, N, D, H, W, 3)
        # 最后的 3 代表 (x, y, z)
        points_ego = points_ego.permute(0, 1, 3, 4, 5, 2)

        return points_ego
    
    def get_geometry_fake(self, rots, trans, intrinsics):
        """
        几何投影 (Geometry Projection)
        作用: 将视锥点从 "图像坐标系 (u,v,d)" 转换到 "车身坐标系 (x,y,z)"。
        
        参数:
            rots: 相机旋转矩阵 (B, N, 3, 3)
            trans: 相机平移向量 (B, N, 3)
            intrinsics: 相机内参矩阵 (B, N, 3, 3)
        """
        # ... (真实场景下这里会有复杂的矩阵乘法：P_ego = Rot * K_inv * P_pix * d + Trans) ...
        
        B, N, _ = trans.shape
        # [模拟] 为了演示运行，这里返回符合 BEV 范围的随机坐标
        # 乘以 10 是为了让点散布在 -20m~20m 附近，确保能落入 BEV 网格 (-50m~50m)
        coords_xyz = torch.randn(B, N, self.D, *self.data_conf['img_size'], 3) * 10
        return coords_xyz

    def get_cam_feats(self, x):
        """
        生成 3D 特征体
        [核心步骤 1: Lift]
        对应论文公式 (1): c_d = alpha_d * c
        作用: 将 2D 图像特征提升为 3D 体积特征。
        """
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W) # 合并 Batch 和 Camera 维度，以便并行处理
        
        # 1. 神经网络提取特征
        # 输入: (B*N, 512, H, W)
        # 输出: (B*N, 105, H, W)  假设 D=41, C=64 -> 41+64=105
        x = self.cam_encode(x) 
        
        # 2. 拆分 Depth (概率) 和 Context (特征)
        depth_logits = x[:, :self.D] # 前 D 个通道是深度 Logits
        context = x[:, self.D:]      # 后 C 个通道是语义特征
        
        # 3. 计算深度概率分布 (Softmax)
        # 对应论文提到的 "alpha"
        depth_probs = depth_logits.softmax(dim=1)
        
        # 4. 外积操作 (Outer Product)
        # 这一步没有可学习参数，纯粹是数学变换。
        # Context (C) * Depth (D) -> Volume (D, C)
        context = context.unsqueeze(1)    # (B*N, 1, C, H, W)
        depth_probs = depth_probs.unsqueeze(2) # (B*N, D, 1, H, W)
        
        # 生成 3D 特征体
        # 含义: 如果某像素在深度 d 的概率很高，那么该处的特征 c 就会被高亮保留。
        cam_feats = context * depth_probs # Shape: (B*N, D, C, H, W)
        
        return cam_feats

    def voxel_pooling(self, geom_feats, x, y, z):
        """
        [核心步骤 2: Splat]
        对应论文 4.2 节: "截锥池化累积和技巧" (CumSum Trick)
        作用: 将散乱的 3D 特征点 "拍扁" 并聚合到规则的 BEV 网格中。
        x，y，z 表示3d空间对应的格子的index，也就是每一个点属于哪一个bev格子
        """
        # geom_feats: (B, N, D, H, W, C) -> 所有特征点
        # x, y, z:    (B, N, D, H, W)    -> 对应的网格索引
        
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W # 总点数 (例如 1*6*41*16*44 ≈ 17万个点)

        # 1. 展平 (Flatten)
        # 使用 contiguous() + view() 也可以，但 reshape 更稳健
        geom_feats = geom_feats.contiguous().view(Nprime, C)
        x = x.reshape(Nprime)
        y = y.reshape(Nprime)
        z = z.reshape(Nprime)

        # 2. 过滤 (Filtering)
        # 丢弃那些投影后飞出 BEV 边界的点
        # 假设 grid 大小是 200x200
        mask = (x >= 0) & (x < 200) & (y >= 0) & (y < 200) & (z >= 0) & (z < 1)
        x, y, z = x[mask], y[mask], z[mask]
        geom_feats = geom_feats[mask]

        # 3. 排序 (Sorting) - 关键加速步骤
        # 计算每个点的一维索引 (Flat Index)
        indices = y * 200 + x # 这里忽略了 Batch 维度 (假设 B=1)
        
        # argsort: 把将要落入同一个格子的点排在一起
        ranks = indices.argsort()
        x, y, z = x[ranks], y[ranks], z[ranks]
        geom_feats = geom_feats[ranks]
        indices = indices[ranks]

        # 4. 累积和 (CumSum) - 论文核心技巧
        # 这一步是为了避免使用慢速的 for 循环来做加法
        
        # keep 标记了每个新 voxel 的起始位置 (当 index 发生变化时为 True)
        keep = torch.ones_like(indices, dtype=torch.bool)
        keep[:-1] = (indices[1:] != indices[:-1])
        
        # 先对所有特征做累加
        cumsum = torch.cumsum(geom_feats, 0)
        # 取出边界处的累加值
        cumsum = cumsum[keep]
        
        # 做差分: 当前边界值 - 上一个边界值 = 该 voxel 内部所有点的和
        # (Concatenate 是为了处理第一个元素)
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        # 5. 填回 BEV 图片 (Scatter) C=64
        final_bev = torch.zeros((1, 200, 200, C), device=x.device)
        
        # 只在有数据的格子填入计算好的 sum
        if cumsum.shape[0] > 0:
            # indices[keep] 得到了去重后的、有效的 voxel 索引
            final_bev.view(-1, C)[indices[keep]] = cumsum
        
        # 调整维度顺序以符合 PyTorch 标准 (B, C, H, W)
        return final_bev.permute(0, 3, 1, 2)

    def forward(self, x, rots, trans, intrinsics):
        """
        前向传播流程
        """
        # x shape: (B, N, 512, H, W) -> 来自 Backbone 的特征
        B, N, C, H, W = x.shape
        
        # --- Step 1: Lift (图像 -> 3D 特征) ---
        # 输出: (B*N, D, C, H, W)
        cam_feats = self.get_cam_feats(x)
        
        # [关键调整] 维度重排以适配 Splat
        # 我们需要把 C (64) 放到最后，因为 voxel_pooling 是对空间点做操作
        # (B*N, D, C, H, W) -> view -> (B, N, D, C, H, W) -> permute -> (B, N, D, H, W, C)
        cam_feats = cam_feats.view(B, N, self.D, self.C, H, W)
        cam_feats = cam_feats.permute(0, 1, 2, 4, 5, 3) 
        
        # --- Step 2: Geometry (计算每个特征点的 3D 坐标) ---
        # 输出: (B, N, D, H, W, 3) 的 xyz 坐标 -> 自车坐标系
        geom_xyz = self.get_geometry(rots, trans, intrinsics)
        
        # --- Step 3: Splat (3D 特征 -> BEV 地图) ---
        
        # 坐标离散化 (Quantization): 物理坐标 -> 网格索引
        # 公式: index = (value - min_bound) / interval
        # 这里从3d坐标变成了网格的下标，除以每个格子的大小，最后得到的是当前位置所在第几个格子
        geom_xyz = ((geom_xyz - (self.grid_conf['xbound'][0])) / self.grid_conf['xbound'][2]).long()
        x_idx, y_idx, z_idx = geom_xyz[..., 0], geom_xyz[..., 1], geom_xyz[..., 2]
        
        # 执行 Voxel Pooling
        bev_feature = self.voxel_pooling(cam_feats, x_idx, y_idx, z_idx)
        
        return bev_feature
    
if __name__ == "__main__":
    print("正在初始化 LSS 模型...")

    # BEV 网格设置: 100m x 100m 范围, 0.5m 分辨率 -> 200x200 网格
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],   
        'ybound': [-50.0, 50.0, 0.5],   
        'zbound': [-10.0, 10.0, 20.0], 
        'dbound': [4.0, 45.0, 1.0],     
    }
    
    # 数据设置: 缩小后的图像尺寸
    data_conf = {
        'img_size': (16, 44), 
        'dbound': grid_conf['dbound'],
        'D': 41 # (45 - 4) / 1 = 41
    }
    
    # 模拟参数
    C_backbone = 512 # 假设输入特征通道
    out_channels = 64 # 输出 BEV 特征通道
    
    # 初始化模型
    lss = LSS_Core(grid_conf, data_conf, input_channels=C_backbone, num_classes=out_channels)

    # 构造伪数据
    B, N = 1, 6
    H_feat, W_feat = data_conf['img_size']

    fake_features = torch.randn(B, N, C_backbone, H_feat, W_feat)
    fake_rots = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)
    fake_trans = torch.zeros(1, 1, 3).expand(B, N, 3)
    fake_intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(B, N, 3, 3)

    print(f"输入特征形状: {fake_features.shape}")
    print("开始前向传播 (Forward)...")

    # 运行
    bev_map = lss(fake_features, fake_rots, fake_trans, fake_intrinsics)

    print("-" * 30)
    print("运行成功！")
    print(f"输出 BEV 特征形状: {bev_map.shape}") # 应该输出 (1, 64, 200, 200)
    print("-" * 30)