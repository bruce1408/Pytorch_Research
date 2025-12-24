import torch
import torch.nn as nn

# ==========================================
# 1. 模拟组件 (Mock Modules) - 为了让代码跑起来
# ==========================================

class MockBackbone(nn.Module):
    """模拟 ResNet/Swin，将图像下采样 16 倍"""
    def __init__(self, out_channels=512):
        super().__init__()
        # 假设输入 3 通道，输出 512 通道，下采样 16 倍 (stride=16)
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=16, padding=1)

    def forward(self, x):
        return self.conv(x)

class MockBEVEncoder(nn.Module):
    """模拟 BEV 编码器 (如 ResNet)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class MockHead(nn.Module):
    """模拟检测头 (CenterPoint Head)"""
    def __init__(self, in_channels):
        super().__init__()
        # 假设预测 10 个类别 + 8 个回归量 = 18 通道
        self.conv = nn.Conv2d(in_channels, 18, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 2. 核心算法实现 (LSS + BEVDet)
# ==========================================

class LSSViewTransformer(nn.Module):
    def __init__(self, grid_conf, out_channels):
        super(LSSViewTransformer, self).__init__()
        self.grid_conf = grid_conf
        self.dx, self.bx, self.nx = self.gen_dx_bx(self.grid_conf['xbound'], 
                                                   self.grid_conf['ybound'], 
                                                   self.grid_conf['zbound'])
        
        # 深度预测网络 # 41
        self.D = int((grid_conf['dbound'][1] - grid_conf['dbound'][0]) / grid_conf['dbound'][2])
        
        # 41 + 64
        self.depth_net = nn.Conv2d(512, self.D + out_channels, kernel_size=1)
        
        self.out_channels = out_channels

    def gen_dx_bx(self, xbound, ybound, zbound):
        
        # 每个格子的物理尺寸（长、宽、高）
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        
        # 第一个格子的中心点在物理世界中的坐标
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        
        # 在 X, Y, Z 方向上分别有多少个格子
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        
        return dx, bx, nx

    def create_frustum(self):
        # 假设特征图大小为 16x44
        H_feat, W_feat = 16, 44 
        
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat, W_feat)
        D, _, _ = ds.shape
        
        xs = torch.linspace(0, W_feat - 1, W_feat, dtype=torch.float).view(1, 1, W_feat).expand(D, H_feat, W_feat)
        ys = torch.linspace(0, H_feat - 1, H_feat, dtype=torch.float).view(1, H_feat, 1).expand(D, H_feat, W_feat)

        # 堆叠 (D, H, W, 3)
        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape
        
        # 1. 生成视锥 (D, H, W, 3)
        frustum = self.create_frustum().to(trans.device)
        
        # 扩展维度 (B, N, D, H, W, 3)
        points = frustum.view(1, 1, *frustum.shape).repeat(B, N, 1, 1, 1, 1)

        # 2. 抵消图像增强
        points = points - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # 3. 图像 -> 相机 -> 自车
        # 实际上是在执行 (u, v, d) -> (u*d, v*d, d) 的变换，
        # 这是从像素坐标 (u,v) 和深度 d 映射到相机坐标系下点的前置步骤。
        points = points.squeeze(-1)
        
        # 3. 图像 -> 相机 (利用透视关系恢复 X, Y)
        # 现在 points 是 6维，最后一维是 3 (u, v, d)
        uv = points[..., :2]
        d  = points[..., 2:3]
        
        # Xc = u * d, Yc = v * d
        uv_scaled = uv * d
        points = torch.cat((uv_scaled, d), dim=-1)
        
        # 4. 相机 -> 自车
        # 这里再次 unsqueeze 进行矩阵乘法，算完再 squeeze 回来
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def voxel_pooling(self, geom_feats, x, y, z):
        # ============================================
        # 【关键修正】先全部展平，避免维度不匹配问题
        # ============================================
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W

        # 将 x, y, z 展平为 (Nprime, )
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        # 将特征展平为 (Nprime, C)
        geom_feats = geom_feats.reshape(-1, C)

        # 1. 坐标平移并量化
        x = ((x - self.bx[0]) / self.dx[0]).long()
        y = ((y - self.bx[1]) / self.dx[1]).long()
        z = ((z - self.bx[2]) / self.dx[2]).long()
        
        # 2. 过滤出界点, nx 网格数目，x表示当前网格的id，不能越界
        valid = (x >= 0) & (x < self.nx[0]) & \
                (y >= 0) & (y < self.nx[1]) & \
                (z >= 0) & (z < self.nx[2])
        
        x = x[valid]
        y = y[valid]
        z = z[valid]
        geom_feats = geom_feats[valid]

        # 3. 排序 (Sort)
        # x + y * 128 + z * (125 * 125)
        ranks =  x + y * self.nx[0] + z * (self.nx[0] * self.nx[1])
        
        # 对ranks进行排序
        sort_idx = ranks.argsort()
        
        x, y, z, ranks, geom_feats = x[sort_idx], y[sort_idx], z[sort_idx], ranks[sort_idx], geom_feats[sort_idx]

        # 4. CumSum (前缀和)
        keep = torch.ones_like(ranks, dtype=torch.bool)
        keep[:-1] = (ranks[1:] != ranks[:-1])
        
        cumsum = torch.cumsum(geom_feats, dim=0)
        cumsum = cumsum[keep]
        
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        # 5. Scatter 回 BEV 网格 -> [1, C, 1, 125, 125]
        final_bev = torch.zeros((1, C, self.nx[2], self.nx[1], self.nx[0]),
                                device=geom_feats.device)
        
        if x.shape[0] > 0:
            final_bev[0, :, z[keep], y[keep], x[keep]] = cumsum.permute(1, 0)

        # 压缩 Z 轴
        return final_bev.sum(2) 

    def forward(self, img_feats, rots, trans, intrins, post_rots, post_trans):
        # ============================================
        # 【这就是报错说找不到的函数，必须包含在类里面】
        # ============================================
        
        # [1, 6, 512, 16, 44]
        B, N, C, H, W = img_feats.shape  
        print(f"  [LSS] 输入特征形状: {img_feats.shape} (B, N, C, H, W)")
        
        # === Lift === 
        # [6, 64 + 41 = 105, 16, 44]
        x = self.depth_net(img_feats.view(B*N, C, H, W))
        
        depth_digit = x[:, :self.D].softmax(dim=1)
        
        # tran_feat [6, 64, 16, 44]
        tran_feat = x[:, self.D:]  
        
        # [6, 41, 16, 44]
        print(f"  [LSS] 预测深度形状: {depth_digit.shape} (BN, D, H, W)")  
        
        # [6, 64, 16, 44] 
        print(f"  [LSS] 预测语义形状: {tran_feat.shape} (BN, C_out, H, W)")
        
        outer = depth_digit.unsqueeze(1) * tran_feat.unsqueeze(2)
        outer = outer.view(B, N, self.out_channels, self.D, H, W)
        feat_outer = outer.permute(0, 1, 3, 4, 5, 2) 
        
        # [1, 6, 41, 16, 44, 64]
        print(f"  [LSS] 视锥特征体形状 (Lift后): {outer.shape} (B, N, D, H, W, C)")

        # === Geometry ===
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        print(f"  [LSS] 3D坐标点形状: {geom.shape} (B, N, D, H, W, 3)")
        
        # === Splat ===
        x = self.voxel_pooling(feat_outer, geom[..., 0], geom[..., 1], geom[..., 2])
        print(f"  [LSS] BEV特征形状 (Splat后): {x.shape} (B, C, H_bev, W_bev)")
        return x
    

class BEVDet(nn.Module):
    def __init__(self, grid_conf):
        super(BEVDet, self).__init__()
        self.out_channels = 64 # BEV 特征通道数
        
        # 1. 图像 Backbone (ResNet-like)
        self.img_backbone = MockBackbone(out_channels=512)
        
        # 2. LSS 转换器
        self.view_transformer = LSSViewTransformer(grid_conf, self.out_channels)
        
        # 3. BEV Encoder
        self.bev_encoder = MockBEVEncoder(in_channels=64, out_channels=256)
        
        # 4. 检测头
        self.head = MockHead(in_channels=256)

    def bev_data_augmentation(self, x):
        # 简单的模拟：不做实际操作，仅打印
        # 实际上这里会对 BEV 特征图进行 flip/rotate
        return x

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        print(f"1. 输入图像形状: {imgs.shape} (B, N, 3, H_img, W_img)")
        
        # 1. 提取图像特征 -> [1, 6, 3, 256, 704]
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        x = self.img_backbone(imgs)
        # Reshape 回 (B, N, ...) -> [6, 512, 16, 44]
        x = x.view(B, N, x.shape[1], x.shape[2], x.shape[3])
        print(f"2. 图像特征形状 (Backbone后): {x.shape} (B, N, C_feat, H_feat, W_feat)")
        
        # 2. LSS 视角转换 -> [1, 6, 512, 16, 44]
        x_splat_feat = self.view_transformer(x, rots, trans, intrins, post_rots, post_trans)
        
        # 3. BEV 数据增强
        x = self.bev_data_augmentation(x_splat_feat)
        
        # 4. BEV 编码
        x = self.bev_encoder(x)
        print(f"3. BEV特征形状 (Encoder后): {x.shape} (B, C_out, Y_bev, X_bev)")
        
        # 5. 检测头
        x = self.head(x)
        print(f"4. 最终输出形状 (Head后): {x.shape} (B, C_task, Y_bev, X_bev)")
        
        return x

# ==========================================
# 3. 运行脚本 (Run it!)
# ==========================================

if __name__ == "__main__":
    # 配置：BEV 范围 [-50m, 50m]，格子大小 0.8m
    # 深度范围 [4m, 45m]，间隔 1m (共 41 个深度格子)
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.8],   # X 轴范围和分辨率
        'ybound': [-50.0, 50.0, 0.8],   # Y 轴范围和分辨率
        'zbound': [-10.0, 10.0, 20.0],  # Z 轴 (Pillar 模式通常只有一层)
        'dbound': [4.0, 45.0, 1.0],     # 深度范围
    }

    # 初始化模型
    model = BEVDet(grid_conf)
    
    # 构造 Dummy Data (Batch=1, N=6个相机)
    B, N = 1, 6
    H_img, W_img = 256, 704 # 常见的输入尺寸 (如 nuScenes)
    
    # 随机输入图像
    imgs = torch.randn(B, N, 3, H_img, W_img)
    
    # 相机参数 (模拟值)
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)     # 旋转矩阵
    trans = torch.zeros(1, 1, 3).repeat(B, N, 1)                # 平移向量
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)  # 内参矩阵
    
    # 图像增强参数 (模拟没有增强，即单位矩阵和零向量)
    post_rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    post_trans = torch.zeros(1, 1, 3).repeat(B, N, 1)

    print("=== 开始运行 BEVDet Pipeline ===")
    output = model(imgs, rots, trans, intrins, post_rots, post_trans)
    print("=== 运行结束 ===")