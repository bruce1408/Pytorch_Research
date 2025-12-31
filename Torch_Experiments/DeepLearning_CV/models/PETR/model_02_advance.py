import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================
# 0) 工具函数 (Utils)
# =========================================================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    反 Sigmoid 函数 (Logit 变换)。
    作用: 将 [0, 1] 区间的概率值映射回 (-inf, +inf) 的实数空间。
    原因: 在位置编码中，直接输入 [0,1] 的线性坐标给 MLP 学习效果通常不好。
          将其拉伸到 Logit 空间能增加梯度的区分度，是 DETR3D/PETR 的标准 Trick。
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

def normalize_pc_range(xyz: torch.Tensor, pc_range) -> torch.Tensor:
    """
    坐标归一化。
    输入: 世界坐标系下的 xyz (..., 3)
    输出: 归一化到 [0, 1] 区间内的坐标
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mins = torch.tensor([x_min, y_min, z_min], device=xyz.device, dtype=xyz.dtype)
    maxs = torch.tensor([x_max, y_max, z_max], device=xyz.device, dtype=xyz.dtype)
    return (xyz - mins) / (maxs - mins)

# =========================================================
# 1) 全局配置 (Config)
# =========================================================
class PETRConfig:
    # --- 相机与图像参数 ---
    num_cams = 6          # 环视相机数量
    img_h = 256           # 输入图像高度
    img_w = 704           # 输入图像宽度

    # --- 特征提取参数 ---
    stride = 16           # Backbone 下采样倍率 (256/16=16, 704/16=44)
    embed_dim = 256       # Transformer 特征维度 (Channel)

    # --- 3D 坐标生成参数 ---
    num_depth = 4         # 沿视锥射线的深度采样点数量 (模拟视锥)
    depth_values = [10.0, 20.0, 30.0, 40.0]  # 具体的深度值 (米)

    # --- 3D 空间范围 (X, Y, Z) ---
    # 用于将真实世界坐标归一化到 [0, 1]
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # --- Transformer 参数 ---
    num_queries = 900     # Object Queries 数量 (潜在物体最大数)
    num_decoder_layers = 6
    num_heads = 8
    ff_dim = 1024         # FFN 中间层维度
    dropout = 0.1

    # --- 检测头参数 ---
    num_classes = 10      # 类别数 (如 nuScenes 10类)
    box_dim = 10          # 回归目标: x, y, z, w, l, h, sin, cos, vx, vy


# =========================================================
# 2) 简单的 Backbone (Mock ResNet)
# =========================================================
class SimpleBackbone(nn.Module):
    """
    一个简化的卷积网络，模拟 ResNet 的下采样过程。
    输入: (B*N, 3, H, W) -> 原始图像
    输出: (B*N, C, H/16, W/16) -> 提取的特征图
    """
    def __init__(self, cfg: PETRConfig):
        super().__init__()
        C = cfg.embed_dim
        self.net = nn.Sequential(
            # Stage 1: Stride 2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  
            nn.ReLU(inplace=True),
            # Stage 2: Stride 2 (Total /4)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      
            # Stage 3: Stride 2 (Total /8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Stage 4: Stride 2 (Total /16)
            nn.Conv2d(128, C, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3) PETR 核心模型实现
# =========================================================
class PETR(nn.Module):
    def __init__(self, cfg: PETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone(cfg)

        # 计算特征图尺寸 (例如: 256/16=16, 704/16=44)
        self.feat_h = cfg.img_h // cfg.stride
        self.feat_w = cfg.img_w // cfg.stride
        self.num_tokens_per_cam = self.feat_h * self.feat_w

        # --- 1. 生成初始的像素网格 (u, v) ---
        # 这里的 grid 是基于特征图大小生成的，但数值对应回原始图像坐标
        # 使用 +0.5 实现像素中心对齐 (Pixel Center Alignment)
        ys = torch.arange(self.feat_h, dtype=torch.float32) + 0.5
        xs = torch.arange(self.feat_w, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        
        # 乘以 stride，还原到原始图像分辨率 (H, W) 下的坐标
        uu = uu * cfg.stride
        vv = vv * cfg.stride
        
        # 堆叠得到 (Hf, Wf, 2) -> 最后一维是 (u, v)
        grid_uv = torch.stack([uu, vv], dim=-1)
        
        # 【重要技巧】使用 register_buffer
        # 1. 它会自动随模型移动到 GPU (model.to(device))
        # 2. 它不是参数 (Parameter)，不会被优化器更新
        # 3. persistent=False 表示它不会被保存到权重文件 state_dict 中 (因为它是动态生成的常量)
        self.register_buffer("grid_uv", grid_uv, persistent=False)

        # --- 2. 深度 Buffer ---
        # 形状 (1, 1, 1, D) 用于后续广播
        depth = torch.tensor(cfg.depth_values, dtype=torch.float32).view(1, 1, 1, cfg.num_depth)
        self.register_buffer("depth_values", depth, persistent=False)

        # --- 3. 3D 位置编码器 (3D Coordinates -> Embedding) ---
        # 输入维度: 3 (x,y,z) * D (深度数)。PETR 将所有深度的坐标 flatten 后一起编码。
        in_dim = 3 * cfg.num_depth
        self.position_encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )

        # --- 4. Transformer Decoder (官方实现) ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=False,  # 注意: PyTorch 默认是 (Seq_Len, Batch, Dim)
            norm_first=True     # Pre-Norm 通常收敛更好
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)

        # --- 5. Object Queries (可学习的锚点) ---
        # 形状: (Num_Queries, Embed_Dim)
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        # --- 6. 检测头 (Prediction Heads) ---
        self.cls_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.reg_head = nn.Linear(cfg.embed_dim, cfg.box_dim)

        # 可选: 对 Backbone 输出特征做 LayerNorm，稳定训练
        self.feat_ln = nn.LayerNorm(cfg.embed_dim)

    def backproject_to_world(self, rots, trans, intrins):
        """
        【PETR 核心算法 1】: 将 2D 像素反投影到 3D 世界坐标系。
        
        逻辑流程:
        Pixel (u,v) -> Camera (x,y,z) -> World (x,y,z)
        
        参数:
          rots: (B,N,3,3) 相机到世界的旋转矩阵 (Cam -> World)
          trans:(B,N,3)   相机到世界的平移向量
          intrins:(B,N,3,3) 相机内参
        
        返回:
          xyz_world: (B, N, HW, D, 3) 
                     B=Batch, N=Cams, HW=FeaturePixels, D=Depths, 3=Coords
        """
        B, N = trans.shape[:2]
        Hf, Wf = self.feat_h, self.feat_w
        HW = Hf * Wf
        D = self.cfg.num_depth

        # 1. 构造齐次像素坐标
        # grid_uv: (Hf, Wf, 2)
        uv = self.grid_uv
        
        # 扩展出第3维 '1' -> (Hf, Wf, 1)
        ones = torch.ones((Hf, Wf, 1), device=uv.device, dtype=uv.dtype)
        
        # 拼接 -> (Hf, Wf, 3)，并展平 -> (1, 1, HW, 3) 以便广播
        uv1 = torch.cat([uv, ones], dim=-1).view(1, 1, HW, 3)
        
        # 复制给每个 Batch 和 每个 Camera -> (B, N, HW, 3)
        uv1 = uv1.repeat(B, N, 1, 1)

        # 2. 像素坐标 -> 相机归一化平面 (Depth=1)
        # 公式: P_cam = K^-1 @ P_pixel
        # K_inv: (B, N, 3, 3)
        K_inv = torch.linalg.inv(intrins)
        
        # 矩阵乘法: (B,N,3,3) @ (B,N,HW,3,1) -> (B,N,HW,3)
        # 这里的 unsqueeze 是为了匹配矩阵乘法的维度
        ray = (K_inv.unsqueeze(2) @ uv1.unsqueeze(-1)).squeeze(-1)

        # 3. 引入深度信息
        # ray 是单位深度的方向向量。我们需要把它扩展到 D 个深度。
        # ray: (B,N,HW,3) -> (B,N,HW,1,3)
        # depth_values: (1,1,1,D,1)
        # 结果: (B,N,HW,D,3)
        ray_d = ray.unsqueeze(3) * self.depth_values.unsqueeze(-1)

        # 4. 相机坐标 -> 世界坐标 (Cam -> Ego/World)
        # 公式: P_world = R @ P_cam + T
        # rots: (B,N,1,1,3,3) 广播到 HW 和 D 维度
        R = rots.view(B, N, 1, 1, 3, 3)
        t = trans.view(B, N, 1, 1, 3)
        
        # 旋转: (3,3) @ (3,1) -> (3)
        xyz = (R @ ray_d.unsqueeze(-1)).squeeze(-1) + t
        
        # 最终形状: (B, N, HW, D, 3)
        return xyz

    def build_pos_embed(self, xyz_world):
        """
        【PETR 核心算法 2】: 将 3D 坐标编码为 Embedding (MLP)。
        
        参数:
          xyz_world: (B, N, HW, D, 3)
        返回:
          pos_embed: (B, N, HW, C)
        """
        B, N, HW, D, _ = xyz_world.shape

        # 1. 归一化: 将真实世界坐标映射到 [0, 1]
        xyz01 = normalize_pc_range(xyz_world, self.cfg.pc_range).clamp(0.0, 1.0)
        
        # 2. Inverse Sigmoid Trick (关键!)
        # 将 [0,1] 映射回实数域，方便 MLP 学习
        xyz_pe = inverse_sigmoid(xyz01)

        # 3. 展平深度维度
        # 我们不希望 Transformer 区分深度维度，而是希望每个像素拥有一个包含所有深度信息的 Embedding
        # (B, N, HW, D, 3) -> (B, N, HW, 3*D)
        xyz_pe = xyz_pe.permute(0, 1, 2, 4, 3).contiguous().view(B, N, HW, 3 * D)

        # 4. MLP 映射到特征维度 C
        pos = self.position_encoder(xyz_pe)  # (B, N, HW, C)
        return pos

    def forward(self, imgs, rots, trans, intrins):
        """
        前向传播
        参数:
          imgs: (B, N, 3, H, W)
          rots: (B, N, 3, 3)
          trans:(B, N, 3)
          intrins:(B, N, 3, 3)
        """
        B, N, _, H, W = imgs.shape

        # --- 1. 提取图像特征 (Backbone) ---
        # 将 Batch 和 Camera 维度合并处理: (B*N, 3, H, W)
        x = imgs.view(B * N, 3, H, W)
        feat = self.backbone(x)  # 输出: (B*N, C, Hf, Wf)
        _, C, Hf, Wf = feat.shape

        # --- 2. 准备 Transformer 输入 (Tokenization) ---
        # 恢复维度: (B, N, C, Hf, Wf)
        feat = feat.view(B, N, C, Hf, Wf)
        
        # 展平 spatial dimensions: (B, N, HW, C)
        # flatten(3) 把 Hf 和 Wf 合并
        feat_tok = feat.flatten(3).permute(0, 1, 3, 2).contiguous()
        
        # LayerNorm (Optional but recommended)
        feat_tok = self.feat_ln(feat_tok)

        # --- 3. 生成 3D 位置编码 (3D PE) ---
        # 计算每个像素对应的 3D 坐标
        xyz_world = self.backproject_to_world(rots, trans, intrins)  # (B,N,HW,D,3)
        # 将坐标转换为 Embedding
        pos = self.build_pos_embed(xyz_world)                        # (B,N,HW,C)

        # --- 4. 特征融合 (Inject 3D Info) ---
        # PETR 的灵魂: 2D 特征 + 3D 位置编码
        # 现在的 tok 既包含了视觉信息，又包含了 3D 几何信息
        tok = feat_tok + pos  # (B,N,HW,C)

        # --- 5. 构建 Transformer Memory (Key/Value) ---
        # 将所有相机的特征拼接到一起，形成一个超长的序列
        # (B, N*HW, C) -> Permute to (Seq_Len, Batch, Dim) for PyTorch API
        memory = tok.view(B, N * self.num_tokens_per_cam, C).permute(1, 0, 2).contiguous()
        
        # --- 6. 准备 Transformer Queries ---
        # query_embed 是可学习的参数，本身不含位置信息(Content Query)，
        # 我们直接把它当作 Target 输入给 Decoder
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Q, B, C)

        # --- 7. Transformer Decoding ---
        # Decoder 内部会进行 Cross-Attention:
        # Query (tgt) 去查询 Memory (图像特征+3D位置)
        hs = self.decoder(tgt=tgt, memory=memory)  # 输出: (Q, B, C)
        
        # 转回 (B, Q, C) 以便过 Head
        hs = hs.permute(1, 0, 2).contiguous()

        # --- 8. 输出预测结果 ---
        cls = self.cls_head(hs)  # 分类: (B, Q, num_classes)
        box = self.reg_head(hs)  # 回归: (B, Q, box_dim)

        return cls, box


# =========================================================
# 4) 测试代码 (Smoke Test)
# =========================================================
def main():
    # 初始化配置
    cfg = PETRConfig()
    model = PETR(cfg)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval() # 设置为评估模式

    # --- 模拟输入数据 ---
    B = 1
    N = cfg.num_cams
    # 模拟图像
    imgs = torch.randn(B, N, 3, cfg.img_h, cfg.img_w, device=device)

    # 模拟相机参数 (单位阵 + 平移为0)
    rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    trans = torch.zeros(B, N, 3, device=device)

    # 模拟内参 (简单的针孔模型参数)
    intrins = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    intrins[:, :, 0, 0] = 500.0 # fx
    intrins[:, :, 1, 1] = 500.0 # fy
    intrins[:, :, 0, 2] = 352.0 # cx (W/2)
    intrins[:, :, 1, 2] = 128.0 # cy (H/2)

    # --- 前向推理 ---
    print("=== 开始 PETR 推理测试 ===")
    with torch.no_grad():
        cls, box = model(imgs, rots, trans, intrins)

    # --- 验证输出形状 ---
    print(f"Input Images: {imgs.shape}")
    print(f"Output Cls  : {cls.shape} (预期: [{B}, {cfg.num_queries}, {cfg.num_classes}])")
    print(f"Output Box  : {box.shape} (预期: [{B}, {cfg.num_queries}, {cfg.box_dim}])")
    
    if cls.shape == (B, cfg.num_queries, cfg.num_classes) and \
       box.shape == (B, cfg.num_queries, cfg.box_dim):
        print("✅ 测试通过: 输出维度正确!")
    else:
        print("❌ 测试失败: 输出维度不匹配")

if __name__ == "__main__":
    main()