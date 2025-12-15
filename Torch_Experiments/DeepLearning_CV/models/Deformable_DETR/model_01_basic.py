import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 模块 1: 多尺度特征提取器 (模拟 Backbone + FPN)
# ==============================================================================
class MultiScaleFeatureExtractor(nn.Module):
    """
    这个模块模拟了 Deformable DETR 的前置部分：从一个骨干网络 (如 ResNet) 
    的输出中，生成 Transformer 所需的多尺度特征图。

    主要功能:
    1. 接收来自 Backbone 不同阶段的特征图 (C3, C4, C5)。
    2. 使用 1x1 卷积将它们的通道数统一映射到 d_model (例如 256)。
    3. 在 C5 的基础上，通过一个步长为 2 的卷积生成一个更粗糙的特征图 C6。
    4. 最终输出 4 个尺度的特征图 (srcs) 和它们对应的掩码 (masks)。
    """
    
    def __init__(self, d_model=256):
        
        super().__init__()
        # 假设输入的 Backbone 特征维度 (来自 ResNet 的 C3, C4, C5 阶段)
        self.backbone_dims = [512, 1024, 2048]
        
        # 1. 输入投影层 (Input Projection):
        # 使用 1x1 卷积，将不同通道数的特征图统一映射到 Transformer 需要的维度 d_model。
        # 这是连接 Backbone 和 Transformer 的桥梁。
        self.input_proj = nn.ModuleList([
            nn.Conv2d(dim, d_model, kernel_size=1) for dim in self.backbone_dims
        ])
        
        # 2. C6 特征图生成器:
        # 在 C5 (通道为2048) 的基础上，通过一个 3x3、步长为 2 的卷积来进一步下采样，
        # 生成一个分辨率更低但感受野更大的 C6 特征图。
        self.c6_conv = nn.Conv2d(2048, d_model, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        """
        输入: 
            features (list): 一个包含3个特征图的列表, [C3, C4, C5]。
        输出: 
            srcs (list): 4个处理后的特征图，通道数均为 d_model。
            masks (list): 4个对应的掩码。
        """
        srcs = []
        masks = []
        
        # 遍历并处理 C3, C4, C5 特征图
        for l, feat in enumerate(features):
            # 通过 1x1 卷积投影通道维度到 d_model
            src = self.input_proj[l](feat)
            srcs.append(src)
            
            # 简化的 Mask 生成：假设 batch 里没有 padding，所有像素都有效。
            # 因此创建一个全为 False (有效) 的掩码。
            B, C, H, W = src.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=src.device)
            masks.append(mask)

        # 单独处理 C6 (由 C5 生成)
        c6 = self.c6_conv(features[-1]) # 使用最后一个输入特征 (C5)
        srcs.append(c6)
        mask_c6 = torch.zeros((c6.shape[0], c6.shape[2], c6.shape[3]), dtype=torch.bool, device=c6.device)
        masks.append(mask_c6)
        
        return srcs, masks
    
# ==============================================================================
# 辅助函数 1: 构建编码器输入
# ==============================================================================
def build_encoder_inputs(srcs, masks, pos_embeds):
    """
    这个函数是数据准备的关键。它将多层级的 2D 特征图“拉平”成 1D 序列，
    并为 Deformable Attention 模块准备好所有必需的元数据。
    """
    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    
    # 遍历每个层级的特征、掩码和位置编码
    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        B, C, H, W = src.shape
        
        # 1. 记录该层级的空间形状 (H, W)，这个信息在后面计算偏移量时至关重要。
        spatial_shapes.append((H, W))
        
        # 2. 展平特征: [B, C, H, W] -> [B, H*W, C] (符合 Transformer 输入格式)
        src = src.flatten(2).transpose(1, 2)
        
        # 3. 展平掩码: [B, H, W] -> [B, H*W]
        mask = mask.flatten(1)
        
        # 4. 展平位置编码: [B, C, H, W] -> [B, H*W, C]
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        
        src_flatten.append(src)
        mask_flatten.append(mask)
        lvl_pos_embed_flatten.append(pos_embed)

    # 5. 将所有层级的数据在序列长度维度上拼接起来
    src_flatten = torch.cat(src_flatten, 1)          # Shape: [B, Total_Len, C]
    mask_flatten = torch.cat(mask_flatten, 1)        # Shape: [B, Total_Len]
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # Shape: [B, Total_Len, C]
    
    # 6. 转换形状列表为 Tensor，并计算每个层级在长序列中的起始索引
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    
    # `level_start_index` 记录了每一层特征在 `src_flatten` 这个长序列中的起始位置。
    # 例如: [0, H0*W0, H0*W0+H1*W1, ...]
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    
    return src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index

# ==============================================================================
# 核心模块 2: 可变形注意力 (Deformable Attention)
# ==============================================================================
class DeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        初始化可变形注意力模块。

        参数:
            d_model (int): 特征维度。
            n_levels (int): 特征层级数。
            n_heads (int): 注意力头数。
            n_points (int): 每个头在每个层级上的采样点数 (论文中默认为 4)。
        """
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 1. 预测采样偏移量的线性层 (Sampling Offsets)
        # 输入 query，输出每个头、每个层级、每个采样点的 (x, y) 偏移量。 8 * 4 * 4 * 2 = 256
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # 2. 预测注意力权重的线性层 (Attention Weights)
        # 输入 query，输出每个采样点的权重。 8 * 4 * 4 = 128
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # 3. 输出投影层 256 -> 256
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index, input_padding_mask=None):
        """
        前向传播。

        参数:
            query (Tensor): [B, Nq, C] 查询向量 (Encoder时是src+pos，Decoder时是object query)。
            reference_points (Tensor): [B, Nq, 2] 归一化的参考中心点 (cx, cy)。
            input_flatten (Tensor): [B, Total_Len, C] 作为 Key 和 Value 的源特征。
        """
        B, Nq, C = query.shape
        
        # --- A. 从 Query 生成采样偏移量和注意力权重 ---
        # sampling_offsets -> [B, Nq, n_heads, n_levels, n_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)
        
        # attention_weights -> [B, Nq, n_heads, n_levels * n_points]
        attention_weights = self.attention_weights(query).view(B, Nq, self.n_heads, self.n_levels * self.n_points)
        # 在所有采样点上做 Softmax
        attention_weights = F.softmax(attention_weights, -1).view(B, Nq, self.n_heads, self.n_levels, self.n_points)

        # --- B. 计算最终的采样点坐标 ---
        # `offset_normalizer` 用于将预测的相对偏移量，根据不同层级的特征图大小，缩放到正确的尺度。
        # 这样，一个相同的偏移量预测值，在小特征图上移动的距离就小，在大特征图上移动的距离就大。
        if reference_points.shape[-1] == 2:
            # `spatial_shapes` 是 (H, W)，但坐标是 (x, y)，所以 stack 时要用 (W, H)。
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # Shape: [L, 2]
            
            # `sampling_locations` = 参考点 + 缩放后的偏移量
            # 扩展维度以利用广播机制进行计算。 
            sampling_locations = reference_points[:, :, None, None, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        # --- C. 核心采样与聚合 ---
        # 调用辅助函数，执行真正的多尺度采样。
        # 这一步在官方实现中是单个 CUDA 算子，这里我们用 PyTorch 的 `grid_sample` 循环模拟。
        output = self.multi_scale_grid_sample(
            input_flatten, spatial_shapes, level_start_index, 
            sampling_locations, attention_weights
        )
        
        # 最终通过输出投影层
        return self.output_proj(output)

    def multi_scale_grid_sample(self, input_flatten, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        """
        使用 F.grid_sample 模拟多尺度采样过程。
        """
        B, Nq, C = input_flatten.shape[0], sampling_locations.shape[1], input_flatten.shape[2]
        
        # `split` 可以根据每层特征的长度 (H*W)，将扁平的 `input_flatten` 切分成一个列表。
        input_split = input_flatten.split([h*w for h, w in spatial_shapes], dim=1) # 按照4个level图层分成4个tensor
        
        output = 0
        
        # 遍历每个特征层级 (Level)
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 1. 从列表中取出当前层级的特征，并恢复成 2D 图像形状 [B, C, H, W]。
            feat_map = input_split[lvl].transpose(1, 2).view(B, C, H, W)
            
            # 2. 取出该层对应的采样点坐标，并归一化到 [-1, 1] 以满足 `grid_sample` 的要求。
            grid = sampling_locations[:, :, :, lvl, :, :] # Shape: [B, Nq, Heads, Points, 2]
            grid = 2 * grid - 1
            
            # 3. 调整特征图和网格的维度以适配 `grid_sample` 的多头计算。
            # 目标特征图形状: [B*Heads, C_head, H, W] --> [B*Heads, head_dim, H, W] --> [16, 32, 100, 100]
            feat_map_per_head = feat_map.view(B, self.n_heads, self.head_dim, H, W).flatten(0, 1)
            
            # 目标网格形状: [B*Heads, Nq, Points, 2] (H_out=Nq, W_out=Points)
            grid_per_head = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)  #[16, 13294, 4, 2]

            # 4. 执行双线性插值采样 (Grid Sample)
            # `grid_sample` 会根据 `grid_per_head` 中的坐标，在 `feat_map_per_head` 上采样。
            # 输出形状: [B*Heads, head_dims, Nq, Points] --> [16, 32, 13294, 4]
            sampled_feat = F.grid_sample(
                feat_map_per_head, 
                grid_per_head, 
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            
            # 5. 加权聚合
            # 将采样到的特征 reshape，以便和注意力权重相乘。
            sampled_feat = sampled_feat.view(B, self.n_heads, self.head_dim, Nq, self.n_points)
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2) # -> [B, Nq, Heads, Points, head_dims]
            
            # 取出该层的注意力权重并扩展维度。
            attn_weights = attention_weights[:, :, :, lvl, :].unsqueeze(-1) # -> [B, Nq, Heads, Points, 1]
            
            # 加权求和 (在采样点维度 `dim=3` 上求和)，并将结果累加到 output。
            output += (sampled_feat * attn_weights).sum(dim=3) # -> [B, Nq, Heads, head_dims]

        # 最终，合并所有注意力头的输出。
        output = output.flatten(2) # -> [B, Nq, Heads*C_head] -> [B, Nq, C]
        return output

# ==============================================================================
# 模块 3: Deformable DETR 演示模型
# ==============================================================================
class DemoDeformableDETR(nn.Module):
    """
    一个简化的演示模型，将上述所有模块连接起来，展示一个完整的 Encoder 计算流程。
    """
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.backbone = MultiScaleFeatureExtractor(self.d_model)
        
        # 简化的 Encoder Layer: Deformable Attention + Add&Norm + FFN
        self.deform_attn = DeformableAttention(d_model=256)
        self.norm = nn.LayerNorm(256)
        self.ffn = nn.Sequential(
            nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 256)
        )
        
    def forward(self, features):
        print(">>> 1. Backbone 多尺度提取")
        srcs, masks = self.backbone(features)
        for i, src in enumerate(srcs):
            print(f"    Level {i} Shape: {src.shape}")
            
        # 模拟生成位置编码 (在真实实现中，这会是一个专门的模块，如 PositionEmbeddingSine)
        pos_embeds = [torch.randn_like(src) for src in srcs]  # TODO
        
        print("\n>>> 2. 扁平化与元数据构建")
        src_flat, mask_flat, pos_flat, shapes, starts = build_encoder_inputs(
            srcs, masks, pos_embeds
        )
        print(f"    Src Flatten: {src_flat.shape}")
        print(f"    Shapes: \n{shapes}")
        print(f"    Start Index: {starts}")

        print("\n>>> 3. Encoder 计算 (Deformable Attention)")
        
        # 3.1 生成参考点 (Reference Points)
        # 在 Encoder 的自注意力中，每个像素的参考点就是它自己的归一化坐标。
        ref_points = self.get_encoder_ref_points(shapes, src_flat.device)
        ref_points = ref_points.unsqueeze(0).repeat(src_flat.shape[0], 1, 1) # -> [B, L_total, 2]
        
        # 3.2 调用核心 Attention
        # Deformable DETR 的 Query 是特征 + 位置编码
        query = src_flat + pos_flat  # [2, 13294, 256] + [2, 13294, 256]
        
        # Value 是原始的特征
        output = self.deform_attn(
            query=query, 
            reference_points=ref_points, 
            input_flatten=src_flat,
            spatial_shapes=shapes, 
            level_start_index=starts
        )
        
        # 3.3 残差连接，层归一化，和前馈网络
        output = self.norm(src_flat + output)
        output = self.ffn(output) + output # 第二个残差连接
        
        print(f"    Encoder Output: {output.shape}")
        return output

    def get_encoder_ref_points(self, spatial_shapes, device):
        """为 Encoder 的每个特征点生成归一化参考坐标"""
        ref_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 生成从 0.5 到 H-0.5 的网格点
            y = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
            x = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
            # 使用 'ij' 索引以消除警告，并确保 y 是行，x 是列。
            ref_y, ref_x = torch.meshgrid(y, x, indexing='ij')
            
            # 归一化并堆叠成 (x, y) 坐标
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W 
            ref = torch.stack((ref_x, ref_y), -1)
            ref_points_list.append(ref)
            
        # 拼接所有层级的参考点
        ref_points = torch.cat(ref_points_list, 1)
        return ref_points.squeeze(0) # -> [Total_Len, 2]

# ==========================================
# 运行演示
# ==========================================
if __name__ == "__main__":
    
    # 模拟输入：3层来自假想 ResNet 的特征图 输入 2, 3, 800, 800
    B = 2
    feats = [
        torch.randn(B, 512, 100, 100), # 模拟 C3 8倍下采样  输出 2, 512, 100, 100
        torch.randn(B, 1024, 50, 50),  # 模拟 C4 16倍下采样 输出 2, 1024, 50, 50
        torch.randn(B, 2048, 25, 25)   # 模拟 C5 32倍下采样 输出 2, 2048, 25, 25
    ]
    
    # 实例化并运行模型
    model = DemoDeformableDETR()
    out = model(feats)
