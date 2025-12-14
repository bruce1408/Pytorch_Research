# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class PositionEmbeddingSine(nn.Module):
#     """
#     标准的 2D 正弦位置编码。
#     逻辑：根据 (x, y) 坐标生成 unique 的向量。
#     """
#     def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats # d_model 的一半
#         self.temperature = temperature
#         self.normalize = normalize
#         self.scale = 2 * math.pi

#     def forward(self, mask):
#         """
#         Args:
#             mask: [B, H, W] BoolTensor. True 代表 Padding (无效区域).
#         Returns:
#             pos: [B, 256, H, W]
#         """
#         not_mask = ~mask # 1 代表真实像素  2,100,100
        
#         # 1. 累加计算物理坐标 (y, x)
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)

#         # 2. 归一化到 [0, 2pi] (如果开启)
#         if self.normalize:
#             eps = 1e-6
#             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

#         # 3. 计算频率项 (1/10000^(2i/d))
#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

#         # 4. 计算 Sin/Cos
#         # pos_x: [B, H, W, 128]
#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
        
#         # 技巧：在最后一个维度交错 stack sin 和 cos
#         # 结果: [sin, cos, sin, cos...]
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

#         # 5. 拼接 X 和 Y (各 128 维 -> 总共 256 维)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos
    
# class DeformableTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
#         super().__init__()
        
#         # 1. Self-Attention 模块 (使用 Deformable Attention)
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
        
#         # 2. FFN (Feed Forward Network) 模块
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = nn.ReLU() # 或 GELU
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
#         """
#         src: [B, Total_Len, 256] -> 内容流 (Content)
#         pos: [B, Total_Len, 256] -> 位置流 (Position)
#         reference_points: [B, Total_Len, 2] -> 每个点的归一化参考坐标
#         """
        
#         # --- A. 准备 Q, K, V ---
#         # Query = Content + Position
#         # 我们需要知道“我是谁(src)”以及“我在哪(pos)”才能决定去哪里采样
#         q = src + pos 
        
#         # Value = Content (src)
#         # 我们只采样图像特征，不需要采样位置编码
#         v = src 

#         # --- B. 第一部分: Attention + Residual + Norm ---
#         # 这里的 self_attn 不需要 K，它通过 q 预测 offset，然后直接去 v (src) 里采样
#         src2 = self.self_attn(
#             query=q, 
#             reference_points=reference_points, 
#             input_flatten=v, 
#             spatial_shapes=spatial_shapes, 
#             level_start_index=level_start_index
#         )
        
#         # 残差连接 (src + dropout(src2)) -> Norm
#         src = self.norm1(src + self.dropout1(src2))

#         # --- C. 第二部分: FFN + Residual + Norm ---
#         # FFN 的输入纯粹是 src (已经融合了上下文信息)
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = self.norm2(src + self.dropout3(src2))
        
#         return src
    
    
# class MSDeformAttn(nn.Module):
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
#         super().__init__()
#         self.n_heads = n_heads
#         self.n_levels = n_levels
#         self.n_points = n_points
#         self.head_dim = d_model // n_heads
        
#         # 1. 预测偏移量 (Offsets): 每个Head, 每个Level, 每个Point 都有 (x,y) 偏移
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
#         # 2. 预测注意力权重 (Weights): 每个采样点的重要性
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
#         # 3. 输出投影
#         self.output_proj = nn.Linear(d_model, d_model)

#     def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index):
#         """
#         query: [B, Len_Q, C] (Src + Pos)
#         input_flatten: [B, Len_In, C] (Src) -> 这里 Len_Q == Len_In
#         reference_points: [B, Len_Q, 2]
#         """
#         B, Len_Q, _ = query.shape
        
#         # --- 1. 生成 Offsets 和 Weights ---
#         # shape: [B, Len_Q, Heads, Levels, Points, 2]
#         sampling_offsets = self.sampling_offsets(query).view(
#             B, Len_Q, self.n_heads, self.n_levels, self.n_points, 2) # 256 -> 128
        
#         # shape: [B, Len_Q, Heads, Levels, Points]
#         attention_weights = self.attention_weights(query).view(
#             B, Len_Q, self.n_heads, self.n_levels * self.n_points) # 256 -> 128
        
#         attention_weights = F.softmax(attention_weights, -1).view(
#             B, Len_Q, self.n_heads, self.n_levels, self.n_points) # 128 -> 128
        

#         # --- 2. 计算绝对采样坐标 ---
#         # 公式: Ref + Offset / Size
#         # 只要 Ref 确定，Offset 确定，采样点就确定了。不需要和 Key 做点积。
#         offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#         sampling_locations = reference_points[:, :, None, None, None, :] \
#                              + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
#         # --- 3. 采样并加权聚合 ---
#         output = self.multi_scale_grid_sample(
#             input_flatten, spatial_shapes, sampling_locations, attention_weights
#         )
        
#         return self.output_proj(output)

#     def multi_scale_grid_sample(self, input_flatten, spatial_shapes, sampling_locations, attention_weights):
#         """
#         修复后的多尺度采样函数 (Heads as Batch Trick)
#         """
#         B, Len_Q, C = input_flatten.shape
#         # 拆分回多层特征图
#         input_split = input_flatten.split([h*w for h, w in spatial_shapes], dim=1)
#         output = 0
        
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             # 1. 准备 Feature Map: 将 Heads 维度移到 Batch 维度
#             # [B, C, H, W] -> [B, Heads, Head_Dim, H, W] -> [B*Heads, Head_Dim, H, W]
#             feat_map = input_split[lvl].transpose(1, 2).view(B, C, H, W)
#             feat_map = feat_map.view(B, self.n_heads, self.head_dim, H, W).flatten(0, 1)
            
#             # 2. 准备 Grid: 同样将 Heads 移到 Batch 维度
#             # sampling_locations: [B, Len_Q, Heads, Levels, Points, 2]
#             grid = sampling_locations[:, :, :, lvl, :, :] # [B, Len_Q, Heads, Points, 2]
            
#             # 归一化 grid 到 [-1, 1]
#             grid = 2 * grid - 1
            
#             # 调整 grid 形状以适配 grid_sample
#             # [B, Len_Q, Heads, Points, 2] -> [B, Heads, Len_Q, Points, 2] -> [B*Heads, Len_Q, Points, 2]
#             grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)
            
#             # 3. 采样 (Grid Sample)
#             # 输入: [B*Heads, Head_Dim, H, W]
#             # 网格: [B*Heads, Len_Q, Points, 2] (把 Len_Q 当作 H_out, Points 当作 W_out)
#             # 输出: [B*Heads, Head_Dim, Len_Q, Points]
#             sampled_feat = F.grid_sample(
#                 feat_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False
#             )
            
#             # 4. 维度恢复与加权
#             # [B*Heads, Head_Dim, Len_Q, Points] -> [B, Heads, Head_Dim, Len_Q, Points]
#             sampled_feat = sampled_feat.view(B, self.n_heads, self.head_dim, Len_Q, self.n_points)
            
#             # 调整顺序以便加权: [B, Len_Q, Heads, Points, Head_Dim]
#             sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)
            
#             # 取出权重: [B, Len_Q, Heads, Points, 1]
#             weights = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
            
#             # 加权求和 (在 Points 维度求和)
#             # sampled_feat * weights -> [B, Len_Q, Heads, Points, Head_Dim]
#             # sum(dim=3) -> [B, Len_Q, Heads, Head_Dim]
#             output += (sampled_feat * weights).sum(dim=3)
            
#         # 最后合并 Heads: [B, Len_Q, Heads*Head_Dim] -> [B, Len_Q, C]
#         return output.flatten(2)
    
# class DeformableDETR_Demo(nn.Module):
#     def __init__(self, num_levels=4, d_model=256):
#         super().__init__()
#         self.d_model = d_model
        
#         # 1. 位置编码生成器 (基础)
#         self.pos_generator = PositionEmbeddingSine(d_model // 2)
        
#         # 2. 层级编码 (Learnable Level Embeddings)
#         self.level_embed = nn.Parameter(torch.randn(num_levels, d_model))
        
#         # 3. Transformer Encoder Block (这里只演示一层)
#         self.encoder_layer = DeformableTransformerEncoderLayer(d_model=d_model, n_levels=num_levels)

#     def forward(self, feature_list):
#         """
#         feature_list: List[[B, 256, H, W], ...] (来自 Backbone)
#         """
#         B = feature_list[0].shape[0]
        
#         # --- Step 1: 准备输入数据 (Src, Mask, Pos) ---
#         srcs = []
#         masks = []
#         pos_embeds = []
#         spatial_shapes = []
        
#         for lvl, src in enumerate(feature_list):
#             B, C, H, W = src.shape
#             spatial_shapes.append((H, W))
            
#             # 生成 Mask (假设无 Padding)
#             mask = torch.zeros((B, H, W), dtype=torch.bool, device=src.device)
            
#             # 生成基础 Pos Embed (Sin/Cos)
#             pos = self.pos_generator(mask) # [B, 256, H, W]
            
#             # 加上 Level Embed (广播相加)
#             # Level Embed [256] -> [1, 256, 1, 1]
#             pos = pos + self.level_embed[lvl].view(1, -1, 1, 1)
            
#             srcs.append(src)
#             masks.append(mask)
#             pos_embeds.append(pos)

#         # --- Step 2: 扁平化与拼接 (Flatten & Concat) ---
#         # 将所有层的数据拼成长序列
#         src_flatten = torch.cat([x.flatten(2).transpose(1, 2) for x in srcs], 1)
#         pos_flatten = torch.cat([x.flatten(2).transpose(1, 2) for x in pos_embeds], 1)
        
#         # 记录形状和偏移索引
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
#         # 生成参考点 (归一化中心坐标)
#         # Encoder 中，参考点就是每个像素自己的坐标
#         ref_points = self.get_reference_points(spatial_shapes, device=src_flatten.device)
#         ref_points = ref_points.unsqueeze(0).repeat(B, 1, 1) # [B, Total_Len, 2]

#         print(f"输入 Transformer 的 Src 形状: {src_flatten.shape}")
#         print(f"输入 Transformer 的 Pos 形状: {pos_flatten.shape}")
        
#         # --- Step 3: 送入 Transformer Block ---
#         # 这里的 src 会作为 Q (加pos后) 和 V
#         memory = self.encoder_layer(
#             src=src_flatten,
#             pos=pos_flatten,
#             reference_points=ref_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index
#         )
        
#         return memory

#     def get_reference_points(self, spatial_shapes, device):
#         """生成每个层级每个像素的归一化中心坐标 (0.5, 0.5)"""
#         points_list = []
#         for (H, W) in spatial_shapes:
#             y = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
#             x = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
#             # indexing='ij' 消除警告
#             ref_y, ref_x = torch.meshgrid(y, x, indexing='ij') 
#             ref_y = ref_y.reshape(-1)[None] / H
#             ref_x = ref_x.reshape(-1)[None] / W
#             ref = torch.stack((ref_x, ref_y), -1)
#             points_list.append(ref)
#         return torch.cat(points_list, 1).squeeze(0)

# # ==========================================
# # 运行测试
# # ==========================================
# if __name__ == "__main__":
#     # 模拟 4 层特征图
#     B = 2
#     C = 256
#     feats = [
#         torch.randn(B, C, 100, 100),
#         torch.randn(B, C, 50, 50),
#         torch.randn(B, C, 25, 25),
#         torch.randn(B, C, 13, 13)
#     ]
    
#     model = DeformableDETR_Demo()
#     output = model(feats)
#     print(f"Transformer 输出形状: {output.shape}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 位置编码
# ==============================================================================
class PositionEmbeddingSine(nn.Module):
    """
    标准的 2D 正弦位置编码。
    逻辑：根据 (x, y) 坐标生成 unique 的向量。
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # d_model 的一半
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        """
        Args:
            mask: [B, H, W] BoolTensor. True 代表 Padding (无效区域).
        Returns:
            pos: [B, 256, H, W]
        """
        not_mask = ~mask  # 1 代表真实像素
        
        # 1. 累加计算物理坐标 (y, x)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # 2. 归一化到 [0, 2pi] (如果开启)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 3. 计算频率项 (1/10000^(2i/d))
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 4. 计算 Sin/Cos
        pos_x = x_embed[:, :, :, None] / dim_t   # [B,H,W,128]
        pos_y = y_embed[:, :, :, None] / dim_t

        # 交错 stack sin/cos
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        # 5. 拼接 X 和 Y (各 128 维 -> 总共 256 维)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ==============================================================================
# 2. 核心模块：多尺度可变形注意力 MSDeformAttn
# ==============================================================================
class MSDeformAttn(nn.Module):
    """
    这个版本既支持 encoder self-attn（Len_Q == Len_In），
    也支持 decoder cross-attn（Len_Q != Len_In）。
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 1. 预测偏移量 (Offsets): 每个Head, 每个Level, 每个Point 都有 (x,y) 偏移
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # 2. 预测注意力权重 (Weights): 每个采样点的重要性
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # 3. 输出投影
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index):
        """
        query:           [B, Len_Q, C]     (Src+Pos，或 Decoder 的 tgt+query_pos)
        input_flatten:   [B, Len_In, C]    (Encoder memory 或 Encoder 输入)
        reference_points:[B, Len_Q, 2]     归一化参考坐标
        spatial_shapes:  [L, 2]           每层 (H, W)
        """
        B, Len_Q, C = query.shape

        # --- 1. 生成 Offsets 和 Weights ---
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_Q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points
        )

        # --- 2. 计算绝对采样坐标 ---
        # offset_normalizer: [L,2] = (W_l, H_l)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        ).to(query.device)  # 确保在同一 device

        # reference_points: [B,Len_Q,2] → [B,Len_Q,1,1,1,2] 以便广播
        sampling_locations = reference_points[:, :, None, None, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        # --- 3. 采样并聚合 ---
        output = self.multi_scale_grid_sample(
            input_flatten, spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def multi_scale_grid_sample(self, input_flatten, spatial_shapes, sampling_locations, attention_weights):
        """
        使用 grid_sample 实现多尺度采样。
        注意这里 Len_Q 来自 sampling_locations，而不是 input_flatten，
        这样就支持 decoder cross-attention。
        """
        B, Len_Q = sampling_locations.shape[0], sampling_locations.shape[1]
        C = input_flatten.shape[2]

        # 把 flatten 的特征按每层长度切开
        lengths = [int(h * w) for (h, w) in spatial_shapes]
        input_split = input_flatten.split(lengths, dim=1)

        output = 0

        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)

            # 1) 取出当前 level 的特征，恢复成 [B,C,H,W]
            feat_map = input_split[lvl].transpose(1, 2).view(B, C, H, W)

            # 2) 取出该层的采样点坐标： [B,Len_Q,Heads,Points,2]
            grid = sampling_locations[:, :, :, lvl, :, :]   # [B,Len_Q,Hd,P,2]
            grid = 2 * grid - 1                             # [0,1] → [-1,1]

            # 3) 把 Heads 维度合并到 Batch 维，方便一次性 grid_sample
            # feat_map: [B,C,H,W] → [B,Heads,HeadDim,H,W] → [B*Heads,HeadDim,H,W]
            feat_map = feat_map.view(B, self.n_heads, self.head_dim, H, W).flatten(0, 1)

            # grid: [B,Len_Q,Heads,P,2] → [B,Heads,Len_Q,P,2] → [B*Heads,Len_Q,P,2]
            grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)

            # 4) 采样：输出 [B*Heads,HeadDim,Len_Q,P]
            sampled_feat = F.grid_sample(
                feat_map,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            # 5) 维度还原 & 加权
            sampled_feat = sampled_feat.view(
                B, self.n_heads, self.head_dim, Len_Q, self.n_points
            )
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [B,Len_Q,Heads,P,HeadDim]

            # 当前 level 的权重: [B,Len_Q,Heads,P,1]
            weights = attention_weights[:, :, :, lvl, :].unsqueeze(-1)

            # 按 point 维度求和 → [B,Len_Q,Heads,HeadDim]
            output += (sampled_feat * weights).sum(dim=3)

        # 合并 heads → [B,Len_Q,C]
        return output.flatten(2)


# ==============================================================================
# 3. Encoder Layer：Deformable Self-Attention + FFN
# ==============================================================================
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        """
        src: [B, Total_Len, C]
        pos: [B, Total_Len, C]
        reference_points: [B, Total_Len, 2]
        """
        q = src + pos
        v = src

        # Deformable Self-Attention
        src2 = self.self_attn(
            query=q,
            reference_points=reference_points,
            input_flatten=v,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        src = self.norm1(src + self.dropout1(src2))

        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


# ==============================================================================
# 4. Decoder Layer：Self-Attn + Deformable Cross-Attn + FFN
# ==============================================================================
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # 自注意力（queries 之间）
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力（queries 去 Encoder memory 上取特征）
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points,
                memory, memory_pos, spatial_shapes, level_start_index):
        """
        tgt:            [B, N_q, C]  当前 decoder 层的 query 特征
        query_pos:      [B, N_q, C]  object query 的位置编码
        reference_points:[B,N_q,2]   decoder 中每个 query 的参考点
        memory:         [B, Len_in, C] encoder 输出
        memory_pos:     [B, Len_in, C] encoder 的 pos（用于 cross-attn）
        """

        # 1. Self-Attention (queries↔queries)
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Deformable Cross-Attention (queries↔memory)
        q = tgt + query_pos
        v = memory + memory_pos
        tgt2 = self.cross_attn(
            query=q,
            reference_points=reference_points,
            input_flatten=v,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


# ==============================================================================
# 5. 完整 DeformableDETR Demo：2 层 Encoder + 2 层 Decoder
# ==============================================================================
class DeformableDETR_Demo(nn.Module):
    def __init__(self, 
                 num_levels=4,
                 d_model=256,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 num_queries=100):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.num_queries = num_queries

        # 位置编码 + level embedding
        self.pos_generator = PositionEmbeddingSine(d_model // 2)
        self.level_embed = nn.Parameter(torch.randn(num_levels, d_model))  # 4*256

        # Encoder / Decoder 堆叠
        self.encoder_layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model=d_model,
                d_ffn=1024,
                n_levels=num_levels, 
                n_heads=8, 
                n_points=4
            ) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model=d_model,
                d_ffn=1024,
                n_levels=num_levels,
                n_heads=8,
                n_points=4
            ) for _ in range(num_decoder_layers)
        ])

        # Decoder 使用的 object queries & reference points
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.refpoint_embed = nn.Embedding(num_queries, 2)  # (cx,cy) ∈ [0,1] 通过 sigmoid

        # 这里先不加分类/回归头，如果后面你要就很好接上：
        # self.class_embed = nn.Linear(d_model, num_classes+1)
        # self.bbox_embed = ...

    def get_reference_points(self, spatial_shapes, device):
        """Encoder 用：为每个像素生成归一化中心坐标"""
        points_list = []
        for (H, W) in spatial_shapes:
            H, W = int(H), int(W)
            y = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
            x = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
            ref_y, ref_x = torch.meshgrid(y, x, indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)  # [1,H*W,2]
            points_list.append(ref)
        return torch.cat(points_list, 1).squeeze(0)  # [Total_Len,2]

    def forward(self, feature_list):
        """
        feature_list: List[[B, 256, H, W], ...] (来自 Backbone/FPN 的多尺度特征)
        返回:
          - memory: [B, Len_in, C] Encoder 输出
          - hs:     [num_decoder_layers, B, N_q, C] 每层 decoder 输出
        """
        B = feature_list[0].shape[0]

        # --- Step 1: 准备 src / mask / pos ---
        srcs = []
        masks = []
        pos_embeds = []
        spatial_shapes = []

        for lvl, src in enumerate(feature_list):
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))

            mask = torch.zeros((B, H, W), dtype=torch.bool, device=src.device)
            pos = self.pos_generator(mask)                    # [B,256,H,W]
            pos = pos + self.level_embed[lvl].view(1, -1, 1, 1)

            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)

        # flatten [2, 13294, 256]
        src_flatten = torch.cat(
            [x.flatten(2).transpose(1, 2) for x in srcs], 1
        )  # [B,Len_in,C]
        
        pos_flatten = torch.cat(
            [x.flatten(2).transpose(1, 2) for x in pos_embeds], 1
        )  # [B,Len_in,C]

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # Encoder reference points
        ref_points_enc = self.get_reference_points(spatial_shapes, device=src_flatten.device)
        ref_points_enc = ref_points_enc.unsqueeze(0).repeat(B, 1, 1)  # [B,Len_in,2]

        print(f"[Encoder] src_flatten: {src_flatten.shape}")

        # --- Step 2: Encoder（2 层）---
        memory = src_flatten
        for layer in self.encoder_layers:
            memory = layer(
                src=memory,
                pos=pos_flatten,
                reference_points=ref_points_enc,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )

        # --- Step 3: Decoder 准备 ---
        N_q = self.num_queries
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B,N_q,C]
        tgt = torch.zeros(B, N_q, self.d_model, device=memory.device)

        ref_points_dec = self.refpoint_embed.weight.sigmoid().unsqueeze(0).repeat(B, 1, 1)  # [B,N_q,2]

        # --- Step 4: Decoder（2 层）---
        hs = []
        for layer in self.decoder_layers:
            tgt = layer(
                tgt=tgt,
                query_pos=query_pos,
                reference_points=ref_points_dec,
                memory=memory,
                memory_pos=pos_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            hs.append(tgt)

        hs = torch.stack(hs)  # [num_decoder_layers,B,N_q,C]
        print(f"[Decoder] hs shape: {hs.shape}")

        return {
            "memory": memory,  # Encoder 输出
            "hs": hs           # Decoder 每一层输出
        }


# ==============================================================================
# 6. 运行测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟 4 层特征图
    B = 2
    C = 256
    feats = [
        torch.randn(B, C, 100, 100),
        torch.randn(B, C, 50, 50),
        torch.randn(B, C, 25, 25),
        torch.randn(B, C, 13, 13),
    ]

    model = DeformableDETR_Demo(num_levels=4, 
                                d_model=256,
                                num_encoder_layers=2,
                                num_decoder_layers=2,
                                num_queries=100)
    out = model(feats)
    print("Encoder memory shape:", out["memory"].shape)
    print("Decoder hs shape:", out["hs"].shape)

