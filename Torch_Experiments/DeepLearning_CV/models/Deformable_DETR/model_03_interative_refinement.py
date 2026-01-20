import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础工具函数
# ==========================================
def inverse_sigmoid(x, eps=1e-5):
    """
    反 Sigmoid 函数（Inverse Sigmoid）
    
    功能：
    - Sigmoid 将 (-∞, +∞) 映射到 (0, 1)
    - inverse_sigmoid 将 (0, 1) 映射回 (-∞, +∞)
    
    用途：
    - 在 Deformable DETR 中，用于坐标空间的转换
    - 参考点（reference points）在 [0,1] 范围，但预测偏移量在 (-∞, +∞)
    - 需要先转换到同一空间才能相加，然后再转回 [0,1]
    
    数学公式：
    inverse_sigmoid(x) = log(x / (1 - x))
    
    Args:
        x: (...,) 输入张量，范围应该在 [0, 1]
        eps: 防止 log(0) 的小常数
    
    Returns:
        (...,) 输出张量，范围在 (-∞, +∞)
    """
    x = x.clamp(min=0, max=1)  # 确保在 [0,1] 范围内
    x1 = x.clamp(min=eps)       # 防止 x=0 导致 log(0)
    x2 = (1 - x).clamp(min=eps) # 防止 (1-x)=0 导致 log(0)
    return torch.log(x1 / x2)


def _get_clones(module, N):
    """
    克隆 N 个相同的模块
    
    功能：
    - Transformer 的 Encoder/Decoder 通常由多个相同结构的层堆叠
    - 使用 deepcopy 确保每个层是独立的（参数不共享）
    
    Args:
        module: 要克隆的模块（例如一个 EncoderLayer）
        N: 克隆的数量
    
    Returns:
        nn.ModuleList: 包含 N 个独立副本的模块列表
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# ==========================================
# 2. 位置编码 (复用)
# ==========================================
class PositionEmbeddingSine(nn.Module):
    """
    正弦位置编码（Sinusoidal Position Embedding）
    
    功能：
    - 为特征图的每个空间位置生成位置编码
    - 帮助模型理解像素之间的相对位置关系
    - 使用正弦/余弦函数，可以编码任意长度的序列
    
    原理：
    - 不同频率的正弦/余弦波组合，编码不同维度的位置信息
    - 偶数维度用 sin, 奇数维度用 cos;
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        """
        Args:
            num_pos_feats: 位置编码的维度（通常是 d_model 的一半，因为 x 和 y 各占一半）
            temperature: 温度参数，控制频率范围
            normalize: 是否归一化位置坐标到 [0, 2π]
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi  # 归一化到 [0, 2π]

    def forward(self, mask):
        """
        生成位置编码
        Args:
            mask: (B, H, W) bool 张量，True 表示 padding（无效位置）
        Returns:
            pos: (B, d_model, H, W) 位置编码
        """
        
        # 计算每个位置的累积和（cumsum），得到 y 和 x 坐标
        not_mask = ~mask
        
        # 沿着 H 维度累积
        y_embed = not_mask.cumsum(1, dtype=torch.float32) 
        
        # 沿着 W 维度累积
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        # 归一化到 [0, 2π]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # 偶数维度用 sin，奇数维度用 cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # 拼接 y 和 x 的位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos

# ==========================================
# 3. 核心注意力机制 (复用 + 修复版)
# ==========================================
class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力（Multi-Scale Deformable Attention）
    
    核心思想：
    - 传统注意力：每个 query 要关注所有 key（计算量大）
    - 可变形注意力：每个 query 只关注少数几个"可学习的采样点"（计算量小）
    
    优势：
    - 计算复杂度从 O(N²) 降到 O(N*K)，K 是采样点数（通常 K=4）
    - 可以处理多尺度特征（FPN 的多个 level）
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Args:
            d_model: 特征维度（例如 256）
            n_levels: 多尺度特征的数量（例如 4 个 FPN level）
            n_heads: 注意力头数
            n_points: 每个 level 每个 head 的采样点数（例如 4 个点）
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        
        # 预测采样偏移量（offsets）：每个 query 预测在哪里采样
        # 输出维度：n_heads * n_levels * n_points * 2（2 表示 x, y 坐标）
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # 预测注意力权重：每个采样点的重要性
        # 输出维度：n_heads * n_levels * n_points
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index):
        """
        可变形注意力前向传播
        
        Args:
            query: (B, Len_Q, d_model) Query 特征
            reference_points: (B, Len_Q, 2) 参考点坐标（归一化到 [0,1]）
            input_flatten: (B, Len_V, d_model) 展平的多尺度特征 = src
            spatial_shapes: (n_levels, 2) 每个 level 的 (H, W)
            level_start_index: (n_levels,) 每个 level 在 input_flatten 中的起始索引
        
        Returns:
            output: (B, Len_Q, d_model) 注意力输出
        """
        
        B, Len_Q, _ = query.shape
        
        # ========== 步骤1：预测采样偏移量和注意力权重 ==========
        # 从 query 特征预测：在哪里采样（offsets）+ 每个采样点的重要性（weights） 
        # 简单的线性层
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points, 2)
        # shape: (B, Len_Q, n_heads, n_levels, n_points, 2)

        # 简单的线性层
        attention_weights = self.attention_weights(query).view(
            B, Len_Q, self.n_heads, self.n_levels * self.n_points)
        # shape: (B, Len_Q, n_heads, n_levels * n_points)

        # 对注意力权重做 softmax（所有 level 的所有点加起来归一化）
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points)

        # 2. 计算绝对采样坐标
        if reference_points.shape[-1] == 2:
            
            # reference_points 是归一化的 [0,1]，需要转换到像素坐标
            # offset_normalizer: (n_levels, 2) - 每个 level 的 (W, H)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            # 绝对采样坐标 = 参考点 + 归一化的偏移量
            sampling_locations = reference_points[:, :, None, None, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # shape: (B, Len_Q, n_heads, n_levels, n_points, 2)
        else:
            raise ValueError("Reference points should be (x, y)")
            
        # 3. 采样，多尺度采样
        output = self.multi_scale_grid_sample(
            input_flatten, spatial_shapes, sampling_locations, attention_weights
        )
        
        return self.output_proj(output)

    def multi_scale_grid_sample(self, input_flatten, spatial_shapes, sampling_locations, attention_weights):
        """
        多尺度网格采样
        
        功能：
        - 在每个 FPN level 上，根据采样坐标进行双线性插值采样
        - 加权聚合所有 level 的采样结果
        """
        
        B, Len_V, C = input_flatten.shape
        Len_Q = sampling_locations.shape[1]
        
        # 按 level 分割特征
        input_split = input_flatten.split([h*w for h, w in spatial_shapes], dim=1)
        
        output = 0
        
        # 遍历每个 FPN level
        for lvl, (H, W) in enumerate(spatial_shapes):
            
            # 恢复 2D 特征图形状
            feat_map = input_split[lvl].transpose(1, 2).view(B, C, H, W)
            
            # Grouped Sampling Trick: 将 Heads 融合进 Batch
            # ========== 分组采样技巧（Grouped Sampling Trick）==========
            # 将多个 head 融合进 batch 维度，一次性采样，提高效率
            feat_map = feat_map.view(B, self.n_heads, self.head_dim, H, W).flatten(0, 1)
            
            # 获取当前 level 的采样坐标
            grid = sampling_locations[:, :, :, lvl, :, :]
            grid = 2 * grid - 1 # 归一化到 [-1, 1] grid_sample的要求
            grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)
            
            # 双线性插值采样
            sampled_feat = F.grid_sample(
                feat_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            
            # 恢复形状
            sampled_feat = sampled_feat.view(B, self.n_heads, self.head_dim, Len_Q, self.n_points)
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)
            
            # 加权聚合 = 注意力权重 + 采样特征
            # (B, Len_Q, n_heads, n_points, 1)
            weights = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
            output += (sampled_feat * weights).sum(dim=3)
        
        return output.flatten(2)

# ==========================================
# 4. Encoder 实现 (层 + 整体)
# ==========================================
class DeformableTransformerEncoderLayer(nn.Module):
    """
        Deformable Transformer Encoder 的一层
        
        结构：
        1. Self-Attention 可变形注意力 : 特征图内部的自注意力
        2. FFN -> 前馈网络 : 两层 MLP
    """
    
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # Self-Attention（可变形注意力）
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, 
                features, 
                pos_embed,
                reference_points, 
                spatial_shapes,
                level_start_index
            ):
        """
        Args:
            features: (B, Len, d_model) 输入特征
            pos_embed: (B, Len, d_model) 位置编码
            reference_points: (B, Len, 2) 参考点
            spatial_shapes: (n_levels, 2) 空间形状
            level_start_index: (n_levels,) level 起始索引
        """
        
        # Self-Attention：Query = Key = Value = src + pos
        query_with_pos = features + pos_embed
        
        attended_features = self.self_attn(
            query_with_pos, 
            reference_points=reference_points, 
            input_flatten=features, 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        features = self.norm1(features + self.dropout1(attended_features))
        
        # FFN
        ffn_output = self.linear2(self.dropout2(self.activation(self.linear1(features))))
        features = self.norm2(features + self.dropout3(ffn_output))
        
        return features

class DeformableTransformerEncoder(nn.Module):
    """
    Deformable Transformer Encoder (多层堆叠)
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, 
                features, 
                pos_embed, 
                reference_points, 
                spatial_shapes, 
                level_start_index
            ):
        
        output_features = features

        for layer in self.layers:
            
            # -- 明确 layer 调用的参数名 --
            output_features = layer(
                features=output_features,
                pos_embed=pos_embed,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        return output_features

# ==========================================
# 5. Decoder 实现 (关键补充部分)
# ==========================================
class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder 的一层
    
    结构：
    1. Self-Attention：Object Queries 之间的自注意力
    2. Cross-Attention（可变形注意力）：Query 关注 Encoder 输出
    3. FFN：前馈网络
    """
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # Self-Attention: 标准 MHSA (因为 Object Queries 数量少，不需要 sparse)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-Attention: Deformable Attention
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

    def forward(self,
                object_queries,             # tgt
                query_pos_embed,            # query_pos,
                query_reference_points,     # reference_points,
                memory,                     # src,
                memory_spatial_shapes,      # src_spatial_shapes,
                memory_level_start_index,   # level_start_index
            ):
        
        """
        Args:
            tgt: (B, Nq, d_model) Object Query 的内容特征
            query_pos_embed: (B, Nq, d_model) Object Query 的位置编码（可学习）
            reference_points: (B, Nq, 2) 参考点坐标（用于可变形注意力）
            src: (B, Len, d_model) Encoder 输出
            src_spatial_shapes: (n_levels, 2) 空间形状
            level_start_index: (n_levels,) level 起始索引
        """
        
        # 1. Self Attention
        # Object Queries 之间的自注意力（让它们互相"交流"）
        # query_pos_embed 是可学习的位置编码
        q_with_pos = k_with_pos = object_queries + query_pos_embed
        attended_queries = self.self_attn(
                    query=q_with_pos,
                    key=k_with_pos,
                    value=object_queries  # Value 是不含位置信息的原始 Object Query
                )[0]       
         
        object_queries = self.norm1(object_queries + self.dropout1(attended_queries))
        
        # 2. Cross Attention (Deformable)
        # Query = 内容(tgt) + 位置(query_pos_embed)
        # Reference Points = 动态预测的坐标
        # Value = Encoder Output (src)
        attended_memory = self.cross_attn(
            query=object_queries + query_pos_embed,
            reference_points=query_reference_points,
            input_flatten=memory,  # Value 来自 Encoder 的输出
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index
        )
        
        object_queries = self.norm2(object_queries + self.dropout2(attended_memory))
        
        # 3. FFN
        ffn_output = self.linear2(self.dropout3(self.activation(self.linear1(object_queries))))
        object_queries = self.norm3(object_queries + self.dropout4(ffn_output))
        
        return object_queries

class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder（多层堆叠 + 迭代优化）
    
    核心创新：Iterative Refinement（迭代优化）
    - 每一层 Decoder 都会预测边界框的偏移量
    - 用这个偏移量更新参考点（reference points）
    - 下一层基于更新后的参考点继续优化
    - 这样逐步精炼，最终得到更准确的边界框
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        
        # 用于每一层修正参考点的预测头 (Box Refinement)
        # 这里的 bbox_embed 在外部定义并传入
        self.bbox_embed = None 

    def forward(self, 
                object_queries,             # 原名: tgt
                reference_points,
                memory,                     # 原名: src
                memory_spatial_shapes,      # 原名: src_spatial_shapes
                level_start_index,
                query_pos_embed,            # 原名: query_pos
                bbox_embed_layer_list                
            ):
        """
        Args:
            tgt: (B, Nq, d_model) Object Query 初始内容
            reference_points: (B, Nq, 2) 初始参考点
            src: (B, Len, d_model) Encoder 输出
            src_spatial_shapes: (n_levels, 2) 空间形状
            level_start_index: (n_levels,) level 起始索引
            query_pos: (B, Nq, d_model) Query 位置编码
            bbox_embed_layer_list: List[nn.Module] 每层的边界框预测头
        
        Returns:
            decoder_outputs: (num_layers, B, Nq, d_model) 每层的输出
            decoder_ref_points: (num_layers, B, Nq, 2) 每层的参考点
        """
        output = object_queries
        decoder_outputs = []
        decoder_ref_points = []
        
        # ========== 逐层处理 + 迭代优化 ==========
        for layer_idx, layer in enumerate(self.layers):
            # 获取当前参考点 (Detach 掉，不反向传播 Ref Point 的梯度到上一层，稳定训练)
            # [B, Nq, 2]
                        
            # 运行 Decoder Layer [B, Nq, 256]
            output = layer(
                object_queries=output,
                query_pos_embed=query_pos_embed,
                query_reference_points=reference_points, 
                memory=memory,
                memory_spatial_shapes=memory_spatial_shapes,
                memory_level_start_index=level_start_index
            )
            
            # --- Iterative Bounding Box Refinement ---
            # 1. 每一层都有一个预测头，预测相对偏移量 (dx, dy, dw, dh)
            # 使用 inverse_sigmoid 进行坐标空间的变换
            
            # 获取当前层的预测头 (通常是共享权重的)
            current_refinement_head = bbox_embed_layer_list[layer_idx]
            
            # 2. 模型预测出的偏移量, 特征空间范围在 -inf ~ +inf
            # 预测偏移量 (dx, dy, dw, dh)
            # 注意：这里只预测 (dx, dy)，用于更新参考点的位置
            pred_delta_box = current_refinement_head(output)
            
            # 保存中间结果 (Prediction)
            decoder_ref_points.append(reference_points) 
            decoder_outputs.append(output)
            
            # 3. 然后再更新，给下一层用，参考点的更新不应该反向传播到上一层的预测，避免训练不稳定
            reference_points = reference_points.detach() # 记得 detach
            
            # 4. 更新参考点 (只更新 cx, cy)
            # 公式: sigmoid( inverse_sigmoid(ref) + delta )，reference_point 范围在[0-1]转换到坐标空间
            # 将参考点从 [0,1] 转换到特征空间 (-∞, +∞)
            ref_points_inv_sigmoid = inverse_sigmoid(reference_points)
            
            # 5. 更新后的坐标 (未归一化).,只关心 便宜点的位置，不用考虑w和h
            new_ref_points_inv = ref_points_inv_sigmoid + pred_delta_box[..., :2]
            
            # 归一化回去 -> [0, 1]，不断的更新这个reference_points
            reference_points = new_ref_points_inv.sigmoid()
            
            
            # ========== 关键理解 ==========
            # 每一层都会：
            # 1. 基于当前参考点做 Cross-Attention
            # 2. 预测一个偏移量
            # 3. 更新参考点
            # 4. 下一层基于更准确的参考点继续优化
            # 这样逐步精炼，最终得到准确的边界框位置

        return torch.stack(decoder_outputs), torch.stack(decoder_ref_points)

# ==========================================
# 6. Deformable DETR 完整模型
# ==========================================
class DeformableDETR(nn.Module):
    """
    Deformable DETR 完整模型
    
    流程：
    1. Backbone 提取多尺度特征（C3, C4, C5, C6）
    2. Encoder：用可变形注意力增强特征
    3. Decoder：Object Queries 通过可变形注意力关注 Encoder 输出
    4. 迭代优化：每一层 Decoder 都更新参考点，逐步精炼边界框
    5. 预测头：输出类别和边界框
    """
    def __init__(self, 
                 num_classes=91, 
                 num_queries=300, 
                 num_feature_levels=4,
                 num_layers=2
                ):
        
        super().__init__()
        self.d_model = 256
        self.num_queries = num_queries # 300 
        self.num_feature_levels = num_feature_levels  # 4
        
        # --- Backbone (Mock) ---
        # 模拟生成 C3, C4, C5 + C6
        self.input_proj = nn.ModuleList([
            nn.Conv2d(512, self.d_model, 1),
            nn.Conv2d(1024, self.d_model, 1),
            nn.Conv2d(2048, self.d_model, 1),
            nn.Conv2d(2048, self.d_model, 3, 2, 1) # C6
        ])
        
        # --- Position Embedding ---
        self.pos_trans = PositionEmbeddingSine(self.d_model // 2)  # 256 / 2
        
        # Level Embedding：区分不同 FPN level [4, 256]
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, self.d_model))
        
        # --- Query Embedding ---
        # 300 个 Object Query，每个query 包含两部分：
        # - 位置编码（query_pos）：可学习的，表示 Query 的初始位置
        # - 内容特征（tgt）：可学习的，表示 Query 的初始内容
        self.query_embed = nn.Embedding(num_queries, self.d_model * 2) # [300, 512]
        
        # --- Encoder layer ---
        encoder_layer = DeformableTransformerEncoderLayer(self.d_model)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_layers)
        
        # --- Decoder layer ---
        decoder_layer = DeformableTransformerDecoderLayer(self.d_model)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers)
        
        # --- Prediction Heads ---
        # Class Head: 输出类别 Logits
        self.class_embed = nn.Linear(self.d_model, num_classes)
        
        # Box Head: 输出 (dx, dy, dw, dh) 偏移量
        self.bbox_embed_layer_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 4) # 输出 [dx, dy, dw, dh]
            ) for _ in range(num_layers)
        ])
        
        # 初始化技巧：让初始预测的 w, h 接近 0（稳定训练）
        nn.init.constant_(self.bbox_embed_layer_list[0][-1].bias.data[2:], -2.0)
        
        self.transformer_weights_init()

    def transformer_weights_init(self):
        # 初始化 trick (略)
        pass
        
    def get_encoder_reference_points(self, spatial_shapes, device):
        """
        生成 Encoder 的参考点（网格点）
        
        功能：
        - Encoder 的每个特征图位置都需要一个参考点
        - 参考点是归一化的网格中心点坐标 [0, 1]
        
        Returns:
            (Len, 2) 所有特征图位置的参考点
        """
        points_list = []
        
        for (H, W) in spatial_shapes:
            
            H = int(H.item())
            W = int(W.item())
            # 网格中心点
            y = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
            x = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
            ref_y, ref_x = torch.meshgrid(y, x, indexing='ij')
            
            # 归一化到 [0, 1]
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            points_list.append(ref)
            
        return torch.cat(points_list, 1).squeeze(0)

    def forward(self, samples):
        # samples: List of Tensors or Tensor [B, 3, H, W]
        # 这里简化假设输入已经是 Tensor [B, 3, 800, 800]
        """
        前向传播
        
        Args:
            samples: (B, 3, H, W) 输入图像
        
        Returns:
            outputs_class: (num_layers, B, num_queries, num_classes) 每层的类别预测
            outputs_coord: (num_layers, B, num_queries, 4) 每层的边界框预测
        """
        x = samples
        B = x.shape[0]
        
        # 1. 模拟 Backbone 特征提取
        # 假设 4 层特征图
        feats = [
            torch.randn(B, 512, 100, 100, device=x.device),
            torch.randn(B, 1024, 50, 50, device=x.device),
            torch.randn(B, 2048, 25, 25, device=x.device),
            torch.randn(B, 2048, 25, 25, device=x.device) # Input for C6
        ]
        
        # 2. 准备 Multi-scale Features
        srcs = []               # 投影后的特征图
        masks = []              # padding_mask
        pos_embeds = []         # 位置编码
        
        for l, feat in enumerate(feats):
            
            # 投影到统一维度
            src = self.input_proj[l](feat)
            srcs.append(src)
            
            # 创建mask，这里简化假设没有padding
            mask = torch.zeros(
                (B, 
                src.shape[2], 
                src.shape[3]),
                dtype=torch.bool,
                device=x.device)
            
            masks.append(mask)
            
            # 生成位置编码
            pos = self.pos_trans(mask)
            
            # level embedding 层用来区分不同的FPN level + pos_embedding
            pos = pos + self.level_embed[l].view(1, -1, 1, 1) # 加 Level Embed
            pos_embeds.append(pos)

        # 3. 展平与元数据构建 (Encoder Input)
        src_flatten = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
        pos_flatten = torch.cat([pos.flatten(2).transpose(1, 2) for pos in pos_embeds], 1)
        mask_flatten = torch.cat([mask.flatten(1) for mask in masks], 1)
        
        # 空间形状 和 level 起始索引
        spatial_shapes = torch.as_tensor([(s.shape[2], s.shape[3]) for s in srcs], 
                                         dtype=torch.long, device=x.device)
        
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), 
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # 4. 运行 Encoder, 生成 Encoder 网格参考点
        enc_ref_points = self.get_encoder_reference_points(spatial_shapes, device=x.device)
        enc_ref_points = enc_ref_points.unsqueeze(0).repeat(B, 1, 1)
        
        # encoder 
        memory = self.encoder(
            features=src_flatten,
            pos_embed=pos_flatten,
            reference_points=enc_ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # 5. 准备 Decoder Input, query_embed 拆分为 query_pos [300, 256] 和 tgt [300, 256]
        query_pos, tgt = torch.split(self.query_embed.weight, self.d_model, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)
        
        # 初始参考点: 由 query_pos 经过 Linear -> Sigmoid 生成
        # 这里做一个简化模拟: 直接取 query_pos 的前两维 sigmoid
        dec_ref_points = torch.sigmoid(query_pos[..., :2]) 
        
        # 6. 运行 Decoder (Iterative Refinement) 迭代优化 =======
        decoder_outputs_stack, decoder_ref_points_stack = self.decoder(
            object_queries=tgt,
            reference_points=dec_ref_points,
            memory=memory,
            memory_spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            query_pos_embed=query_pos,
            bbox_embed_layer_list=self.bbox_embed_layer_list # 传入预测头用于中间层修正
        )
        
        # 7. 生成最终输出
        # hs shape: [Num_Layers, B, Num_Queries, C]
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(decoder_outputs_stack.shape[0]):
            # 这里的 reference 是该层修正后的参考点
            reference = decoder_ref_points_stack[lvl]
            
            # 转回 特征空间以便相加
            reference = inverse_sigmoid(reference) 
            
            # 预测类别
            outputs_class = self.class_embed(decoder_outputs_stack[lvl])
            outputs_classes.append(outputs_class)
            
            # 预测框偏移量
            tmp_bbox = self.bbox_embed_layer_list[lvl](decoder_outputs_stack[lvl])
            
            # 坐标变换: ref + offset -> sigmoid
            if reference.shape[-1] == 4:
                tmp_bbox += reference
            else:
                # 只更新 (x, y)，(w, h) 从偏移量直接得到
                assert reference.shape[-1] == 2
                tmp_bbox[..., :2] += reference
            
            outputs_coord = tmp_bbox.sigmoid()
            outputs_coords.append(outputs_coord)
            
        outputs_class = torch.stack(outputs_classes) # [Num_Layers, B, Nq, Class]
        outputs_coord = torch.stack(outputs_coords)  # [Num_Layers, B, Nq, 4]
        
        return outputs_class, outputs_coord

# ==========================================
# 7. 测试运行
# ==========================================
if __name__ == "__main__":
    # 模拟输入
    dummy_img = torch.randn(4, 3, 800, 800)
    
    # 实例化模型 (2层 Encoder, 2层 Decoder)
    model = DeformableDETR(num_layers=2)
    
    # 前向传播
    out_cls, out_box = model(dummy_img)
    
    print("-" * 30)
    print("模型输出检查:")
    print(f"输入 Batch Size: {dummy_img.shape[0]}")
    print(f"层数 (Layers): {out_cls.shape[0]}")
    print(f"查询数 (Queries): {out_cls.shape[2]}")
    print(f"分类 Logits Shape: {out_cls.shape}") # [2, 4, 300, 91]
    print(f"回归 Box Shape:    {out_box.shape}") # [2, 4, 300, 4]
    print("-" * 30)
    print("代码运行成功！")