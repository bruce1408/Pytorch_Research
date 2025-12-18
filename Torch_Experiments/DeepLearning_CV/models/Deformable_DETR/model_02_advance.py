import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你本地有 scipy，可以打开这行，用真正的匈牙利算法
# from scipy.optimize import linear_sum_assignment


# ============================================================
# 一、工具函数 & 位置编码
# ============================================================

def compute_level_start_index(spatial_shapes: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    计算多尺度特征在展平后，每个尺度特征图的起始索引。
    这在 Deformable Attention 中非常重要，因为它需要知道从哪个位置开始是属于哪个尺度的特征。

    Args:
        spatial_shapes (torch.Tensor): 一个形状为 [L, 2] 的张量，
                                       其中 L 是特征层级的数量，
                                       每一行包含一个特征层级的 (高度, 宽度)。

    Returns:
        Tuple[torch.Tensor, int]:
            - level_start_index (torch.Tensor): 形状为 [L] 的张量，表示每个层级展平后在整个序列中的起始索引。
            - total_len (int): 所有层级展平后的总长度。
    """
    starts = []  # 用于存储每个层级的起始索引
    start = 0
    
    # 遍历每个特征层级
    for l in range(spatial_shapes.shape[0]):
        
        # 获取当前层级的高度和宽度
        H = int(spatial_shapes[l, 0])
        W = int(spatial_shapes[l, 1])
        
        # 将当前起始索引添加到列表中
        starts.append(start)
        start += H * W
        
    # 将列表转换为张量，并返回总长度
    return torch.tensor(starts, dtype=torch.long, device=spatial_shapes.device), start


class PositionEmbeddingSine(nn.Module):
    """
    这是一个用于生成 2D 正弦/余弦位置编码的模块，与原始 DETR 中的实现完全相同。
    位置编码为模型提供了关于图像中像素位置的信息，这对于理解物体的空间关系至关重要。

    Args:
        num_pos_feats (int): 位置编码的维度的一半。最终输出的通道数是 2 * num_pos_feats。
        temperature (int): 一个用于缩放位置编码的温度参数，控制编码的频率范围。
        normalize (bool): 是否对位置编码进行归一化。
        scale (float): 如果 normalize 为 True，用于缩放归一化后的编码。
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        # 如果传入了 scale，则 normalize 必须为 True
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        
        # 如果没有提供 scale，默认使用 2 * pi
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: torch.Tensor):
        """
        前向传播函数。

        Args:
            mask (torch.Tensor): 一个形状为 [B, H, W] 的布尔张量，
                                 其中 True 表示该位置是填充（padding），无效区域。

        Returns:
            torch.Tensor: 生成的位置编码，形状为 [B, 2*num_pos_feats, H, W]。
        """
        assert mask is not None
        # not_mask: 有效点=1, padding=0
        not_mask = ~mask  # [B,H,W] 都是有效点，没有padding
        
        # 沿着 y 轴（高度）和 x 轴（宽度）累加，生成每个点的坐标
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 对 height 累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 对 width 累加
        
        # 如果需要归一化，将坐标值归一化到 [0, scale] 范围内
        if self.normalize:
            eps = 1e-6
            # y_embed[:, -1:, :] 取的是最后一行的累加值，即每个 y 坐标的最大值
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

            # x_embed[:, :, -1:] 取的是最后一列的累加值，即每个 x 坐标的最大值
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # [B, H, W, 1] / [C] -> [B, H, W, C]
        pos_x = x_embed[:, :, :, None] / dim_t   # [B,H,W,C] -> [4, 100, 100, 128]
        pos_y = y_embed[:, :, :, None] / dim_t  

        # 偶数维用 sin，奇数维用 cos  [4, 100, 100, 128]
        # 将偶数维度应用 sin，奇数维度应用 cos
        # pos_x[:, :, :, 0::2] 取所有偶数索引的维度
        # pos_x[:, :, :, 1::2] 取所有奇数索引的维度
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # [4, 100, 100, 128]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # 将 y 和 x 的位置编码拼接起来，并调整维度顺序以匹配 PyTorch 的 [B, C, H, W] 格式
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B,2C,H,W]
        return pos


# ============================================================
# 二、核心：多尺度可变形注意力 MSDeformAttn
# ============================================================

def ms_deform_attn_pytorch(value, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights, n_heads):
    """
    多尺度可变形注意力的 PyTorch 原生实现。

    核心思想：
      对于每一个 query，它不会与所有的 value (key) 进行交互。
      相反，它会预测一小部分采样点的位置（sampling_locations），
      然后只从这些位置提取特征（通过双线性插值），
      最后用预测的注意力权重（attention_weights）对这些采样的特征进行加权求和。
      这个过程在多个特征层级上独立进行，然后将结果相加。

    Args:
        value (torch.Tensor): 输入的值（来自于多尺度特征图），形状为 [B, sum(H_l*W_l), D]。
        spatial_shapes (torch.Tensor): 多尺度特征图的形状 [L, 2]。
        level_start_index (torch.Tensor): 每个尺度在 value 中的起始索引 [L]。
        sampling_locations (torch.Tensor): 归一化的采样点位置 (x,y)，形状为 [B, Len_q, n_heads, L, n_points, 2]。
        attention_weights (torch.Tensor): 采样点的注意力权重，形状为 [B, Len_q, n_heads, L, n_points]。
        n_heads (int): 注意力头的数量。

    Returns:
        torch.Tensor: 注意力模块的输出，形状为 [B, Len_q, D]。
    """
    B, Len_in, D = value.shape
    B2, Len_q, Hh, L, n_points, _ = sampling_locations.shape
    assert B == B2 and Hh == n_heads
    D_head = D // n_heads  # 32

    output = 0
    for lvl in range(L):
        H_l, W_l = spatial_shapes[lvl].tolist()
        start = level_start_index[lvl].item()
        end = start + H_l * W_l

        # 把当前 level 的 value 切出来，reshape 为 [B*n_heads, D_head, H_l, W_l]
        value_l = value[:, start:end, :].view(B, H_l, W_l, n_heads, D_head)
        value_l = value_l.permute(0, 3, 4, 1, 2)                      # [B, n_heads, D_head, H_l, W_l]
        value_l = value_l.reshape(B * n_heads, D_head, H_l, W_l)      # [B*n_heads, D_head, H_l, W_l]

        # 当前 level 上的采样位置 grid_l（归一化 [0,1] → [-1,1] 以便 grid_sample）
        grid_l = sampling_locations[:, :, :, lvl]                     # [B, Len_q, n_heads, n_points, 2]--> [4, 13294, 8, 4, 2]
        grid_l = 2.0 * grid_l - 1.0                                   # -> [-1, 1]
        grid_l = grid_l.permute(0, 2, 1, 3, 4)                        # [B, n_heads, Len_q, n_points, 2]
        grid_l = grid_l.reshape(B * n_heads, Len_q, n_points, 2)      # [B*n_heads, Len_q, n_points, 2]

        # 在每个 head 对应的特征图上用 grid_sample 采样
        # [B*n_heads, D_head, Len_q, n_points]-->[32, 32, 13269, 4]
        sampled = F.grid_sample(
            value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False
        )  

        sampled = sampled.view(B, n_heads, D_head, Len_q, n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)                      # [B,Len_q,n_heads,n_points,D_head]

        # 4. 用 attention_weights 加权求和
        # [B, Len_q, n_heads, n_points] -> [B, n_heads, Len_q, n_points, 1]
        attn_w_l = attention_weights[:, :, :, lvl].unsqueeze(-1)      # [B,Len_q,n_heads,n_points,1]

        # ( [B, n_heads, Len_q, n_points, D_head] * [B, n_heads, Len_q, n_points, 1] ).sum(-2)
        # -> [B, n_heads, Len_q, D_head]
        out_l = (sampled * attn_w_l).sum(-2)                          # [B,Len_q,n_heads,D_head]
        if isinstance(output, int):  # 第一次计算out_l 赋值给 output
            output = out_l
        else:
            output = output + out_l

    output = output.reshape(B, Len_q, D)
    return output


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力模块的封装。
    它包含一个线性层，用于从输入的 query 中预测出采样点的偏移量 (offsets) 和注意力权重 (attention weights)。
    然后调用 `ms_deform_attn_pytorch` 函数来执行实际的注意力计算。
      - 输入 query： [B, Len_q, C]
      - 预测 sampling_offsets & attention_weights
      - 根据 reference_points+offset 在各 level 特征图上采样，再加权和
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 线性层，用于对输入的 value 进行投影
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 线性层，用于对最终输出进行投影
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 对每个 query 预测: n_heads * n_levels * n_points * 2 个 offset
        # 学习从 query 生成采样点偏移量的线性层
        # 每个 query，在每个头、每个层级、每个采样点都需要一个 2D 偏移
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # 对每个 query 预测对应的 attention 权重
        # 学习从 query 生成注意力权重的线性层
        # 每个 query，在每个头、每个层级、每个采样点都需要一个权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化采样偏移的权重和偏置为0，这样在训练初期，采样点就等于参考点
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        
        # 初始化注意力权重的权重和偏置为0
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        
        # 使用 Xavier 初始化 value 和 output 的投影层
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        """
        query: [B, Len_q, d_model]
        reference_points: [B, Len_q, n_levels, 2] in [0,1]  (每个 query 在每个 level 的 base 位置)
        value: [B, Len_in, d_model]  (encoder 的 memory 或者 encoder 输入)，这里不需要位置编码
        spatial_shapes: [n_levels, 2] (H_l, W_l)
        level_start_index: [n_levels]
        """
        B, Len_q, _ = query.shape
        Bv, Len_in, d = value.shape
        assert B == Bv and d == self.d_model

        # 先对 value 做线性变换，相当于生成 K,V
        # 1. 对 value 做线性变换
        value = self.value_proj(value)  # [4, 13294, 256]

        # 2. 从 query 预测采样偏移和注意力权重
        # sampling_offsets: [B, Len_q, n_heads * n_levels * n_points * 2]
        # 预测 offsets & attention weights [4, 13269, 8, 4, 4, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        ) 
        
        # attention_weights: [B, Len_q, n_heads * n_levels * n_points]
        # [4, 13269, 8, 4, 4]
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        ) 
        
        # 对权重进行 softmax 归一化。
        # 注意 softmax 是在 (n_levels * n_points) 这个维度上做的
        # [4, 13269, 8, 4, 4]
        attention_weights = F.softmax(
            attention_weights.view(B, Len_q, self.n_heads, -1), dim=-1
        ).view(B, Len_q, self.n_heads, self.n_levels, self.n_points)

        # offsets 原本是像素单位（相对特征图大小），这里除以 (W,H) 归一化到 [0,1]
        # 偏移量需要根据特征图的大小进行归一化
        # [n_levels, 2] -> [1, 1, 1, n_levels, 1, 2]
        spatial_shapes_ = spatial_shapes.to(query.device).float()   # [num_level, 2] (H,W)
        wh = spatial_shapes_[:, [1, 0]]                             # (W, H)
        normalizer = wh[None, None, None, :, None, :]               # [1, 1, 1, num_level, 1, 2]
        
        # 偏移量归一化
        sampling_offsets = sampling_offsets / normalizer  

        # reference_points: [B, Len_q, L, 2] → [B,Len_q,1,L,1,2]
        reference_points = reference_points[:, :, None, :, None, :]
        
        # [4, 13269, 8, 4, 4, 2] --> [B,Len_q,n_heads,L,n_points,2] 获取采样位置
        sampling_locations = reference_points + sampling_offsets     
        
        # 4. 调用核心函数进行注意力计算, 这里的value是src，不带pos的token编码
        out = ms_deform_attn_pytorch(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, self.n_heads
        )
        
        # 5. 对输出进行线性变换
        out = self.output_proj(out)
        return out


# ============================================================
# 三、Encoder / Decoder Layer
# ============================================================

class DeformableTransformerEncoderLayer(nn.Module):
    """
    一个 Encoder 层：Deformable Self-Attn + FFN
    """

    def __init__(self, 
                 d_model=256, 
                 d_ffn=1024,
                 n_heads=8,
                 n_levels=4,
                 n_points=4,
                 dropout=0.1):
        
        super().__init__()
        
        # 多尺度可变形自注意力模块
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        
        # 标准的前馈网络 (Feed-Forward Network)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        
        # Layer Normalization 和 Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = F.relu

    def forward(self, src, pos, reference_points,
                spatial_shapes, level_start_index, src_mask=None):
        """
        Args:
            src (torch.Tensor): 输入的特征，[B, Len_in, C]。
            pos (torch.Tensor): 位置编码，[B, Len_in, C]。
            reference_points (torch.Tensor): 每个特征点在每个 level 的参考点，[B, Len_in, L, 2]。
            spatial_shapes (torch.Tensor): 多尺度特征图的形状，[L, 2]。
            level_start_index (torch.Tensor): 每个尺度在输入序列中的起始索引，[L]。
            src_mask (torch.Tensor): 输入的 padding mask，[B, Len_in]。
        """
        # Deformable Self-Attention: 对 src+pos 做自注意力（K=V=src）
        # 这里是query带位置编码，在src里面进行查询，src 不需要位置编码
        q = k = src + pos 
        
        # 注意力模块的输入：query, 参考点, value (这里是 src 本身) --> [4, 13294, 256]
        attn_out = self.self_attn(q, reference_points, src, spatial_shapes, level_start_index)
        
        # Add & Norm
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # 2. Feed-Forward Network
        ffn_out = self.linear2(self.dropout(self.relu(self.linear1(src))))
        
        # Add & Norm
        src = src + self.dropout2(ffn_out)
        src = self.norm2(src)
        return src

# ============================================================
# 三、Encoder / Decoder Layer
# ============================================================

class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer 的 Decoder 层。
    它由一个标准的多头自注意力模块 (用于 object queries 之间交互)、
    一个多尺度可变形交叉注意力模块 (用于从 encoder 输出中提取信息)
    和一个前馈神经网络 (FFN) 组成。
    """
    
    
    def __init__(self, 
                 d_model=256,
                 d_ffn=1024,
                 n_heads=8,
                 n_levels=4,
                 n_points=4,
                 dropout=0.1):
        
        super().__init__()
        # 1. Self-Attention for object queries
        # 让不同的 object query 互相通信，避免预测出重复的框
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 2. Deformable Cross-Attention
        # 让 object query 从 encoder 的输出 (memory) 中提取特征
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        # 3. Feed-Forward Network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

        # Layer Normalization 和 Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.relu = F.relu

    def forward(self, tgt, query_pos,
                reference_points, src, src_spatial_shapes, src_level_start_index,
                src_pos=None, src_mask=None):
        """
        Args:
            tgt (torch.Tensor): decoder 的输入，即 object queries 的特征，[B, N_q, C]。
            query_pos (torch.Tensor): object queries 的位置编码，[B, N_q, C]。
            reference_points (torch.Tensor): decoder query 对应的参考点，[B, N_q, L, 2]。
            src (torch.Tensor): encoder 的输出 (memory)，[B, Len_in, C]。
            src_spatial_shapes, src_level_start_index: src 的形状和索引信息。
            src_pos (torch.Tensor): encoder 输出的位置编码，[B, Len_in, C]。
            src_mask (torch.Tensor): src 的 padding mask，[B, Len_in]。
        """
        
        # 1) Self-attn：queries 之间互相看（不看图）
        # 1. Self-Attention (在 object queries 之间)
        q = k = tgt + query_pos
        self_attn_out, _ = self.self_attn(q, k, value=tgt)
        
        # Add & Norm
        tgt = tgt + self.dropout1(self_attn_out)
        tgt = self.norm1(tgt)

        # 2) Cross-attn：queries 在 encoder memory 上做 deformable attention
        if src_pos is None:
            src_pos = torch.zeros_like(src)
        
        
        src2 = self.cross_attn(
            tgt + query_pos,       # query
            reference_points,      # 每个 query 的参考点
            src + src_pos,         # value
            src_spatial_shapes,
            src_level_start_index
        )
        
        
        # Add & Norm
        tgt = tgt + self.dropout2(src2)
        tgt = self.norm2(tgt)

        # 3. Feed-Forward Network
        ffn_out = self.linear2(self.dropout(self.relu(self.linear1(tgt))))
        
        # Add & Norm
        tgt = tgt + self.dropout3(ffn_out)
        tgt = self.norm3(tgt)
        return tgt


# ============================================================
# 四、Deformable Transformer 主体
# ============================================================

class DeformableTransformer(nn.Module):
    """
    将 Encoder 和 Decoder 组装起来，构成完整的 Deformable Transformer。
    """
    def __init__(self, 
                 d_model=256,
                 n_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=1024,
                 n_levels=4,
                 n_points=4,
                 num_queries=100):
        
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # 创建多个 Encoder Layer
        self.encoder_layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model, dim_feedforward, n_heads, n_levels, n_points
            ) for _ in range(num_encoder_layers)
        ])
        
        # 创建多个 Decoder Layer
        self.decoder_layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model, dim_feedforward, n_heads, n_levels, n_points
            ) for _ in range(num_decoder_layers)
        ])

        # decoder 的 reference points（每个 query、每个 level 的 (x,y)）
        # 为 decoder 的 object queries 创建可学习的参考点。
        # 每个 query 在每个 level 上都有一个 2D 参考点，所以是 num_queries * n_levels * 2
        self.ref_point_embed = nn.Embedding(num_queries, n_levels * 2)  # [50, 4*2]

    def get_reference_points_encoder(self, spatial_shapes, device):
        """
            参考点是每个网格中心点的坐标，并进行归一化 [0.5, 0.5]->[0.5, 0.5]/100->[0.005, 0.005]
            为 encoder 输入生成每个位置的 reference point:
            [Len_in, 2], 每个位置对应特征图中的中心点坐标归一化到 [0,1]
        """
        reference_points_list = []
        for l in range(spatial_shapes.shape[0]):
            
            H_l = int(spatial_shapes[l, 0])
            W_l = int(spatial_shapes[l, 1])
            
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_l - 0.5, H_l, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_l - 0.5, W_l, dtype=torch.float32, device=device),
                indexing="ij"
            )
            
            ref = torch.stack((ref_x / W_l, ref_y / H_l), -1)  # [H_l, W_l, 2]
            reference_points_list.append(ref.view(-1, 2))
            
        reference_points = torch.cat(reference_points_list, 0)  # [Len_in, 2]
        return reference_points

    def forward(self, srcs: List[torch.Tensor],    # 多尺度特征: 每个 [B, C, H_l, W_l]
                masks: List[torch.Tensor],         # 每层 [B, H_l, W_l] 的 padding mask
                query_embed: torch.Tensor):        # [N_q, C]
        """
        Returns:
            hs (torch.Tensor): Decoder 各层输出的堆叠, [L_dec, B, N_q, C]。
            memory (torch.Tensor): Encoder 的输出, [B, Len_in, C]。
        """
        assert len(srcs) == self.n_levels

        # 1) flatten 多尺度特征, 预处理：展平多尺度特征和 masks
        src_flatten = []
        mask_flatten = []
        spatial_shapes_list = []
        
        # 2) pos encoding
        pos_encoder = PositionEmbeddingSine(self.d_model // 2)
        
        pos_flatten_list = []
        
        for src, mask in zip(srcs, masks):
            B, C, H, W = src.shape
            spatial_shapes_list.append((H, W))
            
            # 展平特征: [B, C, H, W] -> [B, H*W, C]
            src_flatten.append(src.flatten(2).transpose(1, 2))  
            
            # 展平 mask: [B, H, W] -> [B, H*W]
            mask_flatten.append(mask.flatten(1))  
            
            # 生成并展平位置编码
            pos = pos_encoder(mask)   # [B, C, H, W]
            pos_flatten_list.append(pos.flatten(2).transpose(1, 2))
              
            
        # 拼接所有层级的特征、mask 和位置编码
        src_flatten = torch.cat(src_flatten, 1)    # [B, Len_in, C] --> [4, 13269, 256] 
        mask_flatten = torch.cat(mask_flatten, 1)  # [B, Len_in] --> [4, 13269]
        pos_flatten = torch.cat(pos_flatten_list, 1) # [B, Len_in, C]--> [4, 13269, 256]           
        
        spatial_shapes = torch.tensor(spatial_shapes_list, 
                                      dtype=torch.long, 
                                      device=src_flatten.device)
        
        level_start_index, _ = compute_level_start_index(spatial_shapes)
        
        # 3) encoder reference points
        ref_points_enc = self.get_reference_points_encoder(spatial_shapes, src_flatten.device)
        
        # [1, Len_in, 1, 2]
        ref_points_enc = ref_points_enc.unsqueeze(0).unsqueeze(2)          
        
        # [B, Len_in, L,2]--> [4, 13269, 4, 2]
        ref_points_enc = ref_points_enc.repeat(src_flatten.shape[0], 1, self.n_levels, 1)  
        
        # ============ Encoder ============
        memory = src_flatten  
        # src_flatten       -> [4, 13269, 256] 
        # pos_flatten       -> [4, 13294, 256]
        # ref_points_enc    -> [4, 13269, 4, 2]
        # spatial_shapes    -> [4, 2]
        # level_start_index -> [0, 10000, 12500, 13125](value)
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                pos_flatten,
                ref_points_enc,
                spatial_shapes,
                level_start_index
            )

        # ============ Decoder ============
        N_q = query_embed.shape[0] # 50
        query_embed = query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)  # [B, N_q, C] -> [4, 50, 256]
        tgt = torch.zeros_like(query_embed)                                     # 初始 query 特征

        # decoder reference points（learned）
        ref = self.ref_point_embed.weight                                       # [N_q, L*2]            --> [50, 8]
        ref = ref.sigmoid().view(N_q, self.n_levels, 2)                         # [N_q, L, 2] in [0,1]  --> [50, 4, 2]
        ref_points_dec = ref.unsqueeze(0).repeat(memory.shape[0], 1, 1, 1)      # [B, N_q, L, 2]        --> [4, 50, 4, 2]

        hs = []
        for layer in self.decoder_layers:
            
            # reference_points 在这里是固定的，但在更高级的版本中，它会在每层后被预测和更新
            # tgt           ->[4, 50, 256]
            # query_embed   ->[4, 50, 256]
            # ref_points_dec  [4, 50, 4, 2]
            # memory        ->[4, 13269, 256]
            
            tgt = layer(
                tgt, 
                query_embed,
                ref_points_dec,
                memory,
                spatial_shapes,
                level_start_index,
                src_pos=pos_flatten,
                src_mask=mask_flatten
            )
            hs.append(tgt)

        # [num_decoder_layers, B, N_q, C]
        hs = torch.stack(hs)  
        return hs, memory


# ============================================================
# 五、顶层 DeformableDETR（只关注 Transformer + 头）
# ============================================================

class DeformableDETR(nn.Module):
    """
    顶层模型：Deformable Transformer + 分类/回归头
    """

    def __init__(self, 
                 num_classes,
                 num_queries=100,
                 d_model=256, 
                 n_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=1024, 
                 n_levels=4, 
                 n_points=4):
        
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.transformer = DeformableTransformer(
            d_model=d_model, 
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            n_levels=n_levels,
            n_points=n_points,
            num_queries=num_queries
        )

        # 可学习的 object queries，每个 query 是一个 d_model 维的向量
        self.query_embed = nn.Embedding(num_queries, d_model)  # [50, 256]

        # 分类头：一个线性层，将 query 特征映射到类别概率上。
        # 输出维度是 num_classes + 1，因为要额外包含一个 "no-object" (无物体) 类。
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        
        # Bbox 回归头：一个简单的多层感知机(MLP)，将 query 特征映射到 bbox 坐标。
        # 最后的 sigmoid 保证输出的 (cx, cy, w, h) 都在 [0, 1] 范围内。
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),  # 归一化到 [0,1] 的 (cx, cy, w, h)
        )

    def forward(self, features: List[torch.Tensor], masks: List[torch.Tensor]):
        """
        features: 多尺度特征 [P3,P4,P5,P6]，每个 [B,256,H_l,W_l]
        masks:    每层对应的 padding mask [B,H_l,W_l]（True=padding）
        返回：
            out:
              dict: 包含预测 logits 和 boxes 的字典。
                - "pred_logits": [B, N_q, num_classes+1]
                - "pred_boxes":  [B, N_q, 4]
        """
        
        # 将特征和 object queries 送入 transformer
        # hs: [L_dec, B, N_q, C]        
        hs, memory = self.transformer(features, masks, self.query_embed.weight)  
                
        # 通常只使用 decoder 最后一层的输出进行预测
        # outputs_class: [B, N_q, num_classes+1]
        outputs_class = self.class_embed(hs[-1])  
        
        # outputs_coord: [B, N_q, 4]    
        outputs_coord = self.bbox_embed(hs[-1])

        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }
        
        return out


# ============================================================
# 六、随机 Backbone：只生成随机多尺度特征
# ============================================================

class RandomBackbone(nn.Module):
    """
        一个教学用的“伪”Backbone。
        它不加载任何预训练模型，而是根据输入图像的尺寸，直接生成几层随机的多尺度特征图。
        这使得我们可以不依赖庞大的CNN模型，就能完整地测试和理解 Transformer 部分。
    """

    def __init__(self, d_model=256, n_levels=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        
        # 定义了四个下采样尺度，模仿 FPN (Feature Pyramid Network)
        self.scales = [8, 16, 32, 64]

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            images (torch.Tensor): 输入图像 [B, 3, H, W]。
            masks (torch.Tensor): 图像的 padding mask [B, H, W]。

        Returns:
            - features (List[torch.Tensor]): n_levels 个随机特征图 [B, d_model, H_l, W_l]。
            - feat_masks (List[torch.Tensor]): 对应的 masks [B, H_l, W_l]。
        """
        
        B, _, H, W = images.shape
        features = []
        feat_masks = []

        for s in self.scales:
            H_l = max(H // s, 2)
            W_l = max(W // s, 2)
            
            # 生成随机特征
            feat = torch.randn(B, self.d_model, H_l, W_l, device=images.device)
            features.append(feat)
            
            # mask 用最近邻下采样
            m = F.interpolate(masks.float().unsqueeze(1), size=(H_l, W_l), mode="nearest").squeeze(1).bool()
            feat_masks.append(m)

        return features, feat_masks


# ============================================================
# 七、简单的 Loss & 匹配（toy 版）
# ============================================================

def simple_match(pred_boxes, tgt_boxes):
    """
        一个极度简化的匹配函数，仅用于教学演示。
        它简单地假设前 M 个预测框对应 M 个真实框 (GT)。
        【注意】这在实际应用中是完全不合理的！
        真正的 DETR 使用匈牙利算法 (Hungarian algorithm) 来寻找成本最低的匹配方案，
        成本综合了类别预测的置信度和 bbox 的几何相似度 (如 GIoU)。
    """
    
    M = tgt_boxes.shape[0]
    
    # 返回前 M 个预测的索引和前 M 个 GT 的索引
    pred_indices = torch.arange(M, device=pred_boxes.device)
    gt_indices = torch.arange(M, device=pred_boxes.device)
    return gt_indices, pred_indices


class ToySetCriterion(nn.Module):
    """
    简化版 criterion：
      - 分类: CrossEntropy（含 no-object 类）
      - bbox: L1（不实现 GIoU，只讲结构）
      - 匹配: 用 simple_match（真实实现请用 Hungarian）
    """

    def __init__(self, num_classes, class_weight=1.0, bbox_l1_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        
        # 损失的权重
        self.class_weight = class_weight
        self.bbox_l1_weight = bbox_l1_weight

    def forward(self, outputs, targets):
        """
        outputs:
          - pred_logits: [B,N_q,C+1]
          - pred_boxes:  [B,N_q,4]
        targets: list[dict] 长度 B
          每个 dict:
            - labels: [M_i] in [0,C-1]
            - boxes:  [M_i,4] in [0,1] (cx,cy,w,h)
        """
        pred_logits = outputs["pred_logits"]   # [B,N_q,C+1]
        pred_boxes  = outputs["pred_boxes"]    # [B,N_q,4]
        B, N_q, C1 = pred_logits.shape
        num_classes = C1 - 1

        total_cls_loss = 0.0
        total_l1_loss = 0.0
        total_gt = 0

        device = pred_logits.device

        for i in range(B):
            tgt = targets[i]
            labels = tgt["labels"]      # [M_i]
            boxes  = tgt["boxes"]       # [M_i,4]

            # ✅ 把 target 搬到和预测同一个 device 上
            labels = labels.to(device)
            boxes  = boxes.to(device)

            M = boxes.shape[0]

            # 1. 匹配 (Matching)
            # 对于没有 GT 的图像，所有预测都应是 "no-object"
            if M == 0:
                # 没有 GT，全都当作 no-object
                target_cls = torch.full((N_q,), num_classes, dtype=torch.long, device=device)
                # ✅ 不要 transpose，pred_logits[i] 就是 [N_q, C+1]
                cls_loss = F.cross_entropy(pred_logits[i], target_cls)
                total_cls_loss += cls_loss
                continue

            # 1. 匹配（toy: 前 M 个 query）
            gt_idx, pred_idx = simple_match(pred_boxes[i], boxes)
            # 如果你改用匈牙利，记得保证 gt_idx / pred_idx 也在 device 上：
            # gt_idx  = gt_idx.to(device)
            # pred_idx = pred_idx.to(device)

            # 2. 分类目标：默认 no-object，匹配上的改成对应类，计算分类损失 (Classification Loss)
            # 创建一个大小为 N_q 的目标类别张量，默认填充为 "no-object" 类别
            target_cls = torch.full((N_q,), num_classes, dtype=torch.long, device=device)
            target_cls[pred_idx] = labels[gt_idx]

            # ✅ 正确用法：input [N_q, C+1], target [N_q]
            cls_loss = F.cross_entropy(pred_logits[i], target_cls)

            # 3. bbox L1 只算匹配到的，回归损失 (只对匹配上的计算)
            matched_pred = pred_boxes[i][pred_idx]
            matched_tgt  = boxes[gt_idx]
            l1_loss = F.l1_loss(matched_pred, matched_tgt, reduction="sum")

            total_cls_loss += cls_loss
            total_l1_loss  += l1_loss
            total_gt       += M

        total_cls_loss = total_cls_loss / B
        if total_gt > 0:
            total_l1_loss = total_l1_loss / total_gt
        else:
            total_l1_loss = 0.0

        loss = self.class_weight * total_cls_loss + self.bbox_l1_weight * total_l1_loss
        return {
            "loss_ce": total_cls_loss,
            "loss_bbox": total_l1_loss,
            "loss": loss
        }


# ============================================================
# 八、一个 toy Dataset & 训练/验证循环
# ============================================================

class RandomDetectionDataset(torch.utils.data.Dataset):
    """
    随机生成图像 & 随机 bbox/labels，用来示范训练流程。
    """

    def __init__(self, num_samples=100, num_classes=20, image_size=256, max_boxes=5):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_boxes = max_boxes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机图像
        img = torch.randn(3, self.image_size, self.image_size)
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.bool)  # 无 padding

        # 随机生成 0~max_boxes 个目标
        num_boxes = torch.randint(0, self.max_boxes + 1, (1,)).item()
        boxes = []
        labels = []
        for _ in range(num_boxes):
            cx = torch.rand(1).item()
            cy = torch.rand(1).item()
            w = torch.rand(1).item() * 0.5
            h = torch.rand(1).item() * 0.5
            boxes.append([cx, cy, w, h])
            labels.append(torch.randint(0, self.num_classes, (1,)).item())
        if num_boxes > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels
        }
        return img, mask, target


def collate_fn(batch):
    imgs = []
    masks = []
    targets = []
    for img, mask, tgt in batch:
        imgs.append(img)
        masks.append(mask)
        targets.append(tgt)
    imgs = torch.stack(imgs, 0)
    masks = torch.stack(masks, 0)
    return imgs, masks, targets


def train_one_epoch(model, backbone, criterion, dataloader, optimizer, device):
    model.train()
    backbone.train()
    total_loss = 0.0
    for imgs, masks, targets in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 1. backbone -> 多尺度随机特征
        feats, feat_masks = backbone(imgs, masks)
        
        # 2. transformer + 头
        outputs = model(feats, feat_masks)
        
        # 3. loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, backbone, criterion, dataloader, device):
    model.eval()
    backbone.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks, targets in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            feats, feat_masks = backbone(imgs, masks)
            outputs = model(feats, feat_masks)
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict["loss"].item()
    return total_loss / len(dataloader)


# ============================================================
# 九、main：跑一个 toy 训练/验证
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 20
    num_queries = 50

    # 1. 构建 backbone + DeformableDETR + criterion
    backbone = RandomBackbone(d_model=256, n_levels=4).to(device)
    
    model = DeformableDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=256,
        n_heads=8,
        num_encoder_layers=2,   # 为了快一点，这里用 2 层 encoder/decoder
        num_decoder_layers=2,
        dim_feedforward=512,
        n_levels=4,
        n_points=4
    ).to(device)
    
    criterion = ToySetCriterion(num_classes=num_classes)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(model.parameters()), lr=1e-4, weight_decay=1e-4
    )

    # 2. 构造随机数据的 train/val dataloader
    train_dataset = RandomDetectionDataset(num_samples=50, num_classes=num_classes, image_size=800)
    val_dataset = RandomDetectionDataset(num_samples=20, num_classes=num_classes, image_size=800)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    # 3. 训练几个 epoch
    for epoch in range(3):
        train_loss = train_one_epoch(model, backbone, criterion, train_loader, optimizer, device)
        val_loss = evaluate(model, backbone, criterion, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
