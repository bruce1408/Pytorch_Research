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
    spatial_shapes: [L, 2] (H_l, W_l)
    返回:
      level_start_index: [L] 表示每个 level 在 flatten 后的起始 index
      total_len: 所有 level 展平后的总长度
    """
    starts = []
    start = 0
    for l in range(spatial_shapes.shape[0]):
        H = int(spatial_shapes[l, 0])
        W = int(spatial_shapes[l, 1])
        starts.append(start)
        start += H * W
    return torch.tensor(starts, dtype=torch.long, device=spatial_shapes.device), start


class PositionEmbeddingSine(nn.Module):
    """
    与 DETR 完全一样的 2D 正弦/余弦位置编码:
      输入: mask [B,H,W] (True=padding)
      输出: pos [B,2*num_pos_feats,H,W]
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: torch.Tensor):
        assert mask is not None
        # not_mask: 有效点=1, padding=0
        not_mask = ~mask  # [B,H,W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 对 height 累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 对 width 累加

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t   # [B,H,W,C]
        pos_y = y_embed[:, :, :, None] / dim_t

        # 偶数维用 sin，奇数维用 cos
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B,2C,H,W]
        return pos


# ============================================================
# 二、核心：多尺度可变形注意力 MSDeformAttn
# ============================================================

def ms_deform_attn_pytorch(value, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights, n_heads):
    """
    value: [B, sum(H_l*W_l), D]
    spatial_shapes: [L, 2] (H_l, W_l)
    level_start_index: [L]
    sampling_locations: [B, Len_q, n_heads, L, n_points, 2] in [0,1]
    attention_weights: [B, Len_q, n_heads, L, n_points]
    返回: output: [B, Len_q, D]

    核心思想：
      对每个 query，在每个 level 上只采样 K 个点（而不是对所有 H*W 点做 attention），
      然后用 attention_weights 做加权求和。
    """
    B, Len_in, D = value.shape
    B2, Len_q, Hh, L, n_points, _ = sampling_locations.shape
    assert B == B2 and Hh == n_heads
    D_head = D // n_heads

    output = 0
    for lvl in range(L):
        H_l, W_l = spatial_shapes[lvl].tolist()
        start = level_start_index[lvl].item()
        end = start + H_l * W_l

        # 把当前 level 的 value 切出来，reshape 为 [B*n_heads, D_head, H_l, W_l]
        value_l = value[:, start:end, :].view(B, H_l, W_l, n_heads, D_head)
        value_l = value_l.permute(0, 3, 4, 1, 2)                      # [B,n_heads,D_head,H_l,W_l]
        value_l = value_l.reshape(B * n_heads, D_head, H_l, W_l)      # [B*n_heads,D_head,H_l,W_l]

        # 当前 level 上的采样位置 grid_l（归一化 [0,1] → [-1,1] 以便 grid_sample）
        grid_l = sampling_locations[:, :, :, lvl]                     # [B,Len_q,n_heads,n_points,2]
        grid_l = 2.0 * grid_l - 1.0                                   # -> [-1, 1]
        grid_l = grid_l.permute(0, 2, 1, 3, 4)                        # [B,n_heads,Len_q,n_points,2]
        grid_l = grid_l.reshape(B * n_heads, Len_q, n_points, 2)      # [B*n_heads,Len_q,n_points,2]

        # 在每个 head 对应的特征图上用 grid_sample 采样
        sampled = F.grid_sample(
            value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B*n_heads, D_head, Len_q, n_points]

        sampled = sampled.view(B, n_heads, D_head, Len_q, n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)                      # [B,Len_q,n_heads,n_points,D_head]

        attn_w_l = attention_weights[:, :, :, lvl].unsqueeze(-1)      # [B,Len_q,n_heads,n_points,1]

        out_l = (sampled * attn_w_l).sum(-2)                          # [B,Len_q,n_heads,D_head]
        if isinstance(output, int):
            output = out_l
        else:
            output = output + out_l

    output = output.reshape(B, Len_q, D)
    return output


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力模块：
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

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        # 对每个 query 预测: n_heads * n_levels * n_points * 2 个 offset
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 对每个 query 预测对应的 attention 权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        """
        query: [B, Len_q, d_model]
        reference_points: [B, Len_q, n_levels, 2] in [0,1]  (每个 query 在每个 level 的 base 位置)
        value: [B, Len_in, d_model]  (encoder 的 memory 或者 encoder 输入)
        spatial_shapes: [n_levels, 2] (H_l, W_l)
        level_start_index: [n_levels]
        """
        B, Len_q, _ = query.shape
        Bv, Len_in, d = value.shape
        assert B == Bv and d == self.d_model

        # 先对 value 做线性变换，相当于生成 K,V
        value = self.value_proj(value)

        # 预测 offsets & attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        attention_weights = F.softmax(
            attention_weights.view(B, Len_q, self.n_heads, -1),
            dim=-1
        ).view(B, Len_q, self.n_heads, self.n_levels, self.n_points)

        # offsets 原本是像素单位（相对特征图大小），这里除以 (W,H) 归一化到 [0,1]
        spatial_shapes_ = spatial_shapes.to(query.device).float()   # [L,2] (H,W)
        wh = spatial_shapes_[:, [1, 0]]                             # (W,H)
        normalizer = wh[None, None, None, :, None, :]               # [1,1,1,L,1,2]
        sampling_offsets = sampling_offsets / normalizer

        # reference_points: [B, Len_q, L, 2] → [B,Len_q,1,L,1,2]
        reference_points = reference_points[:, :, None, :, None, :]
        sampling_locations = reference_points + sampling_offsets     # [B,Len_q,n_heads,L,n_points,2]

        out = ms_deform_attn_pytorch(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, self.n_heads
        )
        out = self.output_proj(out)
        return out


# ============================================================
# 三、Encoder / Decoder Layer
# ============================================================

class DeformableTransformerEncoderLayer(nn.Module):
    """
    一个 Encoder 层：Deformable Self-Attn + FFN
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 n_heads=8, n_levels=4, n_points=4, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, pos, reference_points,
                spatial_shapes, level_start_index, src_mask=None):
        """
        src: [B, Len_in, C]         encoder 的特征
        pos: [B, Len_in, C]         位置编码
        reference_points: [B, Len_in, L, 2]  每个 token 在每个 level 的 reference (一般同一 level 的点，只在该 level 有值)
        """
        # Deformable Self-Attention: 对 src+pos 做自注意力（K=V=src）
        q = k = src + pos
        attn_out = self.self_attn(q, reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ffn_out)
        src = self.norm2(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    """
    一个 Decoder 层：
      1) Self-Attn (queries 之间的交互)
      2) Deformable Cross-Attn (queries 从 encoder memory 中取信息)
      3) FFN
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 n_heads=8, n_levels=4, n_points=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, query_pos,
                reference_points, src, src_spatial_shapes, src_level_start_index,
                src_pos=None, src_mask=None):
        """
        tgt: [B, N_q, C]                 当前 decoder 层的 query 表示
        query_pos: [B, N_q, C]          query 的位置编码（object queries）
        reference_points: [B, N_q, L, 2] decoder 层 query 对应的参考点
        src: [B, Len_in, C]             encoder memory
        src_pos: [B, Len_in, C]         encoder 的位置编码
        """
        # 1) Self-attn：queries 之间互相看（不看图）
        q = k = tgt + query_pos
        self_attn_out, _ = self.self_attn(q, k, value=tgt)
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
        tgt = tgt + self.dropout2(src2)
        tgt = self.norm2(tgt)

        # 3) FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ffn_out)
        tgt = self.norm3(tgt)
        return tgt


# ============================================================
# 四、Deformable Transformer 主体
# ============================================================

class DeformableTransformer(nn.Module):
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

        self.encoder_layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model, dim_feedforward, n_heads, n_levels, n_points
            ) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model, dim_feedforward, n_heads, n_levels, n_points
            ) for _ in range(num_decoder_layers)
        ])

        # decoder 的 reference points（每个 query、每个 level 的 (x,y)）
        self.ref_point_embed = nn.Embedding(num_queries, n_levels * 2)

    def get_reference_points_encoder(self, spatial_shapes, device):
        """
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

    def forward(self, srcs: List[torch.Tensor],    # 多尺度特征: 每个 [B,C,H_l,W_l]
                masks: List[torch.Tensor],         # 每层 [B,H_l,W_l] 的 padding mask
                query_embed: torch.Tensor):        # [N_q, C]
        """
        返回：
            hs: [num_decoder_layers, B, N_q, C]
            memory: encoder 输出 [B, Len_in, C]
        """
        assert len(srcs) == self.n_levels

        # 1) flatten 多尺度特征
        src_flatten = []
        mask_flatten = []
        spatial_shapes_list = []
        for src, mask in zip(srcs, masks):
            B, C, H, W = src.shape
            spatial_shapes_list.append((H, W))
            src_flatten.append(src.flatten(2).transpose(1, 2))  # [B,H*W,C]
            mask_flatten.append(mask.flatten(1))                # [B,H*W]

        src_flatten = torch.cat(src_flatten, 1)                 # [B, Len_in, C]
        mask_flatten = torch.cat(mask_flatten, 1)               # [B, Len_in]
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)
        level_start_index, _ = compute_level_start_index(spatial_shapes)

        # 2) pos encoding
        pos_encoder = PositionEmbeddingSine(self.d_model // 2)
        
        pos_flatten_list = []
        for src, mask in zip(srcs, masks):
            pos = pos_encoder(mask)                             # [B,C,H,W]
            pos_flatten_list.append(pos.flatten(2).transpose(1, 2))
        pos_flatten = torch.cat(pos_flatten_list, 1)            # [B, Len_in, C]

        # 3) encoder reference points
        ref_points_enc = self.get_reference_points_encoder(spatial_shapes, src_flatten.device)
        ref_points_enc = ref_points_enc.unsqueeze(0).unsqueeze(2)          # [1, Len_in, 1, 2]
        ref_points_enc = ref_points_enc.repeat(src_flatten.shape[0], 1, self.n_levels, 1)  # [B, Len_in, L,2]

        # ============ Encoder ============
        memory = src_flatten
        for layer in self.encoder_layers:
            memory = layer(memory, pos_flatten, ref_points_enc, spatial_shapes, level_start_index)

        # ============ Decoder ============
        N_q = query_embed.shape[0]
        query_embed = query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)  # [B,N_q,C]
        tgt = torch.zeros_like(query_embed)                                     # 初始 query 特征

        # decoder reference points（learned）
        ref = self.ref_point_embed.weight                                       # [N_q, L*2]
        ref = ref.sigmoid().view(N_q, self.n_levels, 2)                         # [N_q,L,2] in [0,1]
        ref_points_dec = ref.unsqueeze(0).repeat(memory.shape[0], 1, 1, 1)      # [B,N_q,L,2]

        hs = []
        for layer in self.decoder_layers:
            tgt = layer(
                tgt, query_embed,
                ref_points_dec, memory, spatial_shapes, level_start_index,
                src_pos=pos_flatten, src_mask=mask_flatten
            )
            hs.append(tgt)

        hs = torch.stack(hs)  # [num_decoder_layers, B, N_q, C]
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

        # object query embedding
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 分类头 + bbox 头
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
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
              - pred_logits: [B, N_q, num_classes+1]
              - pred_boxes:  [B, N_q, 4]
        """
        hs, memory = self.transformer(features, masks, self.query_embed.weight)  # hs: [L_dec,B,N_q,C]
        outputs_class = self.class_embed(hs[-1])   # 只用最后一层
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
    教学用 backbone：不用 ResNet，直接根据输入尺寸生成随机多尺度特征。
    """

    def __init__(self, d_model=256, n_levels=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        """
        images: [B,3,H,W]
        masks:  [B,H,W]  True=padding, False=valid
        返回:
          features: n_levels 个 [B,d_model,H_l,W_l]
          feat_masks: 对应的 [B,H_l,W_l]
        """
        B, _, H, W = images.shape
        features = []
        feat_masks = []

        # 这里简单地用 1/8,1/16,1/32,1/64 四个尺度
        scales = [8, 16, 32, 64]
        for s in scales:
            H_l = max(H // s, 2)
            W_l = max(W // s, 2)
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
    极简匹配（用于 toy demo）：
      - 假设：只用前 M 个 query 对应 M 个 GT（完全不合理，但结构简单）
      - 真正论文要用匈牙利算法 Hungarian matching。
    """
    M = tgt_boxes.shape[0]
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

            # 2. 分类目标：默认 no-object，匹配上的改成对应类
            target_cls = torch.full((N_q,), num_classes, dtype=torch.long, device=device)
            target_cls[pred_idx] = labels[gt_idx]

            # ✅ 正确用法：input [N_q, C+1], target [N_q]
            cls_loss = F.cross_entropy(pred_logits[i], target_cls)

            # 3. bbox L1 只算匹配到的
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
    train_dataset = RandomDetectionDataset(num_samples=50, num_classes=num_classes, image_size=256)
    val_dataset = RandomDetectionDataset(num_samples=20, num_classes=num_classes, image_size=256)
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
