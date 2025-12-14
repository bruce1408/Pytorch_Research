import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ==========================================
# 1. 基础工具函数
# ==========================================
def inverse_sigmoid(x, eps=1e-5):
    """反 Sigmoid 函数，用于坐标变换"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def _get_clones(module, N):
    """克隆 N 个相同的层"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# ==========================================
# 2. 位置编码 (复用)
# ==========================================
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# ==========================================
# 3. 核心注意力机制 (复用 + 修复版)
# ==========================================
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index):
        B, Len_Q, _ = query.shape
        # 1. 生成 Offsets 和 Weights
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            B, Len_Q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Len_Q, self.n_heads, self.n_levels, self.n_points)

        # 2. 计算绝对采样坐标
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, None, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError("Reference points should be (x, y)")
            
        # 3. 采样
        output = self.multi_scale_grid_sample(
            input_flatten, spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def multi_scale_grid_sample(self, input_flatten, spatial_shapes, sampling_locations, attention_weights):
        B, Len_Q, C = input_flatten.shape
        input_split = input_flatten.split([h*w for h, w in spatial_shapes], dim=1)
        output = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            feat_map = input_split[lvl].transpose(1, 2).view(B, C, H, W)
            # Grouped Sampling Trick: 将 Heads 融合进 Batch
            feat_map = feat_map.view(B, self.n_heads, self.head_dim, H, W).flatten(0, 1)
            grid = sampling_locations[:, :, :, lvl, :, :]
            grid = 2 * grid - 1
            grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)
            
            sampled_feat = F.grid_sample(
                feat_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sampled_feat = sampled_feat.view(B, self.n_heads, self.head_dim, Len_Q, self.n_points)
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)
            
            weights = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
            output += (sampled_feat * weights).sum(dim=3)
        return output.flatten(2)

# ==========================================
# 4. Encoder 实现 (层 + 整体)
# ==========================================
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
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
        q = src + pos
        src2 = self.self_attn(q, reference_points, src, spatial_shapes, level_start_index)
        src = self.norm1(src + self.dropout1(src2))
        
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        for layer in self.layers:
            src = layer(src, pos, reference_points, spatial_shapes, level_start_index)
        return src

# ==========================================
# 5. Decoder 实现 (关键补充部分)
# ==========================================
class DeformableTransformerDecoderLayer(nn.Module):
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index):
        # 1. Self Attention
        # query_pos 是可学习的位置编码
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        
        # 2. Cross Attention (Deformable)
        # Query = 内容(tgt) + 位置(query_pos)
        # Reference Points = 动态预测的坐标
        # Value = Encoder Output (src)
        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            reference_points=reference_points,
            input_flatten=src,
            spatial_shapes=src_spatial_shapes,
            level_start_index=level_start_index
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        
        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        
        # 用于每一层修正参考点的预测头 (Box Refinement)
        # 这里的 bbox_embed 在外部定义并传入
        self.bbox_embed = None 

    def forward(self, tgt, reference_points, src, src_spatial_shapes, level_start_index, query_pos, bbox_embed):
        output = tgt
        intermediate = []
        intermediate_ref_points = []
        
        for layer_idx, layer in enumerate(self.layers):
            # 获取当前参考点 (Detach 掉，不反向传播 Ref Point 的梯度到上一层，稳定训练)
            # [B, Nq, 2]
            reference_points_input = reference_points[:, :, None] * src_spatial_shapes.new_tensor([1., 1.])[None, None, :] 
            
            # 运行 Decoder Layer
            output = layer(
                tgt=output,
                query_pos=query_pos,
                reference_points=reference_points, 
                src=src,
                src_spatial_shapes=src_spatial_shapes,
                level_start_index=level_start_index
            )
            
            # --- Iterative Bounding Box Refinement ---
            # 1. 每一层都有一个预测头，预测相对偏移量 (dx, dy, dw, dh)
            # 使用 inverse_sigmoid 进行坐标空间的变换
            
            # 获取当前层的预测头 (通常是共享权重的)
            layer_bbox_embed = bbox_embed[layer_idx]
            
            # 预测偏移量
            delta_box = layer_bbox_embed(output)
            
            # 2. 更新参考点 (只更新 cx, cy)
            # 公式: sigmoid( inverse_sigmoid(ref) + delta )
            ref_points_inv_sigmoid = inverse_sigmoid(reference_points)
            
            # 更新后的坐标 (未归一化)
            new_ref_points_inv = ref_points_inv_sigmoid + delta_box[..., :2]
            
            # 归一化回去 -> [0, 1]
            reference_points = new_ref_points_inv.sigmoid()
            
            # 保存中间结果 (Prediction)
            intermediate.append(output)
            intermediate_ref_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_ref_points)

# ==========================================
# 6. Deformable DETR 完整模型
# ==========================================
class DeformableDETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=300, num_feature_levels=4, num_layers=2):
        super().__init__()
        self.d_model = 256
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        
        # --- Backbone (Mock) ---
        # 模拟生成 C3, C4, C5 + C6
        self.input_proj = nn.ModuleList([
            nn.Conv2d(512, self.d_model, 1),
            nn.Conv2d(1024, self.d_model, 1),
            nn.Conv2d(2048, self.d_model, 1),
            nn.Conv2d(2048, self.d_model, 3, 2, 1) # C6
        ])
        
        # --- Position Embedding ---
        self.pos_trans = PositionEmbeddingSine(self.d_model // 2)
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, self.d_model))
        
        # --- Query Embedding ---
        # 300 个 Object Query，分为两部分：位置(Query Pos) 和 内容(Tgt)
        self.query_embed = nn.Embedding(num_queries, self.d_model * 2)
        
        # --- Encoder ---
        encoder_layer = DeformableTransformerEncoderLayer(self.d_model)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_layers)
        
        # --- Decoder ---
        decoder_layer = DeformableTransformerDecoderLayer(self.d_model)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers)
        
        # --- Prediction Heads ---
        # Class Head: 输出类别 Logits
        self.class_embed = nn.Linear(self.d_model, num_classes)
        # Box Head: 输出 (dx, dy, dw, dh) 偏移量
        self.bbox_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 4)
            ) for _ in range(num_layers)
        ])
        
        # 这里的 bias 处理是 trick，保证初始训练稳定
        nn.init.constant_(self.bbox_embed[0][-1].layers[-1].bias.data[2:], -2.0)
        self.transformer_weights_init()

    def transformer_weights_init(self):
        # 初始化 trick (略)
        pass
        
    def get_encoder_reference_points(self, spatial_shapes, device):
        # 生成 Encoder 网格参考点
        points_list = []
        for (H, W) in spatial_shapes:
            y = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
            x = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
            ref_y, ref_x = torch.meshgrid(y, x, indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            points_list.append(ref)
        return torch.cat(points_list, 1).squeeze(0)

    def forward(self, samples):
        # samples: List of Tensors or Tensor [B, 3, H, W]
        # 这里简化假设输入已经是 Tensor [B, 3, 800, 800]
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
        srcs = []
        masks = []
        pos_embeds = []
        
        for l, feat in enumerate(feats):
            src = self.input_proj[l](feat)
            srcs.append(src)
            mask = torch.zeros((B, src.shape[2], src.shape[3]), dtype=torch.bool, device=x.device)
            masks.append(mask)
            pos = self.pos_trans(mask)
            pos = pos + self.level_embed[l].view(1, -1, 1, 1) # 加 Level Embed
            pos_embeds.append(pos)

        # 3. 展平与元数据构建 (Encoder Input)
        src_flatten = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
        pos_flatten = torch.cat([pos.flatten(2).transpose(1, 2) for pos in pos_embeds], 1)
        mask_flatten = torch.cat([mask.flatten(1) for mask in masks], 1)
        
        spatial_shapes = torch.as_tensor([(s.shape[2], s.shape[3]) for s in srcs], 
                                         dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # 4. 运行 Encoder
        # 生成 Encoder 网格参考点
        enc_ref_points = self.get_encoder_reference_points(spatial_shapes, device=x.device)
        enc_ref_points = enc_ref_points.unsqueeze(0).repeat(B, 1, 1)
        
        memory = self.encoder(
            src=src_flatten,
            pos=pos_flatten,
            reference_points=enc_ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # 5. 准备 Decoder Input
        # query_embed 拆分为 query_pos 和 tgt (初始为0)
        query_pos, tgt = torch.split(self.query_embed.weight, self.d_model, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)
        
        # 初始参考点: 由 query_pos 经过 Linear -> Sigmoid 生成
        # 这里做一个简化模拟: 直接取 query_pos 的前两维 sigmoid
        dec_ref_points = torch.sigmoid(query_pos[..., :2]) 
        
        # 6. 运行 Decoder (Iterative Refinement)
        hs, inter_references = self.decoder(
            tgt=tgt,
            reference_points=dec_ref_points,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            query_pos=query_pos,
            bbox_embed=self.bbox_embed # 传入预测头用于中间层修正
        )
        
        # 7. 生成最终输出
        # hs shape: [Num_Layers, B, Num_Queries, C]
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            # 这里的 reference 是该层修正后的参考点
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference) # 转回 Logit 空间以便相加
            
            # 预测类别
            outputs_class = self.class_embed(hs[lvl])
            outputs_classes.append(outputs_class)
            
            # 预测框偏移量
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            # 坐标变换: ref + offset -> sigmoid
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            
        outputs_class = torch.stack(outputs_classes) # [Num_Layers, B, Nq, Class]
        outputs_coord = torch.stack(outputs_coords)  # [Num_Layers, B, Nq, 4]
        
        return outputs_class, outputs_coord

# ==========================================
# 7. 测试运行
# ==========================================
if __name__ == "__main__":
    # 模拟输入
    dummy_img = torch.randn(2, 3, 800, 800)
    
    # 实例化模型 (2层 Encoder, 2层 Decoder)
    model = DeformableDETR(num_layers=2)
    
    # 前向传播
    out_cls, out_box = model(dummy_img)
    
    print("-" * 30)
    print("模型输出检查:")
    print(f"输入 Batch Size: {dummy_img.shape[0]}")
    print(f"层数 (Layers): {out_cls.shape[0]}")
    print(f"查询数 (Queries): {out_cls.shape[2]}")
    print(f"分类 Logits Shape: {out_cls.shape}") # [2, 2, 300, 91]
    print(f"回归 Box Shape:    {out_box.shape}") # [2, 2, 300, 4]
    print("-" * 30)
    print("代码运行成功！")