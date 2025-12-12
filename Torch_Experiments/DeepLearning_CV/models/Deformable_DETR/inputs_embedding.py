import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    标准的 2D 正弦位置编码。
    将通道分为两半：一半编码 Y 轴，一半编码 X 轴。
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        """
        初始化函数

        参数:
            num_pos_feats (int): 位置编码向量维度的一半。最终的维度将是 num_pos_feats * 2。
            temperature (int): 一个用于缩放位置编码波长的超参数。
            normalize (bool): 如果为True，则在计算后对位置编码进行归一化。
            scale (float, optional): 如果设置了 normalize=True，则用这个值来缩放归一化的结果。
        """
        super().__init__()
        # num_pos_feats 通常是 d_model 的一半 (例如 256/2 = 128)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        """
        Args:
            mask: [Batch, H, W] 的布尔张量。
                  True 表示是 Padding (无效区域)，False 表示是真实像素。
        Returns:
            pos: [Batch, 2*num_pos_feats, H, W] -> [Batch, 256, H, W]
        """
        not_mask = ~mask # 取反，1 代表真实像素
        
        
        # --- 1. 计算每个像素的物理坐标 (y_embed, x_embed) ---
        # `cumsum` 是累加和。沿着维度1(H)累加，可以得到每个像素的y坐标。
        # 即使有 Padding，`cumsum` 也能正确计算出有效区域内的坐标。
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        
        # 沿着维度2(W)累加，得到每个像素的x坐标。
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # --- 2. 对坐标进行归一化 (可选但推荐) ---
        if self.normalize:
            eps = 1e-6  # 防止除以零的小常数
            
            # y_embed[:, -1:, :] 获取的是每行的有效高度。将y坐标除以总高度，实现归一化。
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            
            # x_embed[:, :, -1:] 获取的是每列的有效宽度。将x坐标除以总宽度，实现归一化。
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale


        # --- 3. 生成用于计算 sin/cos 的频率项 (分母) ---
        # dim_t 是一个序列: [0, 1, 2, ..., 127]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        # 这是位置编码的数学核心。
        # `2 * (dim_t // 2)` 会生成 [0, 0, 2, 2, 4, 4, ...] 的序列。
        # 这确保了 sin 和 cos 函数对使用相同的频率(波长)，只是相位不同。
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        

        # --- 4. 计算正弦和余弦编码 ---
        # `unsqueeze(-1)` 增加一个维度用于广播。
        # [B, H, W] / [128] -> [B, H, W, 128]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        
        # 将 pos_x/pos_y 的奇数位和偶数位分别送入 sin 和 cos 函数。
        # 偶数索引 (0, 2, 4,...) 用于 sin, 奇数索引 (1, 3, 5,...) 用于 cos。
        # `stack` 在新维度(dim=4)上堆叠，然后 `flatten(3)` 将最后两个维度展平。
        # 最终形成 [sin(pos_0), cos(pos_0), sin(pos_1), cos(pos_1), ...] 的交错形式。
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # --- 5. 拼接 X 和 Y 轴的编码，得到最终结果 ---
        # 按照约定，Y轴编码在前，X轴在后。
        # [B, H, W, 128+128] -> [B, H, W, 256]
        # `permute` 调整维度顺序以匹配 PyTorch 的 [B, C, H, W] 格式。
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos
    

class DeformablePositionEmbedding(nn.Module):
    """
    为 Deformable DETR 的多尺度特征图生成混合位置编码。
    该编码由两部分相加而成：
    1. 固定的 2D 正弦空间编码 (Spatial Encoding)
    2. 可学习的层级编码 (Level Embedding)
    """
    
    def __init__(self, num_levels, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # 1. 基础的空间位置编码生成器 (Spatial)
        # d_model 的一半给 X，一半给 Y
        self.spatial_pos_generator = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        
        # 2. 定义可学习的层级编码 (Level Embedding)。这是关键！
        # `nn.Parameter` 将这个张量注册为模型的可学习参数，它会在反向传播中被更新。
        # 形状为 [num_levels, d_model]，例如 [4, 256]。
        # 每一行代表一个特征层级的“身份”向量。
        self.level_embed = nn.Parameter(torch.randn(num_levels, d_model))

    def forward(self, feature_list, mask_list):
        """
        Args:
            feature_list: 包含 4 个特征图的列表，每个形状为 [Batch, 256, H_i, W_i]
            mask_list: 包含 4 个 mask 的列表，每个形状为 [Batch, H_i, W_i]
        Returns:
            all_pos_embeds: 展平并拼接好的位置编码，形状 [Batch, Total_Len, 256]
        """
        pos_embeds_list = []

        # 遍历每一层 (Level 0 ~ Level 3)
        for i, (feature, mask) in enumerate(zip(feature_list, mask_list)):
            
            # --- A. 生成基础 2D 空间位置编码 ---
            # shape: [Batch, 256, H_i, W_i]
            spatial_pos = self.spatial_pos_generator(mask)
            
            # --- B. 获取并加上该层的“层级身份编码” ---
            # 从可学习的 `level_embed` 中取出第 i 层的身份向量。
            # level_vec 形状: [256]
            level_vec = self.level_embed[i]
            
            # 这里是核心：通过广播机制将空间编码和层级编码相加。
            # `level_vec` 被变形为 [1, 256, 1, 1]，以便和 [B, 256, H_i, W_i] 的 spatial_pos 相加。
            # 结果 `final_pos_layer` 的每个位置，既包含了自己在图中的(x,y)信息，也包含了它来自哪个层级的信息。
            final_pos_layer = spatial_pos + level_vec.view(1, -1, 1, 1)
            
            # --- C. 展平 (Flatten)，为送入 Transformer 做准备 ---
            # [B, 256, H, W] -> [B, 256, H*W] -> [B, H*W, 256]
            # `flatten(2)` 将 H, W 维度展平。
            # `transpose(1, 2)` 将通道维度和序列长度维度交换，以符合 Transformer 的输入格式。
            flattened_pos = final_pos_layer.flatten(2).transpose(1, 2)
            
            pos_embeds_list.append(flattened_pos)

        # --- D. 拼接所有层 (Concat) ---
        # 形状: [Batch, Sum(H_i*W_i), 256] -> [Batch, 13294, 256]
        all_pos_embeds = torch.cat(pos_embeds_list, dim=1)
        
        return all_pos_embeds
    

if __name__ == "__main__":
    # --- 模拟输入数据 ---
    batch_size = 2
    d_model = 256
    num_levels = 4
    
    # 假设 4 层特征图的尺寸 (H, W)
    spatial_shapes = [(100, 100), (50, 50), (25, 25), (13, 13)]
    
    # 构造虚假的特征图和 Mask
    features = []
    masks = []
    for (h, w) in spatial_shapes:
        
        # 特征图: [B, 256, H, W]
        features.append(torch.randn(batch_size, d_model, h, w))
        
        # Mask: [B, H, W] (全为 False，表示无 Padding)
        masks.append(torch.zeros((batch_size, h, w), dtype=torch.bool))

    # --- 实例化 Deformable Pos Embed 模块 ---
    deformable_pos_module = DeformablePositionEmbedding(num_levels, d_model)
    
    # --- 前向计算 ---
    output_pos = deformable_pos_module(features, masks)
    
    # --- 打印结果 ---
    print(f"输入层级数: {num_levels}")
    print(f"每层尺寸: {spatial_shapes}")
    
    total_len = sum([h*w for h, w in spatial_shapes]) # 10000+2500+625+169 = 13294
    print(f"预期总序列长度: {total_len}")
    
    print("-" * 30)
    print(f"最终输出的位置编码 Shape: {output_pos.shape}") 
    # 预期: [2, 13294, 256]
    
    # 验证 Level Embedding 是否加上了
    # 我们检查第一层 (Level 0) 的一个点和第二层 (Level 1) 的一个点
    # 理论上，spatial part 是 sin/cos 计算的，level part 是 parameter
    print("\nLevel Embedding 参数形状:", deformable_pos_module.level_embed.shape)