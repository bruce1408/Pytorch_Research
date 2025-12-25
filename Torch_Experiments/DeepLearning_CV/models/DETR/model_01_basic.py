import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Mock Backbone (模拟骨干网络)
# ==========================================
class MockBackbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # 假设我们用 ResNet50，最后输出通道通常是 2048
        # 这里我们模拟输入一张图片，输出降采样 32 倍的特征图
        self.output_dim = 2048 

    def forward(self, x):
        
        # x shape: [batch_size, 3, H, W]
        batch, _, h, w = x.shape
        
        # 模拟 ResNet 最后一层的输出: 尺寸缩小 32 倍
        # 返回随机特征图
        return torch.randn(batch, self.output_dim, h // 32, w // 32)

# ==========================================
# 2. DETR 主模型架构
# ==========================================
class DETR(nn.Module):
    def __init__(self, 
                 num_classes, 
                 hidden_dim=256, 
                 nheads=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 num_queries=100
        ):
        
        super().__init__()
        
        # --- 组件 A: 骨干网络 ---
        self.backbone = MockBackbone()
        
        # --- 组件 B: 投影层 (Channel Mapper) ---
        # 将骨干网的 2048 维特征映射到 Transformer 的 256 维
        self.input_proj = nn.Conv2d(self.backbone.output_dim, hidden_dim, kernel_size=1)
        
        # --- 组件 C: Transformer ---
        # PyTorch 自带的 Transformer 模块
        self.transformer = nn.Transformer(
            d_model = hidden_dim,
            nhead = nheads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            batch_first=False # 注意：PyTorch默认序列维度在第一位 (Seq, Batch, Dim)
        )
        
        # --- 组件 D: Learnable Queries (核心!) ---
        # 这是 DETR 的精髓：100 个可学习的“提问者/锚点”
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # --- 组件 E: 预测头 (Prediction Heads) ---
        # 类别预测 (多加 1 个类别用于 "No Object" 背景类)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # 坐标预测 (center_x, center_y, w, h)
        self.bbox_embed = nn.Linear(hidden_dim, 4) # MLP

        # --- 位置编码 (简化版: 可学习的位置编码) ---
        # 实际论文用的是正弦余弦编码，这里为了代码简洁使用可学习参数代替
        # 分别对 row (y) 和 col (x) 进行编码
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)

    def forward(self, x):
        
        # 1. 骨干网络提取特征
        features = self.backbone(x) # [Batch, 2048, h, w]
        
        # 2. 降维: 2048 -> 256
        h = self.input_proj(features) # [Batch, 256, h, w]
        
        # 3. 构造位置编码 (Positional Encoding)
        B, C, H, W = h.shape
        
        # 生成网格位置编码
        pos_x = self.col_embed(torch.arange(W).to(x.device)).unsqueeze(0).repeat(H, 1, 1)
        pos_y = self.row_embed(torch.arange(H).to(x.device)).unsqueeze(1).repeat(1, W, 1)
        pos = torch.cat([pos_x, pos_y], dim=-1).flatten(0, 1).unsqueeze(1).repeat(1, B, 1) 
        # pos shape: [HW, Batch, 256]

        # 4. 准备 Transformer 输入
        # Flatten: [Batch, 256, H, W] -> [H*W, Batch, 256]
        # Transformer 需要序列作为第一维度 (Seq_Len, Batch, Dim)
        src = h.flatten(2).permute(2, 0, 1)
        
        # 5. 准备 Object Queries
        # [100, 256] -> [100, Batch, 256]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        
        # ================== 修改核心开始 ==================
        # 6. Transformer 前向传播 (修复点)
        # 标准 nn.Transformer 不支持 query_pos_embed 参数
        # 我们需要手动将位置编码(pos)加到 src 上
        # 对于 tgt (Decoder输入)，直接使用 object queries 即可
        
        hs = self.transformer(
            src = src + pos,      # Encoder 输入: 图像特征 + 位置编码
            tgt = query_embed     # Decoder 输入: Object Queries 本身就是“锚点”
        )
        
        # ================== 修改核心结束 ==================

        # hs 输出 shape: [100, Batch, 256]
        
        # 7. 预测头
        # 变回 [Batch, 100, 256]
        hs = hs.permute(1, 0, 2)
        
        # [Batch, 100, num_classes+1]
        outputs_class = self.class_embed(hs)            
        
        # [Batch, 100, 4]
        outputs_coord = self.bbox_embed(hs).sigmoid()   
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

# ==========================================
# 3. 匈牙利匹配器 (Hungarian Matcher)
# ==========================================
# 这是 DETR 训练的核心：如何把 100 个预测框 和 3 个真实框 对应起来？
class SimpleHungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad() # 匹配过程不需要梯度
    def forward(self, outputs, targets):
        """
        outputs: 模型的输出字典
        targets: 真实标签列表 [{'boxes':..., 'labels':...}, ...]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 扁平化所有 Batch 的数据，方便统一计算代价矩阵
        # 概率分布 [Batch * 100, NumClass]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  
        # 预测框 [Batch * 100, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # 2. 拼接所有真实标签
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. 计算代价矩阵 (Cost Matrix)
        # 3.1 分类代价: 取出对应真实类别的预测概率。概率越高，代价越小 (取负号)
        cost_class = -out_prob[:, tgt_ids]

        # 3.2 回归代价: 简单使用 L1 距离 (真实场景通常加 GIoU)
        # cdist 计算成对距离
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 总代价
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        # Reshape 回 [Batch, 100, Num_Target_in_Batch]
        C = C.view(bs, num_queries, -1).cpu()

        # 4. 使用 scipy 的 linear_sum_assignment 进行二分图匹配
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # 返回匹配结果: [(pred_idx_1, tgt_idx_1), (pred_idx_2, tgt_idx_2), ...]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]

# ==========================================
# 4. 运行 Demo Pipeline
# ==========================================
if __name__ == "__main__":
    # --- 配置 ---
    num_classes = 20  # 假设是 VOC 数据集
    batch_size = 2
    
    # 1. 初始化模型
    detr = DETR(num_classes=num_classes, num_queries=100) # 预测 100 个框
    
    # 2. 模拟输入数据 (Batch=2, RGB, 800x800)
    dummy_img = torch.randn(batch_size, 3, 800, 800)
    
    # 3. 模拟 Ground Truth (假设每张图里有随机数量的目标)
    targets = [
        {'labels': torch.tensor([1, 5]), 'boxes': torch.rand(2, 4)}, # 第1张图有2个目标
        {'labels': torch.tensor([3]),    'boxes': torch.rand(1, 4)}  # 第2张图有1个目标
    ]
    
    # --- 前向推理 ---
    print(">>> 开始前向传播...")
    outputs = detr(dummy_img)
    
    print(f"预测 Logits 形状: {outputs['pred_logits'].shape} (Batch, 100, Class+1)")
    print(f"预测 Boxes 形状:  {outputs['pred_boxes'].shape}  (Batch, 100, 4)")
    
    # --- 匈牙利匹配 ---
    print("\n>>> 开始计算二分图匹配 (Assignment)...")
    matcher = SimpleHungarianMatcher()
    indices = matcher(outputs, targets)
    
    # 打印匹配结果
    for i, (src_idx, tgt_idx) in enumerate(indices):
        print(f"Batch {i}:")
        print(f"  模型预测的第 {src_idx.tolist()} 个 Query")
        print(f"  匹配到了第 {tgt_idx.tolist()} 个 真实目标")
        
    print("\n>>> Pipeline 运行成功！接下来就是计算 Loss 反向传播了。")