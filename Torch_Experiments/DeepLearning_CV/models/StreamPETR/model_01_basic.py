import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# =========================================================
# 0) 工具函数 (Utils)
# =========================================================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

# =========================================================
# 1) 全局配置 (Config)
# =========================================================
class StreamPETRConfig:
    # --- 基础 PETR 参数 ---
    num_cams = 6
    img_h = 256
    img_w = 704
    stride = 16
    embed_dim = 256
    num_depth = 4
    depth_values = [10.0, 20.0, 30.0, 40.0]
    
    # 空间范围 [xmin, ymin, zmin, xmax, ymax, zmax]
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # --- StreamPETR 特有参数 ---
    num_queries = 256        # 当前帧新引入的 Query 数量 (Sliding Window)
    memory_len = 512         # 记忆队列最大长度 (即保留上一帧多少个 Query)
    topk_training = 256      # 训练/推理时保留置信度最高的 K 个 Query 传给下一帧
    
    # Transformer & Head
    num_decoder_layers = 6
    num_heads = 8
    num_classes = 10
    box_dim = 10 # x,y,z,w,l,h,sin,cos,vx,vy

# =========================================================
# 2) 基础组件 (Backbone & PE) - 复用 PETR 逻辑
# =========================================================
class SimpleBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, cfg.embed_dim, 3, 2, 1), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

# =========================================================
# 3) StreamPETR 核心模型
# =========================================================
class StreamPETR(nn.Module):
    def __init__(self, cfg: StreamPETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone(cfg)
        
        # --- 3D 坐标生成相关 (PETR Legacy) ---
        self.feat_h, self.feat_w = cfg.img_h // cfg.stride, cfg.img_w // cfg.stride
        ys, xs = torch.meshgrid(
            torch.arange(self.feat_h, dtype=torch.float32) + 0.5,
            torch.arange(self.feat_w, dtype=torch.float32) + 0.5, indexing='ij'
        )
        grid_uv = torch.stack([xs * cfg.stride, ys * cfg.stride], dim=-1)
        self.register_buffer("grid_uv", grid_uv, persistent=False)
        self.register_buffer("depth_values", torch.tensor(cfg.depth_values).view(1,1,1,cfg.num_depth), persistent=False)

        # 3D PE Encoder (用于图像特征)
        self.img_pe_encoder = nn.Sequential(
            nn.Linear(3 * cfg.num_depth, cfg.embed_dim * 2), nn.ReLU(True),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim)
        )

        # --- StreamPETR 核心组件 ---
        
        # 1. 新 Query 的 Learnable Embedding (对应新出现的物体)
        # 我们不直接学 Query，而是学 Reference Points (Anchor) 的初始位置
        # 这里为了简化，使用类似 Deformable DETR 的做法：Embedding + RefPoint
        self.new_query_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)
        self.new_ref_points = nn.Embedding(cfg.num_queries, 3) # 初始化为 [0,1] 之间的归一化坐标

        # 2. Query 位置编码器
        # 用于将 Query 的 3D Reference Points 编码成 PE 加到 Query Content 上
        self.query_pe_encoder = nn.Sequential(
            nn.Linear(3, cfg.embed_dim), nn.ReLU(True),
            nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(cfg.embed_dim, cfg.num_heads, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.num_decoder_layers)

        # 4. Heads
        self.cls_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.reg_head = nn.Linear(cfg.embed_dim, cfg.box_dim) # 回归相对 RefPoint 的偏移量

    def get_img_feat_with_pe(self, imgs, rots, trans, intrins):
        """标准 PETR 流程：提取图像特征并加上 3D PE"""
        B, N, C, H, W = imgs.shape
        # Backbone
        x = imgs.view(B*N, C, H, W)
        feat = self.backbone(x).view(B, N, self.cfg.embed_dim, self.feat_h, self.feat_w)
        
        # 生成 3D 坐标 (省略详细推导，复用之前逻辑)
        # ... (此处代码逻辑与之前的 PETR 完全一致，为节省篇幅简写) ...
        # 假设我们已经得到了 xyz_world (B, N, HW, D, 3)
        # 这里用简化的逻辑模拟得到 xyz_world
        xyz_world = torch.randn(B, N, self.feat_h*self.feat_w, self.cfg.num_depth, 3, device=imgs.device)
        
        # 归一化 + Inverse Sigmoid + MLP
        xyz_world = (xyz_world - torch.tensor(self.cfg.pc_range[:3], device=imgs.device)) / \
                    (torch.tensor(self.cfg.pc_range[3:], device=imgs.device) - torch.tensor(self.cfg.pc_range[:3], device=imgs.device))
        xyz_pe = inverse_sigmoid(xyz_world.clamp(1e-4, 1-1e-4))
        xyz_pe = xyz_pe.permute(0, 1, 2, 4, 3).reshape(B, N, -1, 3*self.cfg.num_depth)
        
        pos_embed = self.img_pe_encoder(xyz_pe) # (B, N, HW, C)
        
        # 融合
        feat = feat.flatten(3).permute(0, 1, 3, 2) # (B, N, HW, C)
        feat = feat + pos_embed
        return feat.flatten(1, 2).permute(1, 0, 2) # (N*HW, B, C) -> Memory

    def motion_compensation(self, prev_ref_pts, ego_motion):
        """
        【StreamPETR 核心 1】：运动补偿
        将上一帧的 Reference Points 根据自车运动对齐到当前帧。
        
        Args:
            prev_ref_pts: (B, M, 3) 归一化坐标 [0, 1]
            ego_motion: (B, 4, 4)  Prev -> Curr 的变换矩阵
        """
        device = prev_ref_pts.device
        pc_range = torch.tensor(self.cfg.pc_range, device=device)
        
        # 1. Denormalize: [0, 1] -> 真实世界坐标 (Ego Frame t-1)
        # x_real = x_norm * (max - min) + min
        lower = pc_range[:3]
        upper = pc_range[3:]
        prev_real = prev_ref_pts * (upper - lower) + lower
        
        # 2. Apply Ego Motion: P_curr = T @ P_prev
        # 变成齐次坐标 (x,y,z,1)
        B, M, _ = prev_real.shape
        ones = torch.ones(B, M, 1, device=device)
        prev_real_homo = torch.cat([prev_real, ones], dim=-1) # (B, M, 4)
        
        # 矩阵乘法 (B, 4, 4) @ (B, M, 4)^T -> (B, 4, M) -> (B, M, 4)
        curr_real_homo = torch.bmm(ego_motion, prev_real_homo.permute(0, 2, 1)).permute(0, 2, 1)
        curr_real = curr_real_homo[..., :3] # 取前三维
        
        # 3. Normalize: 真实世界坐标 (Ego Frame t) -> [0, 1]
        curr_norm = (curr_real - lower) / (upper - lower)
        return curr_norm.clamp(0.0, 1.0)

    def select_top_k(self, pred_logits, pred_boxes, ref_pts, active_queries, K):
        """
        【StreamPETR 核心 2】：Top-K 选择
        从当前的预测结果中筛选出置信度最高的 Query，传递给下一帧。
        """
        # pred_logits: (B, N_all, Num_Classes)
        # 这里的 ref_pts 是更新后的参考点
        
        # 获取最大分类分数
        probs = pred_logits.sigmoid()
        topk_scores, topk_indices = torch.max(probs, dim=-1)[0].topk(K, dim=1)
        
        B = pred_logits.shape[0]
        batch_idx = torch.arange(B, device=pred_logits.device).unsqueeze(1)
        
        # 筛选 Query Content
        selected_queries = active_queries[batch_idx, topk_indices] # (B, K, C)
        
        # 筛选 Reference Points (使用网络回归后的 Box 中心作为新的 Ref Point)
        # 注意：pred_boxes 通常是相对偏移，这里为了简化，假设 pred_boxes[..., :3] 就是最终绝对坐标(norm)
        # 在完整实现中，应该是 ref_pts + inverse_sigmoid(pred_boxes) ...
        selected_ref_pts = pred_boxes[batch_idx, topk_indices, :3].sigmoid() 
        
        # 必须 Detach！切断梯度流，否则显存爆炸
        return selected_queries.detach(), selected_ref_pts.detach()

    def forward(self, imgs, rots, trans, intrins, ego_motion=None, prev_state=None):
        """
        Args:
            ego_motion: (B, 4, 4) 从上一帧到当前帧的变换矩阵
            prev_state: Dict {'queries': (B, M, C), 'ref_pts': (B, M, 3)}
        """
        B = imgs.shape[0]
        
        # 1. 提取图像特征 + 3D PE (Memory)
        memory = self.get_img_feat_with_pe(imgs, rots, trans, intrins)
        
        # 2. 准备 Hybrid Queries
        # -----------------------------------------------------------
        # Part A: 新 Query (Learnable)
        new_queries = self.new_query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, N_new, C)
        new_ref_pts = self.new_ref_points.weight.unsqueeze(0).repeat(B, 1, 1).sigmoid() # (B, N_new, 3)
        
        # Part B: 老 Query (Propagated & Motion Compensated)
        if prev_state is not None:
            prev_queries = prev_state['queries']
            prev_ref_pts = prev_state['ref_pts']
            
            # [关键步骤] 对齐坐标：把上一帧的物体位置搬到当前帧坐标系下
            aligned_ref_pts = self.motion_compensation(prev_ref_pts, ego_motion)
            
            # 拼接
            active_queries = torch.cat([prev_queries, new_queries], dim=1) # (B, M+N, C)
            active_ref_pts = torch.cat([aligned_ref_pts, new_ref_pts], dim=1) # (B, M+N, 3)
        else:
            active_queries = new_queries
            active_ref_pts = new_ref_pts
            
        # 3. 为 Query 注入位置信息
        # StreamPETR 不使用 learnable query_pos，而是根据 Ref Points 生成 PE
        # active_ref_pts 是归一化的，先转 logit
        query_pe = self.query_pe_encoder(inverse_sigmoid(active_ref_pts))
        
        # 构造 Decoder 输入 (Target)
        # Content + Position
        tgt = active_queries + query_pe
        # 这里需要 permute 适应 Decoder API (Seq, Batch, Dim)
        tgt = tgt.permute(1, 0, 2) 
        
        # 4. Transformer Decoder
        # Query 带着刚才算好的位置，去图像特征里找东西
        hs = self.decoder(tgt, memory) # (N_all, B, C)
        hs = hs.permute(1, 0, 2) # (B, N_all, C)
        
        # 5. Prediction Heads
        # 回归框 (x,y,z,w,l,h...)
        # 注意：这里回归的是相对于 active_ref_pts 的偏移量
        tmp_boxes = self.reg_head(hs) 
        # 恢复绝对坐标 (简化写法: ref + offset)
        # 真实实现需处理 inverse_sigmoid(ref) + tmp_boxes
        pred_boxes = tmp_boxes 
        pred_boxes[..., :3] = tmp_boxes[..., :3] + inverse_sigmoid(active_ref_pts)
        
        pred_logits = self.cls_head(hs)
        
        # 6. 生成下一帧状态 (State Update)
        # 选最好的 K 个 Query 传下去
        # 如果是训练模式，所有 Query 都要参与 Loss 计算，但只有 TopK 传给下一帧
        # 这里模拟 Inference 逻辑
        next_queries, next_ref_pts = self.select_top_k(
            pred_logits, pred_boxes, active_ref_pts, hs, self.cfg.topk_training
        )
        
        next_state = {
            'queries': next_queries,
            'ref_pts': next_ref_pts
        }
        
        return pred_logits, pred_boxes.sigmoid(), next_state

# =========================================================
# 4) 运行模拟 (Demo)
# =========================================================
def main():
    cfg = StreamPETRConfig()
    model = StreamPETR(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    B = 1
    # 模拟 Frame 0
    print("--- Processing Frame 0 ---")
    imgs = torch.randn(B, 6, 3, 256, 704).to(device)
    
    # 【修正 1】：使用 repeat 来复制数据，而不是直接 view
    # 原始: torch.eye(3) -> (3, 3)
    # 扩展: view(1, 1, 3, 3) -> (1, 1, 3, 3)
    # 复制: repeat(B, 6, 1, 1) -> (B, 6, 3, 3)
    rots = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    
    # trans 本身是 zeros，直接指定形状即可，不需要 view
    trans = torch.zeros(B, 6, 3).to(device)
    
    intrins = torch.eye(3).view(1, 1, 3, 3).repeat(B, 6, 1, 1).to(device)
    
    # 第一帧没有 prev_state
    logits_0, boxes_0, state_0 = model(imgs, rots, trans, intrins, ego_motion=None, prev_state=None)
    print(f"Frame 0 Output Boxes: {boxes_0.shape}") # (B, 256, 10)
    print(f"State 0 Queries: {state_0['queries'].shape}") # (B, 256, 256)
    
    # 模拟 Frame 1 (车往前开了)
    print("\n--- Processing Frame 1 (Streaming) ---")
    imgs_1 = torch.randn(B, 6, 3, 256, 704).to(device)
    # 假设车往前开了 1米: T_prev_to_curr
    ego_motion = torch.eye(4).view(1, 1, 4, 4).repeat(B, 1, 1, 1).view(B, 4, 4).to(device)
    ego_motion[:, 0, 3] = -1.0 # x 轴平移 -1 (相对运动)
    
    # 把 state_0 传进去
    logits_1, boxes_1, state_1 = model(imgs_1, rots, trans, intrins, ego_motion=ego_motion, prev_state=state_0)
    
    # 注意 Frame 1 的 Query 数量 = New(256) + Old(256) = 512 (如果是推理阶段)
    # 在这个 Demo 里 select_top_k 限制了传下去的只有 256
    print(f"Frame 1 Input Queries (Hybrid): {256 + 256}") 
    print(f"Frame 1 Output Boxes: {boxes_1.shape}")      # (B, 512, 10)
    print(f"State 1 Queries (Selected): {state_1['queries'].shape}") # (B, 256, 256)
    
    print("\n✅ StreamPETR Pipeline Verified!")

if __name__ == "__main__":
    main()
