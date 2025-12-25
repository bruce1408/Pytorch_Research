import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 核心组件: Camera-Aware DepthNet
# ==========================================
class CameraAwareDepthNet(nn.Module):
    """
    BEVDepth 的核心: 相机参数感知的深度预测网络
    """
    def __init__(self, in_channels=256, mid_channels=256, depth_bins=112):
        super().__init__()
        self.depth_bins = depth_bins
        
        # [分支 A] 图像特征处理
        # 简单的卷积层，用来调整特征通道和提取局部信息
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # [分支 B] 相机参数编码 (Camera-Awareness)
        # 输入维度: 27 (例如: 9个内参 + 12个外参 + 6个增强参数)
        self.bn_params = nn.BatchNorm1d(27)
        self.depth_mlp = nn.Sequential(
            nn.Linear(27, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # [融合] SE-Block 风格的注意力机制
        # 利用相机参数生成的权重，对图像特征进行重加权 (Re-weighting)
        self.se_attention = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.Sigmoid() 
        )
        
        # [输出] 深度分类头
        # 输出通道数 = 深度分桶的数量 (depth_bins)
        self.depth_head = nn.Conv2d(mid_channels, depth_bins, kernel_size=1)

    def forward(self, x, mats):
        """
        参数:
            x: (B * N_cam, C, H, W) 图像特征
            mats: (B * N_cam, 27) 相机参数
        返回:
            depth_logits: (B * N_cam, depth_bins, H, W) 深度预测结果(未Softmax)
        """
        # 1. 处理图像特征
        feat = self.reduce_conv(x)
        
        # 2. 处理相机参数
        # 先归一化，再通过 MLP 提取上下文信息
        mats = self.bn_params(mats)  # [12, 27] -> [12, 9内参 + 12外参 + 6增强参数]
        context = self.depth_mlp(mats) # 形状: (B*N, mid_channels)  -> [12, 256]
        
        # 3. 关键步骤: 融合 (Camera-Aware Fusion)
        # 将 context 扩展维度以便与 feat 进行广播乘法
        attn_weight = self.se_attention(context).unsqueeze(-1).unsqueeze(-1) # (B*N, C, 1, 1) -> [12, 256, 1, 1]
        feat = feat * attn_weight 
        
        # 4. 输出深度分布 Logits
        depth_logits = self.depth_head(feat)
        
        return depth_logits

# ==========================================
# 2. 辅助组件: 简单的 LSS View Transformer
# ==========================================
# ==========================================
# 2. 辅助组件: 内存安全的 LSS 模拟器
# ==========================================
class SimpleLSS(nn.Module):
    """
    轻量化版 LSS 模块
    注意：在真实生产环境中，这里必须使用 MMDetection3D 的 VoxelPooling CUDA 算子
    此处为了演示流程跑通，我们使用 插值(Interpolation) 来替代显存爆炸的张量操作
    """
    def __init__(self, out_channels=256, depth_bins=112):
        super().__init__()
        self.depth_bins = depth_bins
        self.out_channels = out_channels
        # 增加一个降维层，模拟从图像空间到BEV空间的特征压缩
        self.bev_compress = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, img_feat, depth_logits):
        """
        img_feat: (B*N, C, H, W)
        depth_logits: (B*N, D, H, W)
        """
        # 方案：直接将图像特征下采样到 BEV 的尺寸 (128x128)
        # 这样避免了构建 (B, C, D, H, W) 的 5维张量
        
        # 1. 先把特征图缩小一点，模拟特征提取的stride
        # 假设输入 256x704 -> 缩小到类似 BEV 的网格大小
        # 这里我们直接强制插值到 128x128，模拟投影结果
        bev_feat = F.interpolate(img_feat, size=(128, 128), mode='bilinear', align_corners=False)
        
        # 2. 利用深度信息加权 (可选，为了让 depth 参与计算)
        # 我们可以计算深度的置信度，用来给特征加权
        # 取深度图最大的概率值作为权重
        with torch.no_grad():
            depth_prob = F.softmax(depth_logits, dim=1) # (B*N, D, H, W)
            # 在深度维度求 max，得到 (B*N, 1, H, W)
            depth_confidence, _ = torch.max(depth_prob, dim=1, keepdim=True)
            # 将权重也插值到 128x128
            depth_weight = F.interpolate(depth_confidence, size=(128, 128), mode='bilinear')
            
        # 3. 特征加权
        bev_feat = bev_feat * depth_weight
        
        return bev_feat # (B*N, C, 128, 128)

# ==========================================
# 3. 完整模型: BEVDepth
# ==========================================
class BEVDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟 Backbone (例如 ResNet-50)
        # 输入假设是 3通道图片，输出 256通道特征
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 核心: DepthNet，都是卷积层的堆叠
        self.depth_net = CameraAwareDepthNet(in_channels=256, mid_channels=256, depth_bins=112)
        
        # 视图转换
        self.view_transformer = SimpleLSS(out_channels=256, depth_bins=112)
        
        # 任务头 (例如 3D 检测头)
        self.det_head = nn.Conv2d(256, 10, kernel_size=1) # 10个类别

    def get_depth_loss(self, depth_logits, gt_depth):
        """
        显式深度监督 Loss (Explicit Depth Supervision)
        """
        # gt_depth: (B*N, H, W), 值为 0~111 的整数, -1 表示无效点(无激光雷达数据)
        mask = gt_depth >= 0 
        
        # 只计算有效像素的 CrossEntropy Loss
        if mask.sum() > 0:
            # 展平以便计算 loss
            valid_logits = depth_logits.permute(0, 2, 3, 1)[mask] # (N_valid, 112)
            valid_gt = gt_depth[mask].long()                      # (N_valid,)
            loss = F.cross_entropy(valid_logits, valid_gt)
        else:
            loss = torch.tensor(0.0, device=depth_logits.device, requires_grad=True)
            
        return loss

    def forward(self, imgs, mats, gt_depth=None):
        B, N, C, H, W = imgs.shape
        # 将 Batch 和 Camera 维度合并
        imgs = imgs.view(B * N, C, H, W)
        mats = mats.view(B * N, -1)
        
        # 1. 提取特征 (B*N, 256, H, W) -> [12, 256, 128, 352]
        feat = self.backbone(imgs) 
        
        # 2. 预测深度 (调用 DepthNet) # (B*N, 112, H, W)
        depth_logits = self.depth_net(feat, mats) 
        
        # 3. 转换到 BEV [12, 256, 128, 128]
        bev_feat_cam = self.view_transformer(feat, depth_logits)
        
        # 4. 多相机特征融合 (这里简单求和，实际需按几何位置拼接) -> [2, 256, 128, 128]
        bev_feat = bev_feat_cam.view(B, N, 256, 128, 128).mean(dim=1)
        
        # 5. 检测头输出 2D 卷积输出10个类别
        preds = self.det_head(bev_feat)
        
        # 6. 计算 Loss (如果是训练模式且有标签)
        loss_dict = {}
        if gt_depth is not None:
            loss_depth = self.get_depth_loss(depth_logits, gt_depth.view(-1, H, W))
            loss_dict['loss_depth'] = loss_depth
            
        return preds, loss_dict

# ==========================================
# 4. 数据模拟与训练流程
# ==========================================
class RandomDataset(Dataset):
    def __init__(self, length=20):
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # 原来是 256, 704 -> 改小为 128, 352
        H_in, W_in = 128, 352 
        
        # 模拟数据
        imgs = torch.randn(6, 3, H_in, W_in)
        mats = torch.randn(6, 27)
        
        gt_depth = torch.randint(-1, 112, (6, H_in, W_in))
        mask = torch.rand(6, H_in, W_in) > 0.1
        gt_depth[mask] = -1
        
        gt_heatmap = torch.randn(10, 128, 128)
        
        return imgs, mats, gt_depth, gt_heatmap

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 初始化模型和优化器
    model = BEVDepthModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 准备数据
    train_loader = DataLoader(RandomDataset(length=10), batch_size=2)
    val_loader = DataLoader(RandomDataset(length=4), batch_size=2)
    
    epochs = 2
    
    # --- 循环开始 ---
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 1. 训练阶段 (Training)
        model.train()
        train_loss_sum = 0
        for batch_idx, (imgs, mats, gt_depth, gt_heatmap) in enumerate(train_loader):
            # [2, 6, 3, 128, 352] [2, 6, 27]
            imgs, mats = imgs.to(device), mats.to(device) 
            
            # [2, 6, 128, 352] [2, 10, 128, 128]
            gt_depth, gt_heatmap = gt_depth.to(device), gt_heatmap.to(device) 
            
            optimizer.zero_grad()
            
            # 前向传播
            preds, loss_dict = model(imgs, mats, gt_depth)
            
            # 计算总 Loss
            # BEVDepth 论文强调深度监督的权重通常设得比较大 (比如 3.0)
            loss_depth = loss_dict['loss_depth']
            loss_det = F.mse_loss(preds, gt_heatmap) # 模拟检测 Loss
            total_loss = loss_det + 3.0 * loss_depth
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item()
            print(f" [Train] Batch {batch_idx}: Depth Loss={loss_depth.item():.4f}, Det Loss={loss_det.item():.4f}")
            
        print(f" -> Avg Train Loss: {train_loss_sum / len(train_loader):.4f}")
        
        # 2. 验证阶段 (Validation)
        model.eval()
        val_loss_sum = 0
        with torch.no_grad(): # 验证不计算梯度，节省显存
            for batch_idx, (imgs, mats, gt_depth, gt_heatmap) in enumerate(val_loader):
                imgs, mats = imgs.to(device), mats.to(device)
                gt_depth, gt_heatmap = gt_depth.to(device), gt_heatmap.to(device)
                
                preds, loss_dict = model(imgs, mats, gt_depth)
                
                loss_depth = loss_dict['loss_depth']
                loss_det = F.mse_loss(preds, gt_heatmap)
                total_loss = loss_det + 3.0 * loss_depth
                
                val_loss_sum += total_loss.item()
                
        print(f" -> Avg Val Loss: {val_loss_sum / len(val_loader):.4f}")

if __name__ == "__main__":
    main()