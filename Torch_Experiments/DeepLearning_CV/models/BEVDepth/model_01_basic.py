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
        mats = self.bn_params(mats)
        context = self.depth_mlp(mats) # 形状: (B*N, mid_channels)
        
        # 3. 关键步骤: 融合 (Camera-Aware Fusion)
        # 将 context 扩展维度以便与 feat 进行广播乘法
        attn_weight = self.se_attention(context).unsqueeze(-1).unsqueeze(-1) # (B*N, C, 1, 1)
        feat = feat * attn_weight 
        
        # 4. 输出深度分布 Logits
        depth_logits = self.depth_head(feat)
        
        return depth_logits

# ==========================================
# 2. 辅助组件: 简单的 LSS View Transformer
# ==========================================
class SimpleLSS(nn.Module):
    """
    简化的 Lift-Splat-Shoot 模块，用于将 2D 特征和深度转换为 BEV 特征
    (注: 工业界通常使用 CUDA 加速的 Voxel Pooling，此处仅为逻辑演示)
    """
    def __init__(self, out_channels=256, depth_bins=112):
        super().__init__()
        self.depth_bins = depth_bins
        self.out_channels = out_channels

    def forward(self, img_feat, depth_logits):
        """
        img_feat: (B*N, C, H, W)
        depth_logits: (B*N, D, H, W)
        """
        # 1. Depth Softmax: 将 Logits 转为概率分布
        depth_probs = F.softmax(depth_logits, dim=1) # (B*N, D, H, W)
        
        # 2. Lift: 外积生成视锥特征 (Frustum Features)
        # 形状: (B*N, C, 1, H, W) * (B*N, 1, D, H, W) -> (B*N, C, D, H, W)
        # 为了演示代码能跑通，我们做一个极其简化的模拟操作：
        # 假设我们将深度维 D 和高度维 H 平均池化掉，模拟投影到平面的过程
        
        # 模拟：特征加权
        # 在真实 LSS 中，这里会生成点云，然后 scatter 到 BEV 网格
        weighted_feat = (img_feat.unsqueeze(2) * depth_probs.unsqueeze(1)).sum(dim=2) # 消掉 D 维度
        
        # 模拟：投影到 BEV (直接插值改变尺寸)
        # 假设 BEV 网格大小是 128x128
        bev_feat = F.interpolate(weighted_feat, size=(128, 128), mode='bilinear')
        
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
        
        # 核心: DepthNet
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
        
        # 1. 提取特征
        feat = self.backbone(imgs) # (B*N, 256, H, W)
        
        # 2. 预测深度 (调用 DepthNet)
        depth_logits = self.depth_net(feat, mats) # (B*N, 112, H, W)
        
        # 3. 转换到 BEV
        bev_feat_cam = self.view_transformer(feat, depth_logits)
        
        # 4. 多相机特征融合 (这里简单求和，实际需按几何位置拼接)
        bev_feat = bev_feat_cam.view(B, N, 256, 128, 128).mean(dim=1)
        
        # 5. 检测头输出
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
        # 模拟数据: Batch=1, 6个相机, 256x704图片
        imgs = torch.randn(6, 3, 256, 704)
        mats = torch.randn(6, 27) # 相机参数
        
        # 模拟 LiDAR 深度真值: 0~111, -1表示空点(稀疏)
        # 只有约 10% 的点有深度值
        gt_depth = torch.randint(-1, 112, (6, 256, 704))
        mask = torch.rand(6, 256, 704) > 0.1
        gt_depth[mask] = -1
        
        # 模拟检测真值 Heatmap
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
            imgs, mats = imgs.to(device), mats.to(device)
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