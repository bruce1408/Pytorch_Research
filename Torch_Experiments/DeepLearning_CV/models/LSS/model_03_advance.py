import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LSS_Core_Complete(nn.Module):
    def __init__(self, grid_conf, data_conf, input_channels, num_classes):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_conf = data_conf
        
        # --- 优化点 1: 使用 register_buffer 自动管理设备 ---
        # 这样当 model.cuda() 时，这些参数会自动转到 GPU，不需要手动 .to(device)
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'], 
                                    self.grid_conf['ybound'], 
                                    self.grid_conf['zbound'])
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)
        
        # 1. Lift: 图像编码器
        self.D = data_conf['D']
        self.C = 64
        self.cam_encode = nn.Conv2d(input_channels, self.D + self.C, kernel_size=1)
        
        # 预计算视锥 (Frustum)
        self.frustum = self.create_frustum()

        # 2. BEV Encoder
        self.bev_encode = nn.Sequential(
            nn.Conv2d(self.C, self.C, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(True),
            nn.Conv2d(self.C, num_classes, 1)
        )

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx.long()

    def create_frustum(self):
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        H, W = self.data_conf['img_size']
        xs = torch.linspace(0, W - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, H - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)
        frustum = torch.stack((xs, ys, ds.expand(D, H, W)), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrinsics):
        B, N, _ = trans.shape
        # --- 修复点 1: 确保模拟数据在正确的设备上 ---
        # 之前的 torch.randn 默认在 CPU，导致与 GPU 上的 bx 运算报错
        coords_xyz = torch.randn(B, N, self.D, *self.data_conf['img_size'], 3, device=trans.device)
        
        # 模拟坐标归一化到 grid 范围
        coords_xyz[..., 0] *= 50.0 
        coords_xyz[..., 1] *= 50.0 
        coords_xyz[..., 2] *= 2.0  
        return coords_xyz

    def get_cam_feats(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        x = self.cam_encode(x)
        depth_logits = x[:, :self.D]
        context = x[:, self.D:]
        depth_probs = depth_logits.softmax(dim=1)
        context = context.unsqueeze(1)
        depth_probs = depth_probs.unsqueeze(2)
        cam_feats = context * depth_probs
        return cam_feats

    def voxel_pooling(self, geom_feats, x, y, z):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        
        # --- 修复点 2: 使用 reshape 替代 view ---
        # 因为 geom_feats 经过了 permute，内存不连续，view 会报错
        geom_feats = geom_feats.reshape(Nprime, C)
        
        x = x.reshape(Nprime)
        y = y.reshape(Nprime)
        z = z.reshape(Nprime)

        # 生成 Batch 索引
        batch_indices = torch.arange(B, device=geom_feats.device).view(B, 1, 1, 1, 1).expand(B, N, D, H, W).reshape(Nprime)

        # 过滤越界点 (self.nx 已经在 GPU 上了)
        mask = (x >= 0) & (x < self.nx[0]) & (y >= 0) & (y < self.nx[1]) & (z >= 0) & (z < self.nx[2])
        
        x, y, z = x[mask], y[mask], z[mask]
        geom_feats = geom_feats[mask]
        batch_indices = batch_indices[mask]

        # 扁平化索引
        indices = batch_indices * (self.nx[0] * self.nx[1]) + y * self.nx[0] + x
        
        # 排序
        ranks = indices.argsort()
        x, y, z = x[ranks], y[ranks], z[ranks]
        geom_feats = geom_feats[ranks]
        indices = indices[ranks]

        # 累加
        keep = torch.ones_like(indices, dtype=torch.bool)
        keep[:-1] = (indices[1:] != indices[:-1])
        
        cumsum = torch.cumsum(geom_feats, 0)
        cumsum = cumsum[keep]
        cumsum = torch.cat((cumsum[:1], cumsum[1:] - cumsum[:-1]))
        
        # 填回 BEV
        final_bev = torch.zeros((B * self.nx[1] * self.nx[0], C), device=geom_feats.device)
        
        if cumsum.shape[0] > 0:
            final_bev[indices[keep]] = cumsum
            
        final_bev = final_bev.view(B, self.nx[1], self.nx[0], C).permute(0, 3, 1, 2)
        
        return final_bev

    def forward(self, x, rots, trans, intrinsics):
        B, N, C, H, W = x.shape
        
        # 1. Lift
        cam_feats = self.get_cam_feats(x) 
        cam_feats = cam_feats.view(B, N, self.D, self.C, H, W)
        cam_feats = cam_feats.permute(0, 1, 2, 4, 5, 3) 
        
        # 2. Geometry
        geom_xyz = self.get_geometry(rots, trans, intrinsics)
        
        # 3. Splat 准备: Metric -> Index
        # 直接使用 self.bx, self.dx (它们已经是 Tensor 且在正确的 device 上)
        geom_xyz = ((geom_xyz - (self.bx - self.dx/2.0)) / self.dx).long()
        x_idx, y_idx, z_idx = geom_xyz[..., 0], geom_xyz[..., 1], geom_xyz[..., 2]
        
        # 4. Splat
        bev_feature = self.voxel_pooling(cam_feats, x_idx, y_idx, z_idx)
        
        # 5. BEV Encode
        output = self.bev_encode(bev_feature)
        
        return output

# --- 以下部分保持不变，用于训练循环 ---
class FakeLSSDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        inputs = torch.randn(6, 512, 16, 44)
        rots = torch.eye(3).unsqueeze(0).expand(6, 3, 3)
        trans = torch.zeros(6, 3)
        intrinsics = torch.eye(3).unsqueeze(0).expand(6, 3, 3)
        target = torch.randint(0, 2, (1, 200, 200)).float()
        return inputs, rots, trans, intrinsics, target

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, rots, trans, intrinsics, targets) in enumerate(loader):
        inputs, rots, trans, intrinsics, targets = [x.to(device) for x in [inputs, rots, trans, intrinsics, targets]]
        
        optimizer.zero_grad()
        preds = model(inputs, rots, trans, intrinsics)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 5 == 0:
            print(f"[Train] Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for batch_idx, (inputs, rots, trans, intrinsics, targets) in enumerate(loader):
            inputs, rots, trans, intrinsics, targets = [x.to(device) for x in [inputs, rots, trans, intrinsics, targets]]
            
            preds = model(inputs, rots, trans, intrinsics)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            probs = torch.sigmoid(preds)
            pred_mask = (probs > 0.5).float()
            intersection = (pred_mask * targets).sum()
            union = pred_mask.sum() + targets.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.item()
            
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    print(f"[Val] Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")
    return avg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1.0],
    }
    data_conf = {'img_size': (16, 44), 'dbound': grid_conf['dbound'], 'D': 41}
    
    model = LSS_Core_Complete(grid_conf, data_conf, input_channels=512, num_classes=1).to(device)
    
    train_dataset = FakeLSSDataset(length=50)
    val_dataset = FakeLSSDataset(length=10)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)