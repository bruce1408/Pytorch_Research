import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 核心组件: 时序对齐模块 (Feature Aligner)
# ==========================================
class BEVAligner(nn.Module):
    """
    BEVDet4D 的灵魂：利用自车运动，将上一帧的 BEV 特征 '扭曲' 到当前帧的坐标系下。
    """
    def __init__(self):
        super().__init__()

    def forward(self, prev_bev, trans, rot, bev_h, bev_w):
        """
        参数:
            prev_bev: 上一帧的 BEV 特征 (B, C, H, W)
            trans: (B, 2) 平移向量 [dx, dy] (单位: 格子/像素, 不是米!)
                   表示上一帧到当前帧的相对位移
            rot:   (B, 1) 旋转角度 [d_theta] (单位: 弧度)
                   表示上一帧到当前帧的相对旋转
        """
        B, C, H, W = prev_bev.shape
        
        # 1. 构建仿射变换矩阵 (Affine Matrix) 2x3
        # PyTorch 的 affine_grid 需要 2x3 矩阵: [R | T]
        # 注意: 这里简化了 3D 变换，只考虑 BEV 平面上的 Rotation(yaw) 和 Translation(x, y)
        
        theta = torch.zeros(B, 2, 3, device=prev_bev.device, dtype=prev_bev.dtype)
        
        cos_r = torch.cos(rot).squeeze(1)
        sin_r = torch.sin(rot).squeeze(1)
        
        # 旋转部分
        theta[:, 0, 0] = cos_r
        theta[:, 0, 1] = -sin_r
        theta[:, 1, 0] = sin_r
        theta[:, 1, 1] = cos_r
        
        # 平移部分 (注意 PyTorch grid_sample 的坐标范围是 [-1, 1])
        # 我们需要把像素位移转换为归一化坐标位移
        # normalize_dx = dx / (W/2)
        theta[:, 0, 2] = trans[:, 0] / (W / 2.0)
        theta[:, 1, 2] = trans[:, 1] / (H / 2.0)
        
        # 2. 生成网格 (Grid Generation)
        # 这一步计算出：为了得到当前帧的图像，我们需要去上一帧图像的哪个坐标去采样？
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        
        # 3. 采样 (Resampling / Warping)
        # 使用双线性插值进行采样
        aligned_bev = F.grid_sample(prev_bev, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return aligned_bev

# ==========================================
# 2. 模拟基础组件 (LSS, Backbone)
# ==========================================
class MockLSS(nn.Module):
    """ 模拟 LSS 将图像转为 BEV (不涉及时序，仅用于跑通代码) """
    def __init__(self, out_channels=64, bev_size=(128, 128)):
        super().__init__()
        self.bev_size = bev_size
        self.conv = nn.Conv2d(256, out_channels, kernel_size=1)
        
    def forward(self, imgs):
        # 假设输入 (B, N, C, H, W)
        B, N, _, _, _ = imgs.shape
        
        # 简单模拟：Pooling + Resize
        feat = imgs.mean(dim=1) # (B, C, H, W)
        feat = F.interpolate(feat, size=self.bev_size)
        feat = self.conv(feat)
        return feat # (B, 64, 128, 128)

# ==========================================
# 3. 完整模型: BEVDet4D
# ==========================================
class BEVDet4D(nn.Module):
    def __init__(self, bev_c=64):
        super().__init__()
        
        # 1. 图像转 BEV 模块
        self.lss = MockLSS(out_channels=bev_c)
        
        # 2. 时序对齐器 (Alignment)
        self.aligner = BEVAligner()
        
        # 3. BEV Encoder (特征融合后进行处理)
        # 输入通道是 2 * bev_c (因为做了 Concat)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_c * 2, bev_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_c),
            nn.ReLU(),
            nn.Conv2d(bev_c, bev_c, kernel_size=3, padding=1)
        )
        
        # 4. 检测头
        self.head = nn.Conv2d(bev_c, 10, kernel_size=1)

    def forward(self, imgs, prev_info=None):
        """
        imgs: 当前帧图像 (B, N, C, H, W)
        prev_info: 一个字典，包含:
            - 'bev': 上一帧的 BEV 特征 (B, C, H, W)
            - 'trans': 相对平移 (B, 2)
            - 'rot': 相对旋转 (B, 1)
        """
        # A. 生成当前帧 BEV
        curr_bev = self.lss(imgs) # (B, 64, 128, 128)
        
        # B. 时序处理 (Temporal Fusion)
        if prev_info is not None and prev_info['bev'] is not None:
            # 1. 获取上一帧 BEV
            prev_bev = prev_info['bev']
            trans = prev_info['trans']
            rot = prev_info['rot']
            
            # 2. 对齐: 把上一帧 BEV 扭曲到当前坐标系
            aligned_prev_bev = self.aligner(prev_bev, trans, rot, curr_bev.shape[2], curr_bev.shape[3])
            
            # 3. 拼接 (Concat): 这是 BEVDet4D 的核心特征
            # (B, 128, 128, 128)
            fused_feat = torch.cat([curr_bev, aligned_prev_bev], dim=1)
            
        else:
            # 如果是序列的第一帧，没有历史信息
            # 策略：直接复制一份当前帧，或者用全0填充
            # 这里演示复制策略，保持输入 Encoder 的通道一致
            fused_feat = torch.cat([curr_bev, curr_bev], dim=1)
            
        # C. 进一步提取特征 (BEV Encoder)
        feat = self.bev_encoder(fused_feat)
        
        # D. 输出结果
        res = self.head(feat)
        
        # E. 返回结果以及当前 BEV (作为下一帧的历史)
        return res, curr_bev.detach() # detach 很重要，不需要对上一帧求梯度

# ==========================================
# 4. 模拟运行流程 (Simulation)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVDet4D(bev_c=64).to(device)
    
    # 模拟输入数据
    B = 1
    # 假设是一个视频序列，有 3 帧
    seq_len = 3
    
    # 初始化历史信息
    history_bev = None
    
    print("Start Processing Sequence...")
    
    for t in range(seq_len):
        # 1. 模拟当前帧图像
        imgs = torch.randn(B, 6, 256, 256, 704).to(device)
        
        # 2. 模拟自车运动 (Ego Motion)
        # 假设车在向前开 (dy 变化), 并且稍微有点转弯 (rot 变化)
        # trans: [dx, dy] 像素单位
        # rot: rad
        if t > 0:
            # 只有第 1 帧以后才有相对运动
            # 模拟：向 Y 轴移动了 5 个像素，旋转了 0.1 弧度
            trans = torch.tensor([[0.0, 5.0]] * B).to(device) 
            rot = torch.tensor([[0.1]] * B).to(device)
            prev_info = {'bev': history_bev, 'trans': trans, 'rot': rot}
        else:
            prev_info = None
            
        # 3. 前向传播
        preds, current_bev = model(imgs, prev_info)
        
        # 4. 更新历史信息 (关键!)
        history_bev = current_bev 
        
        print(f"Time {t}: Output Shape {preds.shape} | Has History? {prev_info is not None}")
        if prev_info:
            print(f"   -> Aligned history using trans={trans.tolist()[0]}, rot={rot.tolist()[0]}")

if __name__ == "__main__":
    main()