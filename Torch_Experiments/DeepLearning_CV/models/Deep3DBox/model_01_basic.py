import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 网络定义 (Network Architecture)
# ==========================================
class Deep3DBoxNet(nn.Module):
    def __init__(self, feature_dim=512, num_bins=2):
        super(Deep3DBoxNet, self).__init__()
        self.num_bins = num_bins
        
        # 假设输入是经过 Backbone + ROI Pooling 后的特征向量
        self.fc1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Head 1: 预测 3D 尺寸 (H, W, L)
        # 输出是 3 个数值，通常是对平均尺寸的残差，这里简化直接输出
        self.dim_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )
        
        # Head 2: 预测朝向 (Multi-Bin Orientation)
        # 论文将 360 度分为 N 个 Bin (通常是2个，覆盖前后)
        # 每个 Bin 输出: 1个置信度 (Confidence) + 2个角度偏移值 (cos, sin)
        self.conf_head = nn.Linear(256, num_bins) # 分类: 属于哪个 Bin
        self.orient_head = nn.Linear(256, num_bins * 2) # 回归: 每个 Bin 的 cos, sin

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        dims = self.dim_head(x)
        
        # 朝向输出
        orientation_conf = self.conf_head(x)
        orientation_offset = self.orient_head(x)
        orientation_offset = orientation_offset.view(-1, self.num_bins, 2)
        
        # 对 offset 做 L2 归一化，保证 cos^2 + sin^2 = 1
        orientation_offset = F.normalize(orientation_offset, dim=2)
        
        return dims, orientation_conf, orientation_offset

# ==========================================
# 2. 训练逻辑 (Training / Loss)
# ==========================================
def compute_loss(pred_dims, pred_conf, pred_offset, target_dims, target_angle, num_bins=2):
    """
    Args:
        target_dims: 真实的 (H, W, L)
        target_angle: 真实的全局偏航角 (radians)
    """
    # 1. 尺寸损失 (MSE Loss)
    dim_loss = F.mse_loss(pred_dims, target_dims)
    
    # 2. 朝向损失 (Multi-Bin Loss)
    # 需要先将真实的 target_angle 转换为 Bin 的分类标签和 Offset
    # 这里为了演示简化，假设我们已经计算好了 gt_bin_idx 和 gt_offset
    # 实际项目中需要编写 geometric utility 函数来计算 GT 属于哪个 bin
    
    # 模拟 GT 标签 (实际使用时需根据 target_angle 计算)
    batch_size = pred_dims.shape[0]
    gt_bin_idx = torch.randint(0, num_bins, (batch_size,)).to(pred_dims.device) # 属于哪个bin
    gt_offset = torch.randn(batch_size, 2).to(pred_dims.device) # 在该bin下的 cos, sin
    
    # Classification Loss (Cross Entropy)
    conf_loss = F.cross_entropy(pred_conf, gt_bin_idx)
    
    # Regression Loss (只计算 GT 所在那个 Bin 的误差)
    # 选取对应 bin 的预测值
    pred_offset_selected = pred_offset[torch.arange(batch_size), gt_bin_idx] 
    orient_reg_loss = F.mse_loss(pred_offset_selected, gt_offset)
    
    total_loss = dim_loss + conf_loss + orient_reg_loss
    return total_loss

# ==========================================
# 3. 几何求解器 (Inference: Solver)
# ==========================================
# 这是论文最核心的部分：已知 Dim, Angle, 2D Box, Camera K -> 求 Translation (x, y, z)

def solve_translation(K, bbox_2d, dim, yaw):
    """
    利用 2D 框紧贴约束求解 3D 位置
    Args:
        K: 相机内参矩阵 (3, 3)
        bbox_2d: [xmin, ymin, xmax, ymax]
        dim: [h, w, l] 预测出的尺寸
        yaw: 预测出的偏航角 (radians)
    Returns:
        location: [x, y, z]
    """
    # 1. 构建物体坐标系下的 3D 框的 8 个顶点 (假设物体中心在原点)
    # 顺序通常是：前后左右上下组合
    h, w, l = dim
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h] # 假设 y 轴向下 (相机坐标系习惯)
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners]) # (3, 8)
    
    # 2. 旋转这些点 (只考虑 Yaw 角旋转)
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    rotated_corners = np.dot(R, corners_3d) # (3, 8)
    
    # 3. 建立线性方程组 Ax = b
    # 论文假设: 2D 框的 4 条边分别被 3D 框的某 4 个顶点碰到。
    # 这是一个超定方程 (Overdetermined System)，或者我们要尝试不同的顶点配置
    # xmin 对应某个顶点投影的 x
    # xmax 对应某个顶点投影的 x
    # ymin 对应某个顶点投影的 y
    # ymax 对应某个顶点投影的 y
    
    # 这里的数学原理：
    # x_img = (f_x * X + c_x * Z) / Z  => x_img * Z = f_x * X + c_x * Z
    # 也就是: f_x * (Rx + Tx) + (c_x - x_img) * (Rz + Tz) = 0
    # 未知数是 Tx, Ty, Tz。我们可以构建矩阵求解。
    
    # 为了简化代码，这里展示核心思路。实际论文中会尝试 64 种顶点组合取误差最小的。
    # 这里我们只取最佳配置的逻辑（假设我们知道哪几个点撑开了框，实际需遍历）。
    
    best_loc = None
    min_error = float('inf')
    
    # 相机参数
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 构建约束矩阵 A 和 b
    # 每一个方程形如: A_row * [tx, ty, tz]^T = b_val
    
    # 这是一个简化的 Solver，实际上需要迭代 8^4 种组合或者使用启发式方法
    # 假设我们选取前4个点分别对应 左、上、右、下 (仅作演示)
    constraints = []
    # 约束 1: Left (xmin)
    # xmin = (fx * (rx + tx) / (rz + tz)) + cx
    # => fx*tx + (cx - xmin)*tz = fx*(-rx) + (xmin - cx)*rz
    constraints.append({'type': 'x', 'val': bbox_2d[0], 'coeff': -1}) 
    constraints.append({'type': 'y', 'val': bbox_2d[1], 'coeff': -1})
    constraints.append({'type': 'x', 'val': bbox_2d[2], 'coeff': 1})
    constraints.append({'type': 'y', 'val': bbox_2d[3], 'coeff': 1})
    
    # 这里用 Least Squares 求解
    A = []
    b = []
    
    # 实际应用中，这里需要遍历顶点组合。
    # 为让代码可运行，这里随机取 4 个不同的顶点索引作为支撑点
    indices = [0, 2, 4, 6] 
    
    for i, idx in enumerate(indices):
        rx, ry, rz = rotated_corners[:, idx]
        
        if i % 2 == 0: # x 约束 (xmin, xmax)
            val = constraints[i]['val'] # u
            # fx * tx + (cx - u) * tz = ...
            A.append([fx, 0, cx - val])
            b.append(val * rz - fx * rx - cx * rz)
        else: # y 约束 (ymin, ymax)
            val = constraints[i]['val'] # v
            # fy * ty + (cy - v) * tz = ...
            A.append([0, fy, cy - val])
            b.append(val * rz - fy * ry - cy * rz)

    A = np.array(A)
    b = np.array(b)
    
    # 求解 Ax = b
    # [tx, ty, tz]
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return result

# ==========================================
# 4. 主流程 (Main Demo)
# ==========================================
if __name__ == "__main__":
    # ------------------
    # A. 训练阶段模拟
    # ------------------
    print("--- Training Phase Simulation ---")
    model = Deep3DBoxNet(feature_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 模拟一个 Batch 的数据
    batch_size = 8
    dummy_features = torch.randn(batch_size, 512) # 随机特征
    dummy_target_dims = torch.abs(torch.randn(batch_size, 3)) # 随机尺寸 GT
    dummy_target_yaw = torch.randn(batch_size) # 随机角度 GT
    
    # Forward
    pred_dims, pred_conf, pred_offset = model(dummy_features)
    
    # Loss & Backward
    loss = compute_loss(pred_dims, pred_conf, pred_offset, dummy_target_dims, dummy_target_yaw)
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("Training step done.\n")

    # ------------------
    # B. 推理阶段模拟
    # ------------------
    print("--- Inference Phase Simulation ---")
    
    # 1. 假设网络输出 (取 Batch 中第一个)
    pred_h, pred_w, pred_l = pred_dims[0].detach().numpy()
    pred_dims_np = [pred_h, pred_w, pred_l]
    
    # 假设解析 Multi-Bin 得到的角度
    pred_yaw = 1.57 # 90度
    
    # 2. 假设输入图像的已知信息
    K = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]]) # 相机内参
    bbox_2d = [500, 200, 700, 400] # [xmin, ymin, xmax, ymax]
    
    print(f"Predicted Dims: {pred_dims_np}")
    print(f"Predicted Yaw: {pred_yaw}")
    print(f"2D BBox: {bbox_2d}")
    
    # 3. 几何求解 (Geometry Solver)
    # 这就是论文所谓的 "Using Geometry"
    location = solve_translation(K, bbox_2d, pred_dims_np, pred_yaw)
    
    print(f"Calculated 3D Location (x, y, z): {location}")
    print("Note: The location is solved purely by geometry constraints.")