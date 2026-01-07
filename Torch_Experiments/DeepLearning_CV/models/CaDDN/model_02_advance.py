"""
CaDDN (Category-Aware Depth Distribution Network) 模型实现
用于基于图像的3D目标检测

主要流程：
1. 图像 -> 特征提取 -> 语义特征 + 深度分布
2. 语义特征 × 深度分布 -> 视锥体特征 (frustum volume)
3. 视锥体特征 -> 体素网格 (voxel grid) -> BEV特征
4. BEV特征 -> 检测头 -> 3D边界框预测
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 0) 工具函数：3D边界框编码/解码 (x,y,z,w,l,h,yaw)
# =========================================================

def limit_period(val, offset=0.5, period=math.pi * 2):
    """
    将角度值限制在一个周期范围内，避免角度溢出问题
    
    参数:
        val: 角度值（可以是tensor）
        offset: 偏移量，默认0.5
        period: 周期长度，默认2π
    
    返回:
        限制后的角度值，范围在 [-period/2, period/2) 之间
    
    例子:
        如果输入角度是 3π，会被限制到 -π 附近
        这样做的目的是让角度预测更稳定，避免360度跳跃
    """
    return val - torch.floor(val / period + offset) * period

def encode_boxes(gt, anchors):
    """
    将真实框(ground truth)编码为相对于锚点(anchor)的偏移量
    
    这是目标检测中的标准做法：不直接预测框的绝对位置，
    而是预测相对于预设锚点的偏移，这样更容易学习
    
    参数:
        gt: 真实框 (..., 7) [x,y,z,w,l,h,yaw]
            x,y,z: 中心点坐标
            w,l,h: 宽度、长度、高度
            yaw: 朝向角
        anchors: 锚点框 (..., 7) 格式相同
    
    返回:
        deltas: (..., 7) 编码后的偏移量 [dx,dy,dz,dw,dl,dh,dyaw]
            dx,dy: 位置偏移（归一化）
            dz: 高度偏移（归一化）
            dw,dl,dh: 尺寸的对数偏移
            dyaw: 角度偏移（考虑周期性）
    """
    # 解包：将最后一维的7个值分别提取出来
    xa, ya, za, wa, la, ha, ra = anchors.unbind(-1)  # anchor的各个分量
    xg, yg, zg, wg, lg, hg, rg = gt.unbind(-1)        # ground truth的各个分量

    # 计算对角线长度，用于归一化x和y的偏移
    # 这样可以让不同大小的框的偏移量在相似范围内
    diagonal = torch.sqrt(la**2 + wa**2)
    
    # 位置偏移：归一化到对角线长度
    dx = (xg - xa) / diagonal
    dy = (yg - ya) / diagonal
    
    # 高度偏移：归一化到锚点高度
    dz = (zg - za) / ha

    # 尺寸偏移：使用对数形式，这样exp后总是正数
    # 为什么用对数？因为尺寸必须是正数，直接预测可能为负
    dw = torch.log(wg / wa)  # log(w_gt / w_anchor)
    dl = torch.log(lg / la)  # log(l_gt / l_anchor)
    dh = torch.log(hg / ha)  # log(h_gt / h_anchor)

    # 角度偏移：考虑周期性（360度 = 0度）
    # 例如：如果gt是350度，anchor是10度，实际只差20度，不是340度
    dyaw = limit_period(rg - ra, offset=0.5, period=2*math.pi)

    # 将所有偏移量堆叠成7维向量
    return torch.stack([dx, dy, dz, dw, dl, dh, dyaw], dim=-1)

def decode_boxes(deltas, anchors):
    """
    将预测的偏移量解码回实际的3D边界框
    
    这是encode_boxes的逆过程
    
    参数:
        deltas: 预测的偏移量 (..., 7) [dx,dy,dz,dw,dl,dh,dyaw]
        anchors: 锚点框 (..., 7)
    
    返回:
        boxes: 解码后的3D边界框 (..., 7) [x,y,z,w,l,h,yaw]
    """
    # 解包锚点和偏移量
    xa, ya, za, wa, la, ha, ra = anchors.unbind(-1)
    dx, dy, dz, dw, dl, dh, dyaw = deltas.unbind(-1)

    # 计算对角线长度（与编码时一致）
    diagonal = torch.sqrt(la**2 + wa**2)
    
    # 解码位置：反向归一化
    x = dx * diagonal + xa
    y = dy * diagonal + ya
    z = dz * ha + za

    # 解码尺寸：exp反向操作
    w = torch.exp(dw) * wa
    l = torch.exp(dl) * la
    h = torch.exp(dh) * ha

    # 解码角度：加上锚点角度，然后限制周期
    yaw = limit_period(dyaw + ra, offset=0.5, period=2*math.pi)
    
    return torch.stack([x, y, z, w, l, h, yaw], dim=-1)

# =========================================================
# 1) 深度分箱器 (Depth Binner)：UD/SID/LID + 标签/索引映射
# =========================================================

class DepthBinner(nn.Module):
    """
    深度分箱器：将连续的深度值离散化为多个深度区间（bins）
    
    为什么需要分箱？
    - 深度估计是一个回归问题，直接预测连续值很难
    - 将其转化为分类问题（预测属于哪个深度区间）更容易学习
    - 论文提出了三种分箱策略：UD（均匀）、SID（间距递增）、LID（线性递增）
    
    三种模式的区别：
    - UD (Uniform Discretization): 均匀分箱，每个区间大小相同
    - SID (Spacing-Increasing Discretization): 间距递增，近处区间小，远处区间大
    - LID (Linear-Increasing Discretization): 线性递增，介于UD和SID之间
    """
    def __init__(self, depth_bins, d_min, d_max, mode="LID"):
        """
        初始化深度分箱器
        
        参数:
            depth_bins: 深度区间的数量（例如80个）
            d_min: 最小深度值（米）
            d_max: 最大深度值（米）
            mode: 分箱模式，"UD"、"SID"或"LID"
        """
        super().__init__()
        assert mode in ["UD", "SID", "LID"], f"未知的分箱模式: {mode}"
        
        self.D = int(depth_bins)      # 深度区间数量
        self.d_min = float(d_min)     # 最小深度
        self.d_max = float(d_max)     # 最大深度
        self.mode = mode              # 分箱模式
        
        # 生成深度区间的边界（edges）和中心点（centers）
        edges = self._make_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])  # 每个区间的中心
        
        # register_buffer: 将tensor注册为模型的一部分，但不参与梯度更新
        # 这些是固定的值，不需要学习
        self.register_buffer("edges", edges)
        self.register_buffer("centers", centers)

    def _make_edges(self):
        """
        根据分箱模式生成深度区间的边界点
        
        返回:
            edges: (D+1,) 深度区间的边界，共D+1个点，形成D个区间
        """
        D, dmin, dmax = self.D, self.d_min, self.d_max
        
        if self.mode == "UD":
            # 均匀分箱：从d_min到d_max均匀分割
            # 例如：[2.0, 4.0, 6.0, ..., 42.0] 共81个点，80个区间
            return torch.linspace(dmin, dmax, D + 1)
        
        if self.mode == "SID":
            # 间距递增分箱：使用指数函数
            # 近处区间小，远处区间大，符合深度感知的特点
            i = torch.arange(0, D + 1, dtype=torch.float32)
            return torch.exp(torch.log(torch.tensor(dmin)) + i / D * torch.log(torch.tensor(dmax / dmin)))
        
        # LID模式：线性递增分箱
        # 使用二次函数，介于UD和SID之间
        i = torch.arange(0, D + 1, dtype=torch.float32)
        return dmin + (dmax - dmin) * (i * (i + 1)) / (D * (D + 1))

    @torch.no_grad()  # 不需要梯度计算
    def depth_to_bin_label(self, depth_map, ignore_value=-1):
        """
        将深度图转换为离散的深度区间标签（用于训练时的监督）
        
        参数:
            depth_map: 深度图，形状任意，单位是米
            ignore_value: 无效深度的标签值（通常是-1）
        
        返回:
            idx: 深度区间索引，形状与depth_map相同，值在[0, D-1]或ignore_value
            valid: 有效深度掩码（布尔值）
        """
        d = depth_map
        
        # 判断哪些像素的深度是有效的（在范围内）
        valid = (d > self.d_min) & (d < self.d_max)
        
        # 使用bucketize找到每个深度值属于哪个区间
        # bucketize: 返回每个值应该插入到edges的哪个位置
        idx = torch.bucketize(d.clamp(self.d_min, self.d_max), self.edges) - 1
        
        # 确保索引在有效范围内 [0, D-1]
        idx = idx.clamp(0, self.D - 1)
        
        # 无效深度设为ignore_value（训练时会忽略这些位置）
        idx[~valid] = ignore_value
        
        return idx, valid

    def depth_to_continuous_index(self, depth):
        """
        将深度值转换为连续的深度索引（用于可微分的采样）
        
        与depth_to_bin_label不同，这里返回的是连续值（浮点数），
        而不是离散的整数标签。这对于可微分的双线性插值很重要。
        
        参数:
            depth: (B,1,N) 深度值，单位是米
        
        返回:
            d_idx: (B,1,N) 连续的深度索引，范围在[0, D-1]
                   例如：如果深度在第一个和第二个区间之间，
                   索引可能是0.7（表示70%在第一个区间，30%在第二个区间）
        """
        # 将深度值限制在有效范围内
        d = depth.clamp(self.d_min, self.d_max)

        if self.mode == "UD":
            # 均匀分箱：直接线性映射
            return (d - self.d_min) / (self.d_max - self.d_min) * (self.D - 1)

        # SID/LID模式：需要分段线性插值
        # 因为区间大小不均匀，需要找到深度值所在的区间，然后计算在该区间内的位置
        d_flat = d.reshape(-1)  # 展平以便处理
        
        # 找到每个深度值属于哪个区间
        k = torch.bucketize(d_flat, self.edges) - 1
        k = k.clamp(0, self.D - 1)
        
        # 获取该区间的两个边界
        e0 = self.edges[k]      # 区间下界
        e1 = self.edges[k + 1]  # 区间上界
        
        # 计算在该区间内的相对位置（0到1之间）
        frac = (d_flat - e0) / (e1 - e0 + 1e-6)  # +1e-6防止除零
        
        # 最终的连续索引 = 区间编号 + 区间内的相对位置
        d_idx = (k.to(d.dtype) + frac).view_as(d)
        return d_idx

# =========================================================
# 2) 骨干网络 + 双头结构（共享特征 -> 语义 + 深度）
# =========================================================

class ConvBNReLU(nn.Module):
    """
    标准的卷积-批归一化-ReLU模块
    
    这是深度学习中非常常用的基础模块：
    - Conv2d: 提取特征
    - BatchNorm2d: 归一化，加速训练，提高稳定性
    - ReLU: 激活函数，引入非线性
    """
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        """
        参数:
            c_in: 输入通道数
            c_out: 输出通道数
            k: 卷积核大小（默认3x3）
            s: 步长（stride，默认1）
            p: 填充（padding，默认1，保持尺寸不变）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),  # bias=False因为后面有BN
            nn.BatchNorm2d(c_out),  # 批归一化
            nn.ReLU(inplace=True),   # inplace=True节省内存
        )
    
    def forward(self, x): 
        return self.net(x)

class TinyBackbone(nn.Module):
    """
    轻量级骨干网络（用于演示）
    
    在实际论文复现中，这里应该替换为ResNet+FPN等更强的骨干网络
    这里使用简单的卷积层是为了代码简洁和快速运行
    
    网络结构：
    - stem: 下采样到1/4分辨率
    - block: 特征提取（不改变分辨率）
    """
    def __init__(self, in_ch=3, base=64):
        """
        参数:
            in_ch: 输入通道数（RGB图像是3）
            base: 基础通道数（会逐渐增加）
        """
        super().__init__()
        
        # stem: 主干部分，负责下采样
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, base, 3, 2, 1),      # 第一次下采样：/2
            ConvBNReLU(base, base, 3, 1, 1),       # 保持尺寸
            ConvBNReLU(base, base*2, 3, 2, 1),     # 第二次下采样：/4
        )
        
        # block: 特征提取块
        self.block = nn.Sequential(
            ConvBNReLU(base*2, base*2, 3, 1, 1),
            ConvBNReLU(base*2, base*2, 3, 1, 1),
        )
        
        self.out_ch = base*2  # 输出通道数
        self.stride = 4        # 总的下采样倍数

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (B, 3, H, W) 输入图像
        
        返回:
            feat: (B, out_ch, H/4, W/4) 特征图
        """
        x = self.stem(x)   # 下采样
        x = self.block(x)   # 特征提取
        return x

class CaDDNHeads(nn.Module):
    """
    CaDDN的双头结构：从共享特征中分别预测语义特征和深度分布
    
    这是CaDDN的核心思想之一：
    - 语义头：提取语义特征（用于后续的3D检测）
    - 深度头：预测每个像素的深度分布（属于哪个深度区间的概率）
    
    两个头共享同一个骨干网络提取的特征，这样可以：
    1. 减少计算量
    2. 让语义和深度信息相互促进学习
    """
    def __init__(self, feat_in, feat_out, depth_bins):
        """
        参数:
            feat_in: 输入特征通道数（来自backbone）
            feat_out: 语义特征输出通道数
            depth_bins: 深度区间数量（深度头的输出通道数）
        """
        super().__init__()
        
        # 语义头：输出语义特征
        self.sem = nn.Sequential(
            ConvBNReLU(feat_in, feat_out, 3, 1, 1),  # 特征提取
            nn.Conv2d(feat_out, feat_out, 1, bias=False),  # 1x1卷积，调整通道
        )
        
        # 深度头：输出深度分布（每个像素的深度区间概率）
        self.depth = nn.Sequential(
            ConvBNReLU(feat_in, feat_out, 3, 1, 1),  # 特征提取
            nn.Conv2d(feat_out, depth_bins, 1, bias=True),  # 输出深度区间数
        )

    def forward(self, feat):
        """
        前向传播
        
        参数:
            feat: (B, feat_in, Hf, Wf) 共享特征
        
        返回:
            sem: (B, feat_out, Hf, Wf) 语义特征
            depth_logits: (B, depth_bins, Hf, Wf) 深度分布的logits（未归一化）
        """
        return self.sem(feat), self.depth(feat)

# =========================================================
# 3) 体素网格（体素中心点） + 视锥到体素的采样
# =========================================================

def make_voxel_grid(grid_size, pc_range, device=None):
    """
    创建3D体素网格的坐标
    
    体素（voxel）是3D空间中的小立方体单元，类似于2D图像中的像素
    
    参数:
        grid_size: (Xn, Yn, Zn) 三个方向的体素数量
        pc_range: (x0, y0, z0, x1, y1, z1) 点云范围
        device: 设备（CPU或GPU）
    
    返回:
        grid: (1, 3, Z, Y, X) 体素网格坐标
              每个位置存储该体素中心的(x, y, z)坐标
    """
    Xn, Yn, Zn = grid_size
    x0, y0, z0, x1, y1, z1 = pc_range
    
    # 计算每个体素的大小
    dx = (x1 - x0) / Xn
    dy = (y1 - y0) / Yn
    dz = (z1 - z0) / Zn
    
    # 生成每个方向的体素中心坐标
    # 从范围中心开始，而不是从边界开始
    xs = torch.linspace(x0 + dx/2, x1 - dx/2, Xn, device=device)
    ys = torch.linspace(y0 + dy/2, y1 - dy/2, Yn, device=device)
    zs = torch.linspace(z0 + dz/2, z1 - dz/2, Zn, device=device)
    
    # 创建3D网格：所有体素中心的坐标组合
    zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing="ij")  # (Z, Y, X)
    
    # 堆叠成 (3, Z, Y, X) 格式，每个位置是(x,y,z)
    grid = torch.stack([xs, ys, zs], dim=-1).permute(3,0,1,2)
    
    return grid.unsqueeze(0)  # 添加batch维度: (1, 3, Z, Y, X)

class FrustumToVoxel(nn.Module):
    """
    将视锥体特征采样到3D体素网格
    
    这是CaDDN的关键步骤：
    1. 视锥体（frustum）：图像空间 + 深度维度，形状是(B,C,D,Hf,Wf)
    2. 体素网格（voxel grid）：3D空间，形状是(B,C,Z,Y,X)
    
    需要做的事情：
    - 将3D体素坐标投影到图像空间
    - 找到对应的图像位置和深度
    - 使用双线性插值从视锥体中采样特征
    """
    def __init__(self, image_hw, feat_stride, depth_binner: DepthBinner):
        """
        参数:
            image_hw: (H, W) 原始图像尺寸
            feat_stride: 特征图的下采样倍数（例如4）
            depth_binner: 深度分箱器
        """
        super().__init__()
        self.img_H, self.img_W = image_hw
        self.stride = feat_stride
        self.db = depth_binner

    def forward(self, frustum_feat, voxel_grid, ego2img):
        """
        将视锥体特征采样到体素网格
        
        参数:
            frustum_feat: (B,C,D,Hf,Wf) 视锥体特征
                          C: 特征通道数
                          D: 深度区间数
                          Hf, Wf: 特征图尺寸
            voxel_grid: (1,3,Z,Y,X) 体素网格坐标（ego坐标系）
            ego2img: (B,4,4) ego坐标系到图像坐标系的变换矩阵
        
        返回:
            voxel_feat: (B,C,Z,Y,X) 体素特征
        """
        
        B, C, D, Hf, Wf = frustum_feat.shape
        device = frustum_feat.device
        
        # 扩展体素网格到batch大小
        vox = voxel_grid.to(device).repeat(B, 1, 1, 1, 1)  # (B, 3, Z, Y, X)
        _, _, Z, Y, X = vox.shape
        
        # 将体素坐标展平: (B,3,N) 其中N=Z*Y*X
        vox_flat = vox.view(B, 3, -1)
        
        # 转换为齐次坐标（添加1）：(B,4,N)
        # 齐次坐标方便进行矩阵变换
        ones = torch.ones((B, 1, vox_flat.shape[-1]), device=device)
        vox_h = torch.cat([vox_flat, ones], dim=1)

        # 投影到图像空间：ego坐标 -> 图像坐标
        img_pts = torch.bmm(ego2img, vox_h)  # (B,4,N)
        
        # 提取深度和图像坐标
        z = img_pts[:, 2:3, :]  # 深度（相机坐标系中的z）
        u = img_pts[:, 0:1, :] / (z + 1e-6)  # 图像x坐标（列）
        v = img_pts[:, 1:2, :] / (z + 1e-6)  # 图像y坐标（行）

        # 判断哪些体素是有效的（在图像范围内且深度合理）
        valid = (z > 1e-3) & \
                (u >= 0) & (u <= self.img_W-1) & \
                (v >= 0) & (v <= self.img_H-1) & \
                (z > self.db.d_min) & (z < self.db.d_max)

        # 转换到特征图坐标系（考虑下采样）
        uf = u / self.stride  # 特征图x坐标
        vf = v / self.stride  # 特征图y坐标

        # 归一化到[-1, 1]范围（grid_sample的要求）
        # align_corners=True: 角点对齐模式
        u_norm = 2.0 * (uf / (Wf - 1)) - 1.0
        v_norm = 2.0 * (vf / (Hf - 1)) - 1.0

        # 深度归一化：将深度值转换为连续的深度索引，然后归一化
        d_idx = self.db.depth_to_continuous_index(z)  # (B,1,N) 连续索引[0,D-1]
        d_norm = 2.0 * (d_idx / (D - 1)) - 1.0  # 归一化到[-1,1]

        # 将无效位置推到范围外（grid_sample会将其设为0）
        out = torch.full_like(u_norm, 2.0)  # 2.0超出[-1,1]范围
        u_norm = torch.where(valid, u_norm, out)
        v_norm = torch.where(valid, v_norm, out)
        d_norm = torch.where(valid, d_norm, out)
        
        # 组合成采样网格: (B,N,3) 每个位置是(u_norm, v_norm, d_norm)
        grid = torch.cat([u_norm, v_norm, d_norm], dim=1).permute(0,2,1)  # (B,N,3)
        grid = grid.view(B, Z, Y, X, 3)  # 恢复空间维度

        # 使用3D双线性插值从视锥体中采样
        voxel_feat = F.grid_sample(
            frustum_feat, grid,
            mode="bilinear",      # 双线性插值
            padding_mode="zeros", # 超出范围填充0
            align_corners=True    # 角点对齐
        )  # 输出: (B,C,Z,Y,X)
        
        return voxel_feat

# =========================================================
# 4) BEV骨干网络 + 基于锚点的检测头（论文风格）
# =========================================================

class BEVBackbone(nn.Module):
    """
    BEV（Bird's Eye View，鸟瞰图）空间的骨干网络
    
    在BEV空间中，我们有了3D特征，现在需要进一步提取特征用于检测
    """
    def __init__(self, c_in=128, c_mid=128):
        """
        参数:
            c_in: 输入通道数
            c_mid: 中间/输出通道数
        """
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(c_in, c_mid, 3, 1, 1),
            ConvBNReLU(c_mid, c_mid, 3, 1, 1),
            ConvBNReLU(c_mid, c_mid, 3, 1, 1),
        )
    
    def forward(self, x): 
        return self.net(x)

class AnchorGenerator:
    """
    锚点生成器：在BEV网格上生成预设的3D边界框（锚点）
    
    锚点（anchor）是预设的边界框模板，检测器会预测：
    - 这个锚点是否包含目标（分类）
    - 如何调整锚点以匹配真实目标（回归）
    
    生成策略：
    - 每个BEV cell（网格单元）一个中心点(x,y)
    - 每个类别一套尺寸(w,l,h)和高度(z)
    - 多个旋转角度（例如0度和90度）
    """
    def __init__(self, pc_range, grid_size_xy, classes_cfg, rotations=(0, math.pi/2)):
        """
        参数:
            pc_range: (x0,y0,z0,x1,y1,z1) 点云范围
            grid_size_xy: (Xn, Yn) BEV网格的X和Y方向大小
            classes_cfg: 类别配置列表，每个元素是dict，包含w,l,h,z
            rotations: 旋转角度列表（弧度）
        """
        self.pc_range = pc_range
        self.Xn, self.Yn = grid_size_xy
        self.classes_cfg = classes_cfg
        self.rotations = rotations

        # 计算每个BEV cell的大小
        x0,y0,_, x1,y1,_ = pc_range
        self.dx = (x1 - x0) / self.Xn
        self.dy = (y1 - y0) / self.Yn

    def generate(self, device):
        """
        生成所有锚点
        
        返回:
            anchors: (A,Y,X,7) 所有锚点，A=类别数×旋转数
            anchor_cls_ids: (A,Y,X) 每个锚点对应的类别ID
        """
        x0,y0,_, x1,y1,_ = self.pc_range

        # 生成BEV网格的中心点坐标
        xs = torch.linspace(x0 + self.dx/2, x1 - self.dx/2, self.Xn, device=device)
        ys = torch.linspace(y0 + self.dy/2, y1 - self.dy/2, self.Yn, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (Y,X)
        
        centers = torch.stack([xx, yy], dim=-1)  # (Y,X,2) 每个位置是(x,y)

        anchors_all = []
        anchor_cls_ids = []

        # 为每个类别和每个旋转角度生成锚点
        for cls_id, cfg in enumerate(self.classes_cfg):
            w, l, h, z = cfg["w"], cfg["l"], cfg["h"], cfg["z"]
            
            for r in self.rotations:
                # 创建锚点: (Y,X,7)
                a = torch.zeros((self.Yn, self.Xn, 7), device=device)
                a[...,0] = centers[...,0]   # x坐标
                a[...,1] = centers[...,1]   # y坐标
                a[...,2] = z                # z坐标（高度）
                a[...,3] = w                # 宽度
                a[...,4] = l                # 长度
                a[...,5] = h                # 高度
                a[...,6] = r                # 旋转角度
                
                anchors_all.append(a)
                # 记录这个锚点对应的类别ID
                anchor_cls_ids.append(torch.full((self.Yn,self.Xn), cls_id, device=device, dtype=torch.long))

        # 堆叠所有锚点
        anchors = torch.stack(anchors_all, dim=0)         # (A,Y,X,7)
        anchor_cls_ids = torch.stack(anchor_cls_ids, 0)   # (A,Y,X)
        
        return anchors, anchor_cls_ids

def boxes_to_aabb_bev(boxes):
    """
    将旋转的3D边界框转换为轴对齐的AABB（Axis-Aligned Bounding Box）
    
    注意：这只是用于fallback的简化版本
    理想情况下应该使用旋转IoU，但这里为了简化使用AABB IoU
    
    参数:
        boxes: (...,7) [x,y,z,w,l,h,yaw] 旋转边界框
    
    返回:
        aabb: (...,4) [x1,y1,x2,y2] 轴对齐边界框
    """
    x, y, _, w, l, _, _ = boxes.unbind(-1)
    
    # 计算半宽和半长
    half_w = w/2
    half_l = l/2
    
    # 简化为轴对齐框（忽略旋转）
    x1 = x - half_l
    y1 = y - half_w
    x2 = x + half_l
    y2 = y + half_w
    
    return torch.stack([x1, y1, x2, y2], -1)

def iou_bev_fallback(anchors, gt):
    """
    计算BEV空间中的IoU（交并比）
    
    注意：这是简化版本，使用AABB IoU而不是旋转IoU
    论文级实现应该使用旋转IoU（例如torchvision的box_iou_rotated）
    
    参数:
        anchors: (N,7) 锚点框
        gt: (M,7) 真实框
    
    返回:
        ious: (N,M) IoU矩阵，每个元素是anchor和gt的IoU
    """
    # 转换为AABB
    a = boxes_to_aabb_bev(anchors)  # (N,4)
    g = boxes_to_aabb_bev(gt)       # (M,4)

    # 计算交集
    N = a.shape[0]
    M = g.shape[0]
    
    # 扩展维度以便批量计算
    a = a[:,None,:].expand(N,M,4)  # (N,M,4)
    g = g[None,:,:].expand(N,M,4)  # (N,M,4)

    # 计算交集的左上角和右下角
    x1 = torch.maximum(a[...,0], g[...,0])  # 交集左边界
    y1 = torch.maximum(a[...,1], g[...,1])  # 交集上边界
    x2 = torch.minimum(a[...,2], g[...,2])  # 交集右边界
    y2 = torch.minimum(a[...,3], g[...,3])  # 交集下边界
    
    # 交集面积
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)

    # 计算并集面积
    area_a = (a[...,2]-a[...,0]).clamp(min=0) * (a[...,3]-a[...,1]).clamp(min=0)
    area_g = (g[...,2]-g[...,0]).clamp(min=0) * (g[...,3]-g[...,1]).clamp(min=0)
    union = area_a + area_g - inter + 1e-6  # +1e-6防止除零
    
    # IoU = 交集 / 并集
    return inter / union

class AnchorHead(nn.Module):
    """
    基于锚点的检测头
    
    输出三个预测：
    1. 分类（cls）：每个锚点属于各个类别的概率
    2. 回归（reg）：如何调整锚点以匹配真实框（7个参数）
    3. 方向（dir）：目标的前后方向（2个bin：前/后）
    """
    def __init__(self, c_in, num_classes, num_anchors):
        """
        参数:
            c_in: 输入特征通道数
            num_classes: 类别数量
            num_anchors: 每个cell的锚点数量（类别数×旋转数）
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 特征提取
        self.conv = ConvBNReLU(c_in, c_in, 3, 1, 1)

        # 三个预测头
        self.cls_head = nn.Conv2d(c_in, num_anchors * num_classes, 1)  # 分类
        self.reg_head = nn.Conv2d(c_in, num_anchors * 7, 1)            # 回归（7个参数）
        self.dir_head = nn.Conv2d(c_in, num_anchors * 2, 1)              # 方向（2个bin）

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (B, c_in, Y, X) BEV特征
        
        返回:
            cls: (B, num_anchors*num_classes, Y, X) 分类预测
            reg: (B, num_anchors*7, Y, X) 回归预测
            direc: (B, num_anchors*2, Y, X) 方向预测
        """
        x = self.conv(x)
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        direc = self.dir_head(x)
        return cls, reg, direc

# =========================================================
# 5) CaDDN主模型（论文级骨架）
# =========================================================

class CaDDN_Paper(nn.Module):
    """
    CaDDN主模型：完整的端到端3D目标检测网络
    
    整体流程：
    1. 图像 -> 骨干网络 -> 共享特征
    2. 共享特征 -> 语义头 + 深度头
    3. 语义特征 × 深度分布 -> 视锥体特征
    4. 视锥体特征 -> 体素网格 -> BEV特征
    5. BEV特征 -> 检测头 -> 3D边界框预测
    """
    def __init__(
        self,
        image_hw=(384,1280),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        voxel_grid_size=(200, 200, 16),
        depth_bins=80,
        depth_range=(2.0, 42.0),
        depth_mode="LID",
        sem_channels=64,
        bev_channels=128,
        classes_cfg=(
            {"name":"Car",     "w":1.6, "l":3.9, "h":1.56, "z":-1.0},
            {"name":"Ped",     "w":0.6, "l":0.8, "h":1.73, "z":-0.9},
            {"name":"Cyclist", "w":0.6, "l":1.76,"h":1.73, "z":-0.9},
        ),
        rotations=(0, math.pi/2),
    ):
        """
        参数:
            image_hw: (H, W) 图像尺寸
            pc_range: (x0,y0,z0,x1,y1,z1) 点云范围
            voxel_grid_size: (X, Y, Z) 体素网格大小
            depth_bins: 深度区间数量
            depth_range: (d_min, d_max) 深度范围
            depth_mode: 深度分箱模式
            sem_channels: 语义特征通道数
            bev_channels: BEV特征通道数
            classes_cfg: 类别配置（尺寸和高度）
            rotations: 旋转角度列表
        """
        super().__init__()
        self.image_hw = image_hw
        self.pc_range = pc_range
        self.grid_size = voxel_grid_size
        
        # 深度分箱器
        self.depth_binner = DepthBinner(depth_bins, depth_range[0], depth_range[1], mode=depth_mode)

        # 图像骨干网络 + 双头
        self.backbone = TinyBackbone(in_ch=3, base=64)
        self.dual_heads = CaDDNHeads(self.backbone.out_ch, sem_channels, depth_bins)

        # 体素网格（预计算，不参与训练）
        vg = make_voxel_grid(voxel_grid_size, pc_range)
        self.register_buffer("voxel_grid", vg, persistent=False)

        # 视锥到体素的采样模块
        self.frustum2voxel = FrustumToVoxel(image_hw, self.backbone.stride, self.depth_binner)

        # BEV特征压缩和骨干网络
        Zn = voxel_grid_size[2]  # Z方向大小
        
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(sem_channels*Zn, bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True)
        )
        
        self.bev_backbone = BEVBackbone(bev_channels, bev_channels)

        # 锚点生成器和检测头
        self.anchor_gen = AnchorGenerator(
            pc_range=pc_range,
            grid_size_xy=(voxel_grid_size[0], voxel_grid_size[1]),
            classes_cfg=list(classes_cfg),
            rotations=rotations
        )
        
        self.num_classes = len(classes_cfg)
        self.num_rot = len(rotations)
        self.num_anchors = self.num_classes * self.num_rot

        self.anchor_head = AnchorHead(
            c_in=bev_channels,
            num_classes=self.num_classes,
            num_anchors=self.num_anchors
        )

    def forward(self, images, ego2img):
        """
        前向传播
        
        参数:
            images: (B, 3, H, W) 输入图像
            ego2img: (B, 4, 4) ego坐标系到图像坐标系的变换矩阵
        
        返回:
            dict包含：
                bev_feat: BEV特征
                cls_map: 分类预测图
                reg_map: 回归预测图
                dir_map: 方向预测图
                depth_logits: 深度分布logits
        """
        # 1. 图像特征提取
        feat = self.backbone(images)  # (B, C, Hf, Wf)
        
        # 2. 双头预测：语义特征 + 深度分布，深度在CaDDN这里有真值可以做有监督学习
        sem, depth_logits = self.dual_heads(feat)
        # sem: (B, sem_channels, Hf, Wf)
        # depth_logits: (B, depth_bins, Hf, Wf)
        
        # 3. 深度分布归一化（softmax）
        depth_prob = F.softmax(depth_logits, dim=1)  # (B, D, Hf, Wf)

        # 4. Lift操作：创建视锥体特征
        # 语义特征在深度维度上加权（权重是深度概率）
        # 这实现了"语义特征 × 深度分布"的融合
        frustum_feat = sem.unsqueeze(2) * depth_prob.unsqueeze(1)
        # sem.unsqueeze(2): (B, C, 1, Hf, Wf)
        # depth_prob.unsqueeze(1): (B, 1, D, Hf, Wf)
        # 结果: (B, C, D, Hf, Wf)

        # 5. 视锥体特征 -> 体素网格 
        voxel_feat = self.frustum2voxel(frustum_feat, self.voxel_grid, ego2img)
        # (B, C, Z, Y, X)

        # 6. 压缩Z维度 -> BEV特征
        B, C, Z, Y, X = voxel_feat.shape
        bev = voxel_feat.view(B, C*Z, Y, X)  # 将Z维度展平到通道维度
        bev = self.bev_compressor(bev)        # 压缩通道数
        bev = self.bev_backbone(bev)          # BEV特征提取

        # 7. 检测头预测
        cls_map, reg_map, dir_map = self.anchor_head(bev)

        return {
            "bev_feat": bev,
            "cls_map": cls_map,           # (B, A*num_cls, Y, X)
            "reg_map": reg_map,           # (B, A*7, Y, X)
            "dir_map": dir_map,           # (B, A*2, Y, X)
            "depth_logits": depth_logits  # (B, D, Hf, Wf)
        }


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Focal Loss：用于解决类别不平衡问题
    
    传统交叉熵对所有样本一视同仁，但focal loss：
    - 对容易分类的样本降低权重（gamma参数）
    - 对正负样本进行加权（alpha参数）
    
    参数:
        logits: (N, num_classes) 预测logits
        targets: (N, num_classes) 目标标签（one-hot）
        alpha: 平衡正负样本的权重
        gamma: 聚焦参数，越大越关注难样本
        reduction: 如何聚合损失
    
    返回:
        loss: 标量或tensor
    """
    p = torch.sigmoid(logits)  # 预测概率
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    
    # p_t: 正确类别的概率
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # focal weight: 难样本权重更大
    loss = ce * ((1 - p_t) ** gamma)
    
    # alpha加权：平衡正负样本
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def reshape_head_outputs(cls_map, reg_map, dir_map, num_classes, num_anchors):
    """
    重塑检测头的输出：从特征图格式转换为列表格式
    
    参数:
        cls_map: (B, A*C, Y, X) 分类预测图
        reg_map: (B, A*7, Y, X) 回归预测图
        dir_map: (B, A*2, Y, X) 方向预测图
        num_classes: 类别数
        num_anchors: 每个cell的锚点数
    
    返回:
        cls: (B, Y*X*A, C) 分类预测列表
        reg: (B, Y*X*A, 7) 回归预测列表
        direc: (B, Y*X*A, 2) 方向预测列表
        (Y, X): 特征图尺寸
    """
    B, _, Y, X = cls_map.shape
    
    # 分类：重塑为 (B, Y*X*A, C)
    cls = cls_map.view(B, num_anchors, num_classes, Y, X).permute(0,3,4,1,2).contiguous()
    cls = cls.view(B, Y*X*num_anchors, num_classes)

    # 回归：重塑为 (B, Y*X*A, 7)
    reg = reg_map.view(B, num_anchors, 7, Y, X).permute(0,3,4,1,2).contiguous()
    reg = reg.view(B, Y*X*num_anchors, 7)

    # 方向：重塑为 (B, Y*X*A, 2)
    direc = dir_map.view(B, num_anchors, 2, Y, X).permute(0,3,4,1,2).contiguous()
    direc = direc.view(B, Y*X*num_anchors, 2)

    return cls, reg, direc, (Y, X)

@torch.no_grad()  # 不需要梯度
def assign_targets_single(
    anchors, anchor_cls_ids, gt_boxes, gt_labels,
    pos_iou_thr=0.6, neg_iou_thr=0.45
):
    """
    为单个样本分配目标（标签匹配）
    
    这是目标检测训练的关键步骤：
    1. 计算每个锚点与每个真实框的IoU
    2. 根据IoU阈值分配正负样本
    3. 为正样本计算回归目标
    
    参数:
        anchors: (A,Y,X,7) 所有锚点
        anchor_cls_ids: (A,Y,X) 锚点类别ID
        gt_boxes: (M,7) 真实框
        gt_labels: (M,) 真实标签
        pos_iou_thr: 正样本IoU阈值（>=此值为正样本）
        neg_iou_thr: 负样本IoU阈值（<此值为负样本）
    
    返回:
        labels_onehot: (N, C) one-hot标签，N=Y*X*A
        reg_targets: (N, 7) 回归目标
        dir_targets: (N,) 方向目标（0或1）
        reg_mask: (N,) 正样本掩码
    """
    device = anchors.device
    A, Y, X, _ = anchors.shape
    N = A*Y*X
    
    # 展平锚点
    anchors_f = anchors.reshape(N, 7)
    acls_f = anchor_cls_ids.reshape(N)

    # 确定类别数
    C = int(gt_labels.max().item()+1) if gt_labels.numel() > 0 else int(acls_f.max().item()+1)

    # 初始化输出
    labels = torch.zeros((N, C), device=device)
    reg_targets = torch.zeros((N, 7), device=device)
    dir_targets = torch.zeros((N,), device=device, dtype=torch.long)
    reg_mask = torch.zeros((N,), device=device, dtype=torch.bool)

    # 如果没有真实框，所有锚点都是负样本
    if gt_boxes.numel() == 0:
        return labels, reg_targets, dir_targets, reg_mask

    # 类别感知匹配：只比较同类的锚点和真实框
    # 这是CaDDN的核心思想：category-aware
    for c in range(C):
        # 找到该类别的真实框
        gt_idx = (gt_labels == c).nonzero(as_tuple=False).flatten()
        if gt_idx.numel() == 0:
            continue
        
        # 找到该类别的锚点
        anc_idx = (acls_f == c).nonzero(as_tuple=False).flatten()
        if anc_idx.numel() == 0:
            continue

        anc_c = anchors_f[anc_idx]
        gt_c = gt_boxes[gt_idx]

        # 计算IoU（注意：这里用的是简化版AABB IoU）
        ious = iou_bev_fallback(anc_c, gt_c)  # (Na, Ng)

        # 为每个锚点找到最佳匹配的真实框
        best_iou, best_gt = ious.max(dim=1)

        # 分配正负样本
        pos = best_iou >= pos_iou_thr  # 正样本
        neg = best_iou < neg_iou_thr   # 负样本
        # 中间的是忽略样本（不参与训练）

        pos_idx = anc_idx[pos]

        if pos_idx.numel() > 0:
            # 获取匹配的真实框和锚点
            matched_gt = gt_c[best_gt[pos]]
            matched_anc = anchors_f[pos_idx]
            
            # 计算回归目标（编码后的偏移量）
            reg_targets[pos_idx] = encode_boxes(matched_gt, matched_anc)
            reg_mask[pos_idx] = True

            # one-hot分类标签
            labels[pos_idx, c] = 1.0

            # 方向标签：根据yaw角度判断前后
            yaw = matched_gt[:, 6]
            # bin0: [-π/2, π/2), bin1: 其他
            dir_targets[pos_idx] = (limit_period(yaw, offset=0.5, period=2*math.pi) > 0).long()

    return labels, reg_targets, dir_targets, reg_mask

def caddn_loss_paper_level(
    outputs, model: CaDDN_Paper,
    gt_boxes_list, gt_labels_list,
    gt_depth_map_list=None,
    focal_alpha=0.25, focal_gamma=2.0,
    cls_weight=1.0, reg_weight=2.0, dir_weight=0.2, depth_weight=1.0
):
    """
    CaDDN的总损失函数
    
    包含四个部分：
    1. 分类损失（Focal Loss）
    2. 回归损失（Smooth L1 Loss）
    3. 方向损失（Cross Entropy）
    4. 深度损失（Cross Entropy，可选）
    
    参数:
        outputs: 模型输出字典
        model: CaDDN模型
        gt_boxes_list: 真实框列表，每个元素是(Mi,7)
        gt_labels_list: 真实标签列表，每个元素是(Mi,)
        gt_depth_map_list: 深度图列表（可选），用于深度监督
        focal_alpha, focal_gamma: Focal Loss参数
        cls_weight等: 各损失项的权重
    
    返回:
        loss_dict: 包含总损失和各项损失的字典
    """
    cls_map = outputs["cls_map"]
    reg_map = outputs["reg_map"]
    dir_map = outputs["dir_map"]
    depth_logits = outputs["depth_logits"]

    B = cls_map.shape[0]
    
    # 生成锚点
    anchors, anchor_cls_ids = model.anchor_gen.generate(device=cls_map.device)

    # 重塑预测输出
    cls_pred, reg_pred, dir_pred, (Y, X) = reshape_head_outputs(
        cls_map, reg_map, dir_map, model.num_classes, model.num_anchors
    )

    # 初始化损失累加器
    total_cls = 0.0
    total_reg = 0.0
    total_dir = 0.0
    total_depth = 0.0

    # 对每个样本计算损失
    for b in range(B):
        # 分配目标
        labels_onehot, reg_tgt, dir_tgt, reg_mask = assign_targets_single(
            anchors, anchor_cls_ids,
            gt_boxes_list[b].to(cls_map.device),
            gt_labels_list[b].to(cls_map.device)
        )

        # 分类损失：Focal Loss
        total_cls = total_cls + sigmoid_focal_loss(
            cls_pred[b], labels_onehot,
            alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )

        # 回归损失：只对正样本计算
        if reg_mask.any():
            total_reg = total_reg + F.smooth_l1_loss(
                reg_pred[b][reg_mask],
                reg_tgt[reg_mask],
                reduction="mean"
            )
            
            # 方向损失：只对正样本计算
            total_dir = total_dir + F.cross_entropy(
                dir_pred[b][reg_mask],
                dir_tgt[reg_mask],
                reduction="mean"
            )
        else:
            # 如果没有正样本，损失为0（避免nan）
            total_reg = total_reg + reg_pred[b].sum() * 0.0
            total_dir = total_dir + dir_pred[b].sum() * 0.0

        # 深度监督损失（如果有深度图）
        if gt_depth_map_list is not None:
            gt_depth = gt_depth_map_list[b].to(cls_map.device)
            # 将深度图转换为深度区间标签
            labels, _ = model.depth_binner.depth_to_bin_label(gt_depth, ignore_value=-1)
            total_depth = total_depth + F.cross_entropy(
                depth_logits[b:b+1], labels.unsqueeze(0),
                ignore_index=-1, reduction="mean"
            )

    # 平均化损失
    total_cls = total_cls / B
    total_reg = total_reg / B
    total_dir = total_dir / B
    total_depth = total_depth / B if gt_depth_map_list is not None else 0.0

    # 加权求和
    loss = cls_weight*total_cls + reg_weight*total_reg + dir_weight*total_dir + depth_weight*total_depth
    
    return {
        "loss": loss,
        "loss_cls": total_cls.detach(),
        "loss_reg": total_reg.detach(),
        "loss_dir": total_dir.detach(),
        "loss_depth": (total_depth.detach() if gt_depth_map_list is not None else torch.tensor(0.0, device=cls_map.device))
    }


# =========================================================
# 以下部分是测试和演示代码
# =========================================================



# 注意：上面的代码已经包含了所有必要的模块
# 这里不需要重复导入

def build_dummy_ego2img(B, img_H, img_W, device):
    """
    构造一个简化的ego坐标系到图像坐标系的变换矩阵
    
    用于测试和演示，实际使用时应该从相机标定数据中获取
    
    参数:
        B: batch大小
        img_H, img_W: 图像高度和宽度
        device: 设备
    
    返回:
        ego2img: (B,4,4) 变换矩阵
    """
    # ---- 1) 相机内参矩阵K ----
    # 内参定义了像素坐标和相机坐标的映射关系
    fx = 500.0 * (img_W / 640.0)  # 焦距x（像素）
    fy = 500.0 * (img_W / 640.0)  # 焦距y（像素）
    cx = img_W / 2.0               # 主点x（图像中心）
    cy = img_H / 2.0               # 主点y（图像中心）

    K = torch.zeros((B, 3, 3), device=device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    # 扩展为齐次坐标形式
    K_homo = torch.zeros((B, 4, 4), device=device)
    K_homo[:, :3, :3] = K
    K_homo[:, 3, 3] = 1.0

    # ---- 2) 外参：ego坐标系到相机坐标系 ----
    # ego坐标系：X前、Y左、Z上（车辆坐标系）
    # 相机坐标系：X右、Y下、Z前（常见设定）
    # 旋转矩阵：将ego坐标转换为相机坐标
    R = torch.tensor([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0],
    ], device=device).unsqueeze(0).repeat(B, 1, 1)

    # 组合成4x4变换矩阵
    T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    # 平移项（这里设为0，实际应该根据相机安装位置设置）

    # 组合：内参 × 外参 = 完整的投影矩阵
    ego2img = torch.bmm(K_homo, T)  # (B, 4, 4)
    return ego2img


def build_dummy_gt(B, num_classes, pc_range, classes_cfg, device):
    """
    生成假的真实框（ground truth）用于测试
    
    参数:
        B: batch大小
        num_classes: 类别数
        pc_range: 点云范围
        classes_cfg: 类别配置
        device: 设备
    
    返回:
        gt_boxes_list: 真实框列表
        gt_labels_list: 真实标签列表
    """
    x0, y0, z0, x1, y1, z1 = pc_range
    gt_boxes_list = []
    gt_labels_list = []

    for b in range(B):
        # 随机生成3~7个目标
        M = torch.randint(low=3, high=8, size=(1,)).item()
        labels = torch.randint(low=0, high=num_classes, size=(M,), device=device)

        boxes = torch.zeros((M, 7), device=device)
        for i in range(M):
            cid = int(labels[i].item())
            cfg = classes_cfg[cid]

            # 随机生成中心位置
            boxes[i, 0] = torch.empty(1, device=device).uniform_(x0 + 5.0, x1 - 5.0)  # x
            boxes[i, 1] = torch.empty(1, device=device).uniform_(y0 + 5.0, y1 - 5.0)  # y
            boxes[i, 2] = torch.tensor(cfg["z"], device=device)                        # z

            # 使用类别配置的尺寸
            boxes[i, 3] = torch.tensor(cfg["w"], device=device)  # 宽度
            boxes[i, 4] = torch.tensor(cfg["l"], device=device)  # 长度
            boxes[i, 5] = torch.tensor(cfg["h"], device=device)  # 高度

            # 随机生成朝向角度
            boxes[i, 6] = torch.empty(1, device=device).uniform_(-math.pi, math.pi)

        gt_boxes_list.append(boxes)
        gt_labels_list.append(labels)

    return gt_boxes_list, gt_labels_list


def build_dummy_depth_maps(B, Hf, Wf, depth_min, depth_max, device):
    """
    生成假的深度图用于测试
    
    参数:
        B: batch大小
        Hf, Wf: 特征图尺寸
        depth_min, depth_max: 深度范围
        device: 设备
    
    返回:
        depth_maps: 深度图列表
    """
    depth_maps = []
    for _ in range(B):
        # 随机生成深度值
        d = torch.empty((Hf, Wf), device=device).uniform_(depth_min, depth_max)
        # 随机设置一部分为无效值（模拟真实场景中的缺失深度）
        mask = torch.rand((Hf, Wf), device=device) < 0.1
        d[mask] = 0.0  # 无效值（<d_min，训练时会忽略）
        depth_maps.append(d)
    return depth_maps


def main():
    """
    主函数：演示完整的训练流程
    
    包括：
    1. 模型初始化
    2. 构造输入数据
    3. 前向传播
    4. 计算损失
    5. 反向传播
    """
    # 设置随机种子（保证结果可复现）
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =========================================================
    # 1) 配置参数（使用小配置避免内存溢出）
    # =========================================================
    B = 2  # batch大小

    # 图像尺寸（原图尺寸，用于投影）
    img_H, img_W = 192, 640  # stride=4 -> Hf=48, Wf=160

    # 3D空间范围
    pc_range = (0.0, -20.0, -3.0, 50.0, 20.0, 1.0)
    voxel_grid_size = (50, 50, 8)   # X,Y,Z（小网格以节省内存）

    # 深度分箱配置
    depth_bins = 16
    depth_range = (2.0, 42.0)
    depth_mode = "LID"

    # 类别配置（KITTI数据集风格）
    classes_cfg = [
        {"name": "Car",     "w": 1.6, "l": 3.9,  "h": 1.56, "z": -1.0},
        {"name": "Ped",     "w": 0.6, "l": 0.8,  "h": 1.73, "z": -0.9},
        {"name": "Cyclist", "w": 0.6, "l": 1.76, "h": 1.73, "z": -0.9},
    ]
    
    rotations = (0.0, math.pi / 2)  # 0度和90度

    # =========================================================
    # 2) 初始化模型
    # =========================================================
    model = CaDDN_Paper(
        image_hw=(img_H, img_W),
        pc_range=pc_range,
        voxel_grid_size=voxel_grid_size,
        depth_bins=depth_bins,
        depth_range=depth_range,
        depth_mode=depth_mode,
        sem_channels=64,
        bev_channels=128,
        classes_cfg=tuple(classes_cfg),
        rotations=rotations,
    ).to(device)

    model.train()  # 设置为训练模式

    # =========================================================
    # 3) 构造输入数据
    # =========================================================
    # 随机图像（实际使用时应该是真实图像）
    images = torch.randn((B, 3, img_H, img_W), device=device)
    # 投影矩阵（实际使用时应该从相机标定数据中获取）
    ego2img = build_dummy_ego2img(B, img_H, img_W, device=device)

    # =========================================================
    # 4) 构造真实标签（ground truth）
    # =========================================================
    gt_boxes_list, gt_labels_list = build_dummy_gt(
        B=B,
        num_classes=len(classes_cfg),
        pc_range=pc_range,
        classes_cfg=classes_cfg,
        device=device
    )

    # 深度监督：需要对齐到特征图尺寸
    Hf = img_H // model.backbone.stride
    Wf = img_W // model.backbone.stride
    gt_depth_map_list = build_dummy_depth_maps(
        B=B, Hf=Hf, Wf=Wf,
        depth_min=depth_range[0], depth_max=depth_range[1],
        device=device
    )

    # =========================================================
    # 5) 前向传播 + 损失计算 + 反向传播
    # =========================================================
    # 优化器
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 前向传播
    outputs = model(images, ego2img)

    # 计算损失
    loss_dict = caddn_loss_paper_level(
        outputs, model,
        gt_boxes_list, gt_labels_list,
        gt_depth_map_list=gt_depth_map_list,
        cls_weight=1.0, reg_weight=2.0, dir_weight=0.2, depth_weight=1.0
    )

    # 反向传播和参数更新
    optim.zero_grad(set_to_none=True)  # 清零梯度
    loss_dict["loss"].backward()       # 反向传播
    optim.step()                       # 更新参数

    # =========================================================
    # 6) 打印结果
    # =========================================================
    print("\n==== Forward Outputs ====")
    print("bev_feat:", outputs["bev_feat"].shape)
    print("cls_map :", outputs["cls_map"].shape)
    print("reg_map :", outputs["reg_map"].shape)
    print("dir_map :", outputs["dir_map"].shape)
    print("depth_logits:", outputs["depth_logits"].shape)

    print("\n==== Loss ====")
    print("loss      :", float(loss_dict["loss"].detach().cpu()))
    print("loss_cls  :", float(loss_dict["loss_cls"].detach().cpu()))
    print("loss_reg  :", float(loss_dict["loss_reg"].detach().cpu()))
    print("loss_dir  :", float(loss_dict["loss_dir"].detach().cpu()))
    print("loss_depth:", float(loss_dict["loss_depth"].detach().cpu()))

    print("\n✅ main() 运行成功：前向传播 + 损失计算 + 反向传播 + 参数更新完成。")
    print("注意：目前IoU使用的是AABB fallback，想要论文级数值需要替换为旋转IoU/旋转NMS。")


if __name__ == "__main__":
    main()