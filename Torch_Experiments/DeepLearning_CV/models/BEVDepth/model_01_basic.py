import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1) 核心：DepthNet
# =========================
class DepthNet(nn.Module):
    """
    DepthNet 是 BEVDepth / LSS 中最核心的子网络之一。

    它的职责不是做 BEV，也不是做检测，而是：
      - 对每一个像素位置，预测一个【深度分布】（离散 D 个 bin）
      - 同时输出该像素的【语义/上下文特征】，供后续 Lift 使用

    输入:
      img_feat: (B*N, Cin, Hf, Wf)
        - B: batch size
        - N: 相机数量
        - Cin: 图像特征通道数（通常来自 backbone，如 256 / 512）
        - Hf, Wf: 下采样后的特征图尺寸

    输出:
      depth_logits: (B*N, D, Hf, Wf)
        - 对每个像素预测 D 个深度 bin 的 logits（分类）
      context_feat: (B*N, Cout, Hf, Wf)
        - 对应像素的语义特征（会被 Lift 到 3D）
    """

    def __init__(self, in_channels: int, num_depth_bins: int, context_channels: int):
        super().__init__()

        # D = 深度离散 bin 数（如 41：4m~45m, step=1m）
        self.D = num_depth_bins  # 41

        # Cout = Lift 后 BEV 的通道数（如 64）
        self.Cout = context_channels

        # 一个中间隐藏通道数（经验设置，保证容量）
        hidden = max(64, in_channels // 2) # 256

        # stem：共享的特征提取层
        # 目的：
        #   - 在分 depth/context 之前做一层特征“解耦”
        #   - 让两个 head 使用同一份中间表征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )


        # 深度分类 head
        # 输出 D 个通道，每个通道对应一个 depth bin 的 logit
        self.depth_head = nn.Conv2d(hidden, self.D, kernel_size=1)

        # 语义 / 上下文 head
        # 输出 Cout 个通道，会在 Lift 阶段和 depth_prob 结合
        self.context_head = nn.Conv2d(hidden, self.Cout, kernel_size=1)


    def forward(self, img_feat: torch.Tensor):
        """
        前向传播

        输入:
          img_feat: (B*N, Cin, Hf, Wf)

        输出:
          depth_logits: (B*N, D, Hf, Wf)
          context_feat: (B*N, Cout, Hf, Wf)
        """
        x = self.stem(img_feat) # [48, 256, 16, 44]

        # 深度分支：分类 logits（还没 softmax）
        depth_logits = self.depth_head(x)

        # 语义分支：用于 Lift 的上下文特征
        context_feat = self.context_head(x)

        return depth_logits, context_feat


# =========================
# 2) Fake Dataset：随机特征 + 随机深度标签(带mask)
# =========================
class FakeDepthDataset(Dataset):
    """
    这是一个【教学 / 验证流程用】的数据集。

    它模拟了 BEVDepth 训练 depth head 时所需的数据格式：
      - 输入：图像特征（这里用随机数代替 backbone 输出）
      - 监督: 离散深度标签（bin id） + 有效 mask

    注意：
      - 在真实 BEVDepth 中，depth_gt 来自 LiDAR 投影
      - valid mask 表示哪些像素真的被 LiDAR 覆盖
    """

    def __init__(
        self,
        length: int = 200,
        num_cams: int = 6,
        in_channels: int = 512,
        Hf: int = 16,
        Wf: int = 44,
        num_depth_bins: int = 41,
        valid_ratio: float = 0.7,   # 有效像素比例（模拟 LiDAR 稀疏性）
        seed: int = 0
    ):
        super().__init__()
        self.length = length
        self.N = num_cams
        self.Cin = in_channels
        self.Hf, self.Wf = Hf, Wf
        self.D = num_depth_bins
        self.valid_ratio = valid_ratio
        self.rng = torch.Generator().manual_seed(seed)

        super().__init__()
        self.length = length
        self.N = num_cams
        self.Cin = in_channels
        self.Hf, self.Wf = Hf, Wf
        self.D = num_depth_bins
        self.valid_ratio = valid_ratio
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # -------------------------
        # 1) 模拟 backbone 输出
        # -------------------------
        # img_feats: (N, Cin, Hf, Wf)
        img_feats = torch.randn(self.N, self.Cin, self.Hf, self.Wf, generator=self.rng)

                # -------------------------
        # 2) 模拟深度标签
        # -------------------------
        # 每个像素一个 depth bin id ∈ [0, D-1]
        # shape: (N, Hf, Wf)
        depth_gt = torch.randint(
            low=0, high=self.D,
            size=(self.N, self.Hf, self.Wf),
            generator=self.rng
        )

                # -------------------------
        # 3) 模拟 LiDAR 有效 mask
        # -------------------------
        # 真实情况：LiDAR 投影非常稀疏
        # valid=True 的像素才参与 loss
        valid = (torch.rand(self.N, self.Hf, self.Wf, generator=self.rng) < self.valid_ratio)

        return img_feats, depth_gt, valid

        # 也可以模拟“越远越稀疏”：这里保持简单就行
        return img_feats, depth_gt, valid


# =========================
# 3) loss：masked depth CE
# =========================
def masked_depth_ce(depth_logits: torch.Tensor,
                    depth_gt: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """
    带 mask 的深度分类损失（BEVDepth 核心）

    输入:
      depth_logits: (B*N, D, H, W)
      depth_gt:     (B*N, H, W)
      valid:        (B*N, H, W)  bool

    输出:
      标量 loss
    """

    # 每个像素的 CE loss
    # shape: (B*N, H, W)
    per_pixel = F.cross_entropy(depth_logits, depth_gt, reduction="none")

    # 只在 valid 像素上计算 loss
    valid_f = valid.float()
    denom = valid_f.sum().clamp(min=1.0)

    loss = (per_pixel * valid_f).sum() / denom
    return loss



@torch.no_grad()
def depth_metrics(depth_logits, depth_gt, valid):
    """
    计算两个直观指标（用于 sanity check）：

    1) Top-1 Accuracy（在有效像素上）
    2) 平均置信度（max softmax prob）
    """

    prob = F.softmax(depth_logits, dim=1)  # (BN,D,H,W)
    pred = prob.argmax(dim=1)              # (BN,H,W)
    conf = prob.max(dim=1).values          # (BN,H,W)

    valid_f = valid.float()
    denom = valid_f.sum().clamp(min=1.0)

    acc = ((pred == depth_gt).float() * valid_f).sum() / denom
    mean_conf = (conf * valid_f).sum() / denom
    return float(acc.item()), float(mean_conf.item())


# =========================
# 4) 训练与验证
# =========================
def run_train_val(
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 3,
    batch_size: int = 2,
    num_cams: int = 6,
    Cin: int = 512,
    Hf: int = 16,
    Wf: int = 44,
    D: int = 41,
    Cout: int = 64,
):
    print(f"Using device: {device}")

    # -------------------------
    # 数据集 & DataLoader
    # -------------------------
    train_ds = FakeDepthDataset(length=400, num_cams=num_cams,
                                in_channels=Cin, Hf=Hf, Wf=Wf,
                                num_depth_bins=D, valid_ratio=0.7)

    val_ds = FakeDepthDataset(length=120, num_cams=num_cams,
                              in_channels=Cin, Hf=Hf, Wf=Wf,
                              num_depth_bins=D, valid_ratio=0.7, seed=123)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # -------------------------
    # 模型 & 优化器
    # -------------------------
    model = DepthNet(Cin, D, Cout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        # -----------------
        # train
        # -----------------
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        tr_conf = 0.0
        n_batches = 0

        for img_feats, depth_gt, valid in train_loader:
            
            # img_feats: (B, N, Cin, Hf, Wf) -> [8, 6, 512, 16, 44]
            B, N, Cin_, H_, W_ = img_feats.shape
            assert N == num_cams and Cin_ == Cin and H_ == Hf and W_ == Wf

            img_feats = img_feats.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True).long() # [8, 6, 16, 44]
            valid = valid.to(device, non_blocking=True).bool() # [8, 6, 16, 44]

            # 变成 BN 维度（和真实实现一致：对每个相机独立做 depth）
            img_feats_bn = img_feats.view(B * N, Cin, Hf, Wf)
            depth_gt_bn = depth_gt.view(B * N, Hf, Wf)
            valid_bn = valid.view(B * N, Hf, Wf)

            depth_logits, context_feat = model(img_feats_bn)  # depth_logits: (BN,D,Hf,Wf), context: (BN,Cout,Hf,Wf)

            # 只训练 depth（符合你的要求：核心 depthnet）
            loss = masked_depth_ce(depth_logits, depth_gt_bn, valid_bn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            acc, conf = depth_metrics(depth_logits, depth_gt_bn, valid_bn)

            tr_loss += float(loss.item())
            tr_acc += acc
            tr_conf += conf
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc /= max(1, n_batches)
        tr_conf /= max(1, n_batches)


        
        # -----------------
        # val
        # -----------------
        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        va_conf = 0.0
        n_batches = 0

        with torch.no_grad():
            for img_feats, depth_gt, valid in val_loader:
                B, N, Cin_, H_, W_ = img_feats.shape
                img_feats = img_feats.to(device, non_blocking=True)
                depth_gt = depth_gt.to(device, non_blocking=True).long()
                valid = valid.to(device, non_blocking=True).bool()

                img_feats_bn = img_feats.view(B * N, Cin, Hf, Wf)
                depth_gt_bn = depth_gt.view(B * N, Hf, Wf)
                valid_bn = valid.view(B * N, Hf, Wf)

                depth_logits, context_feat = model(img_feats_bn)
                loss = masked_depth_ce(depth_logits, depth_gt_bn, valid_bn)
                acc, conf = depth_metrics(depth_logits, depth_gt_bn, valid_bn)

                va_loss += float(loss.item())
                va_acc += acc
                va_conf += conf
                n_batches += 1

        va_loss /= max(1, n_batches)
        va_acc /= max(1, n_batches)
        va_conf /= max(1, n_batches)

        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"train loss {tr_loss:.4f}, acc {tr_acc:.3f}, conf {tr_conf:.3f} | "
            f"val loss {va_loss:.4f}, acc {va_acc:.3f}, conf {va_conf:.3f}"
        )

    return model


if __name__ == "__main__":
    run_train_val()
