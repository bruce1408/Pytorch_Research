import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def plot_lr_history(lr_history, title, output_dir="lr_plots"):
    """
    使用 matplotlib 绘制学习率变化的历史记录。

    Args:
        lr_history (np.array): 一个 numpy 数组，每一行包含 (训练步数, 学习率)。
        title (str): 图表的标题。
    """
    
    # 1. 确保保存图片的文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 根据标题生成一个安全的文件名 (替换空格和特殊字符)
    safe_filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    filepath = os.path.join(output_dir, safe_filename)


    # 创建一个新的图形，避免在同一张图上重复绘制
    plt.figure(figsize=(10, 6))

    # 绘制学习率曲线，并添加标记点
    plt.plot(lr_history[:, 0], lr_history[:, 1], marker='o', linestyle='-')

    # --- 美化图表 ---
    plt.title(title, fontsize=16)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=10)

    # 显示图表
    # plt.show()
    # 3. 保存图表到文件，而不是显示它
    plt.savefig(filepath)
    print(f"📈 Plot saved to: {filepath}")

    # 关闭当前图表，为下一个图表做准备
    plt.close()


def run_scheduler_test(scheduler_name, scheduler_class, scheduler_params,
                       initial_lr=0.5, num_steps=20):
    """
    通用的测试函数，用于测试各种 PyTorch 学习率调度器。
    它会为每个测试创建独立的模型和优化器，保证测试环境的纯净。
    """
    print(f"\n--- Testing {scheduler_name} ---")

    # 1. 为本次测试创建全新的模型和优化器
    model = nn.Conv2d(3, 64, 3)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    # 2. 根据传入的类和参数，实例化学习率调度器
    scheduler = scheduler_class(optimizer, **scheduler_params)

    lr_history_list = []

    # 3. 模拟训练循环
    for step in range(num_steps):
        # 记录当前步的学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history_list.append((step, current_lr))
        print(f"Step {step}: LR = {current_lr:.5f}")

        # 4. 模拟一次训练迭代 (这是调用 optimizer.step() 的前提)
        optimizer.zero_grad()
        # 创建一个假的输入和计算一个假的损失
        dummy_input = torch.randn(1, 3, 64, 64)
        loss = model(dummy_input).sum()
        loss.backward()
        optimizer.step()

        # 5. 更新学习率 (这是学习率调度器的核心)
        scheduler.step()

    lr_history = np.array(lr_history_list)
    plot_lr_history(lr_history, f"{scheduler_name} Learning Rate Schedule")


def test_warmup(init_lr=0.1, warmup_steps=5, total_steps=20):
    """
    测试手动实现的 Warmup 学习率策略。
    """
    print(f"\n--- Testing Manual Warmup ---")
    lr_history_list = []

    # 模拟训练循环
    for step in range(total_steps):
        if step < warmup_steps:
            # 在 Warmup 阶段，学习率从 0 线性增加到 init_lr
            warmup_percent_done = (step + 1) / warmup_steps
            learning_rate = init_lr * warmup_percent_done
        else:
            # Warmup 结束后，使用预设的学习率
            # 在实际应用中，这里通常会衔接一个学习率衰减策略
            learning_rate = init_lr

        lr_history_list.append((step, learning_rate))
        print(f"Step {step}: LR = {learning_rate:.5f}")

    lr_history = np.array(lr_history_list)
    plot_lr_history(lr_history, f"Manual Warmup (for {warmup_steps} steps)")


# 脚本的主入口
if __name__ == '__main__':
    # --- Test 1: StepLR ---
    # 每隔 step_size 步，将学习率乘以 gamma
    run_scheduler_test(
        scheduler_name="StepLR",
        scheduler_class=optim.lr_scheduler.StepLR,
        scheduler_params={'step_size': 5, 'gamma': 0.5}
    )

    # --- Test 2: MultiStepLR ---
    # 在 milestones 指定的步骤，将学习率乘以 gamma
    run_scheduler_test(
        scheduler_name="MultiStepLR",
        scheduler_class=optim.lr_scheduler.MultiStepLR,
        scheduler_params={'milestones': [5, 10, 15], 'gamma': 0.5}
    )

    # --- Test 3: ExponentialLR ---
    # 每个步骤都将学习率乘以 gamma，实现指数衰减
    run_scheduler_test(
        scheduler_name="ExponentialLR",
        scheduler_class=optim.lr_scheduler.ExponentialLR,
        scheduler_params={'gamma': 0.85}
    )

    # --- Test 4: CosineAnnealingLR ---
    # 学习率按照余弦曲线变化。T_max 是半个周期的步数。
    # 余弦退火通常在更多的步数下效果更明显，这里我们模拟50步。
    run_scheduler_test(
        scheduler_name="CosineAnnealingLR",
        scheduler_class=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={'T_max': 50, 'eta_min': 0.01},
        num_steps=50
    )

    # --- Test 5: Warmup ---
    # 这是手动实现的，所以单独调用测试函数
    test_warmup(init_lr=0.5, warmup_steps=5, total_steps=20)