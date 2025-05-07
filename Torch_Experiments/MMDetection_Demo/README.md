# Registry
## 什么是Registry
Registry（注册表）是一种管理和查找「可插拔组件」的通用机制。框架内的所有可扩展模块（Backbone/Neck/Head/Detector/DataSet/Hook……）都挂在各自的 Registry 上：


BACKBONES ──┐
NECKS       │── Registry  ─── model 构建时一键查找并实例化
HEADS       │
DETECTORS  ─┘


## 它做什么

- 集中管理：把所有组件类（如 ResNet, FPN, FasterRCNN）注册到统一表里

- 动态构建：根据配置文件里的 type='FasterRCNN'，自动去 Registry 里找对应类并 FasterRCNN(**cfg) 实例化


## 使用场景
你写了一个 自定义 Backbone/Head 或全新的 Detector，想让 build_detector、build_dataset、build_backbone 等统一接口识别它，只需在代码里加上装饰器即可。

```py

# demo_registry.py
from mmcv.utils import Registry

# 1. 新建一个专门注册优化器的表
OPTIMS = Registry('optimizer')

# 2. 用装饰器注册自定义优化器类
@OPTIMS.register_module()
class MyOptim:
    def __init__(self, params, lr=0.01):
        self.optim = __import__('torch').optim.SGD(params, lr=lr)
    def step(self): self.optim.step()
    def zero_grad(self): self.optim.zero_grad()

# 3. 动态从 cfg 构建
cfg = dict(type='MyOptim', params=[1,2,3], lr=0.1)
optim = OPTIMS.build(cfg)
print(type(optim))   # <class 'demo_registry.MyOptim'>
```


# build_detector

## 它是什么？
build_detector 是 mmdetection 提供的「一行式」模型构建接口。它在底层调用了 DETECTORS.build(cfg)，并自动帮你合并全局的 train_cfg/test_cfg。

## 它做什么？
统一入口：接收 cfg.model、cfg.train_cfg、cfg.test_cfg，无需你手动拼装各子模块

支持多模型：传入 List[Dict] 时，会自动包装成集成模型（EnsembleDetector）

## 典型场景
在 训练或推理脚本 中，通过读取不同的配置文件，动态构建各种主流检测器（FasterRCNN/RetinaNet/CascadeRCNN…）

快速对比不同模型架构，只需传给脚本不同 --config 参数

```py
# demo_build_detector.py
from mmcv import Config
from mmdet.models import build_detector

# 假设有两份配置，分别定义了不同模型
for cfg_path in ['faster_rcnn_r50_fpn.py','retinanet_r50_fpn.py']:
    cfg = Config.fromfile(cfg_path)
    model = build_detector(
        cfg.model,             # 模型结构字典
        train_cfg=cfg.train_cfg,
        test_cfg=cfg.test_cfg
    )
    model.init_weights()
    print(f"Built {model.__class__.__name__} from {cfg_path}")

```


# Hook & Runner
## 它们是什么？
Runner：训练/验证循环的控制器，内置 epoch/iter 的主流程。

Hook：挂在 Runner 上的回调，能在「运行前后」「每个 epoch 前后」「每次迭代前后」插入自定义逻辑。

```
Runner.run():
  for hook in hooks: hook.before_run()
  for epoch:
    for hook in hooks: hook.before_train_epoch()
    for iter:
      for hook in hooks: hook.before_train_iter()
      → forward/backward/step
      for hook in hooks: hook.after_train_iter()
    for hook in hooks: hook.after_train_epoch()
  for hook in hooks: hook.after_run()

```

## 它们做什么？
Runner 解耦了「训练循环」和「模型实现」，让你无需写 for epoch, for batch

Hook 帮你把日志打印、梯度裁剪、学习率调度、模型保存、可视化等操作都插进去，不改核心训练代码

## 典型场景
打印/持久化：每隔 N 次迭代打印 loss、记录到 TensorBoard

检查点：每隔 M 个 epoch 自动保存模型权重

动态调整：在训练过程中根据指标自动缩减学习率或换数据增强策略

```py
# demo_hook_runner.py
import torch
from mmcv import Config
from mmcv.runner import build_runner, Hook
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

# 自定义 Hook：每 100 iter 打印一次当前 loss
class LossLogger(Hook):
    def __init__(self, interval=100):
        self.interval = interval
    def after_train_iter(self, runner):
        loss = runner.log_buffer.output.get('loss')
        if runner.iter % self.interval == 0:
            print(f"[Iter {runner.iter}] loss={loss:.4f}")

# 1. 加载配置
cfg = Config.fromfile('faster_rcnn_r50_fpn.py')
cfg.work_dir = './work_dirs/demo'
cfg.total_epochs = 1

# 2. 构建模型和数据
model = build_detector(cfg.model, cfg.train_cfg, cfg.test_cfg)
datasets = [build_dataset(cfg.data.train)]

# 3. 构建 Runner
runner = build_runner(
    cfg.runner,
    default_args=dict(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        work_dir=cfg.work_dir,
        logger=None
    )
)

# 4. 注册自定义 Hook
runner.register_hook(LossLogger(interval=50), priority='LOW')

# 5. 启动训练
runner.run(datasets, cfg.workflow)

```

# 三者协同使用流程
Registry：先注册／加载你需要的模型、数据集或 Hook 类

build_detector：由配置文件驱动，一行构建目标 Detector

Hook & Runner：用 Runner.auto_run 解放你的训练循环，在 Hook 里插入监控、保存、调度逻辑


# 什么是回调函数
回调函数和“普通函数直接调用”最大的区别就在于——谁来决定“何时”去调用你的扩展逻辑。
```py
def log_loss(epoch, loss):
    print(f"[Manual] Epoch {epoch}: loss={loss:.4f}")

def train_loop(epochs):
    for epoch in range(1, epochs+1):
        # …训练逻辑…
        loss = 1.0 / epoch
        log_loss(epoch, loss)   # ★ 只能在这里写死调用
```
- 缺点：你只能在 train_loop 里“硬编码”调用 log_loss。

- 如果后来想加更多操作（比如保存模型、动态调参），就得在 train_loop 里挨个插入这些函数调用，主循环逻辑迅速臃肿，且每改一次都要进到核心代码改。


## 回调函数（解耦、动态插拔）
```py
class Trainer:
    def __init__(...):
        self._callbacks = []

    def register_callback(self, fn):
        self._callbacks.append(fn)

    def fit(self, epochs):
        for epoch in range(1, epochs+1):
            # …训练逻辑…
            loss = 1.0 / epoch
            # ★ 在“钩子点”遍历回调，统一触发
            for cb in self._callbacks:
                cb(epoch, loss)
```

### 优点

- 解耦：Trainer 主流程不需要知道回调函数具体做什么，只负责在“after-epoch”时“广播”给所有已注册的回调。

- 灵活扩展：后续要加新的功能（logging、保存、EarlyStopping……），只要写一个新回调 def save_ckpt(...)，再 trainer.register_callback(save_ckpt)，即可无痛插入。

- 配置驱动：可以把回调列表放到配置文件里，启动时按需加载，无需改动训练代码。

- 符合开闭原则，对扩展开放，对修改封闭，不改已有的代码，而是增加新的功能


### 回调函数的使用场景
- 日志 / 可视化
  - 每 N 次迭代或每个 epoch 结束时收集 Loss、精度，写入 TensorBoard、控制台或文件。

- 模型检查点
  - 在指定周期自动保存权重、最优模型、EMA 快照等。

- 学习率调度
  - 根据当前迭代数或监测到的指标，动态调整学习率而无需在主循环里写 if/else。

- EarlyStopping / 动态数据增强
  - 监测验证集性能，让训练提前停止，或在训练中途切换或关闭某些增强策略。


```py
# ——— 普通函数 “硬编码” 版 ———
def log_loss(epoch, loss):
    print(f"[Manual] loss@epoch{epoch}={loss:.4f}")

def save_model(epoch, model):
    model.save(f"ckpt_{epoch}.pth")

def train_loop(model, data, epochs):
    for epoch in range(1, epochs+1):
        # …训练逻辑…
        loss = compute_loss(model, data)
        # ★ 每次想加功能，都要在这里写
        log_loss(epoch, loss)
        save_model(epoch, model)

```

```py
# ——— 回调函数 “插件式” 版 ———
class Trainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.callbacks = []

    def register_callback(self, fn):
        self.callbacks.append(fn)

    def fit(self, epochs):
        for epoch in range(1, epochs+1):
            loss = compute_loss(self.model, self.data)
            # ★ “钩子点”统一广播
            for cb in self.callbacks:
                cb(self, epoch, loss)

# 用户只需要写回调、注册即可
def log_callback(trainer, epoch, loss):
    print(f"[Callback] loss@epoch{epoch}={loss:.4f}")

def save_callback(trainer, epoch, loss):
    trainer.model.save(f"ckpt_{epoch}.pth")

trainer = Trainer(model, data)
trainer.register_callback(log_callback)
trainer.register_callback(save_callback)
trainer.fit(10)
```



