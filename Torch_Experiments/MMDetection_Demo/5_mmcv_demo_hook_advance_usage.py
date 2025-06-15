# ——————————————————————————————————————————————
# 1. Registry: 管理可插拔组件
# 功能：管理可插拔组件，提供注册和构建功能
# ——————————————————————————————————————————————
class Registry:
    def __init__(self, name):
        """
        初始化注册表
        :param name: 注册表名称
        """
        
        self.name = name
        self._modules = {}

    def register(self, cls=None, *, name=None):
        """
        注册组件类
        :param cls: 要注册的类
        :param name: 自定义注册名称
        :ret
        """
        if cls is None:
            # 支持装饰器语法
            return lambda actual_cls: self.register(actual_cls, name=name)
        
        # 确定注册键名
        key = name or cls.__name__
        
        # 存储组件类
        self._modules[key] = cls
        return cls

    def build(self, cfg):
        """
        根据配置构建组件实例
        :param cfg: 配置字典，必须包含'type'键
        :return: 组件实例
        """
        
        # cfg: {'type': 'ClassName', ...init args...}
        # 获取组件类
        cls = self._modules[cfg['type']]
        
        # 提取构造参数
        args = {k: v for k, v in cfg.items() if k != 'type'}
        
        # 创建实例
        return cls(**args)

# 创建一个 DETECTORS 注册表
DETECTORS = Registry('detectors')



# ——————————————————————————————————————————————
# 2. build_detector: 从配置动态构建模型
# ——————————————————————————————————————————————
def build_detector(cfg, train_cfg=None, test_cfg=None):
    """
    构建检测器
    :param cfg: 基础配置
    :param train_cfg: 训练配置
    :param test_cfg: 测试配置
    :return: 检测器实例
    """
    
    # 简单地把 train_cfg/test_cfg 合并到 cfg 中
    full_cfg = cfg.copy()
    if train_cfg:
        full_cfg['train_cfg'] = train_cfg
    if test_cfg:
        full_cfg['test_cfg'] = test_cfg
    
    # 通过注册表构建检测器
    model = DETECTORS.build(full_cfg)
    return model



# ——————————————————————————————————————————————
# 3. Runner + Hook: 控制训练/推理流程，并注入回调
# ——————————————————————————————————————————————
class Hook:
    """定义所有可钩入的回调接口"""
    def before_run(self, runner): 
        pass  # 训练开始前
    
    def after_run(self, runner): 
        pass   # 训练结束后
    
    def before_epoch(self, runner): 
        pass  # 每个epoch开始前
    
    def after_epoch(self, runner): 
        pass   # 每个epoch结束后
    
    def before_iter(self, runner): 
        pass   # 每次迭代前
    
    def after_iter(self, runner): 
        pass    # 每次迭代后

class Runner:
    """
    训练运行器，管理训练流程
    """
    def __init__(self, model, data, max_epochs):
        """
        初始化
        :param model: 模型
        :param data: 数据
        :param max_epochs: 最大epoch数
        """
        
        self.model = model
        self.data = data            # iterable of samples
        self.max_epochs = max_epochs
        self.hooks = []

    def register_hook(self, hook):
        """注册钩子"""
        self.hooks.append(hook)

    def run(self):
        """执行训练流程"""
        
        # 训练/推理主循环
        for hook in self.hooks:
            hook.before_run(self)

        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            for hook in self.hooks:
                hook.before_epoch(self)

            for idx, sample in enumerate(self.data, 1):
                self.iter = idx
                for hook in self.hooks:
                    hook.before_iter(self)

                # —— 模型前向／反向示意 —— #
                loss = self.model.train_step(sample)

                for hook in self.hooks:
                    hook.after_iter(self)

            for hook in self.hooks:
                hook.after_epoch(self)

        for hook in self.hooks:
            hook.after_run(self)



# ——————————————————————————————————————————————
# ===== 定义示例组件，演示如何注册与使用 =====
# ——————————————————————————————————————————————

# 1) 自定义 Detector，并注册到 DETECTORS
@DETECTORS.register()
class FakeDetector:
    """模拟检测器类"""
    def __init__(self, backbone, num_classes, train_cfg=None, test_cfg=None):
        self.backbone = backbone
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        print("Initializing weights...")

    def train_step(self, sample):
        # 假设每个 sample 返回一个随机 loss
        import random
        loss = random.random()
        print(f"[Epoch {runner.epoch}][Iter {runner.iter}] loss={loss:.4f}")
        return loss

# 2) 自定义 Hook，实现简单打印
class PrintHook(Hook):
    # 这里改成了pre_run的话，不会调用这个函数，而是基类hook里面的空的函数
    def pre_run(self, runner):
        print(">> Start training")

    def before_epoch(self, runner):
        print(f">> Epoch {runner.epoch} begin")

    def after_epoch(self, runner):
        print(f">> Epoch {runner.epoch} end\n")

    def after_run(self, runner):
        print(">> Training finished")



# ——————————————————————————————————————————————
# ===== 示范完整流程 =====
# ——————————————————————————————————————————————
if __name__ == "__main__":
    
    # 配置部分：模拟从 config 文件读到的字典
    model_cfg = {
        'type': 'FakeDetector',
        'backbone': 'ResNet50',
        'num_classes': 80
    }
    
    train_cfg = {'lr': 0.001}
    test_cfg  = {'score_thr': 0.5}

    # 1. 构建模型
    model = build_detector(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    model.init_weights()

    # 2. 构造假数据集
    data = list(range(1, 6))  # 5 个 sample

    # 3. 构建 Runner 并注册 Hook
    runner = Runner(model, data, max_epochs=2)
    runner.register_hook(PrintHook())

    # 4. 运行
    runner.run()
    
