# demo_registry.py
from mmcv.utils import Registry, build_from_cfg

# 1. 创建一个新的 Registry，用于注册 “模型” 类
MODELS = Registry('model')

# 2. 定义几个“模型”类，并用装饰器注册到 MODELS 里
@MODELS.register_module()
class LinearModel:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
    def __repr__(self):
        return f'LinearModel({self.in_features}→{self.out_features})'

@MODELS.register_module()
class MLPModel:
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    def __repr__(self):
        return f'MLPModel(hidden={self.hidden_size}, layers={self.num_layers})'

# 3. 写一个小函数，模拟 build_from_cfg 的工作
def build_model(cfg):
    """
    cfg: dict, 必须包含 'type' 字段，在 MODELS 里必须有对应的注册。
    其他字段将作为参数传给模型的 __init__。
    """
    return build_from_cfg(cfg, MODELS)

# 4. 在主逻辑里用配置字典动态构造不同模型
if __name__ == '__main__':
    # 定义两种配置
    cfg1 = dict(type='LinearModel', in_features=128, out_features=10)
    cfg2 = dict(type='MLPModel',    hidden_size=256, num_layers=3)

    # 动态构造
    model1 = build_model(cfg1)
    model2 = build_model(cfg2)

    print(model1)  # LinearModel(128→10)
    print(model2)  # MLPModel(hidden=256, layers=3)
