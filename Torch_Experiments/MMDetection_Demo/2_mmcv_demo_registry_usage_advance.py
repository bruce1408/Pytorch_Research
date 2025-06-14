from typing import Dict, Any, Type, Optional

# 定义一个简单的 Registry 类，用于管理模块注册和构建
class Registry:
    def __init__(self, name: str):
        self._name: str = name # 给注册器类一个名字
        self._module_dict: Dict[str, Type] = {} # 创建一个空字典用于存储注册的模块

    def register_module(self, cls: Type) -> Type:
        """
        通过装饰器将类注册到 _module_dict 中
        检查类名是否存在，避免重复注册
        """
        name: str = cls.__name__ # 获取类名        
        if name in self._module_dict: # 如果类名已经存在，抛出异常
            raise KeyError(f"{name} 已经在 {self._name} 中注册过了")
        
        self._module_dict[name] = cls # 将类添加到注册字典中
        
        # 返回类本身，这样装饰器就不会改变类的定义
        return cls

    def get(self, key: str) -> Optional[Type]:
        """根据名字获取注册的类"""
        
        # 如果key不存在，返回None
        return self._module_dict.get(key, None)

    def build(self, cfg: Dict[str, Any]) -> Any:
        """
        这个 build 其实就是实例化这个类的作用
        根据配置字典构建对象
        配置字典要求至少包含一个 'type' 字段，用于指定构建的类名，
        其他键值作为参数传递给类的构造函数。
        build 函数实际就是在返回一个类的实例
        """
        # 从配置中弹出 'type' 字段
        obj_type: str = cfg.pop("type")
        
        # 获取对应的类
        cls: Optional[Type] = self.get(obj_type)
        
        # 如果类不存在，抛出异常
        if cls is None:
            raise KeyError(f"{obj_type} 没有在 {self._name} 中注册")
        
        # 使用剩余的配置作为参数实例化类
        return cls(**cfg)

# 创建一个名为 MODELS 的注册器
MODELS = Registry("models")

# 通过装饰器将 ModelA 注册到 MODELS 中
@MODELS.register_module
class ModelA:
    def __init__(self, param1: float, param2: float):
        # 初始化 ModelA 的参数
        self.param1: float = param1
        self.param2: float = param2

    def forward(self, x: float) -> float:
        # ModelA 的前向传播方法
        return x * self.param1 + self.param2

# 同样，注册另一个示例类 ModelB
@MODELS.register_module
class ModelB:
    def __init__(self, factor: int):
        # 初始化 ModelB 的参数
        self.factor: int = factor

    def forward(self, x: float) -> float:
        
        # ModelB 的前向传播方法
        return x ** self.factor

# 演示如何使用注册器来构建对象
if __name__ == "__main__":
    
    # 通过配置字典构建 ModelA 的实例
    cfg_a: Dict[str, Any] = {"type": "ModelA", "param1": 2, "param2": 3}
    model_a: ModelA = MODELS.build(cfg_a)
    print("ModelA forward(5):", model_a.forward(5))  # 输出: 5 * 2 + 3 = 13

    # 通过配置字典构建 ModelB 的实例
    cfg_b: Dict[str, Any] = {"type": "ModelB", "factor": 3}
    model_b: ModelB = MODELS.build(cfg_b)
    print("ModelB forward(2):", model_b.forward(2))  # 输出: 2 ** 3 = 8
    