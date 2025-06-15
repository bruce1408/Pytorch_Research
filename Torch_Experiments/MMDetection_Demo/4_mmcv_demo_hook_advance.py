from typing import Callable, List, Dict, Any
from enum import Enum
import time

# 定义事件类型枚举
class EventType(Enum):
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    BEFORE_EPOCH = "before_epoch"
    AFTER_EPOCH = "after_epoch"
    BEFORE_ITER = "before_iter"
    AFTER_ITER = "after_iter"

# Hook 管理器类
class HookManager:
    def __init__(self):
        # 存储所有注册的钩子函数
        self._hooks: Dict[EventType, List[Callable]] = { event: [] for event in EventType }
    
    def register_hook(self, event: EventType, hook_func: Callable) -> None:
        """
        注册钩子函数
        Args:
            event: 事件类型
            hook_func: 钩子函数
        """
        if event not in self._hooks:
            raise ValueError(f"未知的事件类型: {event}")
        self._hooks[event].append(hook_func)
    
    def trigger_hooks(self, event: EventType, **kwargs) -> None:
        """
        触发指定事件的所有钩子函数
        Args:
            event: 事件类型
            **kwargs: 传递给钩子函数的参数
        """
        for hook in self._hooks[event]:
            hook(**kwargs)

# 示例训练器类
class Trainer:
    def __init__(self):
        self.hook_manager = HookManager()
        self.epoch = 0
        self.iter = 0
    
    def register_hook(self, event: EventType, hook_func: Callable) -> None:
        """注册钩子函数"""
        self.hook_manager.register_hook(event, hook_func)
    
    def train(self, num_epochs: int) -> None:
        """训练过程"""
        # 触发训练开始前的钩子
        self.hook_manager.trigger_hooks(EventType.BEFORE_TRAIN)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 触发每个epoch开始前的钩子
            self.hook_manager.trigger_hooks(
                EventType.BEFORE_EPOCH, 
                epoch=epoch
            )
            
            # 模拟每个epoch的训练过程
            for iter in range(10):
                self.iter = iter
                # 触发每次迭代开始前的钩子
                self.hook_manager.trigger_hooks(
                    EventType.BEFORE_ITER,
                    epoch=epoch,
                    iter=iter
                )
                
                # 模拟训练步骤
                time.sleep(0.1)
                
                # 触发每次迭代结束后的钩子
                self.hook_manager.trigger_hooks(
                    EventType.AFTER_ITER,
                    epoch=epoch,
                    iter=iter
                )
            
            # 触发每个epoch结束后的钩子
            self.hook_manager.trigger_hooks(
                EventType.AFTER_EPOCH,
                epoch=epoch
            )
        
        # 触发训练结束后的钩子
        self.hook_manager.trigger_hooks(EventType.AFTER_TRAIN)

# 示例钩子函数
def log_training_progress(**kwargs) -> None:
    """记录训练进度的钩子函数"""
    if 'epoch' in kwargs and 'iter' in kwargs:
        print(f"Epoch {kwargs['epoch']}, Iteration {kwargs['iter']}")
    elif 'epoch' in kwargs:
        print(f"Epoch {kwargs['epoch']} completed")

def save_checkpoint(**kwargs) -> None:
    """保存检查点的钩子函数"""
    if 'epoch' in kwargs and kwargs['epoch'] % 2 == 0:
        print(f"Saving checkpoint at epoch {kwargs['epoch']}")

def time_training(**kwargs) -> None:
    """计时钩子函数"""
    if 'epoch' in kwargs and 'iter' in kwargs:
        if kwargs['iter'] == 0:
            print(f"Starting epoch {kwargs['epoch']} at {time.strftime('%H:%M:%S')}")

# 使用示例
if __name__ == "__main__":
    # 创建训练器实例
    trainer = Trainer()
    
    # 注册钩子函数
    trainer.register_hook(EventType.BEFORE_ITER, log_training_progress)
    trainer.register_hook(EventType.AFTER_EPOCH, save_checkpoint)
    trainer.register_hook(EventType.BEFORE_EPOCH, time_training)
    
    # 开始训练
    print("开始训练...")
    trainer.train(num_epochs=3)
    print("训练完成！")