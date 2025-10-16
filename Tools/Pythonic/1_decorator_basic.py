import logging

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 操作分发字典
operation_registry = {}

# 基本操作函数
def operation_dispatcher(value, *args, **kwargs):
    """根据操作类型分发到不同的函数"""
    # 查找操作
    try:
        # 从注册表中调用操作
        return operation_registry[value](*args, **kwargs)
    except KeyError:
        # 如果没有找到操作，返回默认行为
        logger.info("Operation Not Found!")
        return None

# 注册操作函数
def register_operation(value, func):
    """将操作与函数绑定"""
    operation_registry[value] = func

# 定义操作函数
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

# 注册操作
register_operation('add', add)
register_operation('subtract', subtract)
register_operation('multiply', multiply)

# 测试操作
result_add = operation_dispatcher('add', 5, 3)
result_subtract = operation_dispatcher('subtract', 5, 3)
result_multiply = operation_dispatcher('multiply', 5, 3)
result_not_found = operation_dispatcher('divide', 5, 3)

# 输出结果
print(f"Addition Result: {result_add}")
print(f"Subtraction Result: {result_subtract}")
print(f"Multiplication Result: {result_multiply}")
print(f"Operation Not Found Result: {result_not_found}")
