import logging

# 定义装饰器函数
def dispatch_functool(func):
    registry = {}

    # 根据参数value来返回注册的函数
    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func

    # 用于将操作名称与对应函数注册到registry中
    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    # 负责调用dispatch并执行相应的函数
    def wrapper(*args, **kw):
        return dispatch(args[0])(*(args[1:]), **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


# 基本操作函数
@dispatch_functool
def operation_dispatcher(*args, **kwargs):
    logger.info("Operation Not Found!")

# 注册加法（add）操作
@operation_dispatcher.register('add')
def add(a, b):
    return a + b

# 注册减法（subtract）操作
@operation_dispatcher.register('subtract')
def subtract(a, b):
    return a - b

# 注册乘法（multiply）操作
@operation_dispatcher.register('multiply')
def multiply(a, b):
    return a * b

# 注册除法（divide）操作
@operation_dispatcher.register('divide')
def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # 测试加法
    logging.info("Testing 'add' operation")
    result_add = operation_dispatcher('add', 5, 3)
    logging.info(f"Addition result: {result_add}")

    # 测试减法
    logging.info("Testing 'subtract' operation")
    result_subtract = operation_dispatcher('subtract', 5, 3)
    logging.info(f"Subtraction result: {result_subtract}")

    # 测试乘法
    logging.info("Testing 'multiply' operation")
    result_multiply = operation_dispatcher('multiply', 5, 3)
    logging.info(f"Multiplication result: {result_multiply}")

    # 测试除法
    logging.info("Testing 'divide' operation")
    result_divide = operation_dispatcher('divide', 5, 3)
    
