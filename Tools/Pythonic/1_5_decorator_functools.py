import time
from functools import wraps

# 使用装饰器有一个小副作用：它会丢失原函数的一些元信息，
# 比如函数名 (__name__)、文档字符串 (__doc__) 等，因为它们被 wrapper 函数的信息替换了。

def better_timer_decorator(func):
    @wraps(func) # 核心就是这一行
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

@better_timer_decorator
def my_better_function():
    """这是一个示例文档字符串。"""
    pass

print(my_better_function.__name__) # 输出 my_better_function
print(my_better_function.__doc__)  # 输出 这是一个示例文档字符串。