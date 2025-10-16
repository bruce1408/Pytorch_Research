# 写一个统计函数执行时间的装饰器
import time
from functools import wraps
def time_it(func):
    @wraps(func) # 保留函数的元信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"函数 {func.__name__} 执行时间: {end - start} 秒")
        return result
    return wrapper

@time_it
def my_function():
    time.sleep(2)
    print("函数执行完毕")
    
my_function()
print(my_function.__name__) # 输出 wrapper，而不是 my_function


def dispatch_functool(func):
    registry = {}
    
    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func
    
    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)
        
        registry[value] = func
        return func
    
    def wrapper(*args, **kwargs):
        return dispatch(args[0])(*(args[1:]), **kwargs)
    
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry
    
    

    