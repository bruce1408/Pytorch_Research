# 这就需要再加一层函数嵌套。结构是：一个接收装饰器参数的函数，它返回一个真正的装饰器。
# 这是一个装饰器工厂函数
def repeat(num_times):
    
    # 这才是真正的装饰器
    def decorator_repeat(func):
        # 这是最终的包装函数
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_repeat

# 使用带参数的装饰器
@repeat(num_times=3)
def cheer():
    print("Go Go Go!")

cheer()