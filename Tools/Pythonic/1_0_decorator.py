import time
from functools import wraps # 别忘了导入这个！

# 第一层：装饰器函数 (包装厂)
# - 它的工作是接收“产品”(你的函数)，并定义如何“包装”。
def timer_decorator(func): 

    # 第二层：包装函数 (包装盒)
    # - 这是真正执行新功能和调用原函数的地方。
    # - 使用 @wraps(func) 来保留原函数的元信息（名字、文档等）。
    @wraps(func) 
    def wrapper(*args, **kwargs): # 使用 *args, **kwargs 让它能包装任何函数
        
        # --- 附加功能 (包装前) ---
        start_time = time.time()
        print(f"函数 [{func.__name__}] 开始执行...")
        
        # --- 调用核心产品 ---
        result = func(*args, **kwargs) # 调用原始函数，并传入所有参数
        
        # --- 附加功能 (包装后) ---
        end_time = time.time()
        print(f"函数 [{func.__name__}] 执行完毕，耗时 {end_time - start_time:.4f} 秒。")
        
        return result # 把原函数的结果返回出去

    # 第三层：返回“包装好的产品”
    return wrapper


@timer_decorator
def do_something(name):
    """一个处理任务的函数。"""
    time.sleep(1)
    return f"任务完成，处理人：{name}"

# 调用时，实际上是在调用 wrapper 函数
result = do_something("小明") 
print(f"收到的结果是: {result}")
print(f"函数名: {do_something.__name__}") # Благодаря @wraps，这里显示 do_something 而不是 wrapper