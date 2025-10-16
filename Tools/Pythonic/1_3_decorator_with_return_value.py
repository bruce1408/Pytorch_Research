def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # 接收原函数的返回值
        result = func(*args, **kwargs) 
        
        end_time = time.time()
        print(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        
        # 将返回值返回给调用者
        return result
    return wrapper

@timer_decorator
def calculate_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

sum_result = calculate_sum(1000000)
print(f"计算结果: {sum_result}")