import time
from print_utils import print_colored_box

def time_it(func):
    def wrapper(*argc, **kwargs):
        start_time = time.time()
        result = func(*argc, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)  # 保留3位小数
        print_colored_box(f"{func.__name__} taken: {elapsed_time} 秒")
        return result
    return wrapper


if __name__ == "__main__":
    @time_it
    def example_function():
        # 模拟一个耗时的操作
        time.sleep(2)
        print("Example function executed")
        
    # 调用示例函数
    example_function()
