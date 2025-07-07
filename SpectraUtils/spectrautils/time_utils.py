import time
from .print_utils import print_colored_text

def time_it(func):
    def wrapper(*argc, **kwargs):
        start_time = time.time()
        result = func(*argc, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)  # 保留3位小数
        print_colored_text(f"{func.__name__} taken: {elapsed_time} 秒")
        return result
    return wrapper

