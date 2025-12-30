# 1. *args：处理任意数量的位置参数 (Positional Arguments)
# 2. **kwargs：处理任意数量的关键字参数 (Keyword Arguments)

def demo_func(arg1, *args, kwarg1=None, **kwargs):
    print(f"arg1 = {arg1}")
    print(f"args = {args}")
    print(f"kwarg1 = {kwarg1}")
    print(f"kwargs = {kwargs}")


# 定义一个函数，使用 *args 来接收不确定数量的参数
def add_all(*numbers):
    # 在函数内部，numbers 是一个元组，包含了所有传入的参数
    print(f"接收到的参数元组是: {numbers}")
    
    # 初始化总和为 0
    total = 0
    
    # 遍历元组中的每一个数字
    for num in numbers:
        # 把它加到总和上
        total += num
        
    # 返回最终的总和
    return total


# 定义一个函数，使用 **kwargs 来接收不确定数量的关键字参数
def print_user_info(**user_data):
    # 在函数内部，user_data 是一个字典
    print(f"接收到的参数字典是: {user_data}")
    
    # 遍历字典的键和值
    for key, value in user_data.items():
        # 打印每个信息
        print(f"{key}: {value}")

def test_args():
    # 让我们来调用这个函数试试
    print(f"两个数的和: {add_all(1, 2)}")
    print("-" * 20)
    print(f"五个数的和: {add_all(10, 20, 30, 40, 50)}")
    print("-" * 20)
    print(f"没有参数时: {add_all()}")
    

def test_kwargs():
    print("用户 Alice 的信息:")
    print_user_info(name="Alice", age=30, status="active")
    print("-" * 20)

    print("用户 Bob 的信息:")
    print_user_info(name="Bob", city="New York", occupation="Developer")


def test_func():
    demo_func(1, 2, 3, 4, kwarg1=5, extra='extra')
    demo_func(1, name="Alice", age=30, status="active")


# test_args()
# test_kwargs()
test_func()