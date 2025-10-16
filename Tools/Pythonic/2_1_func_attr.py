# 1. 定义一个简单的核心函数（我们的“电钻”）
def say_hello(name):
    """打印一句问候语。"""
    print(f"Hello, {name}!")
    
    # 每次调用时，访问并增加它自己的 counter 属性
    say_hello.call_count += 1

# 2. 直接给函数对象添加“属性”，就像给工具箱贴标签
#    我们给它挂上一个数据属性：计数器
say_hello.call_count = 0
#    我们也可以给它挂上一个描述
say_hello.description = "一个带调用计数器的问候函数。"


# 3. 我们甚至可以定义另一个函数，然后把它“挂”到 say_hello 上，作为它的“方法”
def reset_counter():
    """一个独立的函数，用于重置计数器。"""
    print("--- 计数器已重置 ---")
    say_hello.call_count = 0

# 把 reset_counter 函数本身赋值给 say_hello 的一个新属性 .reset
say_hello.reset = reset_counter


# --- 现在来使用这个“功能增强”后的函数 ---

print("函数描述:", say_hello.description)
print("初始调用次数:", say_hello.call_count)
print("-" * 20)

# 调用几次核心功能
say_hello("Alice")
say_hello("Bob")
say_hello("Charlie")

print("-" * 20)
print("调用后的次数:", say_hello.call_count)

# 现在，调用我们“挂”上去的 reset 方法
say_hello.reset() 

print("重置后的次数:", say_hello.call_count)

# 再次调用核心功能，验证计数器是否从0开始
say_hello("David")
print("再次调用后的次数:", say_hello.call_count)