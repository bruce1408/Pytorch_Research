# 1. 这是我们的 “礼物” (原函数)
def say_hello():
    print("Hello, world!")

# 2. 这是我们的 “包装工” (装饰器函数)
def logger_decorator(func):
    # 定义一个内部函数(wrapper)，这就是“包装纸+蝴蝶结”
    def wrapper():
        print("准备执行函数...") # 功能增强：在原函数前执行
        func()  # 调用原函数
        print("函数执行完毕。") # 功能增强：在原函数后执行
    
    # 装饰器返回这个包装好的新函数
    return wrapper

# 3. 手动进行“包装”
# 将 say_hello 函数和它的增强功能绑定起来
say_hello = logger_decorator(say_hello)

# 4. 现在调用 say_hello，它已经是被装饰过的版本了
say_hello()


# 使用 @ 语法糖，代码更简洁、更Pythonic
@logger_decorator
def say_goodbye():
    print("Goodbye, world!")

say_goodbye()