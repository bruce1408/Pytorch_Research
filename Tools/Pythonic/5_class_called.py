class MyCallableClass:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        print(f"Calling the instance with value: {self.value}")
        print(f"Arguments: {args}")
        print(f"Keyword Arguments: {kwargs}")
        return self.value * 2

# 创建类的实例
my_instance = MyCallableClass(10)

# 直接像调用函数一样调用类实例
result = my_instance(5, name="test")

print("Result:", result)
