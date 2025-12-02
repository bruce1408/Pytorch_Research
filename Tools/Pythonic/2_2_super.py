class Father:
    def __init__(self, name):
        print("正在初始化父类...")
        self.name = name
        self.house = "大别墅"  # 父类的资产

class Son(Father):
    def __init__(self, name, hobby):
        # 1. 这里的 super().__init__(name) 就是调用父类的初始化方法
        # 把 name 传给父亲，让父亲把 self.name 和 self.house 设好
        super().__init__(name)  
        
        # 2. 然后再处理子类独有的逻辑
        print("正在初始化子类...")
        self.hobby = hobby

# 测试
son = Son("小明", "打游戏")

print(f"姓名: {son.name}")   # 继承自父类逻辑
print(f"资产: {son.house}")  # 继承自父类
print(f"爱好: {son.hobby}")  # 子类独有