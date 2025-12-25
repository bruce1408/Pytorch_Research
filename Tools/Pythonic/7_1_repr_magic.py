# 它的主要目标是返回一个对象的“官方”的、无歧义的字符串表示形式。
# 这个字符串的设计初衷是给开发者看的，主要用于调试和记录日志。

def bad_example():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # 创建一个 Point 类的实例（我们称之为“对象”）
    p = Point(10, 20)

    # 打印这个对象
    print(p) 
    # 这个输出非常不友好！它只告诉我们这是 Point 类的一个对象，以及它在内存中的地址。
    # 但我们完全看不出这个点的重要信息，比如它的坐标 x 和 y 是多少。在调试时，这样的信息几乎是无用的。

def good_example():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        # 这就是魔法发生的地方！
        def __repr__(self):
            """
            返回一个能准确描述如何创建这个对象的字符串。
            """
            # 我们使用 f-string 来格式化字符串，非常方便
            return f"Point(x={self.x}, y={self.y})"
        
    # 再次创建和打印对象
    p = Point(10, 20)
    print(p)


def str_repr_example():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __repr__(self):
            # 给开发者的“官方”表示
            return f"Point(x={self.x}, y={self.y})"

        def __str__(self):
            # 给用户的“友好”表示
            return f"一个位于 ({self.x}, {self.y}) 的点"

    p = Point(3, 4)

    # print() 会优先使用 __str__
    print(p)

    # str() 也会优先使用 __str__
    print(str(p))

    # repr() 会强制使用 __repr__
    print(repr(p))

    # 在Python交互式环境中直接输入变量名，会使用 __repr__
    # >>> p
    # Point(x=3, y=4)

    # 打印包含对象的列表，也会对里面的元素使用 __repr__
    points_list = [p]
    print(points_list)


def torch_repr():
    import torch

    print("--- 默认行为 ---")
    t1 = torch.ones(2, 3, 2, 2)
    # 如果直接打印 t1, 会刷屏！
    # 我们把它放到列表里，列表会调用 repr，效果一样
    print([t1]) 
    print("\n" + "="*30 + "\n")


    # === 执行猴子补丁 ===
    torch.Tensor.__repr__ = lambda self: f"<Tensor shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}>"
    # ====================


    print("--- 修改后的行为 ---")
    t2 = torch.ones(2, 3, 64, 64)
    print([t2])


if __name__ == "__main__":
    # <__main__.Point object at 0x7f8c1d3e3a90>
    # bad_example()
    
    # Point(x=10, y=20)
    # good_example()
    
    # str_repr_example()
    
    torch_repr()