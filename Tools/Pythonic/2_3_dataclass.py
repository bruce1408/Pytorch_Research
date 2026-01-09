from dataclasses import dataclass, field

@dataclass
class Config_dataclass:
    num_cams: int
    img_h: int
    img_w: int
    
    
class Config_simple:
    def __init__(self, num_cams, img_h, img_w):
        self.num_cams = num_cams
        self.img_h = img_h
        self.img_w = img_w
    
    # 没有dataclass，要自己写
    def __repr__(self):
        # 手写一个“打印友好”的字符串
        return (f"Config(num_cams={self.num_cams}, "
                f"img_h={self.img_h}, img_w={self.img_w}")
        

@dataclass
class BadConfig:
    
    # ================= error demo ====================
    # scales: list = []  # ❌ 坑：所有实例会共享这一个列表
    # 每次实例化都会调用 lambda 生成一个新的列表
    # a = BadConfig()
    # b = BadConfig()

    # a.scales.append(4)
    # print(b.scales)   # 也会看到 [4]，因为它们是同一个 list
    # ================= error demo ====================
    pass


@dataclass
class GoodConfig:
    # 每次实例化都会调用 lambda 生成一个新的列表
    scales: list = field(default_factory=lambda: [4, 8, 16])
    feat_dims: list = field(default_factory=lambda: [64, 128, 256])


def test_basic_func():
    # Python 会自动帮你生成 __init__
    # 对比刚才的传统写法：
    # 不再需要手写 __init__，构造函数自动生成；
    # 不再需要手写 __repr__，打印自动是 Config(...) 的格式；
    # == 比较自动按字段来比，而不是按对象内存地址。
    # field(default_factory=...) 算是默认写法
    cfg_datacla = Config_dataclass(num_cams=6, img_h=256, img_w=704)
    cfg_simple = Config_simple(num_cams=6, img_h=256, img_w=704)
    
    # 1) default：简单类型可以直接给默认值
    lr: float = 1e-4

    # 2) default_factory：可变类型用它
    scales: list = field(default_factory=lambda: [4, 8, 16])

    # 3) init=False：这个字段不在 __init__ 参数里出现
    created_at: str = field(default="now", init=False)

    # 4) repr=False：打印 __repr__ 时不展示这个字段
    secret: str = field(default="xxx", repr=False)

    # 5) compare=False：做 == 比较时忽略这个字段
    cache_key: str = field(default="", compare=False)

    print(cfg_datacla)
    print(cfg_simple)
    
    
def test_share_list():
    c1 = GoodConfig()
    c2 = GoodConfig()

    c1.scales.append(32)
    print(c1.scales)  # [4, 8, 16, 32]
    print(c2.scales)  # [4, 8, 16]  —— 不受影响，实例互相独立
    


if __name__ == "__main__":
    
    # demo 1
    # test_basic_func()
    
    # demo 2
    test_share_list()