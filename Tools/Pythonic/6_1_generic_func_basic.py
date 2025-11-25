# 泛型函数就是这样一种函数：它拥有一个统一的函数名，但内部会根据传入参数的类型（或值），自动选择并调用一个专门为该类型定制的具体实现版本。
def describe_traditional(data):
    """用传统方式描述数据。"""
    if isinstance(data, int):
        return f"这是一个整数，值为 {data}。"
    elif isinstance(data, str):
        return f"这是一个字符串，长度为 {len(data)}。"
    elif isinstance(data, list):
        return f"这是一个列表，包含 {len(data)} 个元素。"
    else:
        return "未知类型的数据。"

print(describe_traditional(100))
print(describe_traditional("hello world"))
print(describe_traditional([1, 2, 3]))

