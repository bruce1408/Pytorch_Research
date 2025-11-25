import os
from functools import wraps

def dispatch_functool(func):
 
    registry = {}
    
    @wraps(func)
    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func
    
    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func
    
    def wrapper(*args, **kw):
        return dispatch(args[0])(*args[1:], **kw)
    

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry
    
    return wrapper


#  --- 使用我们创建的工具 ---

# 1. 创建一个默认函数
@dispatch_functool
def make_sound(animal_name, volume):
    # 这是默认情况，如果 registry 里找不到 animal_name 就会执行这里
    print(f"The {animal_name} makes a sound at volume {volume}.")

# 2. 为 'dog' 注册一个特殊版本
@make_sound.register('dog')
def make_dog_sound(volume):
    print(f"The dog barks: Woof! Woof! at volume {volume}.")

# 3. 为 'cat' 注册一个特殊版本
@make_sound.register('cat')
def make_cat_sound(volume):
    print(f"The cat meows: Meow! at volume {volume}.")

# --- 现在来调用它！---
make_sound('dog', 10)
# 输出: The dog barks: Woof! Woof! at volume 10.

make_sound('cat', 5)
# 输出: The cat meows: Meow! at volume 5.

# make_sound('bird', 7)
# 输出: The bird makes a sound at volume 7.

print("\n我们可以看看它的注册表:")
print(make_sound.registry)
