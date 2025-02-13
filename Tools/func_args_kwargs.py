def example(arg1, *args, kwarg1=None, **kwargs):
    print(f"arg1: {arg1}")
    print(f"args: {args}")
    print(f"kwarg1: {kwarg1}")
    print(f"kwargs: {kwargs}")

example(1, 2, 3, 4, kwarg1=5, extra='extra')
