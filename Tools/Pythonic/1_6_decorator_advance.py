def dispatch_functool(func):
    registry = {}
    
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
