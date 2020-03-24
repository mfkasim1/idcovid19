from functools import wraps

def memoize(method):
    """Wrap a function such that its result is computed only once.
    See https://en.wikipedia.org/wiki/Memoization
    """
    @wraps(method)
    def wrapper(obj):
        attrname = '_' + method.__name__
        result = getattr(obj, attrname, None)
        if result is None:
            result = method(obj)
        setattr(obj, attrname, result)
        return result
    return wrapper
