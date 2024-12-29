'''Various functions to use throughout the folder'''
from typing import Callable, TypeVar, Any
import time


T = TypeVar('T')


def timing(func: Callable[..., T], use_logger: bool = False
           ) -> Callable[..., T]:
    '''Decorator for timing a function. Run the function, recording
    execution time. Return the original return value, print the
    execution time'''
    def wrapper(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        ret_val = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        print(f'{func.__name__} took {elapsed_time}s')
        return ret_val
    return wrapper
