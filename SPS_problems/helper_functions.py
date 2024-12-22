'''Various functions to use throughout the folder'''
from typing import Callable, TypeVar, Any
import time


T = TypeVar('T')


def timing(func: Callable[..., T], use_logger: bool = False
           ) -> Callable[..., tuple[T, float]]:
    '''Decorator for timing a function. Return both the original
    function return value and the execution time'''

    def wrapper(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        ret_val = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        return ret_val, elapsed_time
    return wrapper
