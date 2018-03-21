"""Docstring for the tools."""
import datetime as dt
from typing import Callable


def timestamp(return_obj: bool=False):
    """Create and return a timestamp.

    Args:
        return_obj (bool): if True, return the datetime object instead of the
            string. Useful for datetime differences.
    """
    if return_obj:
        return dt.datetime.now()

    else:
        return str(dt.datetime.now())


def function_call_counter(func: Callable) -> Callable:
    """Function wrapper to count the calls a decorated function has."""
    def helper(*args, **kwargs):
        """Helper function to actually count the calls."""
        helper.calls += 1
        return func(*args, **kwargs)

    helper.calls = 0
    return helper
