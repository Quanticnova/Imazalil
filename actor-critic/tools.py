"""Docstring for the tools."""
import datetime as dt
from typing import Callable
from collections import ChainMap


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


# for deleting elements not just in the first layer
class DeepChainMap(ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes."""

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)
