"""Docstring for the tools."""
import sys
import datetime as dt
from typing import Callable, Optional
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


# function to be used as decorator
def type_check(*, argument_to_check: str, type_to_check: type):
    """To be used as decorator to check if a function argument is a certain type."""
    def wrap(func):
        def wrapped_func(*args, **kwargs):  # we want to only use kwargs, hence the '*'
            # check if argument is used
            if argument_to_check in kwargs.keys():
                if not isinstance(kwargs[argument_to_check], type_to_check):
                    raise TypeError("Argument {} is not of type {}."
                                    "".format(argument_to_check, type_to_check))

            else:  # if argumet is not in kwargs
                raise KeyError("{} was not found in function kwargs.".format(argument_to_check))

            # if everything went fine until now, we can call the function
            return func(*args, **kwargs)

        return wrapped_func
    return wrap


def keyboard_interrupt_handler(*, save: Optional[Callable],
                               abort: Optional[Callable]):
    """To be used as decorator to handle keyboard interrupts while running a simulation.

    The given function objects for save and abort will be called depending on the choice of aborting the simulation directly or saving the contents first.
    """
    def wrap(func):
        def wrapped_func(*args, **kwargs):
            try:
                func(*args, **kwargs)  # e.g. the simulation main function
            except KeyboardInterrupt:
                store = None
                # ignoring everything else than y,n and emptystring
                while((store != "y") and (store != "n")):
                    store = input("\n: Stopping.. store simulation data? (y/N): ")
                    store = store.lower()  # not case sensitive
                    if store == "":
                        store = "n"

                if store == "n":
                    print(": Aborting simulation...")
                    if abort is not None:
                        abort()
                    sys.exit()

                else:
                    print(": Storing simulation data...")
                    if save is not None:
                        save()
                    sys.exit()

        return wrapped_func
    return wrap


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
