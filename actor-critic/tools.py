"""Docstring for the tools."""
import datetime as dt


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
