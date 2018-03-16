"""Providing the environment."""

import numpy as np
from collections import ChainMap, namedtuple
from typing import Union

class Environment:
    """The environment class providing necessary functionaliy.

    Hurr durr more docstring foo.
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['_dim', '_max_pop', '_env', '_agents_dict']

    # init --------------------------------------------------------------------
    def __init__(self, *, dim: tuple,
                 densities: Union[float, namedtuple]=0.2,
                 food_reserve: Union[int, namedtuple]=3,
                 max_food_reserve: Union[int, namedtuple]=None,
                 p_breed: Union[float, namedtuple]=0.4,
                 p_flee: Union[float, namedtuple]=0.5):
        """Initialize the environment.

        Given parameters supply the necessary parameters for the agent init.
        """
        # initialize attributes
        self._densities = 0.0
        self._food_reserve = 3
        self._max_food_reserve = None
        self._p_breed = 1.0
        self._p_flee = 0.0

        # set propertie managed attributes
        self.dim = dim
        self.food_reserve = food_reserve
        self.p_breed = p_breed
        self.p_flee = p_flee

        if max_food_reserve:
            self.max_food_reserve = max_food_reserve

        # calculate attributes
        self._max_pop = np.prod(self.dim)  # prod of dimensions = max agents

# properties ------------------------------------------------------------------
    @property
    def dim(self) -> Union[float, tuple]:
        """Return the dimension attribute."""
        return self._dim

    @dim.setter
    def dim(self, dim: Union[float, tuple]) -> None:
        if not isinstance(dim, tuple):
            raise TypeError("dim has to be of type tuple, (X,Y), but {} was"
                            " given.".format(type(dim)))

        elif np.any([not isinstance(x, int) for x in dim]):
            raise ValueError("dim entries must be of type int but one or more"
                             " entries in {} are not.".format(dim))
        elif len(dim) > 2:
            raise NotImplementedError("dim > 2 is currently unavailable. Sorry"
                                      " for dimension lock.")

        else:
            self._densities = dim
