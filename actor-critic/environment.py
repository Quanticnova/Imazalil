"""Providing the environment."""

import numpy as np
from collections import ChainMap, namedtuple
from typing import Union, Dict, NamedTuple

from agents import Agent, Predator, Prey


class Environment:
    """The environment class providing necessary functionaliy.

    Hurr durr more docstring foo.
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['_dim', '_densities', '_food_reserve', '_max_food_reserve',
                 '_p_breed', '_p_flee', '_max_pop', '_env', '_agents_dict']

    # init --------------------------------------------------------------------
    def __init__(self, *, dim: tuple, **agent_kwargs: Union[int, float, None]):
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

        # set the attributes that match the slot values
        # values not specified will carry the default values from above
        # TODO: handle not slottet kwargs
        # TODO: remove properties from environment. only handle attributes in
        # agent classes. pass arguments via dict.
        for k, v in agent_kwargs.items():
            if "_" + k in self.__slots__:
                setattr(self, k, v)

        # calculate attributes
        self._max_pop = np.prod(self.dim)  # prod of dimensions = max agents

    # properties --------------------------------------------------------------
    # dimension
    @property
    def dim(self) -> Union[float, tuple]:
        """Return the dimension attribute."""
        return self._dim

    @dim.setter
    def dim(self, dim: Union[float, tuple]) -> None:
        """Set the dimensions."""
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
            self._dim = dim

    # densities
    @property
    def densities(self) -> Union[float, namedtuple]:
        """Return the densities for the agents."""
        return self._densities

    @densities.setter
    def densities(self, densities) -> None:
        """Set the densities."""
        if not (isinstance(densities, float) or isinstance(densities, tuple)):
            raise TypeError("densities must be of type float or (named) tuple"
                            ", but {} was given".format(type(densities)))

        elif self.densities:
            raise RuntimeError("densities already set.")

        elif np.sum(densities) > 1:
            raise ValueError("densities must sum up to <= 1, but the urrent sum"
                             " is {}.".format(np.sum(densities)))

        else:
                self._densities = densities

    # food reserve
    @property
    def food_reserve(self) -> int:
        """The food reserve of the agent."""
        return self._food_reserve

    @food_reserve.setter
    def food_reserve(self, food_reserve: int) -> None:
        """The food reserve setter."""
        if not isinstance(food_reserve, int):
            raise TypeError("food_reserve can only be of type integer, but"
                            " type {} was given".format(type(food_reserve)))

        elif food_reserve < 0:
            raise ValueError("food_reserve must be positive, but {} was given."
                             "".format(food_reserve))

        else:
            self._food_reserve = food_reserve

    # max_food_reserve
    @property
    def max_food_reserve(self) -> int:
        """The maximal food reserve of the agent."""
        return self._max_food_reserve

    @max_food_reserve.setter
    def max_food_reserve(self, max_food_reserve: int) -> None:
        """The maximal food reserve setter."""
        if not isinstance(max_food_reserve, int):
            raise TypeError("max_food_reserve can only be of type integer, "
                            "but type {} was given"
                            "".format(type(max_food_reserve)))

        elif max_food_reserve < self.food_reserve:
            raise ValueError("max_food_reserve must be greater or equal than"
                             " food_reserve={}, but {} was given."
                             "".format(self.food_reserve, max_food_reserve))

        elif self.max_food_reserve:
            raise RuntimeError("max_food_reserve is already set.")

        else:
            self._max_food_reserve = max_food_reserve

    # breeding probability
    @property
    def p_breed(self) -> float:
        """The breeding probability of the Agent."""
        return self._p_breed

    @p_breed.setter
    def p_breed(self, p_breed: float) -> None:
        """The breeding probability setter."""
        if not isinstance(p_breed, float):
            raise TypeError("p_breed must be of type float, but {} was given."
                            "".format(type(p_breed)))

        elif p_breed < 0 or p_breed > 1:
            raise ValueError("p_breed must be between 0 and 1 but {} was"
                             " given.".format(p_breed))

        else:
            self._p_breed = p_breed

    # staticmethods -----------------------------------------------------------

    # methods -----------------------------------------------------------------
