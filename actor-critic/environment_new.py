"""Providing the environment."""

import numpy as np
from collections import ChainMap, namedtuple
from typing import Union, Callable

from agents import Agent, Predator, Prey


class Environment:
    """The environment class.

    more docstring to come!
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['_dim', '_densities', '_agent_types', '_agent_kwargs',
                 '_max_pop', '_env', '_agents_dict', '_agent_named_properties']

    # init --------------------------------------------------------------------
    def __init__(self, *, dim: tuple, agent_types: Union[Callable, tuple],
                 densities: Union[float, tuple],
                 **agent_kwargs: Union[int, float, None]):
        """Initialize the environment.

        more init-docstring to come.
        """
        # initialize attributes
        self._dim = None
        self._densities = None
        self._agent_kwargs = {}
        self._agent_types = None
        self._agents_dict = None  # to be set later

        # set property managed attribute(s)
        self.dim = dim
        self.densities = densities
        self.agent_types = agent_types

        # set named tuple type
        if isinstance(self.agent_types, Callable):
            self.agent_types = [self.agent_types]  # ensure iterable

        # get the names of agent types and create named tuple template
        fn  = [at.__name__ for at in self.agent_types]
        self._agent_named_properties = namedtuple('property', fn)
        for _ in fn:
            
        # store agent_kwargs as attributes
        self.agent_kwargs = agent_kwargs

        # calculate maximum population size
        self._max_pop = np.prod(self.dim)

    # properties -------------------------------------------------------------
    # dimensions
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

    # agent_types
    @property
    def agent_types(self) -> Union[Callable, tuple]:
        """Return the callable agent type(s)."""
        return self._agent_types

    @agent_types.setter
    def agent_types(self, agent_types) -> None:
        """Set the agent types for this environment."""
        # check whether agent_types are Callable or tuple of Callable
        if not (isinstance(agent_types, Callable) or
                isinstance(agent_types, tuple) and
                np.all([isinstance(at, Callable) for at in agent_types])):
            raise TypeError("agent_types must be of type Callable or "
                            "tuple of Callables, but {} was given."
                            "".format(type(agent_types)))

        elif self.agent_types:
            raise RuntimeError("agent_types already set.")

        else:
            self._agent_types = agent_types

    # agent_kwargs
    @property
    def agent_kwargs(self) -> dict:
        """Return the agent_kwargs dictionary."""
        return self._agent_kwargs

    @agent_kwargs.setter
    def agent_kwargs(self, agent_kwargs) -> None:
        """Set the agent kwargs for this run."""
        if agent_kwargs:
            if not isinstance(agent_kwargs, dict):
                raise TypeError("agent_kwargs must be of type dict but {} was"
                                " given.".format(type(agent_kwargs)))

            else:
                self._agent_kwargs = agent_kwargs

    # max_pop
    @property
    def max_pop(self) -> int:
        """Return the maximum population of the grid."""
        return self._max_pop

    # staticmethods


class Grid(Environment):
    """humpdy dumpdy docstring."""

    def __init__(self, *, dim: tuple, agent_types: Union[Callable, tuple],
                 densities: Union[float, tuple],
                 **agent_kwargs: Union[int, float, None]):
        """Initialise the grid."""
        # call parent init function
        super().__init__(dim=dim, agent_types=agent_types, densities=densities,
                         **agent_kwargs)

        # initialise empty environment
        self._env = np.empty(self.max_pop, dtype='U33')  # FIXME: no hardcoding

        # populate the grid
        self._populate()

    # methods
    def _populate(self) -> None:
        """Populate the Environment with given agents & values."""
        # multiply fractions with maximum number of population
        num_agents = np.array([self.densities]) * self.max_pop
        num_agents = np.array(num_agents, dtype=int).ravel()  # ensure values

        # consistency check
        if len(self.agent_types) != len(num_agents):
            raise RuntimeError("Mismatch of Dimensions - densities and"
                               " agent_types must have same length, but"
                               " len(densities) = {} and len(agent_types) = {}"
                               " were given.".format(len(self.densities),
                                                     len(self.agent_types)))

        idx = np.arange(self.max_pop)  # generate indices
        np.random.shuffle(idx)  # shuffle the indices

        # loop over the agent_types, and create as many agents as specified in
        # num_agents. The second for loop manages the right intervals of the
        # shuffled indices.
        for i, (num, at) in enumerate(zip(num_agents, self.agent_types)):
            for _ in idx[sum(num_agents[:i]) : sum(num_agents[:i+1])]:
                a = at(**self.agent_kwargs)  # create new agent isinstance
                self.env[_] = a.uuid  # add the uuid to the environment

        self.env = self.env.reshape(self.dim)

    # env
    @property
    def env(self) -> np.ndarray:
        """Return the grid as numpy array with uuids."""
        return self._env

    @env.setter
    def env(self, env: np.ndarray) -> None:
        """Set the environment."""
        if type(self._env) is not type(env):
            raise TypeError("Type mismatch - env must be of type {} but {} was"
                            " given.".format(type(self._env), type(env)))

        # if any value in env is not '' then env is already initialized
        # NOTE: this is currently not working :-)
        # elif np.any(np.logical_not(np.isin(self._env, ''))):
        #     raise RuntimeError("env already initialized.")

        else:
            self._env = env
