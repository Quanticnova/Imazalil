"""Providing the environment."""

import numpy as np
from collections import ChainMap, namedtuple
from typing import Union, Callable
from gym.spaces import Discrete  # for the discrete action space of the agents


class Environment:
    """The environment class.

    more docstring to come!
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['_dim', '_densities', '_agent_types', '_agent_kwargs',
                 '_max_pop', '_env', '_agents_dict', '_agent_named_properties',
                 '_agents_tuple']

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

        # store agent_kwargs as attributes
        self.agent_kwargs = agent_kwargs

        # calculate maximum population size
        self._max_pop = np.prod(self.dim)

        # create named tuple
        agnts = namedtuple('agent_types', [a.__name__ for a in
                           self.agent_types])
        # initialise with empty dicts
        self._agents_tuple = agnts(*[{} for _ in self.agent_types])
        self._agents_dict = ChainMap(*self._agents_tuple)

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

    # staticmethods -----------------------------------------------------------

    # methods -----------------------------------------------------------------


class GridPPM(Environment):
    """humpdy dumpdy docstring.

    TODO: think about the action lookup. the thing below looks awful!   ACTION_LOOKUP = {0: Environment.move(0),
                     1: Environment.move(1),
                     2: Environment.move(2),
                     3: Environment.move(3),
                     4: Environment.move(4),
                     5: Environment.move(5),
                     6: Environment.move(6),
                     7: Environment.move(7)}

    New idea: lookup key is: LU, U, RU ... left up, up, right up, ...
    """

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

    # properties --------------------------------------------------------------
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

    # staticmethods -----------------------------------------------------------
    @staticmethod
    def _direction_to_value(direction: str) -> np.ndarray:
        """Staticmethod that converts a direction string to a value.

        Multiple passes of DULR as well as other characters are ignored.
        TODO: other stepsize?
        """
        # no typechecking - this has to happen earlier
        dirs = list(direction)
        delta = np.array([0, 0])
        if not dirs:
            return delta
        else:  # NOTE: first Y, then X coordinate
            if "D" in dirs:
                delta += np.array([-1, 0])
            if "U" in dirs:
                delta += np.array([1, 0])
            if "L" in dirs:
                delta += np.array([0, -1])
            if "R" in dirs:
                delta += np.array([0, 1])

            return delta

    # methods -----------------------------------------------------------------
    # populate
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
                name = at.__name__
                a = at(**self.agent_kwargs)  # create new agent isinstance
                self.env[_] = a.uuid  # add the uuid to the environment
                getattr(self._agents_tuple, name)[a.uuid] = a

        self.env = self.env.reshape(self.dim)

    # argument testing
    def _argument_test_str(func: Callable) -> Callable:
        """Function wrapper to check whether argument of decorated function is valid str."""
        def helper(self, s: str):
            """Helper function to actually check if argument is string."""
            if isinstance(s, str):
                return func(self, s)

            else:
                raise TypeError("Argument must be of type {}, but {} was "
                                "given.".format(str, type(s)))

        return helper
    # the code below doesn't work as decorator atm...
    '''
    # index test ndarray
    def _index_test_ndarray(func: Callable) -> Callable:
        """Function wrapper to check whether argument of decorated function is valid np.ndarray."""
        def helper(self, index: np.ndarray):
            """Helper function to actually check if index is ndarray."""
            if isinstance(index, np.ndarray):
                return func(self, index)
            else:
                raise TypeError("Index must be of type {} but {} was given."
                                "".format(np.ndarray, type(index)))

        return helper
    '''
    # neighbourhood
    def neighbourhood(self, index: np.ndarray) -> np.ndarray:
        """Return the 9 neighbourhood for a given index and the index values."""
        if not isinstance(index, np.ndarray):
            raise TypeError("Index must be of type {} but {} was given."
                            "".format(np.ndarray, type(index)))

        # "up" or "down" in the sense of up and down on screen
        delta = np.array([[-1, -1], [-1, 0], [-1, 1],  # UL, U, UR
                          [0, -1], [0, 0], [0, 1],  # L, _, R
                          [1, -1], [1, 0], [1, 1]])  # DL, D, DR

        neighbour_idc = (index + delta) % self.dim  # ensure bounds
        neighbourhood = self.env[tuple(neighbour_idc.T)]  # numpy magic for correct indexing

        return neighbourhood, neighbour_idc

    # moving
    @_argument_test_str
    def move(self, direction: str) -> Callable:
        """Return a function to which an agent index can be passed to move the agent."""
        def move_agent(index: np.ndarray) -> None:
            """Move the given agent to previously specified direction.

            Directions can be (A is agent and corresponds to direction ''):
                LU U  RU
                L  A  R
                LD D  RD
            From these input strings the direction is calculated.

            If the desired location is already occupied, do nothing.
            TODO: add a negative reward for trying to access an already occupied
                space?
            """
            if not isinstance(index, np.ndarray):
                raise TypeError("Index must be of type {} but {} was given."
                                "".format(np.ndarray, type(index)))

            # check if move is possible
            delta = self._direction_to_value(direction)
            new_pos = (index + delta) % self.dim  # taking care of bounds
            if self.env[tuple(new_pos)] != '':
                # to be excepted later
                raise RuntimeError("Direction is already occupied.")

            else:
                # FIXME: this won't work in the 1D case
                # moving
                self.env[tuple(new_pos)] = self.env[tuple(index)]
                self.env[tuple(index)] = ''  # clearing the previous position

        return move_agent

    # eating
    @_argument_test_str
    def eat(self, direction: str) -> Callable:
        """Return a function to which an agent index can be passed and that agent tries to eat."""
        def eat_and_move(index: np.ndarray) -> None:
            pass
