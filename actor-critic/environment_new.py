"""Providing the environment."""

import numpy as np
from collections import namedtuple
from typing import Union, Callable
from gym.spaces import Discrete  # for the discrete action space of the agents

from tools import DeepChainMap


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
        self._agents_dict = DeepChainMap(*self._agents_tuple)

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
    def _target_to_value(target: str) -> np.ndarray:
        """Staticmethod that converts a target string to a value.

        Multiple passes of DULR as well as other characters are ignored.
        TODO: other stepsize?
        """
        # no typechecking - this has to happen earlier
        dirs = list(target)
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

    # the code below doesn't work as decorator atm because I wanted to apply it to a function inside a function

    # just works for 'regular' methods and not for functions inside functions
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

    # dying
    @_index_test_ndarray
    def _die(self, index: np.ndarray) -> None:
        """Delete the given index from the environment and replace its position with empty string ''."""
        uuid = self.env[tuple(index)]
        if uuid != '':
            del self._agents_dict[uuid]
            self.env[tuple(index)] = ''
        else:
            pass  # TODO: maybe warnings.warn?

    # add agent to _agents_tuple
    def _add_to_agents_tuple(self, *, newborn: Callable) -> None:
        """Add the given agent in the corresponding subtuple dictionary.

        The added agent is then also available in GridPPM._agents_dict.
        """
        getattr(self._agents_tuple, newborn.kin)[newborn.uuid] = newborn

    # add agent to Environment
    def add_to_env(self, *, target_index: np.ndarray, newborn: Callable) -> None:
        """Add the given agent to the environment using target_index.

        The agent is added to the corresponding subtuple dictionary, and to the environment array.
        """
        self._add_to_agents_tuple(newborn=newborn)
        self.env[tuple(target_index)] = newborn.uuid  # we assume that the index is not occupied

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
    def move(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed to move the agent."""
        def move_agent(index: np.ndarray) -> None:
            """Move the given agent to previously specified target.

            targets can be (A is agent and corresponds to target ''):
                LU U  RU
                L  A  R
                LD D  RD
            From these input strings the target is calculated.

            If the desired location is already occupied, do nothing.
            TODO: add a negative reward for trying to access an already occupied
                space?
            """
            if not isinstance(index, np.ndarray):
                raise TypeError("Index must be of type {} but {} was given."
                                "".format(np.ndarray, type(index)))

            # check if move is possible
            delta = self._target_to_value(target)
            new_pos = (index + delta) % self.dim  # taking care of bounds
            if self.env[tuple(new_pos)] != '':
                # to be excepted later
                # raise RuntimeError("target is already occupied.")
                pass  # TODO: penalty!

            else:
                # FIXME: this won't work in the 1D case (if desired..)
                # moving
                self.env[tuple(new_pos)] = self.env[tuple(index)]
                self.env[tuple(index)] = ''  # clearing the previous position

        return move_agent

    # eating
    @_argument_test_str
    def eat(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed and that agent tries to eat."""
        def eat_and_move(index: np.ndarray) -> None:
            """Try to eat the prey in target with probability p_flee as agent from index.

            targets are the same as for `move`.
            TODO: maybe change p_flee to p_eat, such that it's easier to call.
            """
            if not isinstance(index, np.ndarray):
                raise TypeError("Index must be of type {} but {} was given."
                                "".format(np.ndarray, type(index)))

            # check if eating move is possible:
            # fetch agent
            agent_uuid = self.env[tuple(index)]
            agent = self._agents_dict[agent_uuid]

            # initialize target
            target_agent = None

            if type(agent) not in self.agent_types:
                raise RuntimeError("The current agent {} of kintype {} is not "
                                   "in the list agent_types. This should not "
                                   "have happened!".format(agent_uuid,
                                                           agent.kin))

            # now we have to check if target is eatable or if there is
            # space to move to
            delta = self._target_to_value(target)
            target_index = (index + delta) % self.dim  # bounds again!
            target_uuid = self.env[tuple(target_index)]  # == agent_uuid if delta is [0,0]

            if target_uuid != '':
                target_agent = self._agents_dict[target_uuid]  # get the target

            if agent.kin == "Predator":
                if not target_agent:
                    pass  # TODO: penalty! don't try to eat empty space

                # if agent targets not itself i.e. moves
                elif delta.any() and (target_agent.kin == "Predator"):
                    print("do not eat your own kind!")
                    pass  # TODO: more penalty! one does not simply eat his own kind!

                elif not delta.any():
                    print("don't try to eat yourself.")
                    pass  # TODO: penalty! don't try to eat yourself.

                else:
                    roll = np.random.rand()
                    if roll <= agent.p_eat:
                        print("p_roll = {}".format(roll))
                        agent.food_reserve += 3  # FIXME: no hardcoding!
                        self._die(target_index)  # remove the eaten prey
                        self.move(target)(index)

            elif agent.kin == "Prey":
                # prey just eats
                if target_uuid == '':
                    agent.food_reserve += 2  # FIXME: no hardcoding!
                    self.move(target)(index)

                elif not delta.any():
                    print("just standing around and eating.")
                    self.food_reserve += 2  # just standing around and eating

                else:
                    pass  # TODO: penalty! a Prey can't eat other agents

            else:
                # TODO: implement more agent types?
                pass

        return eat_and_move

    # procreating
    @_argument_test_str
    def procreate(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed and that agent tries to procreate with probability p_breed."""
        def procreate_and_move(index: np.ndarray) -> None:
            """Try to have offspring in `target` with probability p_breed.

            `targets` are the same as in `move` and `eat`. If `target` is not
            empty, there should be a penalty (TODO). Also for trying to
            procreate without having enough food_reserve is penalized (TODO).
            """
            if not isinstance(index, np.ndarray):
                raise TypeError("Index must be of type {} but {} was given."
                                "".format(np.ndarray, type(index)))

            # fetch agent
            agent_uuid = self.env[tuple(index)]
            agent = self._agents_dict[agent_uuid]

            if type(agent) not in self.agent_types:
                raise RuntimeError("The current agent {} of kintype {} is not "
                                   "in the list agent_types. This should not "
                                   "have happened!".format(agent_uuid,
                                                           agent.kin))

            # now we have to check if target space is free or if it is occupied
            delta = self._target_to_value(target)
            target_index = (index + delta) % self.dim  # bounds again!
            target_uuid = self.env[tuple(target_index)]  # == agent_uuid if delta is [0,0]

            if agent.food_reserve >= 5:  # FIXME: no hardcoding!
                if target_uuid != '':
                    pass  # TODO: penalty!

                else:
                    # try to breed
                    roll = np.random.rand()
                    if roll <= agent.p_breed:
                        # create new instance of <agent>
                        newborn = agent.procreate(food_reserve=3)  # FIXME hardcoded
                        self.add_to_env(target_index=target_index,
                                        newborn=newborn)
                        agent.food_reserve -= 3  # reproduction costs enery
            else:
                pass  # TODO: penalty!

        return procreate_and_move
