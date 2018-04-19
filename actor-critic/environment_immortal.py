"""Providing the environment."""
import warnings

import random as rd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import Union, Callable, NamedTuple
from gym.utils import seeding

from tools import type_check, timestamp

hist = namedtuple('history', ('Predator', 'Prey'))  # history of agent memory


class Environment:
    """The environment class.

    It has the following attributes:
        - dim, a float or tuple describing the dimensions of the grid
        - densities, float or tuple desc. the agent densities on the grid
        - agent_types, Callable or tuple of Callables, contains agent functors
        - agent_kwargs, a dictionary to be passed down to the agents
        - max_pop, the maximal population on the grid = prod(dim) because only
            only one agent is allowed per cell
        - env, numpy array with shape=dim, contains the agent objects in their
            corresponding cell
        - agents_set, a set of all agents on the grid at the moment
        - agents_tuple, named tuple with one set for each agent_type
        - _np_random, a variable needed for seeding
        - history, a named tuple with one deque each agent_type to store all
            experiences an agent undergoes in its life

    Most of the attributes are property managed.
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['_dim', '_densities', '_agent_types', '_agent_kwargs',
                 '_max_pop', '_env', '_agents_set', '_agents_tuple',
                 '_np_random', '_history']  # _agent_named_properties

    # init --------------------------------------------------------------------
    def __init__(self, *, dim: tuple, agent_types: Union[Callable, tuple],
                 densities: Union[float, tuple], history: tuple=None,
                 **agent_kwargs: Union[int, float, None]):
        """Initialize the environment.

        more init-docstring to come.
        """
        # initialize attributes
        self._dim = None
        self._densities = None
        self._agent_kwargs = {}
        self._agent_types = None
        self._np_random = None
        self._history = None  # keeps every memory of every agent

        # set property managed attribute(s)
        self.dim = dim
        self.densities = densities
        self.agent_types = agent_types

        # set named tuple type
        if isinstance(self.agent_types, Callable):
            self.agent_types = [self.agent_types]  # ensure iterable

        # store agent_kwargs as attributes
        self.agent_kwargs = agent_kwargs

        # calculate maximum population size
        self._max_pop = np.prod(self.dim)

        # create named tuple
        agnts = namedtuple('agent_types', [a.__name__ for a in
                           self.agent_types])

        # initialise with empty sets
        self._agents_tuple = agnts(*[set() for _ in self.agent_types])
        self._agents_set = set()
        for s in self._agents_tuple:
            self._agents_set.update(s)

        # initialize history
        if history:
            self.history = history
        else:
            self.history = hist(deque(), deque())  # empty history

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

    @property
    def history(self) -> NamedTuple:
        """Return the list of recorded deeds."""
        return self._history

    @history.setter
    def history(self, history: tuple) -> None:
        """Define the past by setting history."""
        if not isinstance(history, tuple):
            raise TypeError("history must be of type list but {} was given."
                            "".format(type(history)))
        elif self.history is not None:
            raise RuntimeError("history has already started, there is no "
                               "forgiveness anymore.")
        else:
            self._history = history

    # staticmethods -----------------------------------------------------------

    # methods -----------------------------------------------------------------
    def seed(self, seed=None):
        """Set the seed for the random generator for the simulation."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, *args, **kwargs):
        """Dummy method, to be implemented in the derived classes."""
        raise NotImplementedError("Use a derived class that implemented this"
                                  "function")

    def reset(self, *args, **kwargs):
        """Dummy method, to be implemented in the derived classes."""
        raise NotImplementedError("Use a derived class that implemented this"
                                  "function")

    def render(self, *args, **kwargs):
        """Dummy method, to be implemented in the derived classes."""
        raise NotImplementedError("Use a derived class that implemented this"
                                  "function")


class GridPPM(Environment):
    """The Predator Prey Model Grid class.

    This class provides the functionality for PPMs in connection with neural networks and learning. Some basic functionality are methods like `move`, `eat` and `procreate`. The class also provides methods necessary for the learning process, like `reset`, `step` and (hopefully soon) `render`.

    It has the following attributes:
        - action_lookup, a dict which maps numbers between 0 and 26 to actions
            like moving, eating and procreating
        - shuffled_agent_list, a list of shuffled array indices where agents
            are placed on the grid at the creation time of the list
        - _nbh_lr, int, the neighbourhood lower range, needed for slicing the
            right neighbourhood for a given agent from env.
        - _nbh_ur, int, the neighbourhood upper range.
        - state, numpy array, containing the neighbourhood + food reserve of
            the currently active agent.
        - eaten_prey, a list, each prey that got eaten is put there to be
            handled in a special way.
        - _nbh_type, int, specifies the kind of neighbourhood, i.e. 9, 25, 49..
        - _nbh_range, int, used in conjunction with the other _nbh_* attributes

    class constants:
        - REWARDS, a dictionary that maps representations of actions to actual
            rewards.
        - KIN_LOOKUP, a dictionary that maps agent.__name__'s to int.

    Only nbh_type is property managed, all other _nbh_* attributes are set within the nbh_type property method.
    """

    REWARDS = {"wrong_action": -1,  # for every wrong action
               "default_prey": 2,  # for moving/eating
               "default_predator": 1,  # for not dying in that round
               "succesful_predator": 3,  # for eating
               "offspring": 5,  # for succesful procreation
               "death_starvation": -3,  # starvation
               "death_prey": -3,  # being eaten
               "indifferent": 0,
               "default": 1}  # for both prey and predator

    KIN_LOOKUP = {"Predator": -1, "Prey": 1}

    __slots__ = ['action_lookup', 'shuffled_agent_list', '_nbh_lr', '_nbh_ur',
                 'state', 'eaten_prey', '_nbh_type', '_nbh_range']

    # @type_check(argument_to_check="rewards", type_to_check=dict)
    def __init__(self, *, dim: tuple, agent_types: Union[Callable, tuple],
                 densities: Union[float, tuple], rewards: dict=None,
                 neighbourhood: int=9, **agent_kwargs: Union[int, float, None]):
        """Initialise the grid."""
        # call parent init function
        super().__init__(dim=dim, agent_types=agent_types, densities=densities,
                         **agent_kwargs)

        # initialise empty environment
        self._env = np.empty(self.max_pop, dtype=object)

        # initialize other variables
        self.shuffled_agent_list = None
        self.state = None
        self.eaten_prey = deque()

        # neighbourhood stuff
        self._nbh_type = None
        self._nbh_range = None
        self._nbh_lr = None  # lower range
        self._nbh_ur = None  # upper range

        # set neighbourhood variables (all in setter)
        self.nbh_type = neighbourhood

        # populate the grid + initial shuffled agent list
        self._populate()
        self.create_shuffled_agent_list()

        # update the rewards
        if rewards is not None:
            if isinstance(rewards, dict):
                for k, v in rewards.items():
                    if k not in self.REWARDS:
                        warnings.warn("Key {} was not in rewards dictionary."
                                      " Skipping update for this key..."
                                      "".format(k), RuntimeWarning)
                    else:
                        self.REWARDS[k] = v

            else:
                raise TypeError("rewards should always be of type dict, but"
                                " {} was given.".format(type(rewards)))

        # setup of the action ACTION_LOOKUP
        self.action_lookup = {0: self.move('LU'), 1: self.move('U'),
                              2: self.move('RU'), 3: self.move('L'),
                              4: self.move(''), 5: self.move('R'),
                              6: self.move('LD'), 7: self.move('D'),
                              8: self.move('RD'), 9: self.eat('LU'),
                              10: self.eat('U'), 11: self.eat('RU'),
                              12: self.eat('L'), 13: self.eat(''),
                              14: self.eat('R'), 15: self.eat('LD'),
                              16: self.eat('D'), 17: self.eat('RD'),
                              18: self.eat('LU'), 19: self.procreate('U'),
                              20: self.procreate('RU'),
                              21: self.procreate('L'), 22: self.procreate(''),
                              23: self.procreate('R'), 24: self.procreate('LD'),
                              25: self.procreate('D'),
                              26: self.procreate('RD')}

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

        else:
            self._env = env

    @property
    def nbh_type(self) -> int:
        """Return the neighbourhood type, i.e. 9, 25, ..."""
        return self._nbh_type

    @nbh_type.setter
    def nbh_type(self, nbh_type: int) -> None:
        """Set the nbh_type, as well as the values for nbh_range, nbh_lr and nbh_ur."""
        if self.nbh_type is not None:
            raise RuntimeError("neighbourhood type already set!")

        elif not isinstance(nbh_type, int):
            raise TypeError("neighbourhood type must be of type int, but {}"
                            " was given.".format(type(nbh_type)))

        elif not (np.sqrt(nbh_type) == int(np.sqrt(nbh_type))):
            raise RuntimeError("neighbourhood type must be a square of an odd"
                               " number.")

        else:
            self._nbh_type = nbh_type
            self._nbh_range = int(np.sqrt(nbh_type))
            self._nbh_ur = self._nbh_range - int(np.floor(self._nbh_range/2))
            self._nbh_lr = 1 - self._nbh_ur

    # staticmethods -----------------------------------------------------------
    @staticmethod
    def _target_to_value(target: str) -> np.ndarray:
        """Staticmethod that converts a target string to a value.

        Multiple passes of DULR as well as other characters are ignored.
        TODO: other stepsize?
        """
        # no typechecking - this has to happen earlier
        dirs = list(target)
        x = y = 0
        if not dirs:
            return np.array([y, x])
        else:  # first Y, then X coordinate
            if "D" in dirs:
                y -= 1
            if "U" in dirs:
                y += 1
            if "L" in dirs:
                x -= 1
            if "R" in dirs:
                x += 1

            return np.array([y, x])

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
                self.env[_] = a  # add the agent to the environment
                getattr(self._agents_tuple, name).add(a)
                self._agents_set.add(a)

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
    # @_index_test_ndarray
    def _die(self, index: tuple) -> None:
        """Delete the given agent from the environment and replace its position with None."""
        ag = self.env[index]
        if ag is not None:
            if ag.memory.Rewards:
                # record history in the right list
                getattr(self.history, ag.kin).append(ag.memory)

            self._agents_set.remove(ag)  # only deletes the set entry
            getattr(self._agents_tuple, ag.kin).remove(ag)  # same as above
            del ag
            self.env[index] = None

        else:
            warnings.warn("Trying to delete an empty cell", RuntimeWarning)

    # add agent to _agents_tuple
    def _add_to_agents_tuple(self, *, newborn: Callable) -> None:
        """Add the given agent in the corresponding subtuple dictionary.

        The added agent is then also available in GridPPM._agents_set.
        """
        getattr(self._agents_tuple, newborn.kin).add(newborn)
        self._agents_set.add(newborn)

    # add agent to Environment
    def add_to_env(self, *, target_index: tuple, newborn: Callable) -> None:
        """Add the given agent to the environment using target_index.

        The agent is added to the corresponding subtuple dictionary, and to the environment array.
        """
        self._add_to_agents_tuple(newborn=newborn)
        self.env[target_index] = newborn  # we assume that the index is not occupied

    # create shuffled list of agents
    def create_shuffled_agent_list(self) -> list:
        """Return a shuffled deque of (y,x) index arrays where the agents (at deque creation time) are."""
        # numpy is stupid and doesn't allow pep8 syntax like "env is not None"
        y, x = np.where(self.env != None)  # get indices
        agent_list = deque(i for i in zip(y, x))  # create deque
        np.random.shuffle(agent_list)

        self.shuffled_agent_list = agent_list

    # @type_check(argument_to_check="uuid", type_to_check=str)
    def _ag_to_int(self, *, ag: Callable) -> int:
        """Return a integer representation of the agent.

        Predator == -1
        Prey     ==  1
        ''       ==  0
        """
        if ag is not None:
            return self.KIN_LOOKUP[ag.kin]  # if agent, then set value

        else:
            return 0

    # a mapping from index to state
    # @type_check(argument_to_check="index", type_to_check=np.ndarray)
    def index_to_state(self, *, index: tuple, ag: Callable=None) -> tuple:
        """Return neighbourhood and food reserve for a given index.

        If agent was prey and got eaten, index points to '' in env. The return for food_reserve is then None.
        """
        # it can happen, that the index points to an empty space in the environment. This is due to the fact, that an index in the shuffled list doesn't get removed, if a prey got eaten. thus, empty cells are just ignored.
        state = []
        # check if agent has memory:
        if ag is not None:  # if ag is additionally set, directly get state
            if ag.memory.States:
                state = ag.memory.States[-1]
                return state

            else:
                neighbourhood = self.neighbourhood(index=index)
                state = [self._ag_to_int(ag=ag) for ag in neighbourhood]
                state.append(ag.food_reserve)  # got handed an agent
                return np.array(state)

        else:
            active_agent = self.env[index]
            if active_agent.memory.States:
                state = active_agent.memory.States[-1]  # remember the latest state
                return state

            else:
                neighbourhood = self.neighbourhood(index=index)
                state = [self._ag_to_int(ag=ag) for ag in neighbourhood]
                state.append(active_agent.food_reserve)

                return np.array(state)

    # neighbourhood
    def neighbourhood(self, index: tuple) -> np.array:
        """Return the neighbourhood specified in simulation config.

        For the edge cases slice doesn't work so that has to be circumvented
        with this ugly code. Sorry dear reader. :-/
        """
        y, x = index
        idx = np.array(index)  # needed for computation
        if(np.any((idx - self._nbh_range) < 0) or
           np.any((idx + self._nbh_range) >= self.dim)):  # check if edge case
            nbh = deque()
            # manually calculate slice indices
            for j in range(self._nbh_lr, self._nbh_ur):
                for i in range(self._nbh_lr, self._nbh_ur):
                    new_idx = tuple((idx + np.array([j, i])) % self.dim)
                    nbh.append(self.env[new_idx])  # append grid contents
            nbh = np.array(nbh).ravel()  # flatten array

        else:  # directly return sliced and flattened array
            nbh = self.env[slice(y + self._nbh_lr, y + self._nbh_ur),
                           slice(x + self._nbh_lr, x + self._nbh_ur)].ravel()
        return nbh

    '''
    # @_index_test_ndarray
    def neighbourhood2(self, index: tuple) -> tuple:
        """Return the 9 neighbourhood for a given index and the index values."""
        # "up" or "down" in the sense of up and down on screen
        delta = np.array([[-1, -1], [-1, 0], [-1, 1],  # UL, U, UR
                          [0, -1], [0, 0], [0, 1],  # L, _, R
                          [1, -1], [1, 0], [1, 1]])  # DL, D, DR

        neighbour_idc = (np.array(index) + delta) % self.dim  # ensure bounds
        neighbourhood = self.env[tuple(neighbour_idc.T)]  # numpy magic for correct indexing

        return neighbourhood, neighbour_idc
    '''

    # moving
    # @_argument_test_str
    def move(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed to move the agent."""
        def move_agent(index: tuple) -> None:
            """Move the given agent to previously specified target.

            targets can be (A is agent and corresponds to target ''):
                LU U  RU
                L  A  R
                LD D  RD
            From these input strings the target is calculated.

            If the desired location is already occupied, do nothing.
            """
            # check if move is possible
            delta = self._target_to_value(target)
            target_index = tuple((np.array(index) + delta) % self.dim)  # taking care of bounds

            if target == '':
                return self.REWARDS['indifferent']  # just moving

            elif self.env[target_index] is not None:
                return self.REWARDS['wrong_action']  # negative

            else:
                # FIXME: this won't work in the 1D case (if desired..)
                # moving
                self.env[target_index] = self.env[index]
                self.env[index] = None  # clearing the previous position

                return self.REWARDS['default']
        return move_agent

    # eating
    # @_argument_test_str
    def eat(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed and that agent tries to eat."""
        def eat_and_move(index: tuple) -> None:
            """Try to eat the prey in target with probability p_eat = 1 - p_flee as agent from index.

            targets are the same as for `move`.
            """
            # check if eating move is possible:
            # fetch agent
            agent = self.env[index]

            if type(agent) not in self.agent_types:
                raise RuntimeError("The current agent {} of kintype {} is not "
                                   "in the list agent_types. This should not "
                                   "have happened!".format(agent.uuid,
                                                           agent.kin))

            # now we have to check if target is eatable or if there is
            # space to move to
            delta = self._target_to_value(target)
            target_index = tuple((np.array(index) + delta) % self.dim)  # bounds again!
            target_agent = self.env[target_index]  # == agent_uuid if delta is [0,0]

            if agent.kin == "Predator":
                if target_agent is None:
                    # don't try to eat empty space
                    return self.REWARDS['wrong_action']  # negative

                # if agent targets not itself i.e. moves
                elif delta.any() and (target_agent.kin == "Predator"):
                    # don't eat your own kind
                    return self.REWARDS['wrong_action']  # negative

                elif not delta.any():
                    # don't eat yourself
                    return self.REWARDS['wrong_action']  # negative

                else:
                    roll = rd.random()
                    if roll <= agent.p_eat:
                        agent.food_reserve += 3  # FIXME: no hardcoding!
                        self.eaten_prey.append((target_index, target_agent))
                        target_agent.got_eaten = True  # set flag
                        self._die(target_index)  # remove the eaten prey
                        self.move(target)(index)
                        return self.REWARDS['succesful_predator']  # hooray!

                    else:
                        return self.REWARDS['default_predator']  # at least ...

            elif agent.kin == "Prey":
                # prey just eats
                if target_agent is None:
                    agent.food_reserve += 2  # FIXME: no hardcoding!
                    self.move(target)(index)
                    return self.REWARDS['default_prey']  # for eating and moving

                elif not delta.any():
                    agent.food_reserve += 2  # just standing around and eating
                    return self.REWARDS['default_prey']

                else:
                    return self.REWARDS['wrong_action']

            else:
                # for now:
                raise RuntimeError("encountered unknown species of type {} but"
                                   " either Prey or Predator was expected! This"
                                   " should not have happened!"
                                   "".format(agent.kin))

        return eat_and_move

    # procreating
    # @_argument_test_str
    def procreate(self, target: str) -> Callable:
        """Return a function to which an agent index can be passed and that agent tries to procreate with probability p_breed."""
        def procreate_and_move(index: tuple) -> None:
            """Try to have offspring in `target` with probability p_breed.

            `targets` are the same as in `move` and `eat`. If `target` is not
            empty, there should be a negative reward. Also for trying to
            procreate without having enough food_reserve is penalized.
            """
            # fetch agent
            agent = self.env[index]

            if type(agent) not in self.agent_types:

                raise RuntimeError("The current agent {} of kintype {} is not "
                                   "in the list agent_types. This should not "
                                   "have happened!".format(agent.uuid,
                                                           agent.kin))

            # now we have to check if target space is free or if it is occupied
            delta = self._target_to_value(target)
            target_index = tuple((np.array(index) + delta) % self.dim)  # bounds again!
            target_content = self.env[target_index]  # == agent_uuid if delta is [0,0]

            if agent.food_reserve >= 5:  # FIXME: no hardcoding!
                if target_content is not None:
                    # can't procreate without space
                    return self.REWARDS['wrong_action']

                elif target_content == agent:
                    # don't try to create offspring in your own cell
                    return self.REWARDS['wrong_action']

                else:
                    # try to breed
                    roll = rd.random()
                    if roll <= agent.p_breed:
                        # create new instance of <agent>
                        newborn = agent.procreate(food_reserve=3)  # FIXME hardcoded
                        self.add_to_env(target_index=target_index,
                                        newborn=newborn)
                        agent.food_reserve -= 3  # reproduction costs enery
                        return self.REWARDS['offspring']  # a new life...

                    else:
                        if agent.kin == "Prey":
                            return self.REWARDS['default_prey']
                        else:
                            return self.REWARDS['default_predator']
            else:
                # can't procreate without enough energy!
                # return self.REWARDS['wrong_action']
                return self.REWARDS['indifferent']  # testing...

        return procreate_and_move

    # methods for actor-critic ------------------------------------------------
    def reset(self) -> tuple:
        """Reset the environment and return the state and the object of the first popped element of the shuffled agents list."""
        # clear the sets
        self._agents_tuple.Predator.clear()
        self._agents_tuple.Prey.clear()
        self._agents_set.clear()

        # empty Environment
        self._env = np.empty(self.max_pop, dtype=object)

        # populate the grid and agent dicts
        self._populate()

        # create new shuffled agents list
        self.create_shuffled_agent_list()

        # clear eaten prey list
        self.eaten_prey.clear()

        # clear history
        self.history.Predator.clear()
        self.history.Prey.clear()

        # pop list and return state
        index = self.shuffled_agent_list.pop()
        self.state = self.index_to_state(index=index)

        return self.state, index

    def step(self, *, model: Callable, agent: Callable, index: tuple, action: int, returnidx: tuple=None) -> tuple:
        """The method starts from the current state, takes an action and records the return of it."""
        reward = 0  # initialize reward
        # agent.food_reserve -= 1  # reduce food_reserve

        if hasattr(agent, "got_eaten"):
            if agent.got_eaten:
                reward = self.REWARDS['death_prey']

        # if (agent.food_reserve <= 0) and (reward == 0):  # if agent not dead already
            # self._die(index=index)
            # reward = self.REWARDS['death_starvation']  # more death!

        if reward == 0:
            act = self.action_lookup[action]  # select action from lookup
            reward = act(index=index)  # get reward for acting
            # for debugging: checking whether action rewards are valid
            if reward is None:
                raise RuntimeError("reward should not be of type None! The"
                                   " last action was {} by agent {}"
                                   "".format(act, agent))

        # save the reward
        agent.memory.Rewards.append(reward)

        if (len(self._agents_tuple.Predator) and len(self._agents_tuple.Prey)) is 0:
            done = True  # at least one species died out

            # since the episode is now finished, append the rest of the agents'
            # memories to the environments history
            for ag in self._agents_set:
                if ag.memory.Rewards:  # if agent actually has memory
                    getattr(self.history, ag.kin).append(ag.memory)

        else:
            done = False  # no harm in being explicit

        if returnidx is not None:  # keep the old index in the system
            self.state = self.index_to_state(index=returnidx)
            return reward, self.state, done, returnidx

        else:
            # new index, if cell is empty due to eaten prey, repop
            newindex = self.shuffled_agent_list.pop()
            while((self.env[newindex] == "")):
                if len(self.shuffled_agent_list) != 0:
                    newindex = self.shuffled_agent_list.pop()
                else:
                    break

            if self.env[newindex] is not None:
                self.state = self.index_to_state(index=newindex)  # new state
            return reward, self.state, done, newindex

    def render(self, *, episode: int, step: int, figsize: tuple, filepath: str,
               dpi: int, fmt: str, **kwargs):
        """The method visualizes the simulation timesteps."""
        plotarr = np.zeros(shape=np.shape(self.env))
        y, x = np.where(self.env != None)

        for idc in zip(y, x):
            plotarr[idc] = self._ag_to_int(ag=self.env[idc])

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        im = ax.imshow(ma.masked_equal(plotarr, 0), cmap='viridis', vmin=-1,
                       vmax=1)
        cbar = fig.colorbar(mappable=im, ax=ax, fraction=0.047, pad=0.01,
                            ticks=[-1, 1])
        cbar.ax.set_yticklabels(['Predator', 'Prey'])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        info = " Prey: {}, Pred: {}".format(len(self._agents_tuple.Prey),
                                            len(self._agents_tuple.Predator))

        ax.set_title("Episode: {}, Step: {} |".format(episode, step) + info)

        filename = "{}_{:0>3}_{:0>3}.png".format(timestamp(), episode, step)
        fig.savefig(filepath + filename, dpi=dpi, format=fmt)
        plt.close(fig)
