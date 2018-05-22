"""Providing the environment."""
import warnings

import random as rd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import Union, Callable, NamedTuple
from gym.utils import seeding

from tools import type_check, timestamp, function_call_counter

hist = namedtuple('history', ('Predator', 'Prey'))  # history of agent memory


def init(*, goal: str="training", policy_kind: str="conv"):
    """Initialize some global variables to set the environment to act in a specific behaviour.

    Current global variables:
        - goal: set the goal of the simulation; either training or testing
        - conv: define the NN input; True if 'conv', else if 'fc'
    """
    global training
    global conv

    training = True if goal == "training" else False  # quick and dirty
    conv = True if policy_kind == "conv" else False


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
               "default": 1,  # for both prey and predator
               "instadeath": 0}  # for statistical predator death

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
        self.action_lookup = {  # 0: self.move('LU'),
                              0: self.move('U'),
                              # 2: self.move('RU'),
                              1: self.move('L'),
                              2: self.move(''),
                              3: self.move('R'),
                              # 6: self.move('LD'),
                              4: self.move('D'),
                              # 8: self.move('RD'),
                              # 9: self.eat('LU'),
                              5: self.eat('U'),
                              # 11: self.eat('RU'),
                              6: self.eat('L'),
                              7: self.eat(''),
                              8: self.eat('R'),
                              # 15: self.eat('LD'),
                              9: self.eat('D'),
                              # 17: self.eat('RD'),
                              # 18: self.eat('LU'),
                              10: self.procreate('U'),
                              # 20: self.procreate('RU'),
                              11: self.procreate('L'),
                              12: self.procreate(''),
                              13: self.procreate('R'),
                              # 24: self.procreate('LD'),
                              14: self.procreate('D'),
                              # 26: self.procreate('RD')
                              }

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
            for _ in idx[sum(num_agents[:i]): sum(num_agents[:i+1])]:
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
            if ag.memory.Rewards and training:
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
        # Pylama is stupid and doesn't allow pep8 syntax like "env is not None"
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
        if ag is not None:  # if ag is additionally set, directly get state
            # this condition is fulfilled, if an agent gets eaten, because then
            # the agent option is explicitely set
            if ag.memory.States:  # check if agent has memory
                state = ag.memory.States[-1]
                return state

            else:
                neighbourhood = self.neighbourhood(index=index)
                if conv:
                    shape = neighbourhood.shape
                    neighbourhood = neighbourhood.ravel()

                state = [self._ag_to_int(ag=ag) for ag in neighbourhood]

                if conv:
                    state = np.array(state).reshape(shape)
                    state = [state, np.array([ag.food_reserve])]  # got handed an agent
                    return state
                else:
                    state.append(ag.food_reserve)
                    return np.array(state)

        else:
            active_agent = self.env[index]
            neighbourhood = self.neighbourhood(index=index)
            if conv:
                shape = neighbourhood.shape
                neighbourhood = neighbourhood.ravel()

            state = [self._ag_to_int(ag=ag) for ag in neighbourhood]

            if conv:
                state = np.array(state).reshape(shape)
                state = [state, np.array([active_agent.food_reserve])]  # got handed an agent
                return state
            else:
                state.append(active_agent.food_reserve)
                return np.array(state)

            # state = [self._ag_to_int(ag=ag) for ag in neighbourhood]
            # state.append(active_agent.food_reserve)
            # return np.array(state)

            """
            active_agent = self.env[index]
            if active_agent.memory.States:
                state = active_agent.memory.States[-1]  # remember the latest state
                return state

            else:
                neighbourhood = self.neighbourhood(index=index)
                state = [self._ag_to_int(ag=ag) for ag in neighbourhood]
                state.append(active_agent.food_reserve)

                return np.array(state)
            """

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
            nbh = np.array(nbh)

        else:  # directly return sliced and flattened array
            nbh = self.env[slice(y + self._nbh_lr, y + self._nbh_ur),
                           slice(x + self._nbh_lr, x + self._nbh_ur)]

        if conv:
            return nbh.reshape(self._nbh_range, self._nbh_range)  # needed for conv input layer

        else:
            return nbh.ravel()  # flatten array

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
        @function_call_counter
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
        @function_call_counter
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
        @function_call_counter
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
                return self.REWARDS['wrong_action']
                # return self.REWARDS['indifferent']  # testing...

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
        instadeath = False
        # reduce food_reserve
        agent.food_reserve -= 1 if self.agent_kwargs['mortality'] else 0

        # check whether this is the final action or not
        final_action = False if len(self.shuffled_agent_list) != 0 else True

        # if agent got eaten, set the reward
        if hasattr(agent, "got_eaten"):
            if agent.got_eaten:
                reward = self.REWARDS['death_prey']

        # if mortality is set, then check if agent starved
        if self.agent_kwargs['mortality']:
            if (agent.food_reserve <= 0) and (reward == 0):  # if agent not dead already
                self._die(index=index)
                reward = self.REWARDS['death_starvation']  # more death!

        # statistical death
        if (agent.kin == "Predator") and (len(self._agents_tuple.Predator)
                                          > 1):
            death_roll = rd.random()
            if death_roll <= self.agent_kwargs['instadeath']:
                self._die(index=index)
                reward = self.REWARDS['instadeath']  # should be zero
                instadeath = True

        if (reward == 0) and not instadeath:
            act = self.action_lookup[action]  # select action from lookup
            reward = act(index=index)  # get reward for acting
            # for debugging: checking whether action rewards are valid
            if reward is None:
                raise RuntimeError("reward should not be of type None! The"
                                   " last action was {} by agent {}"
                                   "".format(act, agent))

        # save the reward
        if training:
            agent.memory.Rewards.append(reward)

        # check if one species died out and if so, save history of living agents
        if (len(self._agents_tuple.Predator) and len(self._agents_tuple.Prey)) is 0:
            done = True  # at least one species died out

            # since the episode is now finished, append the rest of the agents'
            # memories to the environments history
            if training:
                for ag in self._agents_set:
                    if ag.memory.Rewards:  # if agent actually has memory
                        getattr(self.history, ag.kin).append(ag.memory)

        else:
            done = False  # no harm in being explicit

        if returnidx is not None:  # keep the old index in the system
            self.state = self.index_to_state(index=returnidx)
            return reward, self.state, done, returnidx

        elif not final_action:
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

        else:
            # since the last action already happened, we can just return some
            # data mambo jumbo since step always returns 4 values
            return 0, 0, done, 0

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


# -------------------------------------------------------------------------
class GridOrientedPPM(Environment):
    """Again a docstring.

    The config file for this kind of environment should look a little bit different. The size of the viewing grid should be specified directly.
    """

    REWARDS = {"wrong_action": -3,
               "default_prey": 1,
               "default_predator": 1,
               "succesful_predator": 5,
               "offspring": 20,
               "death_starvation": -5,
               "death_prey": -10,
               "indifferent": 0,
               "default": 1,
               "instadeath": 0}

    KIN_LOOKUP = {"Predator": -1, "Prey": 1, "OrientedPredator": -1,
                  "OrientedPrey": 1}

    # rotation matrices for the possible directions
    # since our indices are (Y, X) the coordinate system is left handed.
    TURNS = {'left': np.array([[0, 1], [-1, 0]]),
             'right': np.array([[0, -1], [1, 0]]),
             'around': np.array([[-1, 0], [0, -1]])}

    # slots -------------------------------------------------------------------
    __slots__ = ['action_lookup', 'shuffled_agent_list', 'state',
                 'eaten_prey', '_view', '_bounds']

    # init --------------------------------------------------------------------
    def __init__(self, *, dim: tuple, agent_types: Union[Callable, tuple],
                 densities: Union[float, tuple], rewards: dict=None,
                 view: tuple=(7, 7), **agent_kwargs: Union[int, float, None]):
        """Initialise the grid."""
        # call parent init function
        super().__init__(dim=dim, agent_types=agent_types, densities=densities,
                         **agent_kwargs)

        # initialize empty environment
        self._env = np.empty(self.max_pop, dtype=object)

        # initialize other variables
        self.shuffled_agent_list = None
        self.state = None
        self._view = None
        self._bounds = ()  # is filled once by setting view
        self.eaten_prey = deque()

        # populate the grid + initial shuffled agent list
        self._populate()
        self.create_shuffled_agent_list()

        # view
        self.view = view

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
        self.action_lookup = {  # TODO: Do me!
                              }

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

    # agent view
    @property
    def view(self) -> tuple:
        """Return the agents' view of the grid as (X, Y) tuple."""
        return self._view

    @view.setter
    def view(self, view: tuple) -> None:
        """Set the agents' view of the grid."""
        if not isinstance(view, tuple):
            raise TypeError("view must be of type tuple but type {} was given"
                            "".format(type(view)))

        else:
            self._view = view

            # set the _bounds list
            view = np.array(view)
            lower_bound = -np.floor(view/2)  # mind the minus
            upper_bound = np.ceil(view/2)
            bounds = np.stack([lower_bound, upper_bound])  # stack of bounds
            center = np.array(self.dim)//2  # more or less the center
            # this only is problematic, if the grid is smaller than the viewing range
            centered_bounds = (center + bounds).T.astype(int)  # if array is rolled, these are the bounds to use for slicing
            self._bounds = (bounds, center, centered_bounds)

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
            for _ in idx[sum(num_agents[:i]): sum(num_agents[:i+1])]:
                name = at.__name__
                a = at(**self.agent_kwargs)  # create new agent isinstance
                self.env[_] = a  # add the agent to the environment
                getattr(self._agents_tuple, name).add(a)
                self._agents_set.add(a)

        self.env = self.env.reshape(self.dim)

    # dying
    def _die(self, index: tuple) -> None:
        """Delete the given agent from the environment and replace its position with None."""
        ag = self.env[index]
        if ag is not None:
            if ag.memory.Rewards and training:
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
        # Pylama is stupid and doesn't allow pep8 syntax like "env is not None"
        y, x = np.where(self.env != None)  # get indices
        agent_list = deque(i for i in zip(y, x))  # create deque
        np.random.shuffle(agent_list)

        self.shuffled_agent_list = agent_list

    # convert agent to integer
    def _ag_to_int(self, *, ag: Callable) -> int:
        """Return a integer representation of the agent.

        Predator         == -1
        Prey             ==  1
        ''               ==  0
        OrientedPredator == -1
        OrientedPrey     ==  1
        """
        if ag is not None:
            return self.KIN_LOOKUP[ag.kin]  # if agent, then set value

        else:
            return 0

    # a mapping from index to state
    def index_to_state(self, *, index: tuple, ag: Callable=None) -> tuple:
        """Return neighbourhood and food reserve for a given index.

        If agent was prey and got eaten, index points to '' in env. The return for food_reserve is then None.
        """
        # it can happen, that the index points to an empty space in the environment. This is due to the fact, that an index in the shuffled list doesn't get removed, if a prey got eaten. thus, empty cells are just ignored.
        state = []
        if ag is not None:  # if ag is additionally set, directly get state
            # this condition is fulfilled, if an agent gets eaten, because then
            # the agent option is explicitely set
            if ag.memory.States:  # check if agent has memory
                state = ag.memory.States[-1]  # directly return state
                return state

            else:  # if agent has no memory, create state
                neighbourhood = self.neighbourhood(index=index)
                if conv:  # if convolutional input layer, take care of shape
                    shape = neighbourhood.shape
                    neighbourhood = neighbourhood.ravel()

                state = [self._ag_to_int(ag=ag) for ag in neighbourhood]

                if conv:  # if conv input layer, type of container for states is important!
                    state = np.array(state).reshape(shape)
                    state = [state, np.array([ag.food_reserve])]  # got handed an agent
                    return state
                else:
                    state.append(ag.food_reserve)
                    return np.array(state)

        else:
            active_agent = self.env[index]
            neighbourhood = self.neighbourhood(index=index)
            if conv:
                shape = neighbourhood.shape
                neighbourhood = neighbourhood.ravel()

            state = [self._ag_to_int(ag=ag) for ag in neighbourhood]

            if conv:
                state = np.array(state).reshape(shape)
                state = [state, np.array([active_agent.food_reserve])]  # active agent
                return state
            else:
                state.append(active_agent.food_reserve)
                return np.array(state)

    # neighbourhood
    def neighbourhood(self, index: tuple) -> np.array:
        """Return the neighbourhood specified in simulation config.

        Edge cases need to be handled separately.
        """
        # get const values
        bounds, center, centered_bounds = self._bounds

        # calculate the bounds shaped accordingly to [[y_low, y_up], [x_low, x_up]]
        shaped_bounds = np.array(2*index).reshape(bounds.shape) + bounds
        shaped_bounds = shaped_bounds.T.astype(int)  # ensure type int

        if np.any(shaped_bounds < 0) or np.any(shaped_bounds > self.dim):
            # edge case
            ybounds, xbounds = centered_bounds
            diff = center - np.array(index)  # difference vector to center

            # recenter array at idc and slice
            nbh = np.roll(self.env, diff, axis=(0, 1))[slice(*ybounds),
                                                       slice(*xbounds)]
        else:
            ybounds, xbounds = shaped_bounds

            nbh = self.env[slice(*ybounds), slice(*xbounds)]  # just slice

        if conv:
            return nbh

        else:  # if no conv input layer is set, the nbh needs to be flattened
            return nbh.ravel()

    # rotate agent
    def turn(self, where: str) -> Callable:
        """Return a functional that turns the agents direction."""
        def turn_agent(index: tuple) -> None:
            """Turn the agent at index in the given direction.

            Possible directions are: 'left', 'right' & 'around'. Other
            directions raise errors.
            """
            if where not in self.TURNS.keys():
                raise RuntimeError("Unknown turn direction '{}' was given."
                                   "".format(where))

            else:
                ag = self.env[index]
                ag.orient = tuple(self.TURNS[where].dot(ag.orient).astype(int))
                return self.REWARDS['indifferent']

        return turn_agent

    # move
    def move(self, stand_still=False) -> Callable:
        """Return a functional that, when called, moves the agent in direction of orientation."""
        def move_agent(index: tuple) -> None:
            """Move the agent.

            If stand_still is set, the agent isn't moved at all. Otherwise
            the agent is moved (or at least tries to move) in direction of its
            orientation.
            """
            if stand_still:
                return self.REWARDS['indifferent']  # do nothing

            else:
                # get agent
                ag = self.env[index]

                # get target index
                target_index = tuple((np.array(index) + ag.orient) % self.dim)

                if self.env[target_index] is not None:
                    return self.REWARDS['wrong_action']

                else:
                    self.env[target_index] = self.env[index]  # move
                    self.env[index] = None  # clear old position
                    return self.REWARDS['default']  # TODO: rename rewards

        return move_agent

    # eating
