"""This class provides the necessary classes for agents (general), predators and prey."""

# import uuid
from collections import namedtuple, deque
from typing import Callable, NamedTuple
from torch.optim import Adam

memory = namedtuple('Memory', ('States', 'Rewards', 'Actions'))


class Agent:
    """
    This class provides an agent object.

    It has the following attributes:
        - food_reserve
        - max_food_reserve
        - generation, a counter, representing the number of parents
        - p_breed, the breeding probability
        - kin, a string providing the type of agent
        - memory, a named tuple of deques saving the agents' states, rewards
            and actions
        - kwargs, a dictionary storing every additional property that might
            find its way into the class call.

    class constants:
        - HEIRSHIP, a list of properties to pass down to the next generation

    Accessing the attributes is done via properties and setter, if necessary.

    The only necessary argument to specify is the food reserve. The rest is optional.
    """

    # class constants
    HEIRSHIP = ['max_food_reserve', 'generation', 'p_breed', '_kwargs']

    # slots -------------------------------------------------------------------
    __slots__ = ['_food_reserve', '_max_food_reserve', '_generation',
                 '_p_breed', '_kin', '_kwargs', '_memory']

    # Init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1.0, kin: str=None,
                 mem: tuple=None, **kwargs):
        """Initialise the agent instance."""
        # Initialize values
        self._food_reserve = 0
        self._max_food_reserve = None
        self._generation = None
        self._p_breed = 1.0
        self._kin = None
        self._kwargs = kwargs  # just set the value directly here.
        self._memory = None

        # Set property managed attributes
        self.food_reserve = food_reserve
        self.p_breed = p_breed

        if kin:  # if kin is given, set kin
            self.kin = kin

        else:  # otherwise just set 'Agent' as kin
            self.kin = self.__class__.__name__

        if max_food_reserve:
            self.max_food_reserve = max_food_reserve

        if generation is not None:
            self.generation = generation

        if mem is not None:
            self.memory = mem
        else:
            self.memory = memory(deque(), deque(), deque())  # initialize empty lists

    # magic method ------------------------------------------------------------
    def __str__(self) -> str:
        """Return the agents properties."""
        props = ("{}\tID: {}\tgeneration: {}\tfood_reserve: {}\t"
                 "max_food_reserve: {}".format(self.kin,  # self.uuid,
                                               self.generation,
                                               self.food_reserve,
                                               self.max_food_reserve))

        return props

    # Properties --------------------------------------------------------------
    # food_reserve
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

        elif self.max_food_reserve:
            if food_reserve >= self.max_food_reserve:
                self._food_reserve = self.max_food_reserve

            else:
                self._food_reserve = food_reserve

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

    # generation
    @property
    def generation(self) -> int:
        """The generation of the agent."""
        return self._generation

    @generation.setter
    def generation(self, generation: int) -> None:
        """The generation setter."""
        if not isinstance(generation, int):
            raise TypeError("generation can only be of type integer, "
                            "but {} was given.".format(type(generation)))

        elif generation < 0:
            raise ValueError("generation must be positive but {} was given"
                             "".format(generation))

        elif self.generation:
            raise RuntimeError("generation is already set.")

        else:
            self._generation = generation

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
            raise ValueError("p_breed must be between 0 and 1 but {} was given."
                             "".format(p_breed))

        else:
            self._p_breed = p_breed

    # kin
    @property
    def kin(self) -> str:
        """Return kin of the agent."""
        return self._kin

    @kin.setter
    def kin(self, kin: str) -> None:
        """The kin setter for the agent."""
        if not isinstance(kin, str):
            raise TypeError("kin must be of type str, but {} was given."
                            "".format(type(kin)))
        elif self.kin:
            raise RuntimeError("kin is alreday set and cannot be changed on the"
                               " fly.")

        else:
            self._kin = kin

    # memory for learning
    @property
    def memory(self) -> NamedTuple:
        """Hold the history of all states, rewards and actions for a single agent."""
        return self._memory

    @memory.setter
    def memory(self, memory: NamedTuple) -> None:
        """Set the NamedTuple for the memory."""
        if not isinstance(memory, tuple):
            raise TypeError("memory must be of type tuple, but {} was given."
                            "".format(type(memory)))
        elif self._memory is not None:
            raise RuntimeError("memory already set. This should not have "
                               "happened.")
        else:
            self._memory = memory

    # staticmethods -----------------------------------------------------------
    # no staticmethods so far...

    # classmethods ------------------------------------------------------------
    @classmethod
    def _procreate_empty(cls, *, food_reserve: int) -> Callable:
        """The classmethod creates a new "empty" instance of `cls`.

        food_reserve needs to be set explicitely.
        """
        return cls(food_reserve=food_reserve)

    # methods -----------------------------------------------------------------
    def procreate(self, *, food_reserve: int) -> Callable:
        """Take a class instance and inherit all attributes in `HEIRSHIP` from self.

        Return a `cls` instance with attributes set.
        """
        # create empty instance
        offspring = self._procreate_empty(food_reserve=food_reserve)

        # iterate over all attributes
        for attr in self.HEIRSHIP:
            parent_attr = getattr(self, attr)
            if parent_attr is not None:
                # adapt generation counter if set in parent
                if attr == 'generation':
                    setattr(offspring, attr, parent_attr+1)
                else:
                    setattr(offspring, attr, parent_attr)

        return offspring


class Predator(Agent):
    """Predator class derived from Agent.

    This provides (additionally to class Agent):
        - p_eat, the probability to eat a prey agent (should be defined as
            1-p_flee), is here for simplicity. (TODO: fix that)
    """

    # class constants
    HEIRSHIP = Agent.HEIRSHIP + ['p_eat']

    # slots -------------------------------------------------------------------
    __slots__ = ['_p_eat']

    # init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1.0, p_eat: float=1.0,
                 **kwargs):
        """Initialise a Predator instance."""
        super().__init__(food_reserve=food_reserve,
                         max_food_reserve=max_food_reserve,
                         generation=generation,
                         p_breed=p_breed,
                         kin=self.__class__.__name__,
                         **kwargs)

        # initialise new attributes
        self._p_eat = 1.0

        # set new (property managed) attributes
        self.p_eat = p_eat

    # magic method ------------------------------------------------------------
    def __str__(self) -> str:
        """Return the agents properties."""
        props = ("{}\tID: {}\tgen: {}\tfood_res: {}\t"
                 "max_food_res: {}\t p_eat: {}".format(self.kin,  # self.uuid,
                                                       self.generation,
                                                       self.food_reserve,
                                                       self.max_food_reserve,
                                                       self.p_eat))

        return props

    # properties --------------------------------------------------------------
    @property
    def p_eat(self) -> float:
        """The eating probability of the predator."""
        return self._p_eat

    @p_eat.setter
    def p_eat(self, p_eat: float) -> None:
        """The eating probability setter."""
        if not isinstance(p_eat, float):
            raise TypeError("p_eat must be of type float, but {} was given."
                            "".format(type(p_eat)))
        elif p_eat < 0 or p_eat > 1:
            raise ValueError("p_eat must be between 0 and 1 but {} was given."
                             "".format(p_eat))

        else:
            self._p_eat = p_eat


class Prey(Agent):
    """Prey class derived from Agent.

    This provides (additionally to class Agent):
        - p_flee, the fleeing probability
        - got_eaten, boolean flag to specify whether a prey got eaten or not.
    """

    # class constants
    # _UUID_LENGTH = Agent._UUID_LENGTH
    HEIRSHIP = Agent.HEIRSHIP + ['p_flee']

    # slots -------------------------------------------------------------------
    __slots__ = ['_p_flee', '_got_eaten']

    # init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1.0, p_flee: float=0.0,
                 **kwargs):
        """Initialise a Prey instance."""
        super().__init__(food_reserve=food_reserve,
                         max_food_reserve=max_food_reserve,
                         generation=generation,
                         p_breed=p_breed,
                         kin=self.__class__.__name__,
                         **kwargs)

        # initialise new attributes
        self._p_flee = 0.0
        self._got_eaten = False

        # set new (property managed) attributes
        self.p_flee = p_flee

    # magic method ------------------------------------------------------------
    def __str__(self) -> str:
        """Return the agents properties."""
        props = ("{}\tID: {}\tgen: {}\tfood_res: {}\t"
                 "max_food_res: {}\t p_flee: {}\t"
                 " got_eaten: {}".format(self.kin,  # self.uuid,
                                         self.generation,
                                         self.food_reserve,
                                         self.max_food_reserve,
                                         self.p_flee,
                                         self.got_eaten))

        return props

    # properties --------------------------------------------------------------
    @property
    def p_flee(self) -> float:
        """The fleeing probability of the prey."""
        return self._p_flee

    @p_flee.setter
    def p_flee(self, p_flee: float) -> None:
        """The fleeing probability setter."""
        if not isinstance(p_flee, float):
            raise TypeError("p_flee must be of type float, but {} was given."
                            "".format(type(p_flee)))
        elif p_flee < 0 or p_flee > 1:
            raise ValueError("p_flee must be between 0 and 1 but {} was given."
                             "".format(p_flee))

        else:
            self._p_flee = p_flee

    @property
    def got_eaten(self) -> bool:
        """Flag if prey was eaten (needed for actor-critic)."""
        return self._got_eaten

    @got_eaten.setter
    def got_eaten(self, got_eaten: bool) -> None:
        """Set if prey got eaten."""
        if not isinstance(got_eaten, bool):
            raise TypeError("got_eaten must be of type bool, but {} was given."
                            "".format(type(got_eaten)))

        else:
            self._got_eaten = got_eaten


# -----------------------------------------------------------------------------
class SmartGuy:
    """The class provides a new type of agent that carries its own neural network (and optimizer).

    It is meant to be used with the isolation environment, so that its behaviour can be studied there.
    """

    __slots__ = ['_policy', '_optimizer', '_kin', '_action_space']

    def __init__(self, policy: Callable, optimizer: Callable, action_space: dict, kin: str):
        """Initialize all necessary things."""
        self._policy = None
        self._optimizer = None
        self._kin = None
        self._action_space = None

        # property managed attributes
        self.policy = policy
        self.optimizer = optimizer
        self.kin = kin
        self.action_space = action_space

    # properties --------------------------------------------------------------
    @property
    def policy(self) -> Callable:
        """Return the agents' policy."""
        return self._policy

    @policy.setter
    def policy(self, policy: Callable) -> None:
        """Set the agents' policy.

        If policy is not a Callable a type error is raised.
        """
        if not callable(policy):
            raise TypeError("policy needs to be callable but {} was given."
                            "".format(type(policy)))

        elif self.policy is not None:
            raise RuntimeError("policy already set!")

        else:
            self._policy = policy

    @property
    def optimizer(self) -> Callable:
        """Return the agents' optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Callable) -> None:
        """Set the agents' optimizer.

        If optimizer is not a Callable a type error is raised.
        """
        if not isinstance(optimizer, Adam):  # FIXME also accept other optims
            raise TypeError("optimizer needs to be callable but {} was given."
                            "".format(type(optimizer)))

        elif self.optimizer is not None:
            raise RuntimeError("optimizer already set!")

        else:
            self._optimizer = optimizer

    @property
    def kin(self) -> str:
        """Return the agents' kin."""
        return self._kin

    @kin.setter
    def kin(self, kin: str) -> None:
        """Set the agents' kin."""
        if not isinstance(kin, str):
            raise TypeError("kin needs to be string but {} was given."
                            "".format(type(kin)))

        elif self.kin is not None:
            raise RuntimeError("kin is already set!")

        else:
            self._kin = kin

    @property
    def action_space(self) -> dict:
        """Return the agents' action space."""
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: dict) -> None:
        """Set the agents' action space."""
        if not isinstance(action_space, dict):
            raise TypeError("action space needs to be dict but {} was given."
                            "".format(type(action_space)))

        elif self.action_space is not None:
            raise RuntimeError("action space is already set!")

        else:
            self._action_space = action_space
