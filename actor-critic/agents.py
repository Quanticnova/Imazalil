"""This class provides the necessary classes for agents (general), predators and prey."""

import uuid


class Agent:
    """
    This class provides an agent object.

    It has the following attributes:
        - food_reserve
        - max_food_reserve
        - generation
        - uuid
        - p_breed, the breeding probability
        - .... more to come

    Accessing the attributes is done via properties and setter, if necessary.
    """

    # class constants
    _UUID_LENGTH = len(uuid.uuid4().hex)

    # slots -------------------------------------------------------------------
    __slots__ = ['_food_reserve', '_max_food_reserve', '_generation',
                 '_p_breed', '_uuid', '_kin']

    # Init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1, kin: str=None,
                 **kwargs):
        """Initialise the agent instance."""
        # Initialize values
        self._food_reserve = 0
        self._max_food_reserve = None
        self._generation = None
        self._uuid = None
        self._p_breed = 1
        self._kin = None

        # Set property managed attributes
        self.food_reserve = food_reserve
        self.p_breed = p_breed
        self.kin = kin
        self.uuid = self._generate_uuid()

        if max_food_reserve:
            self.max_food_reserve = max_food_reserve

        if generation is not None:
            self.generation = generation

    # str method --------------------------------------------------------------
    def __str__(self):
        """Return the agents properties."""
        props = ("ID: {}\tkin: {}\tgeneration: {}\tfood_reserve: {}\t"
                 "max_food_reserve: {}".format(self.uuid, self.kin,
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

    # id
    @property
    def uuid(self) -> str:
        """The uuid of the agent."""
        return self._uuid

    @uuid.setter
    def uuid(self, uuid: str) -> None:
        """The uuid setter."""
        if not isinstance(uuid, str):
            raise TypeError("uuid can only be of type str, but {} was given."
                            "".format(type(uuid)))

        elif len(uuid) < self._UUID_LENGTH:
            raise ValueError("uuid must be of length {} but given uuid {} has "
                             "length {}.".format(self._UUID_LENGTH, uuid,
                                                 len(uuid)))
        elif self.uuid:
            raise RuntimeError("uuid is already set.")

        else:
            self._uuid = uuid

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

    # staticmethods -----------------------------------------------------------
    @staticmethod
    def _generate_uuid():
        """Generate a uuid for an agent."""
        return uuid.uuid4().hex

    # TODO: think about sensible ways for kin type storage -> plotting etc.


class Predator(Agent):
    """Predator class derived from Agent.

    This provides (additionally to class Agent):
        - specified uuid (leading "J_")
    """

    # class constants
    _UUID_LENGTH = Agent._UUID_LENGTH

    # slots -------------------------------------------------------------------
    __slots__ = ['_food_reserve', '_max_food_reserve', '_generation',
                 '_uuid', '_p_breed', '_kwargs']

    # init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1, **kwargs):
        """Initialise a Predator instance."""
        super().__init__(food_reserve=food_reserve,
                         max_food_reserve=max_food_reserve,
                         generation=generation,
                         p_breed=p_breed,
                         kin=self.__class__.__name__)


class Prey(Agent):
    """Prey class derived from Agent.

    This provides (additionally to class Agent):
        - specified uuid (leading "B_")
        - p_flee, the fleeing probability
    """

    # class constants
    _UUID_LENGTH = Agent._UUID_LENGTH

    # slots -------------------------------------------------------------------
    __slots__ = ['_food_reserve', '_max_food_reserve', '_generation',
                 '_uuid', '_p_breed', '_p_flee', '_kwargs']

    # init --------------------------------------------------------------------
    def __init__(self, *, food_reserve: int, max_food_reserve: int=None,
                 generation: int=None, p_breed: float=1, p_flee: float=0,
                 **kwargs):
        """Initialise a Prey instance."""
        super().__init__(food_reserve=food_reserve,
                         max_food_reserve=max_food_reserve,
                         generation=generation,
                         p_breed=p_breed,
                         kin=self.__class__.__name__)

        # initialise new attributes
        self._p_flee = 0

        # set new (property managed) attributes
        self.p_flee = p_flee

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
