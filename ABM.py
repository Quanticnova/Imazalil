# this script should turn out to be a working example of an Agent Based Model

# TODO:
#   - agent class
#   - predator/ prey class (derived from agent class)
#   - NN functionality / regular rules for predator/prey
#   - Grid functionality, grass growing rate, etc.



# general stuff
import numpy as np
import matplotlib.pyplot as plt
import uuid

Agents = dict()

class Agent:
    """
    This class provides the necessary methods and attributes every agent should
    carry.
    Methods:
        - Move(args): moves the agent around in his neighbourhood
        - FeedOn(args): agent feeds on a given resource - if the resource is a
            prey, it can run away with a given probability. Increases _FoodReserve
            attribute.
        - Die(args): if the _FoodReserve is 0 or an Agent gets eaten it dies and
            is removed.

    Attributes:
        - _FoodReserve: tba
        - _GenCounter: (generation counter) if agents mate and create and offspring,
        this counter increases by 1. Merely a fun paramater as of this writing.
        (if used anyway)
        - _GridPosX/Y: X/Y position on the grid
    """

    # order of arguments in __init__ determines input order of arguments
    def __init__(self, FoodReserve, GridPos, GenCounter=0):
        self._FoodReserve = FoodReserve
        self._GenCounter = GenCounter
        self._GridPosX, self._GridPosY = GridPos
        self._ID = str(uuid.uuid4())
        Agents[self._ID] = tuple(self._GridPosX, self._GridPosY)

    def Die(self):
        """
        If an agent dies, it is removed from the list of agents
        """
        del Agents[self._ID]
