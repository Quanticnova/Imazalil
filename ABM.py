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
import random as rd

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
        Agents[self._ID] = [self._GridPosX, self._GridPosY]

    def Die(self):
        """
        If an agent dies, it is removed from the list of agents
        """
        del Agents[self._ID]

    def getNbh(self):
        """
        This method returns a list of the moore neighbourhood of the agent. The
        numbering is the following:
            6 7 8
            3 4 5
            0 1 2
        So clearly the Agent is placed in cell nr. 4.
        """
        nbh = []
        delta = [-1, 0, 1]
        for dy in delta:
            for dx in delta:
                nbh.append([self._GridPosX + dx, self._GridPosY + dy])
                # TODO: since we assume torodial grid, the above function can
                # cause problems. Ideas: get a function that returns the exact
                # indices of the cells or a function that calculates the right
                # cell positions.
        return nbh

    def getNbdIdc(self, gridObject):
        """
        This method returns a list of the indices of the moore neighbourhood of
        the agent. It should hopefully prevent the problems indicaded in the
        TODO above.
        """
        idc = []
        

class Predator(Agent):
    """
    This class is derived from the Agent class.
    """

    def __init__(self, FoodReserve, GridPos, GenCounter):
        super().__init__(FoodReserve, GridPos, GenCounter)



class Grid:
    """
    hurr durr grid stuff
    """
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._grid = []
        for x in range(self._width):
            for y in range(self._height):
                self._grid.append([x,y])
                # TODO: maybe each cell carries a list of coordinates, and a list
                # what it contains. e.g. [[x, y], [<Agent/Grass/Nothing>]]

    def initialPositions(self, nAgents):
        shuffledGrid = self._grid
        rd.shuffle(shuffledGrid)
        return shuffledGrid[:nAgents]
