# this script should turn out to be a working example of an Agent Based Model

# TODO:
#   - NN functionality / regular rules for predator/prey
#   - grass growing rate, etc.

# general stuff
import numpy as np
import matplotlib.pyplot as plt
import uuid
import random as rd
import copy  # needed for deepcopies
import datetime as dt 

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

    # TODO: order of arguments in __init__ determines input order of arguments
    def __init__(self, agentsdict, FoodReserve, GridPos, GenCounter=0, MaxFoodReserve=None, pBreed=0.2):
        self._FoodReserve = FoodReserve
        self._MaxFoodReserve = MaxFoodReserve
        self._GenCounter = GenCounter
        self._GridPosX, self._GridPosY = GridPos
        self._ID = str(uuid.uuid4())
        agentsdict[self._ID] = self  # add the agent to the dictionary with the ID as key
        self._kin = None 
        self._pBreed = pBreed 

    def Die(self, agentsdict):
        """
        If an agent dies, it is removed from the list of agents
        """
        del agentsdict[self._ID]

    def get_fr(self):
        """
        Getter-function for food reserve value
        """
        return self._FoodReserve

    def get_cgp(self):
        """
        Getter-function for current grid position.
        """
        return [self._GridPosX, self._GridPosY]

    def get_ID(self):
        """
        Getter-function for the agents ID.
        """
        return self._ID

    def get_gen(self):
        """
        Getter-function for the generation counter.
        """
        return self._GenCounter  
        
    def get_kin(self):
        """
        Getter-function for the kin type.
        """
        return self._kin 

    def get_maxfr(self):
        """
        Getter-function for the maximum food reserve.
        """
        return self._MaxFoodReserve
    
    def get_pBreed(self):
        """
        Getter-function for the chance of creating offspring.
        """
        return self._pBreed
        
    def get_Nbh(self, gridObject):
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
        x = self._GridPosX  # current X position
        y = self._GridPosY  # current Y position
        wid = gridObject.get_width()  # grid width
        hei = gridObject.get_height()  # grid height
        for dy in delta:
            for dx in delta:
                nbh.append([(x + dx + wid)%wid, (y + dy + hei)%hei])
        return nbh

    def Move(self, Nbh, NbhAgents, direction=-1):
        """
        This method moves the Agent to one of the neighbouring cells. If default value for direction remains 
        unchanged, the move is random. A prey can only move to empty cells. A predator can move to a cell already
        occupied by a prey to try and eat it with probability 1-pFlee. 
        """
        #nbh = self.get_Nbh(gridObject)  # find neighbouring cells
        #currentPos = gridObject.get_currentPositions(agentsdict)  # get current positions of all agents
        #nbhAgents = [c for c in nbh if c in currentPos[1]]  # find agents in neighbourhood of said agent
        possibleMove = nbh[:]  # copy of neighbourhood for better understanding
        
        #if(self.get_kin() is not None):
        for n in NbhAgents:
            possibleMove.remove(n)  # remove all occupied neighbouring cells
            
        if(len(possibleMove)):
            newPos = rd.choice(possibleMove)  # choose new position randomly
            self._GridPosX, self._GridPosY = newPos

    def Eat(self, gridObject, Nbh, NbhAgents, currentPos, agentsdict):
        """
        This method lets an agent eat; whether its grass or prey. 
        """
        MaxFoodReserve = self.get_maxfr()
        if(self._FoodReserve-1 <= 0):
            self._kin = None
        else:
            self._FoodReserve -= 1
        
        if(self._kin is "Prey" and self._FoodReserve < MaxFoodReserve - 1):
            self._FoodReserve += 2
            if(self._FoodReserve > MaxFoodReserve):
                self._FoodReserve = MaxFoodReserve
        
        elif(self._kin is "Pred"):
            #nbh = self.get_Nbh(gridObject)
            #currentPos = gridObject.get_currentPositions(agentsdict)
            #nbhAgents = [c for c in nbh if c in currentPos[1]]
            if(len(NbhAgents) > 1):  # if neighbours contain more than the central agent 
                for n in NbhAgents:
                    idx = currentPos[1].index(n)
                    if(currentPos[2][idx] is "Pred" or currentPos[2][idx] is not None):
                        NbhAgents.remove(n)  # remove all predator agents in the neighbourhood
                
                if(len(NbhAgents)):
                    foodpos = rd.choice(NbhAgents)
                    # TODO: possibility to flee 
                    food_idx = currentPos[1].index(foodpos)  # index of prey to be eaten in list of current agent pos
                    agentsdict[currentPos[0][food_idx]]._kin = None  # set kin type to None for later cleanup
                    self._GridPosX, self._GridPosY = foodpos  # move pred to preys position
                    self._FoodReserve += 2  # actual eating 
                    if(self._FoodReserve > MaxFoodReserve):
                        self._FoodReserve = MaxFoodReserve
                    
                else:
                    self.Move(Nbh, NbhAgents)
            else:
                pass # if only one Agent is in the neighbourhood, it's the 
    # TODO: fix the problenm with scatter
    # TODO: fix the code above; too complicated and redundant 
    # TODO: write a function for increasing/decreasing the food reserve 
    # TODO: maybe introduce a sorting for the list -> speed improvement? 



    def createOffspring(self, gridObject, agentsdict, newbornDict):
        if(self._FoodReserve > self._MaxFoodReserve/2):
            self._FoodReserve = self._FoodReserve - 4  # TODO: parameterize this !!!!
            nbh = self.get_Nbh(gridObject)  # get neighbourhood positions
            currentPos = gridObject.get_currentPositions(agentsdict)
            nbhAgents = [c for c in nbh if c in currentPos[1]]  # find agents in neighbourhood of said agent
            possiblePlace = nbh[:]  # copy of neighbourhood for better understanding
        
            for n in nbhAgents:
                possiblePlace.remove(n)  # remove all occupied neighbouring cells
            
            if(len(possiblePlace)):
                newPos = rd.choice(possiblePlace)  # choose new position randomly
                newgen = self._GenCounter +1
            
                # offspring -> newborn dictionary
                if(self._kin is "Pred"):                
                    Predator(newbornDict, FoodReserve=4, GridPos=newPos, MaxFoodReserve=8, GenCounter=newgen) 
            
                elif(self._kin is "Prey"):
                    Prey(newbornDict, FoodReserve=4, GridPos=newPos, MaxFoodReserve=8, GenCounter=newgen)
                

            # TODO: test this function!!! 

class Predator(Agent):
    """
    This class is derived from the Agent class.
    """

    def __init__(self, agentsdict, FoodReserve, GridPos, GenCounter=0, MaxFoodReserve=None, pBreed=0.3):
        super().__init__(agentsdict, FoodReserve, GridPos, GenCounter, MaxFoodReserve, pBreed)
        self._kin = "Pred"

class Prey(Agent):
    """
    This class is derived from the Agent class.
    """
    
    def __init__(self, agentsdict, FoodReserve, GridPos, GenCounter=0, MaxFoodReserve=None, pBreed=0.2, pFlee=0.2):
        super().__init__(agentsdict, FoodReserve, GridPos, GenCounter, MaxFoodReserve, pBreed)
        self._kin = "Prey"
        self._pFlee = pFlee  # chance to run away from predator 


class Grid:
    # TODO: docstring!    
    """
    <docstring>
    """
    def __init__(self, width, height, cc=-1):
        self._width = width
        self._height = height
        self._grid = []
        
        if(cc is -1):  # cc = carrying capacity of the grid = number of possible agents
            # if default value of -1 remains unchanged, the maximum number of agents is used
            self._cc = self._width * self._height
        
        # initializing grid positions
        for y in range(self._height):
            for x in range(self._width):
                self._grid.append([x,y])
                # TODO: maybe each cell carries a list of coordinates, and a list
                # what it contains. e.g. [[x, y], [<Agent/Grass/Nothing>]]
    
    def get_width(self):
        """
        Getter-function for the grid width.  
        """
        return self._width

    def get_height(self):
        """
        Getter-function for the grid height.  
        """
        return self._height

    def get_grid(self):
        """
        Getter-function for the grid positions. 
        """
        return self._grid 
 
    def get_cc(self):
        """
        Getter-function for the carrying capacity for the grid
        """
        return self._cc

    def initialPositions(self, nAgents):
        """
        This method returns a list of grid positions randomly selected for a given number of Agents. 
        """
        # TODO: Error handling if too many agents are given. 
        shuffledGrid = copy.deepcopy(self._grid)
        rd.shuffle(shuffledGrid)
        return shuffledGrid[:nAgents]

    def get_currentPositions(self, agentsdict):
        """
        This method returns a list of the currently occupied cell positions and the occupants type (e.g. Predator, 
        Prey). 
        """
        IDs = []
        cgps = []
        kins = []
        for ID, agent in agentsdict.items():
            IDs.append(ID)
            cgps.append(agent.get_cgp())
            kins.append(agent.get_kin())
        return [IDs, cgps, kins]

    def get_NbhAgents(self, agentobject, agentsdict):
        """
        Getter-function to return the 9-neighbourhood and the corresponding Agents in that neighbourhood for a given Agent
        object. 
        """
        Nbh = agentobject.get_Nbh(self)
        currentPos = self.get_currentPositions(agentsdict)
        NbhAgents = [c for c in Nbh if c in currentPos[1]]  # find agents in neighbourhood of said agent
        
        return Nbh, NbhAgents, currentPos 

# general functions for the agent based model below here:
def Agents_cleanup(agentsdict):
    IDs = []
    for _ in agentsdict.keys():
        IDs.append(_)
    
    for ID in IDs:
        if(agentsdict[ID].get_kin() is None):
            agentsdict[ID].Die(agentsdict)

def lifecycle(gridObject, agentsdict, newborndict):
    # death cometh before new life may arise
    IDs = list(agentsdict.keys()) 
    
    # remove all agents who died in the last iteration, i.e. have kintype None.
    for ID in IDs:
        if(agentsdict[ID].get_kin() is None):
            agentsdict[ID].Die(agentsdict)
            
    # new life shall arise from thy ashes:
    agentpos = gridObject.get_currentPositions(agentsdict)
    newagentpos = gridObject.get_currentPositions(newborndict)
    
    for newID, newagent in newborndict.items():
        if newagent.get_cgp() not in agentpos[1]:
            agentsdict[newID] = newagent
        
        else:
            pass
            #maybe I should to something about that. TODO? 
    
    # all newborn who have no place in this world shall now be brought to a better place. 
    newborndict.clear()
