

import numpy as np
import matplotlib.pyplot as plt
import uuid 

class Agent:
    """
    This class provides stuff for more stuff to be stuffed. 
    """

    def __init__(self, FoodReserve=4, Generation=0, MaxFoodReserve=None):
        self._FoodReserve = FoodReserve
        self._MaxFoodReserve = MaxFoodReserve
        self._Generation = Generation 
        self._ID = str(uuid.uuid4())
        #self._Kin = None 

    def get_ID(self):
        """
        Getter function for the ID.
        """
        return self._ID 

    def get_gen(self):
        """
        Getter function for the generation.
        """
        return self._Generation

class Prey(Agent):
    """
    This class is derived from the Agent class.
    """

    def __init__(self, FoodReserve, Generation=0, MaxFoodReserve=None, pBreed=0.2):
        super().__init__(FoodReserve, Generation, MaxFoodReserve)
        self._ID = "B" + self._ID  # Bfor beute. this kinda makes the whole kin thing obsolete.

class Predator(Agent):
    """
    This class is derived from the Agent class.
    """

    def __init__(self, FoodReserve, Generation=0, MaxFoodReserve=None, pBreed=0.2):
        super().__init__(FoodReserve, Generation, MaxFoodReserve)
        self._ID = "J" + self._ID  # J for jäger. this kinda makes the whole kin thing obsolete.
    

class Grid:
    """
    This class provides a different approach to the model, since it doesn't rely on dictionaries nor AgentIDs nor positions
    """

    def __init__(self, width, height, rhoprey, rhopred, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred):
        self._width = width
        self._height = height
        dt = 'U' + str(len(str(uuid.uuid4())) + 1)  # datatype for array 
        self._grid = np.empty(self._width*self._height, dtype=dt)
        self._preddict = dict()
        self._preydict = dict() 


        self._populate(rhoprey, rhopred, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred)
        self._grid = self._grid.reshape(self._height, self._width)  # reshape the grid after population

    def _populate(self, rhoprey, rhopred, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred):
        Npopmax = self._width * self._height  # maximum population
        Nprey = int(rhoprey * Npopmax)  # number of prey
        Npred = int(rhopred * Npopmax)  # number of pred

        idx = np.arange(Npopmax)  # create array of indices
        np.random.shuffle(idx)  # shuffle indices

        for _ in idx[:Nprey]:
            p = Prey(FoodReserve=foodresPrey, MaxFoodReserve=MaxFoodReservePrey, pBreed=pBreedPrey)
            self._grid[_] = p.get_ID() 
            self._preydict[p.get_ID()] = p 
            
        for _ in idx[Nprey:Nprey+Npred]:
            p = Predator(FoodReserve=foodresPred, MaxFoodReserve=MaxFoodReservePred, pBreed=pBreedPred)
            self._grid[_] = p.get_ID()
            self._preddict[p.get_ID()] = p

        
    def get_Nbh(self, index):
        # 9 neighbourhood
        y,x = index 
        delta = [-1,0,1]
        nbh = []
        idx_nbh = []
        for dy in delta:
            for dx in delta:
                j = (y+dy+self._height)%self._height
                i = (x+dx+self._width)%self._width 
                idx_nbh.append([j,i])
                nbh.append(self._grid[j, i]) 
        
        return idx_nbh, nbh

    def Move(self, index):
        idx_nbh, nbh = self.get_Nbh(index):
        possibleMoves = []
        for i, n in zip(idx_nbh, nbh):
            if(n == ""):
                possibleMoves.append(i)

        roll = possibleMoves[np.random.choice(range(len(possibleMoves)))]
        self._grid[roll] = self._grid[index]
        self._grid[index] = ""
        

    def plot(self, title='', figsize=(9,9), colourbar=True, ticks=False, filepath='plots/', filename='', dpi=300, fmt='png'):
        # the code below assumes, that self._grid is a numpy array of strings.
        plotarr = np.zeros(shape=(self._height, self._width))
        
        for j in range(self._height):
            for i in range(self._width):
                if(len(self._grid[j,i])):
                    if(self._grid[j,i][0] == "B"):
                        plotarr[j,i] = 1
                    else:
                        plotarr[j,i] = -1
                    
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(plotarr, cmap='seismic')

        if(colourbar):
            cbar = plt.colorbar(mappable=im, ax=ax, label=r'$\leftarrow \mathrm{Predator\ |\ Prey} \rightarrow$')

        if(not ticks):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        if(len(title)):
            ax.set_title(title)

        if(len(filepath)):
            save = filepath + filename
            fig.savefig(save, dpi=dpi, format=fmt)

        if(colourbar):
            return fig, ax, cbar

        return fig, ax

            


