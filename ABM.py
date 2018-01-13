

import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime as dt 


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

    def get_fr(self):
        """
        Getter function for the food reserve.
        """
        return self._FoodReserve

    def set_fr(self, value):
        """
        Setter function for the food reserve
        """
        self._FoodReserve = value  
    
    def get_maxfr(self):
        """
        Getter function for the maximum food reserve. 
        """
        return self._MaxFoodReserve 
   

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

    def get_num_prey(self):
        return len(self._preydict)

    def get_num_pred(self):
        return len(self._preddict)
 
    def get_max_pop(self):
        return self._width * self._height

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

    def Die(self, index):
        """
        If an Agent dies, remove it from the grid-array and from the corresponding dictionary.
        """
        y, x = index
        ID = self._grid[y,x]
        if(ID[0] == "B"):
            del self._preydict[ID]
        else:
            del self._preddict[ID]
        self._grid[y,x] = ""  # clear the place in the array

    def Move(self, index, direction=None):
        y, x = index
        if(direction):
            j, i = direction 
            self._grid[j,i] = self._grid[y,x]
            self._grid[y,x] = "" 
        
        else:
            if(self._grid[y,x] is not ""):  # if not empty, then move 
                idx_nbh, nbh = self.get_Nbh(index)
                possibleMoves = []
                for i, n in zip(idx_nbh, nbh):
                    if(n == ""):
                        possibleMoves.append(i)
                if(len(possibleMoves)):
                    j, i = possibleMoves[np.random.choice(len(possibleMoves))]
                    self._grid[j,i] = self._grid[y,x]
                    self._grid[y,x] = ""
        
    def Eat(self, index, agent, pFlee):
        y, x = index 
        idx_nbh, nbh = self.get_Nbh(index)
        preys = []
        for n in nbh:
            if(len(n)):
                if n[0] == "B":
                    preys.append(n)

        if(len(preys)):
            roll = np.random.rand() 
            if(roll > pFlee):
                food = preys[np.random.choice(len(preys))]
                foodidx = idx_nbh[nbh.index(food)]
                self.fr_update(agent)
                self.Die(foodidx)
                self.Move(index, foodidx)
        else:
            self.Move(index)


    def createOffspring(self, agent, kin, index, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred,  pBreedPrey, pBreedPred):
        y, x = index
        idx_nbh, nbh = self.get_Nbh(index)
        #ID = self._grid[y,x]
        #if(ID[0] == "B"):
        #    agent = self._preydict[ID]
        #else:
        #    agent = self._preddict[ID]
        
        possibleMoves = []
        for i, n in zip(idx_nbh, nbh):
            if(n == ""):
                possibleMoves.append(i)

        if(len(possibleMoves) > 0):
            k, j = possibleMoves[np.random.choice(len(possibleMoves))]
            agent.set_fr(agent.get_fr() - 3) # reduce foodreserve 
            if(kin == "B"):
                p = Prey(FoodReserve=foodresPrey, MaxFoodReserve=MaxFoodReservePrey, pBreed=pBreedPrey)
                self._grid[k,j] = p.get_ID() 
                self._preydict[p.get_ID()] = p
                #print('*' + p.get_ID())
            else:
                p = Predator(FoodReserve=foodresPred, MaxFoodReserve=MaxFoodReservePred, pBreed=pBreedPred)
                self._grid[k,j] = p.get_ID() 
                self._preddict[p.get_ID()] = p
                #print('*' + p.get_ID())
                
        else: 
            pass

    def fr_update(self, agent):
        fr = agent.get_fr() 
        agent.set_fr(fr + 3)  # TODO make this optional! 
        if(agent.get_fr() > agent.get_maxfr()):
            agent.set_fr(agent.get_maxfr())


    def TakeAction(self, index, pFlee, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred):  
        #TODO  pBreed, pFlee as attributes in agent class? 
        y, x = index  # for given index, get array indices 
        
        if(len(self._grid[y,x])):  # if picked index is not empty
            ID = self._grid[y,x]  # extract ID 
            kin = ID[0]  # extract kintype 
            # get agent object 
            if(kin == "B"):  
                agent = self._preydict[ID]
                pBreed = pBreedPrey 
                
            else:
                agent = self._preddict[ID]
                pBreed = pBreedPred
                
            fr = agent.get_fr()
            # if food reserve is too low, the agent dies 
            if(fr -1 <= 0):
                self.Die(index)
            
            else:
                agent.set_fr(fr-1)  # decrease the foodreserve by 1 
                if(kin == "B"):
                    self.fr_update(agent)  # food reserve update 
                
                else:
                    self.Eat(index, agent, pFlee)  

                
                if(agent.get_fr() > agent.get_maxfr()//2): # if foodreserve is more than half the maximum 
                    roll = np.random.rand()  # pick a random number 
                    if(roll<=pBreed):  # if pick succesfull, breed.
                        self.createOffspring(agent, kin, index, foodresPrey, foodresPred, MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred) # Breeding
                
                if(kin == "B"):
                    self.Move(index)  # otherwise, take a step in a random direction, if possible 
        else: 
            pass 
                
    def plot(self, densities=None, currenttimestep=None, timesteps=1000, title='', figsize=(9,12), colourbar=True, ticks=False, filepath='plots/', filename='', dpi=300, fmt='png'):
        # the code below assumes, that self._grid is a numpy array of strings.
        plotarr = np.zeros(shape=(self._height, self._width))
        
        for j in range(self._height):
            for i in range(self._width):
                if(len(self._grid[j,i])):
                    if(self._grid[j,i][0] == "B"):
                        plotarr[j,i] = 1
                    else:
                        plotarr[j,i] = -1
                    
        if(densities):
            prey, pred = densities
            maxpop = self.get_max_pop()
            Rhoprey = np.array(prey)/maxpop 
            Rhopred = np.array(pred)/maxpop 
            x = np.arange(currenttimestep+2)  # +1 because there is the initial datapoint and because range starts at 0
            fig, (ax, axd) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}) # figure setup
            w, h = figsize 
            fig.set_figheight(h)
            fig.set_figwidth(w)
            #fig.tight_layout()
            fig.subplots_adjust(hspace=0.1)

            axd.plot(x, Rhoprey, color='#fde725', ls='-', label='Prey')
            axd.plot(x, Rhopred, color='#440154', ls='-', label='Predator')
            axd.set_ylabel('Agent density')
            axd.set_xlabel('Timesteps')
            axd.set_xlim([0,timesteps])
            axd.set_ylim([0,1])
            axd.legend(loc=2)

        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        
        im = ax.imshow(plotarr, cmap='viridis', vmin=-1, vmax=1)

        if(colourbar):
            cbar = plt.colorbar(mappable=im, ax=ax, fraction=0.047, pad=0.01, 
                                label=r'$\leftarrow \mathrm{Predator\ |\ Prey} \rightarrow$')

       
        if(not ticks):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        
        info = " Prey: " + str(len(self._preydict)) + ", Pred: " + str(len(self._preddict))
        
        if(len(title)):
            title = title + info 
        ax.set_title(title)

        if(len(filepath)):
            save = filepath + filename + "." + fmt 
            fig.savefig(save, dpi=dpi, format=fmt)

        if(colourbar):
            return fig, ax, cbar

        return fig, ax
    
    def timestamp(self):
        return str(dt.datetime.now())
            


