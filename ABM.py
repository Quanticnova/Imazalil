

import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime as dt


class Agent:
    """
    This class provides an object, which can have the following attributes:
        - FoodReserve
        - MaxFoodReserve
        - Generation
        - ID
    It is also equipped with getter functions for each of the attributes, and a setter function for
    the FoodReserve.
    """

    # slots are used to fix the number of attributes a class can have -> memory/speed improvement
    __slots__ = ['_FoodReserve', '_MaxFoodReserve', '_Generation', '_ID']


    def __init__(self, FoodReserve=4, Generation=0, MaxFoodReserve=None):
        self._FoodReserve = FoodReserve
        self._MaxFoodReserve = MaxFoodReserve
        self._Generation = Generation
        self._ID = str(uuid.uuid4())

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

    __slots__ = ['_FoodReserve', '_MaxFoodReserve', '_Generation', '_ID', '_pBreed', '_pFlee']

    def __init__(self, FoodReserve, Generation=0, MaxFoodReserve=None, pBreed=0.2, pFlee=0.5):
        super().__init__(FoodReserve, Generation, MaxFoodReserve)
        self._ID = "B" + self._ID  # Bfor beute. this kinda makes the whole kin thing obsolete.
        self._pBreed = pBreed
        self._pFlee = pFlee

    def get_pFlee(self):
        """
        getter function for probability to flee.
        """
        return self._pFlee

    def get_pBreed(self):
        """
        getter function for probability to breed.
        """
        return self._pBreed

class Predator(Agent):
    """
    This class is derived from the Agent class.
    """

    __slots__ = ['_FoodReserve', '_MaxFoodReserve', '_Generation', '_ID', '_pBreed']

    def __init__(self, FoodReserve, Generation=0, MaxFoodReserve=None, pBreed=0.2):
        super().__init__(FoodReserve, Generation, MaxFoodReserve)
        self._ID = "J" + self._ID  # J for jäger. this kinda makes the whole kin thing obsolete.
        self._pBreed = pBreed

    def get_pBreed(self):
        """
        getter function for probability to breed.
        """
        return self._pBreed

class Grid:
    """
    This class provides a different approach to the model, since it doesn't rely on dictionaries,
    nor AgentIDs nor positions.
    It provides the following attributes:
        - width
        - height
        - grid - a numpy array of either empty cells ("") or cells with agents ("J.." or "B..")
        - preddict - a dictionary of all predators, with IDs as keys and classobjects as values
        - preydict - same, just with preys

    It also provides the following methods:
        - populate - populate the empty initial grid with Agents, depending on their inital density
        - getter functions for amonut of prey/pred, maximum population size = grid size,
        - get_Nbh - for a given array index, return the indices and contents of the 9-neighbourhood
        - Die - if agent dies, delete it from the dictionary and empty its array space.
        - Move - move an agent in a random direction, if possible. if direction is given, it is
          directly placed there.
        - Eat - if prey, just increase the FoodReserve. If pred, check the Nbh for possible preys,
          and try to eat one of them with probability 1-pFlee. If the pred gets to eat, increase
          its FoodReserve.
        - createOffspring - with pBreed, create an offspring of the own kind, reduce FoodReserve by
          3 (which might be optional later on), and place it on a empty cell in its Nbh.
        - fr_update - wrapped up some loc for easier handling of the FoodReserve update process
        - TakeAction - basically, the actual simulation step. This function determines the given
          index' kintype, and acts accordingly to that with the above methods Move, Eat, Die,
          createOffspring, fr_update...
        - plot - plots the whole goddamn thing. if you have a list of the numbers of prey & pred at
          each timestep (-> get_num_pred, ..), you can use them as an input list to create density
          plots over time.
    """

    __slots__ = ['_width', '_height', '_maxPop', '_grid', '_preddict', '_preydict']

    def __init__(self, width, height, rhoprey, rhopred, foodresPrey, foodresPred,
                 MaxFoodReservePrey, MaxFoodReservePred, pBreedPrey, pBreedPred, pFlee):
        self._width = width
        self._height = height
        self._maxPop = self._width * self._height  # maximal population
        dt = 'U' + str(len(str(uuid.uuid4())) + 1)  # datatype for array
        # initialization of empty grid and dictionaries
        self._grid = np.empty(self._width*self._height, dtype=dt)
        self._preddict = dict()
        self._preydict = dict()

        # populate the empty grid with given parameters
        self._populate(rhoprey, rhopred, foodresPrey, foodresPred, MaxFoodReservePrey,
                       MaxFoodReservePred, pBreedPrey, pBreedPred, pFlee)

    def _populate(self, rhoprey, rhopred, foodresPrey, foodresPred, MaxFoodReservePrey,
                  MaxFoodReservePred, pBreedPrey, pBreedPred, pFlee):
        """
        Populate the empty grid!
        """
        Nprey = int(rhoprey * self._maxPop)  # number of prey
        Npred = int(rhopred * self._maxPop)  # number of pred

        idx = np.arange(self._maxPop)  # create array of indices
        np.random.shuffle(idx)  # shuffle indices

        for _ in idx[:Nprey]:
            p = Prey(FoodReserve=foodresPrey, MaxFoodReserve=MaxFoodReservePrey, pBreed=pBreedPrey,
                     pFlee=pFlee)
            self._grid[_] = p.get_ID()
            self._preydict[p.get_ID()] = p

        for _ in idx[Nprey:Nprey+Npred]:
            p = Predator(FoodReserve=foodresPred, MaxFoodReserve=MaxFoodReservePred, pBreed=pBreedPred)
            self._grid[_] = p.get_ID()
            self._preddict[p.get_ID()] = p

        # after populating:
        self._grid = self._grid.reshape(self._height, self._width)  # reshape to height x width


    def get_grid(self):
        """
        getter function for the grid array.
        """
        return self._grid

    def get_num_prey(self):
        """
        getter function for the number of preys.
        """
        return len(self._preydict)

    def get_num_pred(self):
        """
        getter function for number of preds.
        """
        return len(self._preddict)

    def get_max_pop(self):
        """
        getter function for the maximal number of possible agents on the grid = grid size
        """
        return self._maxPop

    def get_Nbh(self, index):
        """
        Return the array indices of self._grid and their content of the 9-neighbourhood for a given
        index.
        """
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
        """
        check the 9-neighbourhood and pick a random empty place to move to. if none is available,
        do nothing.
        if a direction is given (in form of a tuple or a list of length 2), move the agent directly
        there.
        """
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

    def Eat(self, index, agent):
        """
        find all preys in the Nbh, and pick one at random to eat (or at least try to eat).
        """
        y, x = index
        idx_nbh, nbh = self.get_Nbh(index)
        preys = []
        empties = []
        for n in nbh:
            if(len(n)):
                if n[0] == "B":
                    preys.append(n)
            else:
                empties.append(n)

        if(len(preys)):
            roll = np.random.rand()
            food = preys[np.random.choice(len(preys))]
            if(roll > self._preydict[food].get_pFlee()):
                foodidx = idx_nbh[nbh.index(food)]
                self.fr_update(agent)
                self.Die(foodidx)
                self.Move(index, foodidx)

        # if there are no preys to eat, take a step in a random free direction
        elif(len(empties)):
            self.Move(index, direction=empties[np.random.choice(len(empties) )])

        else:
            pass  # TODO, something to do, if you can't move? hm...

    def createOffspring(self, agent, kin, index, foodresPrey, foodresPred):
        y, x = index
        idx_nbh, nbh = self.get_Nbh(index)
        possibleMoves = []
        for i, n in zip(idx_nbh, nbh):
            if(n == ""):
                possibleMoves.append(i)

        if(len(possibleMoves) > 0):
            k, j = possibleMoves[np.random.choice(len(possibleMoves))]
            agent.set_fr(agent.get_fr() - 3) # reduce foodreserve, TODO, maybe 4?
            if(kin == "B"):
                p = Prey(FoodReserve=foodresPrey, MaxFoodReserve=agent.get_maxfr(),
                         pBreed=agent.get_pBreed())
                self._grid[k,j] = p.get_ID()
                self._preydict[p.get_ID()] = p

            else:
                p = Predator(FoodReserve=foodresPred, MaxFoodReserve=agent.get_maxfr(),
                             pBreed=agent.get_pBreed())
                self._grid[k,j] = p.get_ID()
                self._preddict[p.get_ID()] = p

        else:
            pass

    def fr_update(self, agent):
        fr = agent.get_fr()
        agent.set_fr(fr + 3)  # TODO make this optional!
        if(agent.get_fr() > agent.get_maxfr()):
            agent.set_fr(agent.get_maxfr())


    def TakeAction(self, index, foodresPrey, foodresPred):
        y, x = index  # for given index, get array indices

        if(len(self._grid[y,x])):  # if picked index is not empty
            ID = self._grid[y,x]  # extract ID
            kin = ID[0]  # extract kintype

            # get agent object
            if(kin == "B"):
                agent = self._preydict[ID]

            else:
                agent = self._preddict[ID]

            fr = agent.get_fr()  # if food reserve is too low, the agent dies
            if(fr -1 <= 0):
                self.Die(index)

            else:
                agent.set_fr(fr-1)  # decrease the foodreserve by 1
                if(kin == "B"):
                    self.fr_update(agent)  # food reserve update

                else:
                    self.Eat(index, agent)

                if(agent.get_fr() > agent.get_maxfr()//2): # if foodreserve is > than half the maximum
                    roll = np.random.rand()  # pick a random number
                    if(roll<=agent.get_pBreed()):  # if pick succesfull, breed.
                        self.createOffspring(agent, kin, index, foodresPrey, foodresPred) # Breeding

                if(kin == "B"):
                    self.Move(index)  # otherwise, take a step in a random direction, if possible
        else:
            pass  # something to do for empty grid cells?

    def plot(self, densities=None, currenttimestep=None, timesteps=1000, title='', figsize=(9,12),
             colourbar=True, ticks=False, filepath='plots/', filename='', dpi=300, fmt='png'):
        # the code below assumes, that self._grid is a numpy array of strings.
        plotarr = np.zeros(shape=(self._height, self._width))

        # numpy magic!
        _y, _x = np.where(self._grid != '')

        for j, i in np.array([_y, _x]).T:
            if(self._grid[j,i][0] == 'B'):
                plotarr[j,i] = 1
            else:
                plotarr[j,i] = -1

        if(densities):
            densities = list(densities)  # ensure type

            # figure setup
            fig, (ax, axd) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]})
            w, h = figsize
            fig.set_figheight(h)
            fig.set_figwidth(w)
            fig.subplots_adjust(hspace=0.1)

            # normalization and timesteps
            maxpop = self.get_max_pop()
            x = np.arange(currenttimestep+2)  # +1 because there is the initial datapoint and because range starts at 0

            # colors and labels
            colors = ['#fde725', '#440154']
            labels = ['Prey', 'Predator']
            # TODO: optional colors and kintypes 

            # plotting
            for n, d in enumerate(densities):
                rhod = np.array(d)/maxpop
                axd.plot(x, rhod, ls='-', color=colors[n], label=labels[n])

            axd.set_ylabel('Agent density')
            axd.set_xlabel('Timesteps')
            axd.set_xlim([0,timesteps])
            axd.set_ylim([0,1])
            axd.legend(loc=1)

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
