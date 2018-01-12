

import numpy as np
import matplotlib.pyplot as plt


class Grid:
    """
    This class provides a different approach to the model, since it doesn't rely on dictionaries nor AgentIDs nor positions 
    """
    
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._grid = np.zeros(shape=(self._height, self._width))

    def populate(self, rhoprey, rhopred):
        Npopmax = self._width * self._height
        Nprey = int(rhoprey * Npopmax)
        Npred = int(rhopred * Npopmax)
        prey = np.zeros(Nprey) +1
        pred = np.zeros(Npred) -1
        pop = np.stack([prey, pred]).ravel()
        
        idy = np.arange(self._height)
        idx = np.arange(self._width)
        np.random.shuffle(j)
        np.random.shuffle(i)

        for j in idy:
            for i in idx:
                

