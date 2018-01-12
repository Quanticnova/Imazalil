

import numpy as np
import matplotlib.pyplot as plt


class Grid:
    """
    This class provides a different approach to the model, since it doesn't rely on dictionaries nor AgentIDs nor positions
    """

    def __init__(self, width, height, rhoprey, rhopred):
        self._width = width
        self._height = height
        #self._grid = np.zeros(shape=(self._height, self._width))
        self._grid = np.zeros(self._width*self._height)

        self._populate(rhoprey, rhopred)
        self._grid = self._grid.reshape(self._height, self._width)  # reshape the grid after population

    def _populate(self, rhoprey, rhopred):
        Npopmax = self._width * self._height  # maximum population
        Nprey = int(rhoprey * Npopmax)  # number of prey
        Npred = int(rhopred * Npopmax)  # number of pred
        prey = np.ones(Nprey)  # array of 1s with length of number of prey
        pred = np.ones(Npred) * -1  # same with preds and -1

        idx = np.arange(Npopmax)  # create array of indices
        np.random.shuffle(idx)  # shuffle indices

        for i,p in zip(idx[:Nprey], prey):
            self._grid[i] = p

        for i,p in zip(idx[Nprey:Npred], pred):
            self._grid[i] = p

    def plot(self, title='', figsize=(9,9), colourbar=True, ticks=False, filepath='plots/', filename='', dpi=300, fmt='png'):
        save = filepath + filename
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(self._grid, cmap='seismic')

        if(colourbar):
            cbar = plt.colorbar(mappable=im, ax=ax, label='Agents')

        if(not ticks):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        if(len(title)):
            ax.set_title(title)

        if(len(filepath)):
            fig.savefig(save, dpi=dpi, format=fmt)

        if(colourbar):
            return fig, ax, cbar

        return fig, ax
