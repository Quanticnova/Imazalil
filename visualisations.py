import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# this script should contain all visualisation functionality like plotting routines etc.

# TODO: cleanup of code here; create a base function for plotting, use colors and markers as optional arguments
def plotAgents(ax, positions, label, marker='.', color='k', s=100, zorder=None, preprocess=True):
    if(preprocess):
        positions = np.array(positions).T
    ax.scatter(*positions, marker=marker, color=color, s=s, label=label, zorder=zorder)

def plotNbh(ax, positions, label='Nbh', centerlabel='Nbh center', marker='s', centermarker='x', color='yellow',
            s=200, centers=100, zorder=0, centerzorder=100):

    plotAgents(ax=ax, positions=positions, label=label, marker=marker, color=color, s=s, zorder=zorder)
    center = np.array(positions)[4]
    plotAgents(ax=ax, positions=center, label=centerlabel, marker=centermarker, color='k', s=centers,
               zorder=centerzorder, preprocess=False)

def show_agents(grid, agentsdict, showgrid=False, showlegend=False, figsize=(9,9), savefig=False, title='', numAgents=True):
    preypos = []
    predpos = []
    kinless = []
    for a in agentsdict.values():
        kin = a.get_kin()
        cgp = a.get_cgp()
        if kin is "Prey":
            preypos.append(cgp)
        elif kin is "Pred":
            predpos.append(cgp)
        elif kin is None:
            kinless.append(cgp)

    #plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if(showgrid):
        plotAgents(ax, grid.get_grid(), label='Grid', color='k')
    plotAgents(ax, preypos, label='Prey', color='g', zorder=10, s=20)
    plotAgents(ax, predpos, label='Predator', color='r', zorder=10, s=20)
    if(len(kinless)):
        plotAgents(ax, kinless, label='Kinless', color='k', zorder=1000, s=500)

    if(showlegend):
        ax.legend(bbox_to_anchor=(1.25, 1.0), fontsize=12)

    if(len(title)):
        if(numAgents):
            title = title + ", Pred: " + str(len(predpos)) + ", Prey: " + str(len(preypos))
        ax.set_title(title)

    if(savefig):
        fig.savefig('plots/' + str(dt.datetime.now()) + '.png', dpi=200)
    return fig, ax

def show_nbh(grid, agentsdict, agent_ID, savefig=False):

    fig, ax = show_agents(grid, agentsdict)
    plotNbh(ax, agentsdict[agent_ID].get_Nbh(grid))
    ax.legend(bbox_to_anchor=(1.25, 1.0), fontsize=12)

    if(savefig):
        fig.savefig('plots/' + str(dt.datetime.now()) + '.png', dpi=200)
    # TODO: spacing of plot and legend; legend is cropped at the moment.

def mean_food(agentsdict):
    foodres = []
    for agent in agentsdict.values():
        foodres.append(agent.get_fr())
    return np.mean(foodres)

def count_species(agentsdict):
    prey = 0
    pred = 0
    for agent in agentsdict.values():
        if agent.get_kin() is "Prey":
            prey += 1

        elif agent.get_kin() is "Pred":
            pred += 1

    return prey, pred
