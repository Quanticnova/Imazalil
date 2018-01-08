import random as rd
import yaml
import ABM as abm
import visualisations as vis
import matplotlib.pyplot as plt

with open("simconfig.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#print(cfg['Sim']['Pmate']['Pred'])

Agents = dict() # initialize empty agent dictionary
newborn = dict() # initialize empty newborn dictionary 

# fixed seed for reproducability
rd.seed(a=12345678)

# grid
w = cfg['Grid']['NX']
h = cfg['Grid']['NY']

# general 
food_res = 4

# initialize grid
grid = abm.Grid(w,h)

# initialize number of prey and predators
rhoPrey = cfg['Sim']['RhoPrey']
rhoPred = cfg['Sim']['RhoPred']
CarryingCapacity = grid.get_cc() 
nprey = int(CarryingCapacity * rhoPrey)
npred = int(CarryingCapacity * rhoPred)
num_agents = npred + nprey

# initialize starting positinos for agents
ipos = grid.initialPositions(num_agents)

# init preds
for _ in range(npred):
    abm.Predator(Agents, food_res, ipos[_], MaxFoodReserve=8)

# init preys
for _ in range(npred, num_agents):
    abm.Prey(Agents, food_res, ipos[_], MaxFoodReserve=8)

################# actual simulation below here ############################## 

for _ in range(cfg['Sim']['NEpoch']):
    for ID, a in Agents.items():
        if(a.get_kin() is not None):
            roll = rd.random()
            if(roll < a.get_pBreed()):
                a.createOffspring(grid, Agents, newborn)
            else:
                a.Eat(grid, Agents)
    abm.lifecycle(Agents, newborn, grid)  # cleanup all the dead Agents
    fig, ax = vis.show_agents(grid, Agents, savefig=True, title="timestep " + str(_))
    plt.close(fig)
