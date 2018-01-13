import ABM as abm
import numpy as np
import matplotlib.pyplot as plt 
import yaml
import datetime as dt 

with open("simconfig.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


# fixed random seed for reproducability 
np.random.seed(123456789)

#N = 256

# Sim setup 
w = cfg['Grid']['NX']
h = cfg['Grid']['NY']
rhoprey = cfg['Sim']['RhoPrey']
rhopred = cfg['Sim']['RhoPred']

pFlee = cfg['Prey']['Pflee']
pBreedPrey = cfg['Prey']['Pbreed']
pBreedPred = cfg['Pred']['Pbreed']

FoodReservePrey = cfg['Prey']['FoodReserve']
FoodReservePred = cfg['Pred']['FoodReserve']
MaxFrPrey = cfg['Prey']['FoodReserveMax']
MaxFrPred = cfg['Pred']['FoodReserveMax']



# Grid setup 
grid = abm.Grid(h,                  # height of the grid 
                w,                  # width of the grid 
                rhoprey,            # rhoprey 
                rhopred,            # rhopred
                FoodReservePrey,    # food reserve prey
                FoodReservePred,    # fr pred
                MaxFrPrey,          # max food reserve prey
                MaxFrPred,          # max fr pred 
                pBreedPrey,         # pBreed prey 
                pBreedPred)         # pBreed pred 

## actual sim
cycles = h*w 

ts = cfg['Sim']['Timesteps']  # timesteps 

# initialize empty lists for density plots 
nprey = []
npred = []

# append the first values 
nprey.append(grid.get_num_prey())
npred.append(grid.get_num_pred())

# initial plot, currenttimestep = -1, because range(ts) starts at 0 and internally the x axis is created via np.arange(currenttimestep) 
fig, ax, cbar = grid.plot(densities=[nprey, npred], currenttimestep=-1, timesteps=ts, 
                            filepath='plots/', filename=grid.timestamp()+"_0", title='init')
if(__name__ == '__main__'):
    print(": Simulation start", dt.datetime.now())
    for _ in range(ts):
        start = dt.datetime.now() 
        for it in range(cycles):
            j = np.random.randint(0,h)
            i = np.random.randint(0,w)

            grid.TakeAction([j,i], pFlee, FoodReservePrey, FoodReservePred, MaxFrPrey, MaxFrPred, 
                            pBreedPrey, pBreedPred) 
        
        if(grid.get_num_pred()==0):  # precaution, if predator dies out, the simulation stops 
            print(":: Predator died out !")
            break
        
        print(":: Step ", _, " | Elapsed time: ", dt.datetime.now() - start)
        nprey.append(grid.get_num_prey())
        npred.append(grid.get_num_pred())
        fig, ax, cbar = grid.plot(densities=[nprey, npred], currenttimestep=_, timesteps=ts, 
                                filepath='plots/', filename=grid.timestamp() + "_" + str(_ + 1), title='Step ' + str(_) + ', ')
        plt.close() 
    print(": Simulation stop", dt.datetime.now())



