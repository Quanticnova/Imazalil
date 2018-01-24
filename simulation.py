import ABM as abm
import numpy as np
import matplotlib.pyplot as plt
import yaml
import datetime as dt

with open("simconfig.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


# fixed random seed for reproducability
np.random.seed(123456789)

# Sim setup
w = cfg['Grid']['NX']
h = cfg['Grid']['NY']
rhoprey = cfg['Sim']['RhoPrey']
rhopred = cfg['Sim']['RhoPred']
ff = cfg['Sim']['FastForward']

pFlee = cfg['Prey']['Pflee']
pBreedPrey = cfg['Prey']['Pbreed']
pBreedPred = cfg['Pred']['Pbreed']

FoodReservePrey = cfg['Prey']['FoodReserve']
FoodReservePred = cfg['Pred']['FoodReserve']
MaxFrPrey = cfg['Prey']['FoodReserveMax']
MaxFrPred = cfg['Pred']['FoodReserveMax']

# plot config
filepath = cfg['Plots']['filepath']
figsize = cfg['Plots']['figsize']
DPI = cfg['Plots']['DPI']
fmt = cfg['Plots']['format']

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
                pBreedPred,         # pBreed pred
                pFlee)              # pFlee for prey
## actual sim
cycles = h*w

ts = cfg['Sim']['Timesteps']  # timesteps

# initialize empty lists for density plots
nprey = []
npred = []

# initialize empty step counter
stepcnt=0

# append the first values
nprey.append(grid.get_num_prey())
npred.append(grid.get_num_pred())


if(__name__ == '__main__'):
    # initial plot, currenttimestep = -1, because range(ts) starts at 0 and internally the x axis is created via np.arange(currenttimestep)
    print(": Simulation start", dt.datetime.now())
    fig, ax, cbar = grid.plot(densities=[nprey, npred], currenttimestep=-1, timesteps=ts, dpi=DPI,
                                filepath=filepath, filename=grid.timestamp()+"_0", title='init',
                                fmt=fmt, figsize=figsize)
    plt.close(fig)
    # actual simulation
    for _ in range(ts):
        stepcnt+=1
        start = dt.datetime.now()
        _y, _x = np.where(grid.get_grid() != '')  # indices of agents
        idc = np.array([_y, _x]).T
        np.random.shuffle(idc)  # shuffle the indices
        for idx in idc:
            j, i = idx

            grid.TakeAction([j,i], FoodReservePrey, FoodReservePred)

        if(grid.get_num_pred()==0):  # precaution, if predator dies out, the simulation stops
            print(":: Predator died out !")
            break

        print(":: Step ", _, " | Elapsed time: ", dt.datetime.now() - start)
        nprey.append(grid.get_num_prey())
        npred.append(grid.get_num_pred())
        if(stepcnt >= ff):
            fig, ax, cbar = grid.plot(densities=[nprey, npred], currenttimestep=_, timesteps=ts,
                                      dpi=DPI, filepath=filepath,
                                      filename=grid.timestamp() + "_" + str(_ + 1),
                                      title='Step ' + str(_) + ', ', fmt=fmt, figsize=figsize)
            plt.close()
            stepcnt=0  # reset counter
    print(": Simulation stop", dt.datetime.now())
