"""This is an example simulation file for running a PPM simulation, using actor-critic method."""

import yaml
import numpy as np
import numpy.random as rd
#import warnings
# matplotlib?
#from collections import namedtuple

import torch
#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable

from agents import Predator, Prey
from environment import GridPPM
from tools import timestamp
import actor_critic as ac  # also ensures GPU usage when imported


# load config files.....
with open("simulation_config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

env = GridPPM(agent_types=(Predator, Prey), **cfg['Model'])


# Initialize the policies and optimizer ---------------------------------------
PreyModel = ac.Policy()
PredatorModel = ac.Policy()
PreyOptimizer = optim.Adam(PreyModel.parameters(), lr=3e-2)  # whatever the numbers...
PredatorOptimizer = optim.Adam(PredatorModel.parameters(), lr=3e-2)


# main loop
def main():
    """Trying to pseudo code here."""

    for i_eps in range(episodes):  # for now
        state, idx = env.reset()  # returns state and object of random agent

        while(env.shuffled_agent_list):
            ag = env._agents_dict[env.env[tuple(idx)]]  # agent object
            # select model and action
            model = PreyModel if ag.kin == "Prey" else PredatorModel
            action = ac.select_action(model=model, state=state)
            # take a step
            state, reward, done = env.step(model=model, agent=ag,
                                           index=idx, action=action)



if __name__ == "__main__":
    main()
