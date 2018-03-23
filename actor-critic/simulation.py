"""This is an example simulation file for running a PPM simulation, using actor-critic method."""

import yaml
import numpy as np
import numpy.random as rd
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
from actor_critic import Policy  # also ensures GPU usage when imported


# load config files.....


# Initialize the policies and optimizer ---------------------------------------
PreyModel = Policy()
PredatorModel = Policy()
PreyOptimizer = optim.Adam(PreyModel.parameters(), lr=3e-2)  # whatever the numbers...
PredatorOptimizer = optim.Adam(PredatorModel.parameters(), lr=3e-2)
