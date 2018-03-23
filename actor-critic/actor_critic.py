"""This file provides the functionality for the actor critic neural network."""

import numpy as np
import numpy.random as rd
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# if gpu is to be used --------------------------------------------------------
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# instance of namedtuple to be used in policy
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# defining the classes for the pred/prey actor-critics
class Policy(nn.Module):
    """Create an instance of a neural network policy.

    More docstring to come!
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['saved_actions', 'rewards', 'affine1', 'action_head',
                 'value_head']

    # init --------------------------------------------------------------------
    def __init__(self, inputs=10, outputs=27, *args, **kwargs):
        """Initialize the neural network and some of its attributes."""
        super(Policy, self).__init__()  # call nn.Modules init

        # initialize attributes
        self.saved_actions = []
        self.rewards = []

        # initialize layers
        self.affine1 = nn.Linear(inputs, 128)  # first layer
        self.action_head = nn.Linear(128, outputs)  # 27 possible actions
        self.value_head = nn.Linear(128, 1)

    # methods -----------------------------------------------------------------
    def forward(self, input_vector: torch.Tensor) -> tuple:
        """Forward the given input_vector through the layers and return the two outputs."""
        input_vector = F.relu(self.affine1(input_vector))  # Layer 1
        action_scores = self.action_head(input_vector)  # action layer
        state_values = self.value_head(input_vector)  # state value layer

        # softmax (re)normalizes all input tensors such that the sum of their contents is 1. The smaller a value (also negative), the smaller its softmaxed value.
        return F.softmax(action_scores, dim=-1), state_values


# defining necessary functions - move to a class maybe? /shrug
def select_action(*, model, state):
    """Select an action based on the weighted possibilities given as the output from the model."""
    state = torch.from_numpy(state).float()  # float creates a float tensor
    probs, state_value = model(Variable(state))  # propagate the state as Variable
    cat_dist = Categorical(probs)  # categorical distribution
    action = cat_dist.sample()  # I think I should e-greedy right at this point
    model.saved_actions.append(SavedAction(cat_dist.log_prob(action),
                                           state_value))
    return action.data[0]  # just output a number and not additionally the type


# defining what to do after the episode finished.
