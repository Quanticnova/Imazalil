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
        state_value = self.value_head(input_vector)  # state value layer

        # softmax (re)normalizes all input tensors such that the sum of their contents is 1. The smaller a value (also negative), the smaller its softmaxed value.
        return F.softmax(action_scores, dim=-1), state_value


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
def finish_episode(*, model, optimizer, gamma: float=0.1):
    """Calculate the losses and backprop them through the models NN."""
    # initialize a few variables
    R = 0  # The discounted reward
    policy_losses = []
    state_value_losses = []
    rewards = []
    eps = np.finfo(np.float32).eps  # machine epsilon

    # get saved actions
    saved_actions = model.saved_actions

    # iterate over all rewards that we got during the play
    for r in model.rewards[::-1]:  # backwards to account for the more recent actions
        R = r + gamma * R  # discount!
        rewards.append(R)  # append + [::-1] is faster than insert(0,*)

    rewards = torch.Tensor(rewards[::-1])  # backwardss
    rewards = (rewards - rewards.mean()) / (rewards.st() + eps)  # why eps???

    # now interate over all probability-state value-reward pairs
    for (log_prob, state_value), r in zip(saved_actions, rewards):
        reward = r - state_value.data[0]  # get the value, needs `Variable`
        policy_losses.append(-log_prob * reward)
        # calculate the (smooth) L^1 loss = least absolute deviation
        state_value_losses.append(F.smooth_l1_loss(state_value,
                                                   Variable(torch.Tensor([r]))))

    # empty the gradient of the optimizer
    optimizer.zero_grad()

    # calculate the loss
    loss = torch.stack(policy_losses).sum() + torch.stack(state_value_losses).sum()

    # backpropagate the loss
    loss.backward()
    optimizer.step()

    # clear memory from unneeded variables
    del model.rewards[:]
    del model.saved_actions[:]
