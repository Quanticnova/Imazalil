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
    __slots__ = ['saved_actions', 'rewards', 'affine1', 'affine2', 'affine3',
                 'action_head', 'value_head']

    # init --------------------------------------------------------------------
    def __init__(self, inputs=10, outputs=27, *args, **kwargs):
        """Initialize the neural network and some of its attributes.

        Default input of 10 for 9 neighbourhood and food reserve.
        """
        super(Policy, self).__init__()  # call nn.Modules init

        # initialize attributes
        # self.saved_actions = []
        # self.rewards = []

        # initialize layers
        self.affine1 = nn.Linear(inputs, 64)  # first layer
        self.affine2 = nn.Linear(64, 128)  # second layer
        self.affine3 = nn.Linear(128, 256)  # third layer
        self.action_head = nn.Linear(256, outputs)  # 27 possible actions
        self.value_head = nn.Linear(256, 1)

    # methods -----------------------------------------------------------------
    def forward(self, input_vector: torch.Tensor) -> tuple:
        """Forward the given input_vector through the layers and return the two outputs."""
        input_vector = F.relu(self.affine1(input_vector))  # Layer 1
        input_vector = F.relu(self.affine2(input_vector))  # Layer 2
        input_vector = F.relu(self.affine3(input_vector))  # Layer 3
        action_scores = self.action_head(input_vector)  # action layer
        state_value = self.value_head(input_vector)  # state value layer

        # softmax (re)normalizes all input tensors such that the sum of their contents is 1. The smaller a value (also negative), the smaller its softmaxed value.
        return F.softmax(action_scores, dim=-1), state_value


# defining necessary functions - move to a class maybe? /shrug
def select_action(*, model, agent, state):
    """Select an action based on the weighted possibilities given as the output from the model."""
    agent.memory.States.append(state)  # save the state
    state = torch.from_numpy(state).float()  # float creates a float tensor
    probs, state_value = model(Variable(state))  # propagate the state as Variable
    cat_dist = Categorical(probs)  # categorical distribution
    action = cat_dist.sample()  # I think I should e-greedy right at this point
    agent.memory.Actions.append(SavedAction(cat_dist.log_prob(action),
                                            state_value))
    return action.data[0]  # just output a number and not additionally the type


# defining what to do after the episode finished.
def finish_episode(*, model, optimizer, history, gamma: float=0.1) -> None:
    """Calculate the losses and backprop them through the models NN."""
    # initialize a few variables
    eps = np.finfo(np.float32).eps  # machine epsilon
    losses = []
    for (_, agent_rewards, saved_actions) in history:

        R = 0  # The discounted reward
        rewards = []
        policy_losses = []
        state_value_losses = []

        # get saved actions
        # saved_actions = model.saved_actions

        # iterate over all rewards that we got during the play
        for r in agent_rewards[::-1]:  # backwards to account for the more recent actions
            R = r + gamma * R  # discount!
            rewards.append(R)  # append + [::-1] is faster than insert(0,*)

        rewards = torch.Tensor(rewards[::-1])  # backwardss
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        # I think the eps should take care of my problem of NaNs. Somehow it
        # doesn't work, but the effect is the same as if I just switch the NaNs
        # to 0.
        # converting NaNs to 0.
        rewards[rewards != rewards] = 0  # should convert all NaN to 0

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
        losses.append(torch.stack(policy_losses).sum() + torch.stack(state_value_losses).sum())

    # average all losses
    loss = torch.stack(losses).mean()

    # backpropagate the loss
    loss.backward()
    optimizer.step()

    # TODO return some mean values or so

    # clear memory from unneeded variables
    #del model.rewards[:]
    #del model.saved_actions[:]
