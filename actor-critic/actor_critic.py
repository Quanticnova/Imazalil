"""This file provides the functionality for the actor critic neural network."""

import warnings

import numpy as np
from collections import namedtuple, deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


# set mode for the actor_critic
def init(*, mode: str='cpu', goal: str="training", policy_kind: str="fc"):
    """Set the mode of the actor critic model to either 'cpu' or 'gpu'."""
    global FloatTensor
    global Tensor
    global dtype
    global use_cuda
    global train
    global conv

    # set boolean variable to tell select_action how to process states
    conv = True if policy_kind == "conv" else False

    # set boolean variable to indicate training (or testing if False)
    train = True if goal == "training" else False

    # gpu stuff
    if mode == 'gpu':
        use_cuda = torch.cuda.is_available()

    elif mode == 'cpu':
        use_cuda = False

    else:
        warnings.warn("Unknown mode '{}' given, proceeding in mode 'cpu'."
                      "".format(mode), RuntimeWarning)
        use_cuda = False

    # if gpu is to be used ----------------------------------------------------
    if use_cuda:
        print(": CUDA is available.")
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    # Tensor = FloatTensor
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# instance of namedtuple to be used in policy
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ConvPolicy(nn.Module):
    """Create an instance of a neural network with convolutional input layer."""

    __slots__ = ['conv1', 'hidden1', 'hidden2', 'hidden3', 'action_head',
                 'value_head', 'affine1']

    def __init__(self, *, conv1: dict, affine1: tuple, hidden1: tuple,
                 hidden2: tuple, hidden3: tuple, action_head: tuple,
                 value_head: tuple, **kwargs):
        """Initialize the neural network and its attributes."""
        super(ConvPolicy, self).__init__()  # call nn.Modules init function

        # convolutional tuple should contain {in_channels: 1, out_channels: 6,
        # kernel_size: 3, stride: 1, padding: 1}
        self.conv1 = nn.Conv2d(**conv1)  # for neighbourhood
        self.affine1 = nn.Linear(*affine1)  # for food reserve
        self.hidden1 = nn.Linear(*hidden1)
        self.hidden2 = nn.Linear(*hidden2)
        self.hidden3 = nn.Linear(*hidden3)
        self.action_head = nn.Linear(*action_head)
        self.value_head = nn.Linear(*value_head)

    def forward(self, input_data: tuple) -> tuple:
        """Forward the given input image."""
        image, foodreserve = input_data  # we assume this structure
        process = F.relu(self.conv1(image))  # has now conv dims
        foodres = F.relu(self.affine1(foodreserve))
        process = process.view(process.size(0), -1)  # flatten the tensor
        process = torch.cat([process[0], foodres], -1)  # concatenate layers
        process = F.relu(self.hidden1(process))
        process = F.relu(self.hidden2(process))
        process = F.relu(self.hidden3(process))
        action_scores = self.action_head(process)
        state_value = self.value_head(process)

        return F.softmax(action_scores, dim=-1), state_value


# defining the classes for the pred/prey actor-critics
class Policy(nn.Module):
    """Create an instance of a neural network policy.

    More docstring to come!
    """

    # slots -------------------------------------------------------------------
    __slots__ = ['saved_actions', 'rewards', 'affine1', 'affine2', 'affine3',
                 'action_head', 'value_head']

    # init --------------------------------------------------------------------
    def __init__(self, input: tuple, hidden1: tuple, hidden2: tuple,
                 hidden3: tuple, action_head: tuple, value_head: tuple,
                 *args, **kwargs):
        """Initialize the neural network and some of its attributes.

        Default input of 10 for 9 neighbourhood and food reserve.
        """
        super(Policy, self).__init__()  # call nn.Modules init

        # initialize layers
        '''
        self.affine1 = nn.Linear(inputs, 64)  # first layer
        self.affine2 = nn.Linear(64, 128)  # second layer
        self.affine3 = nn.Linear(128, 256)  # third layer
        self.action_head = nn.Linear(256, outputs)  # 27 possible actions
        self.value_head = nn.Linear(256, 1)
        '''

        self.affine1 = nn.Linear(*input)
        self.affine2 = nn.Linear(*hidden1)
        self.affine3 = nn.Linear(*hidden2)
        self.affine4 = nn.Linear(*hidden3)
        self.action_head = nn.Linear(*action_head)
        self.value_head = nn.Linear(*value_head)

    # methods -----------------------------------------------------------------
    def forward(self, input_vector: torch.Tensor) -> tuple:
        """Forward the given input_vector through the layers and return the two outputs."""
        input_vector = F.relu(self.affine1(input_vector))  # Layer 1
        input_vector = F.relu(self.affine2(input_vector))  # Layer 2
        input_vector = F.relu(self.affine3(input_vector))  # Layer 3
        input_vector = F.relu(self.affine4(input_vector))  # Layer 4
        action_scores = self.action_head(input_vector)  # action layer
        state_value = self.value_head(input_vector)  # state value layer

        # softmax (re)normalizes all input tensors such that the sum of their contents is 1. The smaller a value (also negative), the smaller its softmaxed value.
        return F.softmax(action_scores, dim=-1), state_value


# defining necessary functions - move to a class maybe? /shrug
def select_action(*, model, agent, state) -> float:
    """Select an action based on the weighted possibilities given as the output from the model."""
    # state should be a list of numpy arrays [np.array(nbh), np.array(fr)]
    agent.memory.States.append(state)  # save the state
    if conv:
        for i, s in enumerate(state):
            state[i] = Variable(torch.from_numpy(s).float().type(dtype))
        probs, state_value = model(state)  # propagate the state

    else:
        state = torch.from_numpy(state).float().type(dtype)  # float creates a float tensor
        probs, state_value = model(Variable(state))  # propagate the state as Variable
    cat_dist = Categorical(probs)  # categorical distribution
    action = cat_dist.sample()  # I think I should e-greedy right at this point
    if train:
        agent.memory.Actions.append(SavedAction(cat_dist.log_prob(action),
                                                state_value))
    return action.data[0]  # just output a number and not additionally the type


# defining what to do after the episode finished.
def finish_episode(*, model, optimizer, history, gamma: float=0.1,
                   return_means: bool=False) -> Optional[tuple]:
    """Calculate the losses and backprop them through the models NN."""
    # initialize a few variables
    eps = np.finfo(np.float32).eps  # machine epsilon
    losses = deque()
    returns_to_average = deque()
    for (_, agent_rewards, saved_actions) in history:

        R = 0  # The discounted reward
        rewards = deque()
        policy_losses = deque()
        state_value_losses = deque()

        # reverse rewards (its a deque!)
        agent_rewards.reverse()

        # iterate over all rewards that we got during the play
        for r in agent_rewards:  # backwards to account for the more recent actions
            returns_to_average.append(r)  # for later averaging
            R = r + gamma * R  # discount!
            rewards.appendleft(R)  # deque power baby!

        rewards = torch.Tensor(rewards).type(dtype)  # use gpu if available
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
                                      Variable(torch.Tensor([r]).type(dtype))))

        # empty the gradient of the optimizer
        optimizer.zero_grad()

        # calculate the loss
        losses.append(torch.stack(policy_losses).sum() + torch.stack(state_value_losses).sum())

    # average all losses
    loss = torch.stack(losses).mean()

    # backpropagate the loss
    loss.backward()
    optimizer.step()

    # free memory
    losses.clear()  # its a deque

    # if output is wanted
    if return_means:
        ret_avg = np.mean(returns_to_average)
        returns_to_average.clear()
        return loss, ret_avg


# saving function
def save_checkpoint(state: dict, filename: str) -> None:
    """Save the given model state to file."""
    if filename[-8:] != ".pth.tar":
        filename += ".pth.tar"

    torch.save(state, filename)  # stores the given parameters
