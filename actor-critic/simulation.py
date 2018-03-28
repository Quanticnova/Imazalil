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
env.seed(12345678)

# Initialize the policies and optimizer ---------------------------------------
PreyModel = ac.Policy()
PredatorModel = ac.Policy()
PreyOptimizer = optim.Adam(PreyModel.parameters(), lr=3e-2)  # whatever the numbers...
PredatorOptimizer = optim.Adam(PredatorModel.parameters(), lr=3e-2)

# means list
mean_gens = []  # in step units
mean_prey_rewards = []  # in episode units
mean_pred_rewards = []


# main loop
def main():
    """Trying to pseudo code here."""
    for i_eps in range(cfg['Sim']['episodes']):  # for now
        print("\n: Environment resetting now...")
        state, idx = env.reset()  # returns state and object of random agent

        for _ in range(cfg['Sim']['steps']):
            print(":: Episode {}, Step {}".format(i_eps, _))
            print("::: Plotting current state...")
            env.render(episode=i_eps, step=_, **cfg['Plot'])
            while(env.shuffled_agent_list):
                # if any prey got eaten last round, use it
                if len(env.eaten_prey) != 0:
                    tmpidx, ag = env.eaten_prey.pop()
                    env.state = env.index_to_state(index=tmpidx, ag=ag)
                    if env.state[-1] is None:
                        env.state[-1] = int(ag.food_reserve)
                    state = env.state
                    model = PreyModel
                    action = ac.select_action(model=model, state=state)
                    reward, state, done, idx = env.step(model=model,
                                                        agent=ag,
                                                        index=tmpidx,
                                                        returnidx=idx,
                                                        action=action)
                else:
                    ag = env._agents_dict[env.env[tuple(idx)]]  # agent object

                    # select model and action
                    model = PreyModel if ag.kin == "Prey" else PredatorModel
                    action = ac.select_action(model=model, state=state)
                    # take a step
                    reward, state, done, idx = env.step(model=model,
                                                        agent=ag,
                                                        index=idx,
                                                        action=action)

                model.rewards.append(reward)
                if done:
                    print(":: Breakpoint reached: Predators: {}\t Prey: {}"
                          "".format(len(env._agents_tuple.Predator),
                                    len(env._agents_tuple.Prey)))
                    break

            if done:
                break
            env.create_shuffled_agent_list()
            print("::: Created new shuffled agents list with {} individuals."
                  "".format(len(env.shuffled_agent_list)))
            # mean value output
            gens = []
            for a in env._agents_dict.values():
                gens.append(a.generation)

            prey_foodres = []
            for a in env._agents_tuple.Prey.values():
                prey_foodres.append(a.food_reserve)
            print("::: Mean food_reserve of preys: {}"
                  "".format(np.mean(prey_foodres)))
            print("::: Mean generation: {}".format(np.mean(gens)))
            mean_gens.append(np.mean(gens))
            idx = env.shuffled_agent_list.pop()
            env.state = env.index_to_state(index=idx)
            state = env.state

        mean_prey_rewards.append(np.mean(PreyModel.rewards))
        mean_pred_rewards.append(np.mean(PredatorModel.rewards))
        print(": optimizing now...")
        ac.finish_episode(model=PreyModel, optimizer=PreyOptimizer, gamma=0.01,
                          prnt="Preys")
        ac.finish_episode(model=PredatorModel, optimizer=PredatorOptimizer,
                          gamma=0.01, prnt="Predators")

    for f in [mean_gens, mean_pred_rewards, mean_prey_rewards]:
        np.savetxt(cfg['Plot']['filepath'] + str(f) + ".txt", np.arry(f))


if __name__ == "__main__":
    main()
