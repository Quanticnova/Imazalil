"""This is an example simulation file for running a PPM simulation, using actor-critic method."""

import yaml
import numpy as np
import torch.optim as optim

from agents import Predator, Prey
from environment import GridPPM
from tools import timestamp
import actor_critic as ac  # also ensures GPU usage when imported


# load config files.....
with open("simulation_config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

env = GridPPM(agent_types=(Predator, Prey), **cfg['Model'])
# env.seed(12345678)

# Initialize the policies and optimizer ---------------------------------------
PreyModel = ac.Policy()
PredatorModel = ac.Policy()
PreyOptimizer = optim.Adam(PreyModel.parameters(), lr=1e-4)  # whatever the numbers...
PredatorOptimizer = optim.Adam(PredatorModel.parameters(), lr=1e-4)

# means list
mean_gens = []  # in step units
mean_prey_rewards = []  # in episode units
mean_pred_rewards = []


# main loop
def main():
    """Trying to pseudo code here."""
    inittime = timestamp(return_obj=True)
    for i_eps in range(cfg['Sim']['episodes']):  # for now
        eps_time = timestamp(return_obj=True)
        print("\n: Environment resetting now...")
        state, idx = env.reset()  # returns state and object of random agent

        for _ in range(cfg['Sim']['steps']):
            print(":: Episode {}, Step {}".format(i_eps, _))
            if i_eps % cfg['Plot']['every'] == 0:  # plot every nth episode
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
                    action = ac.select_action(model=model, agent=ag,
                                              state=state)
                    reward, state, done, idx = env.step(model=model,
                                                        agent=ag,
                                                        index=tmpidx,
                                                        returnidx=idx,
                                                        action=action)
                else:
                    ag = env._idx_to_ag(idx)  # agent object

                    # select model and action
                    model = PreyModel if ag.kin == "Prey" else PredatorModel
                    action = ac.select_action(model=model, agent=ag,
                                              state=state)
                    # take a step
                    reward, state, done, idx = env.step(model=model,
                                                        agent=ag,
                                                        index=idx,
                                                        action=action)

                # model.rewards.append(reward)
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

            print("::: Mean generation: {}".format(np.mean(gens)))
            mean_gens.append(np.mean(gens))

            idx = env.shuffled_agent_list.pop()
            env.state = env.index_to_state(index=idx)
            state = env.state

        print("\n: Episode Runtime: {}".format(timestamp(return_obj=True) -
                                               eps_time))
        # mean_prey_rewards.append(np.mean(PreyModel.rewards))
        # mean_pred_rewards.append(np.mean(PredatorModel.rewards))
        print(": optimizing now...")
        ac.finish_episode(model=PreyModel, optimizer=PreyOptimizer,
                          history=env.history.Prey, gamma=0.05)
        ac.finish_episode(model=PredatorModel, optimizer=PredatorOptimizer,
                          history=env.history.Predator, gamma=0.05)

    print("\n: Entire simulation runtime: {}".format(timestamp(return_obj=True) - inittime))
    # for f in [mean_gens, mean_pred_rewards, mean_prey_rewards]:
    np.savetxt(cfg['Plot']['filepath'] + "gens" + ".txt", np.array(mean_gens))


if __name__ == "__main__":
    main()
