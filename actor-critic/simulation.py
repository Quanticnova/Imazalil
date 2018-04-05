#!/home/corvus/docs/uni/bachelor_thesis/code/Imazalil/BA-venv/bin/python

"""This is an example simulation file for running a PPM simulation, using actor-critic method."""

import yaml
import numpy as np
import torch
import torch.optim as optim
import argparse as ap
from collections import deque

from agents import Predator, Prey
from environment import GridPPM
from tools import timestamp, keyboard_interrupt_handler
import actor_critic as ac  # also ensures GPU usage when available

# setup argparse options
parser = ap.ArgumentParser(description="Command line options for the simulation script.")
parser.add_argument("--resume", type=str, default="",
                    help="resume simulation from given state")

# load Args
args = parser.parse_args()
arg_res = args.resume  # resume filepath

# load config files.....
with open("simulation_config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

cfg_res = cfg['Sim']['resume_state_from']  # resume filepath

resume = None  # initialize
# if resume filepath is given, resume simulation from there
if arg_res and cfg_res:
    if arg_res != cfg_res:  # TODO priorize cmd arg over cfg
        raise IOError("resume is specified twice and doesn't match. Don't"
                      " know what to load...\nGiven paths:\n\t{}\n\t{}"
                      "".format(arg_res, cfg_res))
    else:
        print(": Resuming simulation from checkpoint {}".format(arg_res))
        resume = torch.load(arg_res)

elif arg_res:
    print(": Resuming simulation from checkpoint {}".format(arg_res))
    resume = torch.load(arg_res)

elif cfg_res:
    print(": Resuming simulation from checkpoint {}".format(cfg_res))
    resume = torch.load(cfg_res)

# Initialize Grid -------------------------------------------------------------
env = GridPPM(agent_types=(Predator, Prey), **cfg['Model'])
# env.seed(12345678)

# Initialize the policies and averages ----------------------------------------
PreyModel = ac.Policy()
PredatorModel = ac.Policy()
# averages
avg = {'mean_gens': deque(),  # in step units
       'mean_prey_rewards': deque(),  # in episode units
       'mean_pred_rewards': deque(),
       'mean_prey_loss': deque(),  # in episode units
       'mean_pred_loss': deque()}

# simulation parameters
resume_pars = {'last_episode': 0}

if resume is not None:
    print(": Found the following keys: {}".format(resume.keys()))
    # resume models
    PreyModel.load_state_dict(resume['PreyState'])
    PredatorModel.load_state_dict(resume['PredatorState'])
    # resume parameter averages
    for p in avg.keys():
        if p in resume:
            avg[p] = resume[p]

    for p in resume_pars.keys():
        if p in resume:
            resume_pars[p] = resume[p]

PreyOptimizer = optim.Adam(PreyModel.parameters(), lr=1e-4)
PredatorOptimizer = optim.Adam(PredatorModel.parameters(), lr=1e-4)

if resume is not None:  # resume the parameters..
    PreyOptimizer.load_state_dict(resume['PreyOptimizerState'])
    PredatorOptimizer.load_state_dict(resume['PredatorOptimizerState'])

# save function ---------------------------------------------------------------
save_state = {'PreyState': PreyModel.state_dict(),
              'PredatorState': PredatorModel.state_dict(),
              'PreyOptimizerState': PreyOptimizer.state_dict(),
              'PredatorOptimizerState': PredatorOptimizer.state_dict(),
              'mean_gens': avg['mean_gens'],
              'mean_prey_rewards': avg['mean_prey_rewards'],
              'mean_pred_rewards': avg['mean_pred_rewards'],
              'mean_prey_loss': avg['mean_prey_loss'],
              'mean_pred_loss': avg['mean_pred_loss']}


def save():
    """Save the current state of the simulation to a resumeable file."""
    print("\n: Storing the following keys: {}".format(save_state.keys()))
    filename = cfg['Sim']['save_state_to'] + "state_" + timestamp()
    ac.save_checkpoint(state=save_state, filename=filename)


# main loop -------------------------------------------------------------------
@keyboard_interrupt_handler(save=save, abort=None)
def main():
    """Trying to pseudo code here."""
    inittime = timestamp(return_obj=True)  # get initial time datetime object

    for i_eps in range(resume_pars['last_episode'], cfg['Sim']['episodes']):  # if resume is given, start from there
        # add entry to save_dict
        save_state['last_episode'] = i_eps

        eps_time = timestamp(return_obj=True)  # record episode starting time
        print("\n: Environment resetting now...")
        state, idx = env.reset()  # returns state and object of random agent

        # save data
        if i_eps % cfg['Sim']['save_state_every'] == 0:
            save()

        for _ in range(cfg['Sim']['steps']):
            print(":: Episode {}, Step {}".format(i_eps, _))

            if cfg['Plot']['render']:
                if i_eps % cfg['Plot']['every'] == 0:  # plot every nth episode
                    print("::: Plotting current state...")
                    env.render(episode=i_eps, step=_, **cfg['Plot'])

            while(env.shuffled_agent_list):  # as long as there are agents
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
                    # ag = env._idx_to_ag(idx)  # agent object
                    ag = env.env[idx]

                    # select model and action
                    model = PreyModel if ag.kin == "Prey" else PredatorModel
                    action = ac.select_action(model=model, agent=ag,
                                              state=state)
                    # take a step
                    reward, state, done, idx = env.step(model=model,
                                                        agent=ag,
                                                        index=idx,
                                                        action=action)

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
            gens = deque()
            for a in env._agents_set:
                # for a in env._agents_dict.values():
                gens.append(a.generation)

            avg['mean_gens'].append(np.mean(gens))
            print("::: Mean generation: {}".format(avg['mean_gens'][-1]))
            gens.clear()  # free memory

            # prepare next step
            idx = env.shuffled_agent_list.pop()
            env.state = env.index_to_state(index=idx)
            state = env.state

        print("\n: Episode Runtime: {}".format(timestamp(return_obj=True) -
                                               eps_time))
        print("\n: optimizing now...")
        opt_time_start = timestamp(return_obj=True)
        l, mr = ac.finish_episode(model=PreyModel, optimizer=PreyOptimizer,
                                  history=env.history.Prey, gamma=0.05,
                                  return_means=True)
        print(":: [avg] Prey loss:\t{}\t Prey reward: {}"
              "".format(l.data[0], mr))
        avg['mean_prey_loss'].append(l.data[0])
        avg['mean_prey_rewards'].append(mr)

        l, mr = ac.finish_episode(model=PredatorModel,
                                  optimizer=PredatorOptimizer,
                                  history=env.history.Predator, gamma=0.05,
                                  return_means=True)
        print(":: [avg] Predator loss:\t{}\t Predator reward: {}"
              "".format(l.data[0], mr))
        avg['mean_pred_loss'].append(l.data[0])
        avg['mean_pred_rewards'].append(mr)

        print("\n: optimization time: {}".format(timestamp(return_obj=True) - opt_time_start))

    print("\n: Entire simulation runtime: {}".format(timestamp(return_obj=True) - inittime))

    # save everything
    save()


if __name__ == "__main__":
    main()
