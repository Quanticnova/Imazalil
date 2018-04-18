#!/home/corvus/docs/uni/bachelor_thesis/code/Imazalil/BA-venv/bin/python

"""This is an example simulation file for running a PPM simulation, using actor-critic method."""

import yaml
import numpy as np
import torch
import torch.optim as optim
import argparse as ap
from collections import deque

from agents import Predator, Prey
from environment_immortal import GridPPM
from tools import timestamp, keyboard_interrupt_handler
import actor_critic as ac  # also ensures GPU usage when available

# setup argparse options
parser = ap.ArgumentParser(description="Command line options for the simulation script.")
parser.add_argument("--resume", type=str, default="",
                    help="resume simulation from given state")
parser.add_argument("--config", type=str, default="simulation_config.yml",
                    help="load the specified configuration file")

# if gpu is to be used
use_cuda = torch.cuda.is_available()

# load Args
args = parser.parse_args()
arg_res = args.resume  # resume filepath
arg_cfg = args.config  # configuration file

# load config files.....
with open(arg_cfg, "r") as ymlfile:
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
inp = cfg['Model']['neighbourhood'] + 1  # number of inputs to handle
PreyModel = ac.Policy(inputs=inp)
PredatorModel = ac.Policy(inputs=inp)

# use gpu if available --------------------------------------------------------
if use_cuda:
    PreyModel.cuda()
    PredatorModel.cuda()

# averages
avg = {'mean_gens': deque(),  # in step units
       'mean_prey_rewards': deque(),  # in episode units
       'mean_pred_rewards': deque(),
       'mean_prey_loss': deque(),  # in episode units
       'mean_pred_loss': deque()}

# deque of episode/step pairs
epstep = deque()  # list of tuples of episode/step number

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

    if 'epstep' in resume:
        epstep = resume['epstep']

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
              'mean_pred_loss': avg['mean_pred_loss'],
              'epstep': epstep}


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
                    state = env.index_to_state(index=tmpidx, ag=ag)

                    if state[-1] is None:
                        state[-1] = int(ag.food_reserve)

                    env.state = state
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

                    # save the episode number with number of steps
                    epstep.append((i_eps, _))

                    break

            if done:
                break
            env.create_shuffled_agent_list()
            # print("::: Created new shuffled agents list with {} individuals."
            #      "".format(len(env.shuffled_agent_list)))
            # mean value output
            # gens = deque()
            # for a in env._agents_set:
                # for a in env._agents_dict.values():
            #    gens.append(a.generation)

            # avg['mean_gens'].append(np.mean(gens))
            # print("::: Mean generation: {}".format(avg['mean_gens'][-1]))
            # gens.clear()  # free memory

            # print food reserve
            mean_prey_fr = np.mean([ag.food_reserve for ag in env._agents_tuple.Prey])
            mean_pred_fr = np.mean([ag.food_reserve for ag in env._agents_tuple.Predator])
            print("::: Preys:\t{}, mean food reserve: {}".format(
                  len(env._agents_tuple.Prey), mean_prey_fr))
            print("::: Predators:\t{}, mean food reserve: {}".format(
                  len(env._agents_tuple.Predator), mean_pred_fr))

            # prepare next step
            idx = env.shuffled_agent_list.pop()
            env.state = env.index_to_state(index=idx)
            state = env.state

        # if actual timestep limit is reached append agent memory to history
        if not done:
            for ag in env._agents_set:
                if ag.memory.Rewards:  # if agent actually has memory
                    getattr(env.history, ag.kin).append(ag.memory)

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
