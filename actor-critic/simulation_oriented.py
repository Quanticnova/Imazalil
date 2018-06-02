#!/usr/bin/env python

"""Simulation file for oriented agents."""

import warnings

import yaml
import numpy as np
import torch
import torch.optim as optim
import argparse as ap
from collections import deque

# make sure that the path to Imazalil/actor-critic is in $PYTHONPATH
from agents import OrientedPredator, OrientedPrey
import environment as Environment  # init needs to be called
from tools import timestamp, keyboard_interrupt_handler, sum_calls, chunkify
import actor_critic as ac  # init needs to be called

# setup argparse options ------------------------------------------------------
parser = ap.ArgumentParser(description="Command line options for the simulation script.")
parser.add_argument("--resume", type=str, default="",
                    help="resume simulation from given state")
parser.add_argument("--config", type=str, default="simulation_config.yml",
                    help="load the specified configuration file")


# load Args
args = parser.parse_args()
arg_res = args.resume  # resume filepath
arg_cfg = args.config  # configuration file

# load config files
with open(arg_cfg, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

# actor critic init settings --------------------------------------------------
# if gpu is to be used
mode = cfg['Network']['mode']
use_cuda = torch.cuda.is_available() if mode == 'gpu' else False

# if training or testing goal
goal = cfg['Sim']['goal']
training = True if goal == "training" else False

# make sure, that everything is ported to the gpu if one should be used
ac.init(mode=mode, goal=goal, policy_kind=cfg['Network']['kind'])

# Environment init settings ---------------------------------------------------
# simulation goal
Environment.init(goal=goal, policy_kind=cfg['Network']['kind'])

cfg_res = cfg['Sim']['resume_state_from']  # resume filepath

resume = None  # initialize
# if resume filepath is given, resume simulation from there
if arg_res and cfg_res:
    if arg_res != cfg_res:  # cmd line arg > cfg arg
        warnings.warn("resume is specified twice and doesn't match."
                      "\nGiven paths:\n\t{}\n\t{}\nResuming from {} now..."
                      "".format(arg_res, cfg_res, arg_res), UserWarning)
        resume = torch.load(arg_res)
    else:
        print(": [init] Resuming simulation from checkpoint {}".format(arg_res))
        resume = torch.load(arg_res)

elif arg_res:
    print(": [init] Resuming simulation from checkpoint {}".format(arg_res))
    resume = torch.load(arg_res)

elif cfg_res:
    print(": [init] Resuming simulation from checkpoint {}".format(cfg_res))
    resume = torch.load(cfg_res)

# Initialize Grid -------------------------------------------------------------
env = Environment.GridOrientedPPM(agent_types=(OrientedPredator, OrientedPrey),
                                  **cfg['Model'])
# env.seed(12345678)

# Initialize the policies and averages ----------------------------------------
Policy = ac.Policy if cfg['Network']['kind'] == 'fc' else ac.ConvPolicy
PreyModel = Policy(**cfg['Network']['layers'])
PredatorModel = Policy(**cfg['Network']['layers'])
# policy needed for env.step
Policy = {"OrientedPredator": PredatorModel,
          "OrientedPrey": PreyModel}

# use gpu if wanted -----------------------------------------------------------
if use_cuda:
    PreyModel.cuda()
    PredatorModel.cuda()

# averages
avg = {'mean_prey_rewards': deque(),  # in episode units
       'mean_pred_rewards': deque(),
       'mean_prey_loss': deque(),  # in episode units
       'mean_pred_loss': deque()}

# deque of episode/step pairs
epsbatch = deque()  # list of tuples of episode/step number

# simulation parameters
resume_pars = {'last_episode': 0}

if resume is not None:
    print(": [init] Found the following keys: {}".format(resume.keys()))
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

    if 'last_episode' in resume:
        resume_pars['last_episode'] += 1  # if resume, don't rerun the last step

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
              'mean_prey_rewards': avg['mean_prey_rewards'],
              'mean_pred_rewards': avg['mean_pred_rewards'],
              'mean_prey_loss': avg['mean_prey_loss'],
              'mean_pred_loss': avg['mean_pred_loss'],
              'epsbatch': epsbatch}


def save():
    """Save the current state of the simulation to a resumeable file."""
    print("\n: [sim] Storing the following keys: {}".format(save_state.keys()))
    filename = cfg['Sim']['save_state_to'] + "state_" + timestamp()
    ac.save_checkpoint(state=save_state, filename=filename)

    # clear episode/timestep/function call counter
    epsbatch.clear()


# main loop +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@keyboard_interrupt_handler(save=save, abort=None)
def main():
    """The main simulation loop."""
    inittime = timestamp(return_obj=True)  # initial datetime object
    batch = deque()  # initial batch deque to append values to

    # if no resume was given above, this starts from 0: -----------------------
    for i_eps in range(resume_pars['last_episode'], cfg['Sim']['episodes']):
        # add entry to save_dict
        save_state['last_episode'] = i_eps

        eps_time = timestamp(return_obj=True)  # record episode starting time
        print("\n: [env] Resetting now...")
        env.reset()  # returns None in this scenario

        # save data
        if i_eps % cfg['Sim']['save_state_every'] == 0:
            save()

        for ts in range(cfg['Sim']['steps']):  # ts = timestep ----------------
            print("\n:: [sim] Episode {}, Step {}".format(i_eps, ts))

            # if ts should be rendered:
            if cfg['Plot']['render']:
                if i_eps % cfg['Plot']['every'] == 0:  # plot every nth episode
                    print("::: [sim] Plotting current state...")
                    env.render(episode=i_eps, timestep=ts,
                               params=cfg['Plot']['params'])

            # run while there are agents to play with
            while(len(env.shuffled_agent_list) > 0 or len(env.eaten_prey) > 0):
                # take a step
                reward, state, done = env.step(policy=Policy,
                                               select_action=ac.select_action)

                if done or ((ts + 1) % cfg['Sim']['steps'] == 0):
                    print(":: [sim] Breakpoint reached " + 40 * "-")

                    break

            # data analysis and storage ---------------------------------------
            preys = env._agents_tuple.OrientedPrey
            preds = env._agents_tuple.OrientedPredator
            # food reserve
            mean_prey_fr = np.mean([ag.food_reserve for ag in preys])
            mean_pred_fr = np.mean([ag.food_reserve for ag in preds])

            # generation
            mean_prey_gen = np.mean([ag.generation for ag in preys])
            mean_pred_gen = np.mean([ag.generation for ag in preds])

            print("::: Preys:\t{}, mean food reserve: {:.3f}, mean generation:"
                  " {:.3f} ".format(len(preys), mean_prey_fr, mean_prey_gen))

            print("::: Predators:\t{}, mean food reserve: {:.3f}, mean "
                  "generation: {:.3f}".format(len(preds), mean_pred_fr,
                                              mean_pred_gen))

            # storage
            batch.append([mean_pred_fr, mean_prey_fr, len(preds), len(preys),
                          mean_pred_gen, mean_prey_gen])

            # back to simulation ----------------------------------------------
            if done or ((ts + 1) % cfg['Sim']['steps'] == 0):
                # save the episode number with number of steps
                epsbatch.append([(i_eps, ts), batch.copy()])

                # clear current batch deque
                batch.clear()
                break

            # create new shuffled agent list
            env.create_shuffled_agent_list()
        # ---------------------------------------------------------------------

        # append memory of remaining agents to history
        if training:
            for ag in env._agents_set:
                if ag.memory.Rewards:  # if that agent actually has memory
                    getattr(env.history, ag.kin).append(ag.memory)

        print("\n: [sim] Episode Runtime: {}"
              "".format(timestamp(return_obj=True) - eps_time))

        # optimization --------------------------------------------------------
        optimize = all([len(hist) > 0 for hist in env.history]) and training

        if optimize:
            print("\n: [ac] optimizing now...")
            opt_time_start = timestamp(return_obj=True)
            fcalls = {}
            l, mr, sa = ac.finish_episode(model=PreyModel,
                                          optimizer=PreyOptimizer,
                                          history=env.history.OrientedPrey,
                                          gamma=cfg['Network']['gamma'],
                                          return_means=True)
            fcalls['Prey'] = sa
            print(":: [ac] Prey loss:\t{}\t Prey reward: {}"
                  "".format(l.item(), mr))
            avg['mean_prey_loss'].append(l.item())
            avg['mean_prey_rewards'].append(mr)

            l, mr, sa = ac.finish_episode(model=PredatorModel,
                                          optimizer=PredatorOptimizer,
                                          history=env.history.OrientedPredator,
                                          gamma=cfg['Network']['gamma'],
                                          return_means=True)
            fcalls['Predator'] = sa  # sa = selected actions

            print(":: [ac] Predator loss:\t{}\t Predator reward: {}"
                  "".format(l.item(), mr))
            avg['mean_pred_loss'].append(l.item())
            avg['mean_pred_rewards'].append(mr)
            epsbatch[-1].append(fcalls)  # save the function calls
            # FIXME: function calls is still broken!! there is no information
            # about the timesteps! :(

            print("\n: [ac] optimization time: "
                  "{}".format(timestamp(return_obj=True) - opt_time_start))

        else:
            print("\n: [ac] Not enough history to train...")

    print("\n: [sim] Entire simulation runtime: "
          "{}".format(timestamp(return_obj=True) - inittime))

    # save everything
    save()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# actual execution of loop:
if __name__ == "__main__":
    main()
