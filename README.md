# Imazalil
The code base for my Bachelor's thesis at Heidelberg University. The files in the topmost directory are used to run "regular" agent based PPM simulations. The more interesting parts are "hidden" in the [actor-critic](actor-critic) directory. 

### Thesis
The PDF of the thesis can be found [here](thesis/BA_Fabian_Krautgasser.pdf).

## Agent based model for Predator-Prey Simulations

Important files:
- `ABM.py`: grid environment and agents for the simulation
- `simulation.py`: the script to run the simulation
- `simconfig.yml`: the yamlfile to configure the simulation, e.g. gridsize etc.

To run a simulation, make sure that `simulation.py` and `simconfig.yml` are in the same directory and that a directory called `plots` exists. Then invoke
```
python3 simulation.py
```
because the simconfig file will be read in automatically.

## Actor-Critic Reinforcement Learning

Important files in *actor-critic* subdirectory:
- `actor_critic.py`: the policy classes as well as the means for action-selection and training the network (finish episode)
- `agents.py`: `Agent` base-class, and `Predator` and `Prey` classes which inherit from `Agent`, as well as the `OrientedPredator` and `OrientedPrey` classes.
- `environment.py`: `Environment` base-class, and the `GridPPM` and `GridOrientedPPM` classes that inherit from the base class. The environments provide all the interactions between agents, as well as the initial population of a grid. They also provide methods to run a simulation like `reset`, `step` and `render`.
- `simulation.py`: script to run the simulation; takes two optional arguments: `--config <configfile.yml>` and `--resume <simulation_snapshot.pth.tar>`, where the latter resumes from a certain point during training. The frequency of snapshot outputs can be set in the configuration file.
- `tools.py`: a collection of tools used for the simulation, like a input check for the internal functions, or a keyboard interrupt handler (very useful :D)
- `rcParams.yml`: a small file to specify some arguments for matplotlib to make plots look nicer
- `simulation_config.yml`: the acutal simulation config file. specifies things like gridsize, densities, rewards for the agents, sizes of layer in the NN (but not their topology), whether a gpu or cpu should be used.. 
- `simulation_oriented.py`: the script to run the oriented agent simulation, takes the same arguments as above
- `simulation_oriented_config.yml`: configuration file for the oriented PPM simulation; has additional parameters, hence the new file

Additional files are:
- `environment_slow_fR_decrease.py`: a version of the environment with a smaller decrease in foodreserve per turn, unfortunately hardcoded in the file. this issue was fixed in the orientedPPM version
- `simulation_slow_fR_decrease.py`: needed because the different environment file must be loaded initially

To run a simulation, invoke
```
python3 simulation_oriented.py --config simulation_oriented_config.yml
```

The location of the plots can be specified in the config file.

The output are png files of the grid configurations, and additionally every N episodes a `.pth.tar` snapshot containing the current NN weights, and more simulation data like rewards, population sizes, etc.
