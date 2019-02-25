## Dragonfly Experiments


To get started, first follow the instructions in the
[Dragonfly repository](dragonfly.github.io)
to insall Dragonfly.

Then clone this repository and copy it inside the
[`examples`](https://github.com/dragonfly/dragonfly/tree/master/examples)
directory in dragonfly,
i.e. at the same level as the `salsa`, `synthetic` and `supernova` directories.
The Dragonfly `experiments` directory does not be in the root of the Dragonfly 
repository, to run the demos.
```bash
$ cp -r <path_to_dragonfly_root>/examples ./
$ cd examples
$ git clone https://github.com/dragonfly/dragonfly-experiments.git
```
Once this is done, set relevant environment variables.
This will need to be done at the beginning of each session.
```
$ HOME_PATH=$(pwd)
$ PATH=$PATH:$HOME_PATH
$ PYTHONPATH=$HOME_PATH
```

&nbsp;


This repository provides three Python scripts to run experiments.

1. [`euclidean/run_euclidean_experiments.py`](euclidean/run_euclidean_experiments.py):
Executes the experiments on Euclidean domains. This include the synthetic experiments
on the maximum likelihood problem on luminous red galaxies.

2. [`non_euclidean/run_non_euclidean_synthetic_experiments.py`](euclidean/run_non_euclidean_synthetic_experiments.py):
Executes the synthetic experiments on non-Euclidean domains. 

3. [`non_euclidean/run_non_euclidean_realtime_experiments.py`](euclidean/run_non_euclidean_realtime_experiments.py):
Executes the model selection and astrophysical maximum
likelihood experiments on non-Euclidean domains. 

To run these experiments, simply `cd` into the relevant directory and execute the
script. For example, the first script above can be run via the following commands.
You may select the specific experiment in the script.
```bash
(env)$ cd dragonfly-experiments/euclidean
(env)$ python e1_synthetic_bo_packages.py
```
