## Dragonfly Experiments


To get started, first follow the instructions in the
[Dragonfly repository](dragonfly.github.io)
to insall Dragonfly.

Then, clone this repository.
```bash
$ git clone https://github.com/dragonfly/dragonfly-experiments.git
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

The experiments use the examples in the
[dragonfly/examples](https://github.com/dragonfly/dragonfly/tree/master/examples)
directory. You need to specify the path to this directory via the
`DRAGONFLY_EXPERIMENTS_DIR` variable at the beginning of each script.

To run these experiments, simply `cd` into the relevant directory and execute the
script. For example, the second script above can be run via the following commands.
You may select the specific experiment in the script.
```bash
(env)$ cd dragonfly-experiments/euclidean
(env)$ python run_non_euclidean_synthetic_experiments.py
```

Once the experiment is done, you may use
[plotting.py](plotting.py) to plot the results.
```bash
(env)$ python plotting.py --file non_eucildean/syn_results/<file_name>.mat
```


