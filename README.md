## Dragonfly Experiments

** Clone dragonfly-experiments repository **
```bash
$ git clone https://github.com/dragonfly/dragonfly-experiments.git
```

** Install dragonfly in Python Virtual Environment:**
Dragonfly can be pip installed in a python virtualenv, by following the steps below.
```bash
$ virtualenv env        # Python2
$ source env/bin/activate
(env)$ pip install numpy
(env)$ pip install scikit-learn scipy matplotlib
(env)$ pip install git+https://github.com/dragonfly/dragonfly.git
```

### Euclidean Domain Synthetic Experiments
```bash
(env)$ cd dragonfly-experiments/euclidean
(env)$ python e1_synthetic_bo_packages.py
```

### Non Euclidean Domain Synthetic Experiments
```bash
(env)$ cd dragonfly-experiments/non_euclidean
(env)$ python e3_mf_synthetic.py
```

### Non Euclidean Domain Real Experiments
```bash
(env)$ cd dragonfly-experiments/non_euclidean
(env)$ python e3_mf_real.py
```
