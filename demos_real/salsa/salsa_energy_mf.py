"""
  Demo on the Shrunk Additive Least Squares Approximations method (SALSA) on the energy
  appliances dataset.
  -- kandasamy@cs.cmu.edu

  SALSA is Kernel Ridge Regression with special kind of kernel structure and KRR is
  GP Regression in non-Bayesian settings. Hence, we will use kernel.py and gp_core.py from
  the gp directory for this demo.
  We tune for the following parameters in the method.
    - Kernel type: {se, matern0.5, matern1.5, matern2.5}
    - Additive Order: An integer in (1, d) where d is the dimension.
    - Kernel scale: float
    - Bandwidths for each dimension: float
    - L2 Regularisation penalty: float

  If you use this experiment, please cite the following paper.
    - Kandasamy K, Yu Y, "Additive Approximations in High Dimensional Nonparametric
      Regression via the SALSA", International Conference on Machine Learning, 2016.
"""

# pylint: disable=invalid-name

import numpy as np
import pickle
# Local
from demos_real.salsa.salsa import salsa_train_and_validate

# Load data
try:
  import os
  import sys
  curr_dir_path = os.path.dirname(os.path.realpath(__file__))
  data_path = os.path.join(curr_dir_path, 'energy_appliance.p')
  if sys.version_info[0] < 3:
    ENERGY_DATA = pickle.load(open(data_path, 'rb'))
  else:
    ENERGY_DATA = pickle.load(open(data_path, 'rb'), encoding='latin1')
except IOError:
  print(('Could not load file %s. Make sure the file energy_appliance.p is in the same ' +
         'directory as this file or pass the dataset to the function.')%(data_path))


# MAX_DATA_SIZE = 4000 # for debugging
# MAX_DATA_SIZE = 10000
# _MAX_TR_DATA_SIZE = 10000
# _MAX_VA_DATA_SIZE = 2500
MAX_DATA_SIZE = 8000
_MAX_TR_DATA_SIZE = 8000
_MAX_VA_DATA_SIZE = 2000


def _get_tr_dataset_size_from_z0(z0):
  """ Return the training dataset size. """
  return int(np.round(np.exp(z0)))


def salsa_compute_negative_validation_error(hyperparams, num_tr_data_to_use,
                                            X_tr=None, Y_tr=None, X_va=None, Y_va=None,
                                            *args, **kwargs):
  """ Computes the negative validation error. """
  # Check if training data is provided
  if X_tr is None:
    X_tr = ENERGY_DATA['train']['x'][:_MAX_TR_DATA_SIZE]
    Y_tr = ENERGY_DATA['train']['y'][:_MAX_TR_DATA_SIZE]
    X_va = ENERGY_DATA['vali']['x'][:_MAX_VA_DATA_SIZE]
    Y_va = ENERGY_DATA['vali']['y'][:_MAX_VA_DATA_SIZE]
  # Unpack the hyperparams
  kernel_type = hyperparams[0]
  add_order = hyperparams[1]
  kernel_scale = 10 ** hyperparams[2]
  bandwidths = 10 ** np.array(hyperparams[3])
  l2_reg = 10 ** hyperparams[4]
  # call the salsa function
  vali_err = salsa_train_and_validate(X_tr, Y_tr, X_va, Y_va, kernel_type, add_order,
                                      kernel_scale, bandwidths, l2_reg,
                                      num_tr_data_to_use, *args, **kwargs)
  return - vali_err


def objective(z, x):
  """ Objective for SALSA with energy dataset. """
  num_tr_data_to_use = _get_tr_dataset_size_from_z0(z[0])
  return salsa_compute_negative_validation_error(x, num_tr_data_to_use)


def cost(z):
  """ Compute cost. """
  num_tr_data_to_use = _get_tr_dataset_size_from_z0(z[0])
  return (num_tr_data_to_use / float(MAX_DATA_SIZE)) ** 3

