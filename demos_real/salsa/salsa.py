"""
  Demo on the Shrunk Additive Least Squares Approximations method (SALSA) for high
  dimensional regression.
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

from dragonfly.gp.kernel import ESPKernelSE, ESPKernelMatern
from dragonfly.gp.gp_core import GP


def get_salsa_kernel_from_params(kernel_type, add_order, kernel_scale, bandwidths,
                                 problem_dim):
  """ Returns the kernel for SALSA. """
  if kernel_type == 'se':
    return ESPKernelSE(problem_dim, kernel_scale, add_order, bandwidths)
  elif kernel_type.startswith('matern'):
    nu = float(kernel_type[-3:])
    nu_vals = [nu] * problem_dim
    return ESPKernelMatern(problem_dim, nu_vals, kernel_scale, add_order, bandwidths)
  else:
    raise ValueError('Unknown kernel type %s.'%(kernel_type))


def get_salsa_estimator_from_data_and_hyperparams(X_tr, Y_tr, kernel_type, add_order,
                                                  kernel_scale, bandwidths, l2_reg):
  """ Returns an estimator using the data. """
  problem_dim = np.array(X_tr).shape[1]
  kernel = get_salsa_kernel_from_params(kernel_type, add_order, kernel_scale,
                                        bandwidths, problem_dim)
  def _get_mean_func(_mean_func_const_value):
    """ Returns the mean function from the constant value. """
    return lambda x: np.array([_mean_func_const_value] * len(x))
  mean_func = _get_mean_func(np.median(Y_tr))
  gp_obj = GP(X_tr, Y_tr, kernel, mean_func, l2_reg)
  estimator = lambda x_test: gp_obj.eval(x_test, uncert_form='none')[0]
  return estimator


def salsa_train_and_validate(X_tr, Y_tr, X_va, Y_va, kernel_type, add_order,
                             kernel_scale, bandwidths, l2_reg,
                             num_tr_data_to_use=None, num_va_data_to_use=None,
                             shuffle_data_when_using_a_subset=True):
  """ Train and return validation error for SALSA. """
  # pylint: disable=too-many-arguments
  # Prelims
  if num_tr_data_to_use is None:
    num_tr_data_to_use = len(X_tr)
  if num_va_data_to_use is None:
    num_va_data_to_use = len(X_va)
  num_tr_data_to_use = int(num_tr_data_to_use)
  num_va_data_to_use = int(num_va_data_to_use)
  if num_tr_data_to_use < len(X_tr) and shuffle_data_when_using_a_subset:
    X_tr = np.copy(X_tr)
    np.random.shuffle(X_tr)
  if num_va_data_to_use < len(X_va) and shuffle_data_when_using_a_subset:
    X_va = np.copy(X_va)
    np.random.shuffle(X_va)
  # Get relevant subsets
  X_tr = X_tr[:num_tr_data_to_use]
  Y_tr = Y_tr[:num_tr_data_to_use]
  X_va = X_va[:num_va_data_to_use]
  Y_va = Y_va[:num_va_data_to_use]
  print('Training with %d data, and validating with %d data.'%(
        len(X_tr), len(X_va)))
  # Get estimator
  salsa_estimator = get_salsa_estimator_from_data_and_hyperparams(X_tr, Y_tr,
                      kernel_type, add_order, kernel_scale, bandwidths, l2_reg)
  valid_predictions = salsa_estimator(X_va)
  valid_diffs = (valid_predictions - Y_va) ** 2
  avg_valid_err = valid_diffs.mean()
  print('Trained with %d data, and validated with %d data. err=%0.4f'%(
        len(X_tr), len(X_va), avg_valid_err))
  return avg_valid_err


