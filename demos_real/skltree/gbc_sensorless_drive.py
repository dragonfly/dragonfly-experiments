"""
  Tuning the hyperparameters of Gradient boosted classification on the sensorless drive
  data.
  -- kandasamy@cs.cmu.edu

  Dataset: Paschke et al. "Sensorlose Zustandsuberwachung an Synchronmotoren", 2013
"""

# pylint: disable=invalid-name


import numpy as np
# Local
from demos_real.skltree.gbc_sensorless_drive_mf import objective as objective_mf
from demos_real.skltree.gbc_sensorless_drive_mf import MAX_TR_DATA_SIZE


def objective(x):
  """ Define the objective. """
  return objective_mf([np.log(MAX_TR_DATA_SIZE)], x)

