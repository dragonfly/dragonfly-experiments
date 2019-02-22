"""
  Tuning the hyperparameters of Gradient boosted classification on the sensorless drive
  data.
  -- kandasamy@cs.cmu.edu

  Dataset: Paschke et al. "Sensorlose Zustandsuberwachung an Synchronmotoren", 2013
"""

# pylint: disable=invalid-name
# pylint: disable=unexpected-keyword-arg

import pickle
# Local
from demos_real.skltree.skltree import gbc_train_and_validate, get_tr_dataset_size_from_z0

try:
  import os
  import sys
  curr_dir_path = os.path.dirname(os.path.realpath(__file__))
  data_path = os.path.join(curr_dir_path, 'sensorless_drive.p')
  if sys.version_info[0] < 3:
    DATA = pickle.load(open(data_path, 'rb'))
  else:
    DATA = pickle.load(open(data_path, 'rb'), encoding='latin1')
except IOError:
  print(('Could not load file %s. Make sure the file sensorless_drive.p is in the same ' +
         'directory as this file or pass the dataset to the function.')%(data_path))

MAX_TR_DATA_SIZE = 20000
MAX_VA_DATA_SIZE = 5000


def objective(z, x):
  """ Objective. """
  num_tr_data_to_use = get_tr_dataset_size_from_z0(z[0])
  return gbc_train_and_validate(x, DATA, num_tr_data_to_use,
                                MAX_TR_DATA_SIZE, MAX_VA_DATA_SIZE)

def cost(z):
  """ Compute cost. """
  num_tr_data_to_use = get_tr_dataset_size_from_z0(z[0])
  return num_tr_data_to_use / float(MAX_TR_DATA_SIZE)

