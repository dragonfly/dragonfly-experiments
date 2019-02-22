"""
  Function caller for LRG data.

  Cosmological Parameters and their domains and their true values
  Omega_k     : Spatial Curvature [-0.01, 0.03]         : 0
  Omega_Lambda: Dark Energy Fraction [0.7, 0.8]         : 0.762
  omega_C     : Cold Dark Matter Density [0.1, 0.105]   : 0.1045
  omega_B     : Baryon Density [0.02, 0.3]              : 0.02233
  n_s         : Scalar Spectral Index [0.95, 0.96]      : 0.951
  A_s         : Scalar Fluctuation Amplitude[0.65, 0.7] : 0.6845
  alpha       : Running of Spectral Index[-0.02, 0.01]  : 0.0
  b           : Galaxy Bias [1.0, 2.0]                  : 1.908
  Q_nl        : Nonlinear Correction [30.0, 31.0]       : 30.81

  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=unnecessary-lambda

from __future__ import print_function

import os
import time
import numpy as np

# Local imports
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller

class LRGOptFunctionCaller(EuclideanFunctionCaller):
  """ OptFunction for LRG data. """

  #TODO: Implement MF version of the function caller
  def __init__(self, path, in_debug_mode=False):
    """ Function caller for LRG Data  """
    self.path = path
    domain_bounds = np.array([[-0.01, 0.03],
                              [0.7, 0.8],
                              [0.1, 0.105],
                              [0.02, 0.3],
                              [0.95, 0.96],
                              [0.65, 0.7],
                              [-0.02, 0.01],
                              [1.0, 2.0],
                              [30.0, 31.0]])
    self.sf_func = lambda x: self.lrgLogLikl(x)
    super(LRGOptFunctionCaller, self).__init__(self.sf_func, domain_bounds,
                                               descr='lrg', vectorised=False,
                                               to_normalise_domain=True,
                                               noise_type='no_noise')

  def get_sf_func(self):
    """ Return sf function caller. """
    return self.sf_func

  def _eval_single_common_wrap_up(self, true_val, qinfo, noisy, caller_eval_cost):
    """ Override eval_single to write the result to a file.  """
    val, qinfo = super(LRGOptFunctionCaller, self)._eval_single_common_wrap_up( \
      true_val, qinfo, noisy, caller_eval_cost)
    if hasattr(qinfo, 'result_file'):
      self._write_result_to_file(val, qinfo.result_file)
    return val, qinfo

  @classmethod
  def _write_result_to_file(cls, result, file_name):
    """ Writes the result to the file name. """
    file_handle = open(file_name, 'w')
    file_handle.write(str(result))
    file_handle.close()

  def lrgLogLikl(self, evalPts):
    """ This computes the log likelihood of LRG. """

#     LOWESTLOGLIKLVAL = -10000.0
    LOWESTLOGLIKLVAL = -1e17

    # Prelims
    numParams = 9
    cur_dir = os.getcwd()

    # Create sim directory
    sim_dir = self.path

    # Executable file
#     binName = './bings13.out'
    binName = './binlov3'

    # Outfile
    fortOutFile = 'lOut_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'

    os.chdir(sim_dir)
    cmd = binName + ' '
    for j in range(numParams):
      #cmd = cmd + str(evalPts[j]) + ' '
      cmd = cmd + format(evalPts[j], '.10f') + ' '
    cmd = cmd + fortOutFile

#     print('cmd: %s' % cmd)
    os.system(cmd)

    with open(fortOutFile, 'r') as f:
      result = [line.replace(' ', '').rstrip().replace('.', '') for line in f.readlines()]

    if result[-1] == '-NAN' or result[-1] == 'NAN':
      logLiklVals = LOWESTLOGLIKLVAL
    else:
      logLiklVals = float(result[-1])

    os.system('rm -rf ' + fortOutFile)
    os.chdir(cur_dir)
    logLiklVals = logLiklVals/1e13

    return logLiklVals

