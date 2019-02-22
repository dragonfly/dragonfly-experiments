"""
  A synthetic objective function for the electrolyte design example.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name


def electrolyte_synthetic_function(x):
  """ Computes the electrolyte objective. """
  x1 = float(x[0]) * x[4]
  x2 = float(x[1]) * x[5]
  x3 = float(x[2]) * x[6]
  x4 = float(x[3][0]) * x[7][0]
  x5 = float(x[3][1]) * x[7][1]
  x6 = float(x[3][2]) * x[7][2]
  x7 = float(x[3][3]) * x[7][3]
  s1 = x[8][0]
  s2 = x[8][1]
  s3 = x[8][2]
  s0 = 1 - (s1 + s2 + s3)
  ret0 = s0 * (x1 + 2*x2 + 3*x4**2)
  ret1 = s1 * (x2**1.5 + 1.3*x6 + x7)
  ret2 = s2 * (x3*x5 + x1*x2 + x7*x6)
  ret3 = s3 * (x1*x2 + 1.3*x1*x2 + x4**1.2)
  return ret0 + ret1 + ret2 + ret3


def objective(x):
  """ Compute objective. """
  return electrolyte_synthetic_function(x)

