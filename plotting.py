"""
  Reads either pickle or mat files and plots the results.
  -- syiblet@andrew.cmu.edu
  -- kvysyara@andrew.cmu.edu
  -- kandasamy@cs.cmu.edu

  Usage:
  python plotting.py --filelist <file containing list of pickle or mat file paths>
  python plotting.py --file     <pickle or mat file path>
"""

from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import os
import numpy as np
# Local
# from utils.plot_utils import COLOURS, plot_results, get_plot_options, \
#                              load_results, get_file_paths
from dragonfly.utils.plot_utils import COLOURS, plot_results, get_plot_options, \
                                       load_results, get_file_paths


syn_funcs = {'hartmann3': (3.86278, 3),
             'hartmann6': (3.322368, 6),
             'currin_exp': (13.7986850, 2),
             'branin': (-0.39788735773, 2),
             'borehole': (309.523221, 8),
             'park1': (25.5872304, 4),
             'park2': (5.925698, 4)
            }

def get_true_maxval(study_name):
  ''' Returns true max values for high dim synthetic functions. '''
  name = study_name.split('-')[0]
  if name not in syn_funcs:
    return None
  if len(study_name.split('-')) == 1:
    return syn_funcs[name][0]
  group_dim = int(study_name.split('-')[1])
  max_val, domain_dim = syn_funcs[name]
  num_groups = int(group_dim/domain_dim)
  return max_val*num_groups


def main():
  """ Main function. """
#   plot_order = ['rand', 'ga', 'dragonfly']
#   plot_order = ['rand', 'pdoo', 'dragonfly']
  plot_order = ['rand', 'ga']
#   # Cartesian product experiments
#   plot_order = ['rand', 'ga', 'hyperopt', 'smac', 'gpyopt', 'dragonfly', 'dragonfly-mf']
#     # Real experiments
#   plot_order = ['rand', 'ga', 'smac', 'gpyopt', 'dragonfly', 'dragonfly-mf']
#     # Euclidean Methods
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'smac', 'spearmint', 'gpyopt', 'dragonfly']
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'smac', 'gpyopt', 'dragonfly']
#   plot_order = ['rand', 'pdoo', 'hyperopt', 'gpyopt', 'dragonfly']

  # Load options and results
  options = get_plot_options()
  if options.filelist != '':
    file_paths = get_file_paths(options.filelist)
  elif options.file != '':
    file_paths = [os.path.realpath(os.path.abspath(options.file))]
  else:
    raise ValueError('Missing Filelist.')
  to_plot_legend = True
  options.legend_location = 3
  results = load_results(file_paths)
  # Ancillary info for the plot
  study_name = str(np.asscalar(results['study_name']))
  # Check if noisy or not
  if 'is_noisy' in results and results['is_noisy'].flatten()[0]:
    is_noisy = True
  else:
    is_noisy = False
  # GEt true max value
  true_maxval = get_true_maxval(study_name)
  y_bounds = None
  if study_name in ['borehole_6', 'park1_3', 'park2_3', 'park2_4', 'hartmann6_4',
                    'syn_cnn_1', 'syn_cnn_2', 'park1_constrained',
                    'hartmann3_constrained', 'borehole_constrained']:
    y_label = 'Maximum Value'
    x_label = 'Expended Capital'
  else:
    y_label = 'Simple Regret'
    x_label = 'Number of Evaluations'
  # title
  if not hasattr(options, 'title') or options.title is None:
    if study_name == 'borehole':
      plot_title = r'Borehole $(d=8)$'
      if not len(results['methods']) == 7:
        to_plot_legend = True
        options.legend_location = 3
    elif study_name == 'branin':
      plot_title = r'Branin $(d=2)$'
      to_plot_legend = True
      options.legend_location = 3
    elif study_name == 'branin-40':
      plot_title = r'Branin$\times$20 $(d=40)$'
    elif study_name == 'hartmann3':
      plot_title = r'Hartmann3 $(d=3)$'
      to_plot_legend = True
      options.legend_location = 1
    elif study_name == 'park1':
      plot_title = r'Park1 $(d=4)$'
    elif study_name == 'park2':
      plot_title = r'Park2 $(d=4)$'
    elif study_name == 'hartmann6':
      plot_title = r'Hartmann6 $(d=6)$'
      if 'ei' in plot_order:
        to_plot_legend = True
        options.legend_location = 3
    elif study_name == 'hartmann-20':
      plot_title = r'Hartmann6$\times$3 $(d=20)$'
      true_maxval = syn_funcs['hartmann6'][0] * 3
    elif study_name == 'branin-14':
      plot_title = r'Branin$\times7$ $(d=14)$'
      if not is_noisy:
        y_bounds = [20, 800]
    elif study_name == 'borehole-32':
      plot_title = r'Borehole$\times4$ $(d=32)$'
    elif study_name == 'hartmann6-42':
      plot_title = r'Hartmann6$\times7$ $(d=42)$'
    elif study_name == 'park2-20':
      plot_title = r'Park2$\times$5 $(d=20)$'
    elif study_name == 'park1-12':
      plot_title = r'Park1$\times$3 $(d=12)$'
    elif study_name == 'hartmann6':
      plot_title = r'Hartmann6 $(d=6)$'
    elif study_name == 'hartmann3-18':
      plot_title = r'Hartmann3$\times$6 $(d=18)$'
      to_plot_legend = True
      options.legend_location = 3
    elif study_name == 'park2-24':
      plot_title = r'Park2$\times$6 $(d=24)$'
    elif study_name == 'park2-40':
      plot_title = r'Park2$\times$10 $(d=40)$'
    elif study_name == 'branin-50':
      plot_title = r'Branin$\times$25 $(d=50)$'
    elif study_name == 'borehole-96':
      plot_title = r'Borehole$\times$12 $(d=96)$'
    elif study_name == 'park1-108':
      plot_title = r'Park1$\times$27 $(d=108)$'
    # Non-euclidean domains
    elif study_name == 'borehole_6':
      plot_title = r'Borehole_6 $(d=8)$'
      to_plot_legend = True
      options.legend_location = 4
      y_bounds = [100, 295]
    elif study_name == 'hartmann6_4':
      plot_title = r'Hartmann6_4 $(d=6)$'
      to_plot_legend = True
      options.legend_location = 4
    elif study_name == 'park1_3':
      plot_title = r'Park1_3 $(d=4)$'
      y_bounds = [17, 22.5]
    elif study_name == 'park2_3':
      plot_title = r'Park2_3 $(d=5)$'
    elif study_name == 'park2_4':
      plot_title = r'Park2_4 $(d=5)$'
    elif study_name == 'syn_cnn_1':
      plot_title = r'syn_cnn_1'
    elif study_name == 'syn_cnn_2':
      plot_title = r'syn_cnn_2'
    elif study_name == 'park1_constrained':
      plot_title = r'Park1-Constrained $(d=4)$'
      to_plot_legend = True
      options.legend_location = 4
    elif study_name == 'hartmann3_constrained':
      y_bounds = [1.3, 3.8]
      plot_title = r'Hartmann3-Constrained $(d=3)$'
      to_plot_legend = True
      options.legend_location = 4
    elif study_name == 'borehole_constrained':
      plot_title = r'Borehole-Constrained $(d=6)$'
    elif study_name == 'lrg':
      plot_title = r'Luminous Red Galaxies $(d=9)$'
      to_plot_legend = True
      options.legend_location = 4
      y_bounds = [-1400, -999]
    elif study_name == 'supernova':
      plot_title = r'Type Ia Supernova $(d=3)$'
      x_label = 'Wall Clock Time (hours)'
      to_plot_legend = True
      options.legend_location = 4
      y_bounds = [-0.25, 0.075]
      x_label = 'Wall Clock Time (hours)'
    elif study_name == 'salsa':
      plot_title = r'SALSA, Energy Appliances $\,(d=30)$'
      to_plot_legend = True
      options.legend_location = 1
      x_label = 'Wall Clock Time (hours)'
      y_label = 'Validation Error'
    elif study_name == 'rfrnews':
      plot_title = r'Random Forest Regression, News $\,(d=6)$'
      x_label = 'Wall Clock Time (hours)'
      to_plot_legend = True
      options.legend_location = 3
    elif study_name == 'gbrnaval':
      plot_title = r'Gradient Boosted Regression, Naval $\,(d=7)$'
      y_bounds = [1e-5, 1e0]
      x_label = 'Wall Clock Time (hours)'
      to_plot_legend = False
      options.legend_location = 3
    elif study_name == 'gbrprotein':
      plot_title = r'Gradient Boosted Regression, Protein $\,(d=7)$'
      x_label = 'Wall Clock Time (hours)'
      to_plot_legend = False
      options.legend_location = 3
      y_bounds = [0.5, 2e0]
    elif study_name == 'infra':
      plot_title = r'Real Time Stream Processing $\,(d=37)$'
      x_label = 'Wall Clock Time (hours)'
      y_label = 'Latency (ms)'
      to_plot_legend = True
      options.legend_location = 3
    else:
      plot_title = ''
  # Method legend dictionary
  method_legend_colour_marker_dict = {
    # Packages
    "rand": {'legend':'RAND', 'colour':COLOURS['black'], 'marker':'s', 'linestyle':'-'},
    "ga": {'legend':'EA', 'colour':COLOURS['red'], 'marker':'*', 'linestyle':'-'},
    "pdoo": {'legend':'PDOO', 'colour':COLOURS['red'], 'marker':'>', 'linestyle':'-'},
    "hyperopt": {'legend':'HyperOpt', 'colour':COLOURS['orange'], 'marker':'1',
                 'linestyle':'-'},
    "gpyopt": {'legend':'GPyOpt', 'colour':COLOURS['green'], 'marker':'+',
               'linestyle':'-'},
    "smac": {'legend':'SMAC', 'colour':COLOURS['magenta'], 'marker':'x',
             'linestyle':'-'},
    "spearmint": {'legend':'Spearmint', 'colour':COLOURS['yellow'], 'marker':'^',
                  'linestyle':'-'},
    "dragonfly": {'legend':'Dragonfly', 'colour':COLOURS['blue'], 'marker':'o',
                  'linestyle':'-'},
    "dragonfly-mf": {'legend':'Dragonfly+MF', 'colour':COLOURS['cyan'], 'marker':'d',
                     'linestyle':'-'},
    # Custom methods
    "ml": {'legend':'ML', 'colour':COLOURS['red'], 'marker':'>', 'linestyle':'-'},
    "post_sampling": {'legend':'PS', 'colour':COLOURS['green'], 'marker':'s',
                      'linestyle':'-'},
    "ml+post_sampling": {'legend':'ML+PS', 'colour':COLOURS['blue'], 'marker':'.',
                         'linestyle':'-'},
    # Acquisition
    "pi": {'legend':'GP-PI', 'colour':COLOURS['orange'], 'marker':'>', 'linestyle':'-'},
    "ei": {'legend':'GP-EI', 'colour':COLOURS['green'], 'marker':'^',
                  'linestyle':'-'},
    "ttei": {'legend':'TTEI', 'colour':COLOURS['magenta'], 'marker':'1',
                 'linestyle':'-'},
    "ucb": {'legend':'GP-UCB', 'colour':COLOURS['red'], 'marker':'x', 'linestyle':'-'},
    "add_ucb": {'legend':'Add-GP-UCB', 'colour':COLOURS['cyan'], 'marker':'s',
                     'linestyle':'-'},
    "ts": {'legend':'TS', 'colour':COLOURS['blue'], 'marker':'+', 'linestyle':'-'},
  }
  print(results['methods'])
  plot_results(results, plot_order, method_legend_colour_marker_dict, x_label, y_label,
               to_plot_legend=to_plot_legend, true_maxval=true_maxval,
               plot_title=plot_title, options=options, y_bounds=y_bounds)


if __name__ == '__main__':
  main()

