"""
  Synthetic experiments for optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import Namespace
from time import time, clock

# Dragonfly imports
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.experiment_caller import get_multifunction_caller_from_config
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.opt.blackbox_optimiser import blackbox_opt_args
from dragonfly.opt.cp_ga_optimiser import cpga_opt_args
from dragonfly.opt.gp_bandit import get_all_cp_gp_bandit_args, \
                                    get_all_mf_cp_gp_bandit_args
from dragonfly.opt.random_optimiser import cp_random_optimiser_args
from dragonfly.utils.option_handler import load_options
from dragonfly.utils.reporters import get_reporter
# Examples from dragonfly/examples
from synthetic.borehole_6.borehole_6_mf import borehole_6_mf
from synthetic.borehole_6.borehole_6_mf import cost as cost_borehole_6_mf
from synthetic.branin.branin_mf import branin_mf
from synthetic.branin.branin_mf import cost as cost_branin_mf
from synthetic.hartmann3_2.hartmann3_2_mf import hartmann3_2_mf
from synthetic.hartmann3_2.hartmann3_2_mf import cost as cost_hartmann3_2_mf
from synthetic.hartmann6_4.hartmann6_4_mf import hartmann6_4_mf
from synthetic.hartmann6_4.hartmann6_4_mf import cost as cost_hartmann6_4_mf
from synthetic.park2_4.park2_4_mf import park2_4_mf
from synthetic.park2_4.park2_4_mf import cost as cost_park2_4_mf
from synthetic.park2_3.park2_3_mf import park2_3_mf
from synthetic.park2_3.park2_3_mf import cost as cost_park2_3_mf
from synthetic.park1_3.park1_3_mf import park1_3_mf
from synthetic.park1_3.park1_3_mf import cost as cost_park1_3_mf
# Local
from cp_opt_method_evaluator import CPOptMethodEvaluator

# Experiment Parameters ==============================================================
# We won't be changing these parameters much.
# IS_DEBUG = True
IS_DEBUG = False
NUM_TRIALS = 10
NUM_WORKERS = 1
MAX_CAPITAL = 200
TIME_DISTRO = 'const'
SAVE_RESULTS_DIR = './syn_results'

# Choose if evals are noisy -----------------------------------------------------------
# NOISY_EVALS = True
NOISY_EVALS = False

# Choose experiment here --------------------------------------------------------------
# STUDY_NAME = 'borehole_6'
# STUDY_NAME = 'hartmann6_4'
# STUDY_NAME = 'park1_3'
# STUDY_NAME = 'park2_3'
# STUDY_NAME = 'park2_4'
# STUDY_NAME = 'syn_cnn_2'

# Choose methods here ------------------------------------------------------------
METHODS = ['rand', 'ga', 'dragonfly']
# These packages need to be installed. SMAC does not work with Python2 and other packages
# have not been tested with Python3.
# METHODS = ['spearmint']
# METHODS = ['gpyopt']
# METHODS = ['hyperopt']
# METHODS = ['smac']


if not os.path.exists(SAVE_RESULTS_DIR):
    os.makedirs(SAVE_RESULTS_DIR)

def get_prob_params():
  """ Returns the problem parameters. """
  prob = Namespace()
  prob.study_name = STUDY_NAME
  if IS_DEBUG:
    prob.num_trials = 3
    prob.max_num_evals = 10
  else:
    prob.num_trials = NUM_TRIALS
    prob.max_num_evals = MAX_NUM_EVALS
  # Common
  prob.time_distro = TIME_DISTRO
  prob.num_workers = NUM_WORKERS
  _study_params = {
    'branin': ('../../../synthetic/branin/config_mf.json',
               branin_mf, cost_branin_mf, 0.1, 0, 1),
    'hartmann3_2': ('../../../synthetic/hartmann3_2/config_mf.json',
                    hartmann3_2_mf, cost_hartmann3_2_mf, 0.1, 0, 1),
    'hartmann6_4': ('../../../synthetic/hartmann6_4/config_mf.json',
                    hartmann6_4_mf, cost_hartmann6_4_mf, 0.1, 0, 1),
    'borehole_6': ('../../../synthetic/borehole_6/config_mf.json',
                   borehole_6_mf, cost_borehole_6_mf, 1, 0, 1),
    'park2_4': ('../../../synthetic/park2_4/config_mf.json',
                park2_4_mf, cost_park2_4_mf, 0.3, 0, 1),
    'park2_3': ('../../../synthetic/park2_3/config_mf.json',
                park2_3_mf, cost_park2_3_mf, 0.1, 0, 1),
    'park1_3': ('../../../synthetic/park1_3/config_mf.json',
                park1_3_mf, cost_park1_3_mf, 0.5, 0, 1),
    }
  (domain_config_file, raw_func, raw_fidel_cost_func, _fc_noise_scale,
   _initial_pool_size, _) = _study_params[prob.study_name]
  # noisy
  prob.noisy_evals = NOISY_EVALS
  if NOISY_EVALS:
    noise_type = 'gauss'
    noise_scale = _fc_noise_scale
  else:
    noise_type = 'no_noise'
    noise_scale = None
  # Create domain, function_caller and worker_manager
  config = load_config_file(domain_config_file)
  func_caller = get_multifunction_caller_from_config(raw_func, config,
                  raw_fidel_cost_func=raw_fidel_cost_func, noise_type=noise_type,
                  noise_scale=noise_scale)
  # Set max_capital
  if hasattr(func_caller, 'fidel_cost_func'):
    prob.max_capital = prob.max_num_evals * \
                       func_caller.fidel_cost_func(func_caller.fidel_to_opt)
  else:
    prob.max_capital = prob.max_num_evals
  # Store everything in prob
  prob.func_caller = func_caller
  prob.worker_manager = SyntheticWorkerManager(prob.num_workers,
                                               time_distro='caller_eval_cost')
  prob.save_file_prefix = prob.study_name + ('-debug' if IS_DEBUG else '')
  prob.methods = METHODS
  prob.save_results_dir = SAVE_RESULTS_DIR
  prob.reporter = get_reporter('default')
  # evaluation options
  prob.evaluation_options = Namespace(prev_eval_points='none',
                                      initial_pool_size=_initial_pool_size)
  return prob


def get_method_options(prob, capital_type):
  """ Returns the method options. """
  methods = prob.methods
  all_method_options = {}
  for meth in methods:
    if meth == 'rand':
      curr_options = load_options(cp_random_optimiser_args)
    elif meth == 'ga':
      curr_options = load_options(cpga_opt_args)
    elif meth.startswith('dragonfly'):
      if meth.startswith('dragonfly-mf'):
        curr_options = load_options(get_all_mf_cp_gp_bandit_args())
      else:
        curr_options = load_options(get_all_cp_gp_bandit_args())
      meth_parts = meth.split('_')
      if len(meth_parts) == 2:
        curr_options.acq_opt_method = meth_parts[-1]
    elif meth in ['hyperopt', 'gpyopt', 'smac', 'spearmint']:
      curr_options = load_options(blackbox_opt_args)
      curr_options = Namespace(redo_evals_for_true_val=True)
    else:
      raise ValueError('Unknown method %s.'%(meth))

    if meth == 'spearmint':
#       curr_options.exp_dir = '/home/karun/boss/e3_cp/Spearmint/' + \
#                              prob.study_name.split('-')[0]
#       curr_options.pkg_dir = '/home/karun/Spearmint/spearmint'  
      curr_options.exp_dir = '/zfsauton3/home/kkandasa/projects/Boss/boss/e3_cp/Spearmint/' + \
                             prob.study_name.split('-')[0]
      curr_options.pkg_dir = '~/projects/Boss/Spearmint/spearmint'
      curr_options.pkg_dir = '/zfsauton3/home/kkandasa/projects/Boss/Spearmint/spearmint'
    curr_options.capital_type = capital_type
    all_method_options[meth] = curr_options
  return all_method_options


def main():
  """ Main function. """
  prob = get_prob_params()
  method_options = get_method_options(prob, 'return_value')
  # construct evaluator
  evaluator = CPOptMethodEvaluator(study_name=prob.study_name,
                                   func_caller=prob.func_caller,
                                   worker_manager=prob.worker_manager,
                                   max_capital=prob.max_capital,
                                   methods=prob.methods,
                                   num_trials=prob.num_trials,
                                   save_dir=prob.save_results_dir,
                                   evaluation_options=prob.evaluation_options,
                                   save_file_prefix=prob.save_file_prefix,
                                   method_options=method_options,
                                   reporter=prob.reporter)
  # run trials
  start_realtime = time()
  start_cputime = clock()
  evaluator.run_trials()
  end_realtime = time()
  end_cputime = clock()
  prob.reporter.writeln('')
  prob.reporter.writeln('realtime taken: %0.6f'%(end_realtime - start_realtime))
  prob.reporter.writeln('cputime taken: %0.6f'%(end_cputime - start_cputime))


if __name__ == '__main__':
  main()

