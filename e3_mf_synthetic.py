"""
  Synthetic experiments for optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member

import os
from argparse import Namespace
from time import time, clock

# Local demo functions
from demos_synthetic.borehole_6.borehole_6_mf import borehole_6_mf
from demos_synthetic.borehole_6.borehole_6_mf import cost as cost_borehole_6_mf
from demos_synthetic.branin.branin_mf import branin_mf
from demos_synthetic.branin.branin_mf import cost as cost_branin_mf
from demos_synthetic.hartmann3_2.hartmann3_2_mf import hartmann3_2_mf
from demos_synthetic.hartmann3_2.hartmann3_2_mf import cost as cost_hartmann3_2_mf
from demos_synthetic.hartmann6_4.hartmann6_4_mf import hartmann6_4_mf
from demos_synthetic.hartmann6_4.hartmann6_4_mf import cost as cost_hartmann6_4_mf
from demos_synthetic.park2_4.park2_4_mf import park2_4_mf
from demos_synthetic.park2_4.park2_4_mf import cost as cost_park2_4_mf
from demos_synthetic.park2_3.park2_3_mf import park2_3_mf
from demos_synthetic.park2_3.park2_3_mf import cost as cost_park2_3_mf
from demos_synthetic.park1_3.park1_3_mf import park1_3_mf
from demos_synthetic.park1_3.park1_3_mf import cost as cost_park1_3_mf

# Local
from cp_opt_method_evaluator import CPOptMethodEvaluator
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.experiment_caller import get_multifunction_caller_from_config
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.opt.blackbox_optimiser import blackbox_opt_args
from dragonfly.opt.cp_ga_optimiser import cpga_opt_args
from dragonfly.opt.gp_bandit import get_all_cp_gp_bandit_args, get_all_mf_cp_gp_bandit_args
from dragonfly.opt.random_optimiser import cp_random_optimiser_args
from dragonfly.utils.option_handler import load_options
from dragonfly.utils.reporters import get_reporter


# Experiment Parameters ==============================================================

# IS_DEBUG = True
IS_DEBUG = False

NOISY_EVALS = True
# NOISY_EVALS = False

NUM_TRIALS = 20

# STUDY_NAME = 'hartmann3_2'
# STUDY_NAME = 'hartmann6_4'
STUDY_NAME = 'park2_3'
# STUDY_NAME = 'park1_3'
# STUDY_NAME = 'borehole_6'
# STUDY_NAME = 'park2_4'
# STUDY_NAME = 'syn_cnn_1'
# STUDY_NAME = 'syn_cnn_2'

# Other problem parameters - won't be changing these much
NUM_WORKERS = 1
MAX_NUM_EVALS = 200
TIME_DISTRO = 'const'
SAVE_RESULTS_DIR = 'syn_results'

# METHODS = ['smac']
# METHODS = ['rand', 'ga', 'hyperopt', 'gpyopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'dragonfly-mf', 'dragonfly']

# METHODS = ['rand', 'ga']
# METHODS = ['hyperopt']
# METHODS = ['spearmint']
# METHODS = ['dragonfly', 'smac']
# METHODS = ['smac']
# METHODS = ['gpyopt']
# METHODS = ['spearmint']
# METHODS = ['dragonfly', 'rand', 'ga']
METHODS = ['dragonfly']
# METHODS = ['dragonfly-mf']
# METHODS = ['rand', 'ga', 'hyperopt']
# METHODS = ['rand', 'ga', 'hyperopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'hyperopt', 'gpyopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'dragonfly-mf', 'dragonfly']
# METHODS = ['bo', 'bo_ga-direct', 'rand', 'ga']

out_dir = './syn_results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
    'branin': ('./demos_synthetic/branin/config_mf.json',
               branin_mf, cost_branin_mf, 0.1, 0, 1),
    'hartmann3_2': ('./demos_synthetic/hartmann3_2/config_mf.json',
                    hartmann3_2_mf, cost_hartmann3_2_mf, 0.1, 0, 1),
    'hartmann6_4': ('./demos_synthetic/hartmann6_4/config_mf.json',
                    hartmann6_4_mf, cost_hartmann6_4_mf, 0.1, 0, 1),
    'borehole_6': ('./demos_synthetic/borehole_6/config_mf.json',
                   borehole_6_mf, cost_borehole_6_mf, 1, 0, 1),
    'park2_4': ('./demos_synthetic/park2_4/config_mf.json',
                park2_4_mf, cost_park2_4_mf, 0.3, 0, 1),
    'park2_3': ('./demos_synthetic/park2_3/config_mf.json',
                park2_3_mf, cost_park2_3_mf, 0.1, 0, 1),
    'park1_3': ('./demos_synthetic/park1_3/config_mf.json',
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

