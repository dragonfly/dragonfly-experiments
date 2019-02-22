"""
  Synthetic experiments for optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import Namespace
from time import time, clock
from datetime import datetime
# Local demo functions
from demos_real.supernova.snls_mf import objective as supernova_obj_mf
from demos_real.supernova.snls_mf import cost as supernova_cost_mf
from demos_real.salsa.salsa_energy_mf import objective as salsa_obj_mf
from demos_real.salsa.salsa_energy_mf import cost as salsa_cost_mf
from demos_real.skltree.gbc_sensorless_drive_mf import objective as gbcsensorless_obj_mf
from demos_real.skltree.gbc_sensorless_drive_mf import cost as gbcsensorless_cost_mf
from demos_real.skltree.gbr_protein_mf import objective as gbrprotein_obj_mf
from demos_real.skltree.gbr_protein_mf import cost as gbrprotein_cost_mf
from demos_real.skltree.gbr_naval_mf import objective as gbrnaval_obj_mf
from demos_real.skltree.gbr_naval_mf import cost as gbrnaval_cost_mf
from demos_real.skltree.rfr_news_mf import objective as rfrnews_obj_mf
from demos_real.skltree.rfr_news_mf import cost as rfrnews_cost_mf
# Local
from cp_opt_method_evaluator import CPOptMethodEvaluator
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.experiment_caller import get_multifunction_caller_from_config
from dragonfly.exd.worker_manager import RealWorkerManager
from dragonfly.opt.blackbox_optimiser import blackbox_opt_args
from dragonfly.opt.cp_ga_optimiser import cpga_opt_args
from dragonfly.opt.gp_bandit import get_all_cp_gp_bandit_args, get_all_mf_cp_gp_bandit_args
from dragonfly.opt.random_optimiser import cp_random_optimiser_args
from dragonfly.utils.option_handler import load_options
from dragonfly.utils.reporters import get_reporter


# Experiment Parameters ==============================================================

# IS_DEBUG = True
IS_DEBUG = False

NUM_TRIALS = 10
MAX_NUM_EVALS = 2000

# STUDY_NAME = 'supernova'
# STUDY_NAME = 'salsa'
STUDY_NAME = 'gbcsensorless'
# STUDY_NAME = 'gbrprotein'
# STUDY_NAME = 'gbrnaval'
# STUDY_NAME = 'rfrnews'

# Other problem parameters - won't be changing these much
NUM_WORKERS = 1
SAVE_RESULTS_DIR = 'real_results'

# METHODS = ['smac']
# METHODS = ['rand', 'ga', 'hyperopt', 'gpyopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'dragonfly-mf', 'dragonfly']

# METHODS = ['rand', 'ga', 'dragonfly-mf', 'dragonfly']
# METHODS = ['gpyopt']
# METHODS = ['spearmint']
# METHODS = ['smac']

# # METHODS = ['dragonfly-mf', 'dragonfly']
# # METHODS = ['dragonfly-mfexpdecay']
# METHODS = ['dragonfly-mfexpdecay', 'dragonfly-mf', 'dragonfly']
# METHODS = ['dragonfly-mf', 'dragonfly']
# METHODS = ['dragonfly', 'dragonfly-mf']
# METHODS = ['ga', 'gpyopt']
# METHODS = ['rand', 'smac']
# METHODS = ['gpyopt', 'ga']

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
# METHODS = ['dragonfly', 'dragonfly-mf']
# METHODS = ['rand', 'ga', 'hyperopt']
# METHODS = ['rand', 'ga', 'hyperopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'hyperopt', 'gpyopt', 'dragonfly-mf', 'dragonfly']
# METHODS = ['rand', 'ga', 'dragonfly-mf', 'dragonfly']
# METHODS = ['bo', 'bo_ga-direct', 'rand', 'ga']


def get_evaluation_tmp_dir(study_name):
  """ Gets the tmp dir. """
  return 'working_dirs/%s_%s'%(study_name, datetime.now().strftime('%m%d-%H%M%S'))


def get_prob_params():
  """ Returns the problem parameters. """
  prob = Namespace()
  prob.study_name = STUDY_NAME
  if IS_DEBUG:
    prob.num_trials = 3
    prob.max_num_evals = 20
  else:
    prob.num_trials = NUM_TRIALS
    prob.max_num_evals = MAX_NUM_EVALS
  # Common
  prob.num_workers = NUM_WORKERS
  # study_params in order config_file, objective, cost_func, budget in hours.
  _study_params = {
    'supernova': ('../demos_real/supernova/config_mf.json',
                  supernova_obj_mf, supernova_cost_mf, 4.0),
    'salsa': ('../demos_real/salsa/config_salsa_energy_mf.json',
              salsa_obj_mf, salsa_cost_mf, 8.0),
    'gbcsensorless': ('../demos_real/skltree/config_gbc_mf.json',
              gbcsensorless_obj_mf, gbcsensorless_cost_mf, 4.0),
    'gbrprotein': ('../demos_real/skltree/config_gbr_mf.json',
              gbrprotein_obj_mf, gbrprotein_cost_mf, 3.0),
    'gbrnaval': ('../demos_real/skltree/config_naval_gbr_mf.json',
              gbrnaval_obj_mf, gbrnaval_cost_mf, 3.0),
    'rfrnews': ('../demos_real/skltree/config_rfr_mf.json',
              rfrnews_obj_mf, rfrnews_cost_mf, 6.0),
    }
#   _study_params = {
#     'supernova': ('../demos_real/supernova/config_mf_duplicate.json',
#                   supernova_obj_mf, supernova_cost_mf, 2.0),
#     'salsa': ('../demos_real/salsa/config_salsa_energy_mf.json',
#               salsa_obj_mf, salsa_cost_mf, 4.0),
#     }
  domain_config_file, raw_func, raw_fidel_cost_func, budget_in_hours = \
    _study_params[prob.study_name]
  # noisy
  prob.noisy_evals = False
  noise_type = 'no_noise'
  noise_scale = None
  # Create domain, function_caller and worker_manager
  config = load_config_file(domain_config_file)
  func_caller = get_multifunction_caller_from_config(raw_func, config,
                  raw_fidel_cost_func=raw_fidel_cost_func, noise_type=noise_type,
                  noise_scale=noise_scale)
  # Set max_capital
  if IS_DEBUG:
    prob.max_capital = 0.05 * 60 * 60
  else:
    prob.max_capital = budget_in_hours * 60 * 60
  # Store everything in prob
  prob.func_caller = func_caller
  prob.tmp_dir = get_evaluation_tmp_dir(prob.study_name)
  prob.worker_manager = RealWorkerManager(prob.num_workers, prob.tmp_dir)
  prob.save_file_prefix = prob.study_name + ('-debug' if IS_DEBUG else '')
  prob.methods = METHODS
  prob.save_results_dir = SAVE_RESULTS_DIR
  prob.reporter = get_reporter('default')
  # evaluation options
  prob.evaluation_options = Namespace(prev_eval_points='none',
                                      initial_pool_size=0)
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
    elif meth in ['hyperopt', 'gpyopt', 'smac']:
      curr_options = load_options(blackbox_opt_args)
      curr_options = Namespace(redo_evals_for_true_val=False)
    else:
      raise ValueError('Unknown method %s.'%(meth))
    curr_options.capital_type = capital_type
    all_method_options[meth] = curr_options
  return all_method_options


def main():
  """ Main function. """
  prob = get_prob_params()
  method_options = get_method_options(prob, 'realtime')
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

