"""
  Harness for Evaluating methods for optimisation on euclidean spaces.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-self-use

import os
import signal
import time
import timeit
import subprocess
import re
import json
from argparse import Namespace
from collections import OrderedDict
import numpy as np
# Dragonfly imports
from dragonfly.exd.exd_utils import get_euclidean_initial_qinfos
from dragonfly.opt.opt_method_evaluator import OptMethodEvaluator
from dragonfly.opt.gp_bandit import gpb_from_func_caller
import dragonfly.opt.random_optimiser as random_optimiser
from dragonfly.utils.doo import DOOFunction, pdoo_wrap
from dragonfly.utils.general_utils import flatten_list_of_lists


class EucOptMethodEvaluator(OptMethodEvaluator):
  """ Evalution of methods for Euclidean spaces. """

  def _get_initial_pool_qinfos(self):
    """ Gets initial pool. """
    # Do all intialisations at the highest fidelity
    init_qinfos = get_euclidean_initial_qinfos('latin_hc',
                                               self.evaluation_options.initial_pool_size,
                                               self.func_caller.domain.bounds)
    # Add fidel_to_opt
    if hasattr(self.func_caller, 'fidel_to_opt'):
      for qinfo in init_qinfos:
        qinfo.fidel = self.func_caller.fidel_to_opt
    return init_qinfos

  def _optimise_with_method_on_func_caller(self, method, func_caller, worker_manager,
                                           max_capital, meth_options, reporter,
                                           *args, **kwargs):
    """ Run method on the function caller and return. """
    meth_options.mode = 'asy'
    method = method.lower()
    if method == 'rand':
      _, _, history = random_optimiser.random_optimiser_from_func_caller( \
                      self.func_caller, self.worker_manager, self.max_capital, \
                      meth_options.mode, options=meth_options)
    elif method == 'mf_ucb':
      meth_options.acq = 'ucb'
      meth_options.mf_strategy = 'boca'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=True, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'add_mf_ucb':
      meth_options.acq = 'add_ucb'
      meth_options.mf_strategy = 'boca'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=True, options=meth_options, reporter=reporter, \
        domain_add_max_group_size=0, *args, **kwargs)

    elif method in ['slice', 'post_sampling']:
      meth_options.acq = 'ucb-ei-ttei-add_ucb'
      meth_options.gpb_hp_tune_criterion = 'post_sampling'
      meth_options.gpb_post_hp_tune_method = 'slice'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'nuts':
      meth_options.gpb_hp_tune_criterion = 'post_sampling'
      meth_options.gpb_post_hp_tune_method = 'nuts'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'rand_exp_sampling':
      meth_options.acq = 'ucb-ei-ttei-add_ucb'
      meth_options.gpb_hp_tune_criterion = 'ml'
      meth_options.gpb_ml_hp_tune_opt = 'rand_exp_sampling'
      meth_options.kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method in ['direct', 'ml']:
      meth_options.acq = 'ucb-ei-ttei-add_ucb'
      meth_options.gpb_hp_tune_criterion = 'ml'
      meth_options.gpb_ml_hp_tune_opt = 'direct'
      meth_options.kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'ml+post_sampling':
      meth_options.acq = 'ucb-ei-ttei-add_ucb'
      meth_options.kernel_type = 'se'
      meth_options.gpb_hp_tune_criterion = 'ml-post_sampling'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'add-ei':
      meth_options.acq = 'ei'
      meth_options.use_additive_gp = True
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'esp-ei':
      meth_options.acq = 'ei'
      meth_options.kernel_type = 'esp'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'esp-ucb':
      meth_options.acq = 'ucb'
      meth_options.kernel_type = 'esp'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'add-add_ucb':
      meth_options.acq = 'add_ucb'
      meth_options.use_additive_gp = True
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'esp-se':
      meth_options.kernel_type = 'esp'
      meth_options.esp_kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'esp-matern':
      meth_options.kernel_type = 'esp'
      meth_options.esp_kernel_type = 'matern'
      meth_options.esp_matern_nu = 2.5
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'add-se':
      meth_options.use_additive_gp = True
      meth_options.kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'add-matern':
      meth_options.use_additive_gp = True
      meth_options.kernel_type = 'matern'
      meth_options.matern_nu = 2.5
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'se':
      meth_options.kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'matern':
      meth_options.kernel_type = 'matern'
      meth_options.matern_nu = 2.5
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method == 'ensemble-dfl':
      meth_options.acq_probs = 'uniform'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method in ['adaptive-ensemble-dfl', 'dragonfly']:
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method.startswith('dragonfly-mf'):
      if method == 'drafonfly-mf-exp':
        meth_options.fidel_kernel_type = 'expdecay'
      else:
        meth_options.fidel_kernel_type = 'se'
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=True, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    elif method in ['ei', 'ucb', 'add_ucb', 'pi', 'ttei', 'ts', 'ei-ucb-ttei-ts',
                    'ei-ucb-ttei-ts-add_ucb']:
      meth_options.acq = method
      _, _, history = gpb_from_func_caller(self.func_caller, self.worker_manager, \
        self.max_capital, is_mf=False, options=meth_options, reporter=reporter, \
        *args, **kwargs)
    # External packages
    elif method == 'pdoo':
      doo_func_to_max = _get_func_to_max_from_func_caller(self.func_caller)
      doo_obj = DOOFunction(doo_func_to_max, self.func_caller.domain.bounds)
      _, _, history = pdoo_wrap(doo_obj, self.max_capital, return_history=True)
      history = common_final_operations_for_all_external_packages(history,
                  self.func_caller, meth_options)
    elif method == 'hyperopt':
      best, history = self._optimize_by_hyperopt_pkg(self.func_caller, self.max_capital,
                                                     options=meth_options)
      print('Best eval point: {}'.format(best))
    elif method == 'smac':
      history = self._optimize_by_smac_pkg(self.func_caller, self.max_capital,
                                           meth_options)
    elif method == 'spearmint':
      history = self._optimize_by_spearmint_pkg(self.func_caller,
                                                self.max_capital, meth_options)
    elif method == 'gpyopt':
      history = self._optimize_by_gpyopt_pkg(self.func_caller, self.max_capital,
                                             meth_options)
    # Final operations
    return history

  def _get_space_params(self, space, domain_bounds):
    """ Creates space object for hyperopt """
    param_space = []
    for i, bound in enumerate(domain_bounds):
      param_name = 'x_' + str(i)
      param_space.append(space(param_name, bound[0], bound[1]))
    return param_space

  def _optimize_by_hyperopt_pkg(self, func_caller, max_capital, options):
    """ Optimizes the function using hyperopt package """
    try:
      from hyperopt import fmin, Trials
    except ImportError:
      raise ImportError('hyperopt package is not installed')
    space = options.space
    algo = options.algo
    param_space = self._get_space_params(space, func_caller.domain.bounds)
    func_to_min = _get_func_to_min_from_func_caller(func_caller)
    trials = Trials()
    best = fmin(func_to_min, space=param_space, algo=algo,
                max_evals=int(max_capital), trials=trials)
    history = Namespace()
    trial_data = trials.trials
    total_num_queries = len(trial_data)
    history.query_step_idxs = [i for i in range(total_num_queries)]
    pts_in_hypopt_format = [trial_data[i]['misc']['vals'].values()
                            for i in range(total_num_queries)]
    history.query_points = [flatten_list_of_lists(pt)
                            for pt in pts_in_hypopt_format]
    history.query_send_times = \
                           [float(trial_data[i]['book_time'].isoformat().split(':')[-1]) \
                            for i in range(total_num_queries)]
    history.query_receive_times = \
                        [float(trial_data[i]['refresh_time'].isoformat().split(':')[-1]) \
                         for i in range(total_num_queries)]
    losses = [-loss for loss in trials.losses()]
    history.query_vals = losses
    history = common_final_operations_for_all_external_packages(history, self.func_caller,
                                                                options)
    return best, history

  def _optimize_by_smac_pkg(self, func_caller, max_capital, options):
    """ Optimizes the function using smac package """
    try:
      from smac.facade.func_facade import fmin_smac
    except ImportError:
      raise ImportError('smac package is not installed')
    domain_bounds = func_caller.domain.bounds
    bounds = [(float(b[0]), float(b[1])) for b in domain_bounds]
    x0 = [(b[0] + b[1])/2 for b in domain_bounds]
    func_to_min = _get_func_to_min_from_func_caller(func_caller)
    history = Namespace()
    if options.capital_type == 'realtime':
      x, _, trace = fmin_smac(func=func_to_min,  # function
                              x0=x0,
                              bounds=bounds,
                              maxfun=10000,
                              maxtime=max_capital)  # maximum number of evaluations
    else:
      x, _, trace = fmin_smac(func=func_to_min,  # function
                              x0=x0,
                              bounds=bounds,
                              maxfun=max_capital)  # maximum number of evaluations
    runhistory = trace.get_runhistory()
    data = [([int(k.config_id),
              str(k.instance_id) if k.instance_id is not None else None,
              int(k.seed)], list(v))
            for k, v in runhistory.data.items()]
    config_ids_to_serialize = set([entry[0][0] for entry in data])
    configs = {id_: conf.get_dictionary()
               for id_, conf in runhistory.ids_config.items()
               if id_ in config_ids_to_serialize}  # all queried points
    total_num_queries = len(configs)
    history.query_step_idxs = [i for i in range(total_num_queries)]
    history.query_points = [list(configs[i].values())
                            for i in range(1, total_num_queries+1)]
    history.query_send_times = [0] * total_num_queries
    history.query_receive_times = list(range(1, total_num_queries+1))
    history.query_vals = [-vlist[0] for (_, vlist) in data]
    history = common_final_operations_for_all_external_packages(history, self.func_caller,
                                                                options)
    return history

  def _create_domain(self, domain_bounds):
    """ Creates domain object for gpyopt """
    domain = []
    for i, bound in enumerate(domain_bounds):
      space = {}
      space['name'] = 'x_' + str(i)
      space['type'] = 'continuous'
      space['domain'] = (bound[0], bound[1])
      domain.append(space)
    return domain

  def _optimize_by_gpyopt_pkg(self, func_caller, max_capital, options):
    """ Optimizes the function using gpyopt package """
    try:
      import GPyOpt
    except ImportError:
      raise ImportError('GPyOpt package is not installed')
    history = Namespace()
    max_iter = int(max_capital)
    domain = self._create_domain(func_caller.domain.bounds)
    func_to_min = _get_func_to_min_from_func_caller(func_caller)
    gpyopt_func_to_min = lambda x: func_to_min(x.squeeze().tolist())
    Bopt = GPyOpt.methods.BayesianOptimization(f=gpyopt_func_to_min, # func to optimize
                                               domain=domain)
                                               #verbosity=False
    if options.capital_type == 'realtime':
      Bopt.run_optimization(max_iter=100000, max_time=max_capital, eps=-1.0,
                            verbosity=True)
    else:
      Bopt.run_optimization(max_iter=int(max_capital), eps=-1.0, verbosity=True)
    query_vals = []
    query_pts = []
    for _, (x, y) in enumerate(zip(Bopt.X, Bopt.Y)):
      query_vals.append(-1*y[0])
      query_pts.append(x.tolist())
    x = Bopt.x_opt
    history.query_step_idxs = [i for i in range(max_iter)]
    history.query_receive_times = list(range(1, max_iter+1))
    history.query_points = query_pts
    history.query_send_times = list(range(0, max_iter))
    history.query_receive_times = list(range(1, max_iter+1))
    history.query_vals = query_vals
    history = common_final_operations_for_all_external_packages(history, self.func_caller,
                                                                options)
    return history

  def _optimize_by_spearmint_pkg(self, func_caller, max_capital, options):
    """ Optimizes the function using spearmint package """
    exp_dir = options.exp_dir + '_' + time.strftime("%Y%m%d-%H%M%S")
    out_dir = exp_dir + '/output'
    cur_dir = os.getcwd()
    out_file = cur_dir + '/output-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
    history = Namespace()
    total_num_queries = int(max_capital)

    os.system('cp -rf ' + options.exp_dir + ' ' + exp_dir)
    config_file = exp_dir + '/config.json'
    with open(config_file, 'r') as _file:
      config = json.load(_file, object_pairs_hook=OrderedDict)
    config['experiment-name'] = config['experiment-name'] + '-' + \
                                time.strftime("%Y%m%d-%H%M%S")
    if '-' in self.study_name:
      dim = int(self.study_name.split('-')[-1])
    else:
      dim = len(config['variables'])
    variables = OrderedDict()
    for i in range(int(dim/len(config['variables']))):
      for var in config['variables']:
        variables[var + '_' + str(i)] = config['variables'][var]
    config.pop('variables')
    config['variables'] = variables
    with open(config_file, 'w') as _file:
      json.dump(config, _file, indent=4)

    # Create output directory
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    # Start a sub process to run spearmint package
    os.chdir(options.pkg_dir)
    cmd = 'python main.py ' + exp_dir + ' > ' + out_file + ' 2>&1'
    start_time = timeit.default_timer()
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    if options.capital_type == 'realtime':
      count = total_num_queries
      timeout = False
      while True:
        curr_time = timeit.default_timer()
        out_files = os.listdir(out_dir)
        if curr_time - start_time >= count and timeout is False:
          count = len(out_files) + 1
          timeout = True
        if len(out_files) > count:
          os.killpg(os.getpgid(proc.pid), signal.SIGINT)
          break
    else:
      while True:
        out_files = os.listdir(out_dir)
        if len(out_files) > total_num_queries:
          os.killpg(os.getpgid(proc.pid), signal.SIGINT)
          break
    os.chdir(cur_dir)

    # Extract info for plotting
    out_files = sorted(os.listdir(out_dir))[0:total_num_queries]
    query_vals = []
    query_pts = []
    pts = []
    for fname in out_files:
      file_path = out_dir + '/' + fname
      if os.path.getsize(file_path) == 0:
        continue
      point_dict, value = _read_spearmint_query_file(file_path)
      point = [point_dict[k] for k in config['variables'].keys()]
      query_vals.append(value)
      query_pts.append(point)
    query_pts = [flatten_list_of_lists(x) for x in query_pts]
    query_pts = [x for x in query_pts if func_caller.raw_domain.is_a_member(x)]
    history.query_step_idxs = [i for i in range(total_num_queries)]
    history.query_points = [func_caller.get_normalised_domain_coords(x)
                            for x in query_pts]
    history.query_send_times = [0] * total_num_queries
    history.query_receive_times = list(range(1, total_num_queries+1))
    history.query_vals = query_vals
    history = common_final_operations_for_all_external_packages(history, self.func_caller,
                                                                options)
    # Delete temporay files and directories
    os.system('rm -rf ' + out_file)
    os.system('rm -rf ' + exp_dir)
    return history


def common_final_operations_for_all_external_packages(history, func_caller, options,
                                                      raw_func=None):
  """ Final operations for all packages. """
  # query step idxs
  total_num_queries = len(history.query_vals)
  history.query_step_idxs = [i for i in range(total_num_queries)]
  # Query eval times
  history.query_eval_times = \
                         [history.query_receive_times[i] - history.query_send_times[i] \
                          for i in range(len(history.query_receive_times))]
  # query_true_vals
  if options.redo_evals_for_true_val:
    if raw_func is None:
      history.query_true_vals = [func_caller.eval_single(x, noisy=False)[0] for x in
                                 history.query_points]
    else:
      history.query_true_vals = [raw_func(x) for x in history.query_points]
  else:
    history.query_true_vals = history.query_vals
  # Current Optimum values
  history.curr_opt_vals = []
  history.curr_opt_points = []
  curr_max = -np.inf
  curr_opt_point = None
  for idx in range(len(history.query_true_vals)):
    qv = history.query_vals[idx]
    if qv >= curr_max:
      curr_max = qv
      curr_opt_point = history.query_points[idx]
    history.curr_opt_vals.append(curr_max)
    history.curr_opt_points.append(curr_opt_point)
  # Current True optimum values and points
  history.curr_true_opt_vals = []
  history.curr_true_opt_points = []
  curr_max = -np.inf
  curr_true_opt_point = None
  for idx, qtv in enumerate(history.query_true_vals):
    if qtv >= curr_max:
      curr_max = qtv
      curr_true_opt_point = history.query_points[idx]
    history.curr_true_opt_vals.append(curr_max)
    history.curr_true_opt_points.append(curr_true_opt_point)
  # Other data
  history.query_worker_ids = [0] * total_num_queries
  history.query_qinfos = [''] * total_num_queries
  history.job_idxs_of_workers = [None] * total_num_queries
  history.num_jobs_per_worker = [total_num_queries]
  return history


def _read_spearmint_query_file(file_name):
  """ Reads the result from a spearmint query file. """
  file_handle = open(file_name, 'r')
  value_done = False
  point_done = False
  for raw_line in file_handle:
    # value
    line = raw_line.strip()
    if line.startswith('Result'):
      value = float(line.split()[-1])
      value_done = True
    # point
    try:
      add_np_array_line = line.replace('array', 'np.array')
      point_candidate = eval(add_np_array_line)
      if isinstance(point_candidate, dict):
        point_done = True
        point = point_candidate
    except Exception as e:
      pass
    if value_done and point_done:
      break
  file_handle.close()
  return point, value




def _get_func_to_min_from_func_caller(func_caller):
  """ Returns a function to be minimised. """
  return lambda x, *args, **kwargs: - func_caller.eval_single(x)[0]


def _get_func_to_max_from_func_caller(func_caller):
  """ Returns a function to be minimised. """
  return lambda x, *args, **kwargs: func_caller.eval_single(x)[0]

