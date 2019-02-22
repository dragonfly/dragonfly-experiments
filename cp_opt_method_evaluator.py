"""
  Harness for Evaluating methods for optimisation in cartesian product spaces.
  -- kandasamy@cs.cmu.edu
"""

from __future__ import print_function

# pylint: disable=invalid-name

import os
import signal
import time
import timeit
import subprocess
import json
from argparse import Namespace
from collections import OrderedDict
import numpy as np

# Local imports
from dragonfly.opt.opt_method_evaluator import OptMethodEvaluator
from dragonfly.exd.exd_utils import get_cp_domain_initial_qinfos
from dragonfly.opt.random_optimiser import random_optimiser_from_func_caller
from dragonfly.opt.cp_ga_optimiser import cp_ga_optimiser_from_proc_args
from dragonfly.opt.gp_bandit import gpb_from_func_caller
from dragonfly.utils.general_utils import flatten_list_of_lists
from euc_opt_method_evaluator import \
  common_final_operations_for_all_external_packages
# Other packages


class CPOptMethodEvaluator(OptMethodEvaluator):
  """ Evaluation of methods on Cartesian product spaces. """

  def _get_initial_pool_qinfos(self):
    """ Returns the initial qinfos. """
    init_qinfos = get_cp_domain_initial_qinfos(self.func_caller.domain,
                            num_samples=self.evaluation_options.initial_pool_size,
                            dom_euclidean_sample_type='latin_hc',
                            dom_integral_sample_type='latin_hc',
                            dom_nn_sample_type='nn')
    return init_qinfos

  def _optimise_with_method_on_func_caller(self, method, func_caller, worker_manager,
                                           max_capital, meth_options, reporter,
                                           *args, **kwargs):
    """ Run method on the function caller and return. """
    meth_options.mode = 'asy'
    method = method.lower()
    if method == 'rand':
      _, _, history = random_optimiser_from_func_caller(func_caller, worker_manager,
        max_capital, meth_options.mode, options=meth_options, reporter=reporter,
        *args, **kwargs)
    elif method == 'ga':
      _, _, history = cp_ga_optimiser_from_proc_args(func_caller, func_caller.domain,
        worker_manager, max_capital, options=meth_options, reporter=reporter,
        *args, **kwargs)
    elif method.startswith('dragonfly'):
      is_mf = method.startswith('dragonfly-mf') # Check if multi-fidelity
      if method == 'dragonfly-mfexpdecay':
        meth_options.fidel_euc_kernel_type = 'expdecay'
        meth_options.fidel_int_kernel_type = 'expdecay'
      _, _, history = gpb_from_func_caller(func_caller, worker_manager, max_capital,
        is_mf=is_mf, mode=meth_options.mode, options=meth_options, reporter=reporter,
        *args, **kwargs)
    elif method.startswith('gpyopt'):
      history = self._optimise_with_gpyopt(func_caller, max_capital, meth_options)
    elif method.startswith('hyperopt'):
      history = self._optimise_with_hyperopt(func_caller, max_capital, meth_options)
    elif method.startswith('smac'):
      history = self._optimise_with_smac(func_caller, max_capital, meth_options)
    elif method.startswith('spearmint'):
      history = self._optimise_with_spearmint(func_caller, max_capital, meth_options, self.study_name)
    else:
      raise ValueError('Unknown method %s.'%(method))
    return history

  @classmethod
  def _optimise_with_smac(cls, func_caller, max_capital, meth_options):
    """ Optimise with SMAC. """
    # pylint: disable=import-error
    try:
      from smac.facade.smac_facade import SMAC
      from smac.scenario.scenario import Scenario
    except ImportError:
      raise ImportError('smac package is not installed')
    # Determine max_iter
    if hasattr(func_caller, 'fidel_cost_func'):
      max_num_evals = int(max_capital /
                          func_caller.fidel_cost_func(func_caller.fidel_to_opt))
    else:
      max_num_evals = int(max_capital)
    # First get all hyper-parameters
    smac_func_to_min, smac_config_space, smac_convert_pt_back = \
        get_smac_func_and_space(func_caller.eval_single, func_caller.domain)
    scenario_dict = {"run_obj":"quality",
                     "cs":smac_config_space,
                     "deterministic":True,
                     }
    if meth_options.capital_type == 'realtime':
      scenario_dict['wallclock_limit'] = max_capital
      scenario_dict['runcount_limit'] = 10000
      scenario_dict['runcount-limit'] = 10000
    else:
      scenario_dict['runcount_limit'] = max_num_evals
      scenario_dict['runcount-limit'] = max_num_evals
    scenario = Scenario(scenario_dict)
    smac = SMAC(scenario=scenario, tae_runner=smac_func_to_min)
    smac.optimize()
    runhistory = smac.runhistory
    data = [([int(k.config_id),
              str(k.instance_id) if k.instance_id is not None else None,
              int(k.seed)], list(v))
            for k, v in runhistory.data.items()]
    query_vals = [-vlist[0] for (_, vlist) in data]
    config_ids_to_serialize = set([entry[0][0] for entry in data])
    configs = [conf.get_dictionary()
               for id_, conf in runhistory.ids_config.items()
               if id_ in config_ids_to_serialize]
    # Save everything to history here
    history = Namespace()
    history.query_points = [smac_convert_pt_back(x) for x in configs]
    num_smac_queries = len(history.query_points)

    history.query_send_times = [elem for elem in range(num_smac_queries)]
    history.query_receive_times = [elem+1 for elem in range(num_smac_queries)]
    history.query_vals = query_vals
    # Try deleting the directory
    try:
      import shutil
      to_del_dir = smac.output_dir.split('/')[0]
      shutil.rmtree(to_del_dir)
    except OSError:
      pass
    history = common_final_operations_for_all_external_packages(history, func_caller,
                                                                meth_options)
    return history


  @classmethod
  def _optimise_with_gpyopt(cls, func_caller, max_capital, meth_options):
    """ Optimise with GpyOpt. """
    import GPyOpt
    print(meth_options.capital_type)
    gpyopt_func_to_min, gpyopt_space, gpyopt_convert_pt_back = \
      get_gpyopt_func_and_space(func_caller.eval_single, func_caller.domain)
    print(meth_options.capital_type)
    bopt = GPyOpt.methods.BayesianOptimization(f=gpyopt_func_to_min, domain=gpyopt_space)
    print(meth_options.capital_type)
    if hasattr(func_caller, 'fidel_cost_func'):
      max_num_evals = int(max_capital /
                          func_caller.fidel_cost_func(func_caller.fidel_to_opt))
    else:
      max_num_evals = int(max_capital)
    if meth_options.capital_type == 'realtime':
      print('running GPYopt with time budget')
      bopt.run_optimization(max_iter=100000, max_time=max_capital, eps=-1.0,
                            verbosity=True)
    else:
      print('running GPYopt with iter budget')
      bopt.run_optimization(max_iter=int(max_num_evals), eps=-1.0, verbosity=True)
    # Extract the queried points and values.
    query_vals = []
    query_pts_in_gpyopt_fmt = []
    for _, (x, y) in enumerate(zip(bopt.X, bopt.Y)):
      query_vals.append(-1*y[0])
      query_pts_in_gpyopt_fmt.append([x.tolist()])
    # Save everything to history
    total_num_queries = len(query_vals)
    history = Namespace()
    history.query_points = [gpyopt_convert_pt_back(x) for x in query_pts_in_gpyopt_fmt]
    history.query_vals = query_vals
    history.query_send_times = list(range(0, total_num_queries))
    history.query_receive_times = list(range(1, total_num_queries+1))
    history = common_final_operations_for_all_external_packages(history, func_caller,
                                                                meth_options)
    return history

  @classmethod
  def _optimise_with_hyperopt(cls, func_caller, max_capital, meth_options):
    """ Optimise with hyperopt. """
    import hyperopt as hypopt
    hypopt_func_to_min, hypopt_space, hypopt_convert_pt_back = \
      get_hypopt_func_and_space(func_caller.eval_single, func_caller.domain, hypopt)
    hypopt_algo = hypopt.tpe.suggest
    hypopt_trials = hypopt.Trials()
    if hasattr(func_caller, 'fidel_cost_func'):
      max_num_evals = int(max_capital /
                          func_caller.fidel_cost_func(func_caller.fidel_to_opt))
    else:
      max_num_evals = int(max_capital)
    hypopt.fmin(hypopt_func_to_min, space=hypopt_space, algo=hypopt_algo,
                max_evals=max_num_evals, trials=hypopt_trials)
    trial_data = hypopt_trials.trials
    num_hypopt_queries = len(trial_data)
    history = Namespace()
    hypopt_labels = trial_data[0]['misc']['vals'].keys()
    hypopt_labels.sort()
    pts_in_hypopt_format = [[trial_data[i]['misc']['vals'][key] for key in hypopt_labels]
                            for i in range(num_hypopt_queries)]
    history.query_points = [hypopt_convert_pt_back(flatten_list_of_lists(pt))
                            for pt in pts_in_hypopt_format]
    history.query_send_times = [elem for elem in range(num_hypopt_queries)]
    history.query_receive_times = [elem+1 for elem in range(num_hypopt_queries)]
    query_vals = [-loss for loss in hypopt_trials.losses()]
    history.query_vals = query_vals
    history = common_final_operations_for_all_external_packages(history, func_caller,
                                                                meth_options)
    return history

  @classmethod
  def _optimise_with_spearmint(cls, func_caller, max_capital, options, study_name):
    """ Optimise with spearmint. """
    exp_dir = options.exp_dir + '_' + time.strftime("%Y%m%d-%H%M%S")
    out_dir = exp_dir + '/output'
    cur_dir = os.getcwd()
    out_file = cur_dir + '/output-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
    history = Namespace()
    if hasattr(func_caller, 'fidel_cost_func'):
      total_num_queries = int(max_capital /
                          func_caller.fidel_cost_func(func_caller.fidel_to_opt))
    else:
      total_num_queries = int(max_capital)
    os.system('cp -rf ' + options.exp_dir + ' ' + exp_dir)
    config_file = exp_dir + '/config.json'
    with open(config_file, 'r') as _file:
      config = json.load(_file, object_pairs_hook=OrderedDict)
    config['experiment-name'] = config['experiment-name'] + '-' + \
                                time.strftime("%Y%m%d-%H%M%S")
    if '-' in study_name:
      dim = int(study_name.split('-')[-1])
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
    query_true_vals = []
    query_pts = []
    pts = []
    for fname in out_files:
      file_path = out_dir + '/' + fname
      if os.path.getsize(file_path) == 0:
        continue
      point_dict, value, true_value = _read_spearmint_query_file(file_path)
      point = [point_dict[k] for k in config['variables'].keys()]
      query_vals.append(value)
      query_pts.append(point)
      if true_value is None:
        query_true_vals.append(value)
      else:
        query_true_vals.append(true_value)

    query_pts = [flatten_list_of_lists(x) for x in query_pts]
    #query_pts = [x for x in query_pts if func_caller.raw_domain.is_a_member(x)]
    history.query_step_idxs = [i for i in range(total_num_queries)]
    #history.query_points = [func_caller.get_normalised_domain_coords(x)
    #                        for x in query_pts]
    history.query_points = [[]] * total_num_queries
    history.query_send_times = [0] * total_num_queries
    history.query_receive_times = list(range(1, total_num_queries+1))
    history.query_vals = query_vals
    history.query_true_vals = query_true_vals

    # Query eval times
    history.query_eval_times = \
                         [history.query_receive_times[i] - history.query_send_times[i] \
                          for i in range(len(history.query_receive_times))]

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

    # Delete temporay files and directories
    os.system('rm -rf ' + out_file)
    os.system('rm -rf ' + exp_dir)
    return history



# Common Utils -----------------------------------------------------------------------
def _get_var_label_with_counter(_counter):
  """ Returns label. """
  _counter += 1
  return _counter, 'hypopt_x_%04d'%(_counter)

def _convert_single_list_repr_to_cp_domain(point, num_dims_per_domain, cp_dom_types):
  """ Converts the point back into cp_domain format. """
  cum_dims = [0] + list(np.cumsum(num_dims_per_domain))
  ret_x = []
  for idx in range(len(num_dims_per_domain)):
    curr_elem = list(point[cum_dims[idx]:cum_dims[idx+1]])
    if cp_dom_types[idx] == 'integral':
      int_elem = [int(x) for x in curr_elem]
      curr_elem = int_elem
    ret_x.append(curr_elem)
  return ret_x

def _get_convert_single_list_pt_back(num_dims_per_domain, cp_dom_types):
  """ Gets a function to convert a single list point back. """
  return lambda x: _convert_single_list_repr_to_cp_domain(x, num_dims_per_domain,
                                                          cp_dom_types)

def _get_convert_single_list_of_list_pt_back(num_dims_per_domain, cp_dom_types):
  """ Gets a function to convert a single list point back. """
  return lambda x: _convert_single_list_repr_to_cp_domain(x[0], num_dims_per_domain,
                                                          cp_dom_types)

def _convert_indices_to_objects_for_disc_domains(x, disc_spaces):
  """ Converts the discrete indices to objects. """
  ret = list(x[:])
  for idx, ds in enumerate(disc_spaces):
    if ds is not None:
      ret[idx] = ds.get_item(x[idx])
  return ret

def _convert_list_repr_to_cp_domain_repr(x, num_dims_per_domain, cp_dom_types,
                                         disc_spaces):
  """ Converts a list representation to cp_domain representation. """
  return _convert_single_list_repr_to_cp_domain(
           _convert_indices_to_objects_for_disc_domains(x, disc_spaces),
           num_dims_per_domain, cp_dom_types)

def _convert_alphabetical_dict_repr_to_cp_domain_repr(x, num_dims_per_domain,
                                                      cp_dom_types, disc_spaces):
  """ Converts an alphabetical list representation to cp_domain representation. """
  cfg = {k:x[k] for k in x}
  smac_labels = list(cfg.keys())
  smac_labels.sort()
  points = [cfg[key] for key in smac_labels]
  ret = _convert_single_list_repr_to_cp_domain(
         _convert_indices_to_objects_for_disc_domains(points, disc_spaces),
         num_dims_per_domain, cp_dom_types)
  return ret



class DiscItemsToIndexConverter(object):
  """ Class for converting Discrete Items to Indices. """

  def __init__(self, items):
    """ Constructor. """
    self.items = items
    self.indices = list(range(len(items)))

  def get_item(self, idx):
    """ Returns Item. """
    return self.items[int(idx)]

  def get_idx(self, item):
    """ Returns the index. """
    return self.items.index(item)


# SMAC -------------------------------------------------------------------------------
def get_smac_func_and_space(func, cp_domain):
  """ Returns a function to be passed to SMAC. """
  from smac.configspace import ConfigurationSpace
  from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
       UniformFloatHyperparameter, UniformIntegerHyperparameter
  smac_configs = []
  num_dims_per_domain = []
  disc_spaces = []
  cp_dom_types = []
  # Iterate through each domain
  counter = 0
  for dom in cp_domain.list_of_domains:
    dom_type = dom.get_type()
    cp_dom_types.append(dom_type)
    if dom_type == 'euclidean':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        smac_configs.append(UniformFloatHyperparameter(var_label, bds[0], bds[1],
                            default_value=(bds[0] + bds[1])/2))
    elif dom_type == 'integral':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        smac_configs.append(UniformIntegerHyperparameter(var_label, bds[0], bds[1],
                            default_value=(bds[0] + bds[1])//2))
    elif dom_type in ['prod_discrete', 'prod_discrete_numeric']:
      num_dims_per_domain.append(dom.get_dim())
      for lois in dom.list_of_list_of_items:
        curr_disc_to_idx_converter = DiscItemsToIndexConverter(lois)
        disc_spaces.append(curr_disc_to_idx_converter)
        counter, var_label = _get_var_label_with_counter(counter)
        smac_configs.append(CategoricalHyperparameter(var_label,
                            curr_disc_to_idx_converter.indices,
                            default_value=curr_disc_to_idx_converter.indices[0]))
  smac_space = ConfigurationSpace()
  smac_space.add_hyperparameters(smac_configs)
  # The convert back function
  smac_convert_pt_back = lambda x: _convert_alphabetical_dict_repr_to_cp_domain_repr(x,
                                   num_dims_per_domain, cp_dom_types, disc_spaces)
  # Then the function
  smac_func_to_min = lambda x: - func(smac_convert_pt_back(x))[0]
  return smac_func_to_min, smac_space, smac_convert_pt_back


# Hyper-opt --------------------------------------------------------------------------
def get_hypopt_func_and_space(func, cp_domain, hypopt_module):
  """ Returns a Function to be passed into hyperopt. """
  num_dims_per_domain = []
  disc_spaces = []
  cp_dom_types = []
  # First the space
  hypopt_space = []
  # Iterate through each domain
  counter = 0
  for dom in cp_domain.list_of_domains:
    dom_type = dom.get_type()
    cp_dom_types.append(dom_type)
    if dom_type == 'euclidean':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        hypopt_space.append(hypopt_module.hp.uniform(var_label, bds[0], bds[1]))
    elif dom_type == 'integral':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        hypopt_space.append(hypopt_module.hp.quniform(var_label, bds[0], bds[1], 1))
    elif dom_type in ['prod_discrete', 'prod_discrete_numeric']:
      num_dims_per_domain.append(dom.get_dim())
      for lois in dom.list_of_list_of_items:
        curr_disc_to_idx_converter = DiscItemsToIndexConverter(lois)
        disc_spaces.append(curr_disc_to_idx_converter)
        counter, var_label = _get_var_label_with_counter(counter)
        hypopt_space.append(hypopt_module.hp.choice(var_label,
                                                    curr_disc_to_idx_converter.indices))
  # The convert back function
  hypopt_convert_pt_back = lambda x: _convert_list_repr_to_cp_domain_repr(x,
                                     num_dims_per_domain, cp_dom_types, disc_spaces)
  # Then the function
  hypopt_func_to_min = lambda x: - func(hypopt_convert_pt_back(x))[0]
  return hypopt_func_to_min, hypopt_space, hypopt_convert_pt_back


# GPyOpt ---------------------------------------------------------------------------
def get_gpyopt_func_and_space(func, cp_domain):
  """ Returns a function to be passed to Gpyopt. """
  num_dims_per_domain = []
  disc_spaces = []
  cp_dom_types = []
  # Iterate through each domain
  gpyopt_space = []
  counter = 0
  for dom in cp_domain.list_of_domains:
    dom_type = dom.get_type()
    cp_dom_types.append(dom_type)
    if dom_type == 'euclidean':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        gpyopt_space.append({'name':var_label, 'type':'continuous',
                             'domain':(bds[0], bds[1])})
    elif dom_type == 'integral':
      num_dims_per_domain.append(dom.get_dim())
      for bds in dom.bounds:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        gpyopt_space.append({'name':var_label, 'type':'discrete',
                             'domain':tuple(range(bds[0], bds[1]))})
    elif dom_type == 'prod_discrete_numeric':
      num_dims_per_domain.append(dom.get_dim())
      for lois in dom.list_of_list_of_items:
        disc_spaces.append(None)
        counter, var_label = _get_var_label_with_counter(counter)
        gpyopt_space.append({'name':var_label, 'type':'discrete',
                             'domain':tuple(lois)})
    elif dom_type == 'prod_discrete':
      num_dims_per_domain.append(dom.get_dim())
      for lois in dom.list_of_list_of_items:
        curr_disc_to_idx_converter = DiscItemsToIndexConverter(lois)
        disc_spaces.append(curr_disc_to_idx_converter)
        counter, var_label = _get_var_label_with_counter(counter)
        gpyopt_space.append({'name':var_label, 'type':'categorical',
                             'domain':tuple(curr_disc_to_idx_converter.indices)})
  # Convert back function
  gpyopt_convert_pt_back = lambda x: _convert_list_repr_to_cp_domain_repr(x[0],
                                      num_dims_per_domain, cp_dom_types, disc_spaces)
  # Function
  gpyopt_func_to_min = lambda x: - func(gpyopt_convert_pt_back(x))[0]
  return gpyopt_func_to_min, gpyopt_space, gpyopt_convert_pt_back


# Spearmint ---------------------------------------------------------------------------
def _read_spearmint_query_file(file_name):
  """ Reads the result from a spearmint query file. """
  file_handle = open(file_name, 'r')
  true_value = None
  for raw_line in file_handle:
    # value
    line = raw_line.strip()
    if line.startswith('Result'):
      value = -1.0*float(line.split()[-1])
    elif line.startswith('True Result'):
      true_value = -1.0*float(line.split()[-1])
    # point
    try:
      add_np_array_line = line.replace('array', 'np.array')
      point_candidate = eval(add_np_array_line)
      if isinstance(point_candidate, dict):
        point_done = True
        point = point_candidate
    except Exception as e:
      pass
  file_handle.close()
  return point, value, true_value
