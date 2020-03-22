import subprocess, os, signal
import pyprob
from pyprob import RemoteModel
import numpy as np
from pathlib import Path
import torch
from types import SimpleNamespace
from sacred import Experiment

server_address = 'ipc://@FRED'
model_executable = 'FRED'
FRED_HOME = os.environ['FRED_HOME']
HOME = os.environ['HOME']
USER = os.environ['USER']
default_params = None

# Use sacred for command line interface + hyperparams
ex = Experiment()

@ex.config
def my_config():
    # paths
    params_base = 'params'
    level_1 = f'/scratch/{USER}'
    level_2 = 'experiment_name'
    level_3 = 'level_3'
    debug = False
    dump_simulator_log = True

    # Inference-related parameters
    num_traces = 10

    # Simulator parameters
    days = None


def read_param_file(path):
    with open(path, 'r') as f:
        param_lines = f.readlines()
    param_lines = map(lambda x: x.strip(), param_lines)
    param_lines = list(filter(lambda x: not(x.startswith('#') or x == ''), param_lines))
    params_dict = {}
    for param_line in param_lines:
        parts = param_line.split('=')
        name = parts[0].strip()
        value = parts[1].strip()
        params_dict[name] = value
    return params_dict

def get_default_params():
    global default_params
    if default_params is not None:
        return default_params.copy()
    defaults = 'defaults'
    if not os.path.exists(defaults):
        defaults = os.path.join(FRED_HOME, 'input_files', 'defaults')
    if not os.path.exists(defaults):
        raise Exception('could not find defaults file')
    params = read_param_file(defaults)
    default_params = params.copy()
    return params

def dump_parameter_file(path='', args=None, sampled_parameters={}):
    params = get_default_params()
    params.update(read_param_file(args.params))
    params.update(sampled_parameters)
    with open(path, 'w') as f:
        for param, value in params.items():
            f.write(f'{param} = {value}\n')


def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)
    out_dir = Path(args.level_1) / args.level_2 / args.level_3
    out_dir.mkdir(parents=True, exist_ok=args.debug)
    args.out_dir = str(out_dir)
    base_params = read_param_file(args.params_base)

    if args.days is not None:
        base_params['days'] = args.days
    else:
        if 'days' in base_params:
            args.days = int(base_params['days'])
        else:
            args.days = int(get_default_params()['days'])
    
    args.params = os.path.join(out_dir, 'params_generated')
    with open(args.params, 'w') as f:
        for param, value in base_params.items():
            f.write(f'{param} = {value}\n')
    return args

def run(args):
    def model_dispatcher(trace_idx):
        arguments = f'{args.params} {trace_idx} {args.out_dir}'
        if args.dump_simulator_log:
            return subprocess.Popen(f'{model_executable} {server_address} {arguments} 2>&1 > {args.out_dir}/LOG{trace_idx} &', shell=True, preexec_fn=os.setsid)
        else:
            return subprocess.Popen(f'{model_executable} {server_address} {arguments} &', shell=True, preexec_fn=os.setsid)

    try:
        model = RemoteModel(server_address, model_dispatcher=model_dispatcher, restart_per_trace=True)
        traces = model.posterior(num_traces=args.num_traces, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING,
                                 observe={f'obs_{i}': 0.1 for i in range(args.days)})
        for idx, trace in enumerate(traces):
            dump_parameter_file(sampled_parameters={trace.variables[0].name: trace.variables[0].value.item()}, path=os.path.join(args.out_dir, f'params{idx}'), args=args)
    finally:
        if model._model_process is not None:
            print('Done, killing model process: {}'.format(model._model_process.pid))
            os.killpg(os.getpgid(model._model_process.pid), signal.SIGTERM)

@ex.automain
def command_line_entry(_run,_config):
    args = init(_config)
    run(args)
