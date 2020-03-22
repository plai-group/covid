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
fred_home = os.environ['FRED_HOME']
home = os.environ['HOME']
parameter_file = f'{fred_home}/input_files/default'
out_dir = 'OUT_TEST'

# Use sacred for command line interface + hyperparams
ex = Experiment()

@ex.config
def my_config():
    # paths
    out_dir = './dummy'
    params_path = './path'
    level_1 = f'{home}/scratch/covid_results'
    level_2 = 'experiment_name'
    level_3 = 'level_3'

def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)
    out_dir = Path(args.level_1) / args.level_2 / args.level_3
    out_dir.mkdir(parents=True, exist_ok=False)
    args.out_dir = str(out_dir)
    return args


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

def write_parameter_file(path, sampled_parameters={}):
    defaults = 'defaults'
    if not os.path.exists(defaults):
        defaults = os.path.join(fred_home, 'input_files', 'defaults')
    if not os.path.exists(defaults):
        raise Exception('could not find defaults file')
    params = read_param_file(defaults)
    params.update(read_param_file(parameter_file))
    params.update(sampled_parameters)
    with open(path, 'w') as f:
        for param, value in params.items():
            f.write(f'{param} = {value}\n')

def run(args):
    def model_dispatcher(trace_idx):
        arguments = f'{parameter_file} {trace_idx} {args.out_dir}'
        return subprocess.Popen(f'{model_executable} {server_address} {arguments} 2>&1 > {args.out_dir}/LOG{trace_idx} &', shell=True, preexec_fn=os.setsid)

    try:
        model = RemoteModel(server_address, model_dispatcher=model_dispatcher, restart_per_trace=True)
        traces = model.posterior(num_traces=10, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING)
        for idx, trace in enumerate(traces):
            write_parameter_file(sampled_parameters={trace.variables[0].name: trace.variables[0].value.item()}, path=os.path.join(args.out_dir, f'params{idx}'))
    finally:
        if model._model_process is not None:
            print('Done, killing model process: {}'.format(model._model_process.pid))
            os.killpg(os.getpgid(model._model_process.pid), signal.SIGTERM)

@ex.automain
def command_line_entry(_run,_config):
    args = init(_config)
    run(args)
