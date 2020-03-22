import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

server_address = 'ipc://@FRED'
model_executable = 'FRED'

def model_dispatcher(trace_idx):
    out_dir = 'OUT_TEST'
    arguments = f'$FRED_HOME/input_files/default {trace_idx} {out_dir}'
    return subprocess.Popen(f'{model_executable} {server_address} {arguments} > /dev/null &', shell=True, preexec_fn=os.setsid)

if __name__ == '__main__':
    try:
        model = RemoteModel(server_address, model_dispatcher=model_dispatcher, restart_per_trace=True)
        samples = model.posterior_results(num_traces=10, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING)

    finally:
        if model._model_process is not None:
            print('Done, killing model process: {}'.format(model._model_process.pid))
            os.killpg(os.getpgid(model._model_process.pid), signal.SIGTERM)