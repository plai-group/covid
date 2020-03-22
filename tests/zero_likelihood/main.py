import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

server_address = 'ipc://@zero_likelihood'
model_executable = './simulator'
num_traces = 10


def model_dispatcher(trace_idx):
    return subprocess.Popen(f'{model_executable} {server_address} > /dev/null &', shell=True, preexec_fn=os.setsid)

if __name__ == '__main__':
    try:
        model = RemoteModel(server_address, model_dispatcher=model_dispatcher)
        samples = model.posterior_results(num_traces=num_traces, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING,
                                          observe={f'obs_{i}': 0.9 for i in range(10)})
        print(f'model\'s trace_idx = {model.trace_idx} (expected: {num_traces - 1})')

    finally:
        if model._model_process is not None:
            print('Done, killing model process: {}'.format(model._model_process.pid))
            os.killpg(os.getpgid(model._model_process.pid), signal.SIGTERM)