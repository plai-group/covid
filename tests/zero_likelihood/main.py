import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

model_address = 'ipc://@gum'
model_executable = './simulator'

if __name__ == '__main__':
    try:
        #model_process = subprocess.Popen('{} {} > /dev/null &'.format(model_executable, model_address), shell=True, preexec_fn=os.setsid)
        model = RemoteModel(model_address, model_executable=model_executable)
        samples = model.posterior_results(num_traces=10, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING,
                                          observe={f'obs_{i}': 0.9 for i in range(10)})

    finally:
        if model._model_process is not None:
            print('Done, killing model process: {}'.format(model._model_process.pid))
            os.killpg(os.getpgid(model._model_process.pid), signal.SIGTERM)