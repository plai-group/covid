import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

model_address = 'ipc://@gum'
model_executable = './simulator'

if __name__ == '__main__':
    obs = 2 # observed value from the model

    try:
        model_process = subprocess.Popen('{} {} > /dev/null &'.format(model_executable, model_address), shell=True, preexec_fn=os.setsid)
        model = RemoteModel(model_address)
        samples = model.posterior_results(num_traces=5000, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs':obs})
        
        var_gt = 5 * 2 / (5+2)
        mu_posterior = var_gt * (1/5 + obs/2)
        print(f'Mean = {samples.mean} (GT: {mu_posterior})')
        print(f'Standard deviation = {samples.stddev} (GT: {np.sqrt(var_gt)})')
    finally:
        if model_process is not None:
            print('Done, killing model process: {}'.format(model_process.pid))
            os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)