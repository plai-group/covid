import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

model_address = 'ipc://@gum'     # A shared address for inter-process communication (IPC) between the simulator and pyprob.
model_executable = './simulator' # Path to the simulator's executable file.

if __name__ == '__main__':
    try:
        # Start the simulator, passing in the inter-process communication address.
        model_process = subprocess.Popen('{} {} > /dev/null &'.format(model_executable, model_address), shell=True, preexec_fn=os.setsid)
        # Instantiate the Python-side model.
        model = RemoteModel(model_address)
        # Run the simulator 10 times.
        samples = model.prior_results(num_traces=10)
    finally:
        if model_process is not None:
            print('Done, killing model process: {}'.format(model_process.pid))
            os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)