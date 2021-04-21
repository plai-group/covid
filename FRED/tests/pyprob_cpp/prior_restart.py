import pyprob
import subprocess, os, signal
from pyprob import RemoteModel
import numpy as np
import torch

model_executable = './simulator' # Path to the simulator's executable file.
# There is no need for a model_address as it will be randomly generated in this example

if __name__ == '__main__':
    def model_dispatcher(trace_idx, server_address):
        return subprocess.Popen('{} {} > /dev/null &'.format(model_executable, server_address), shell=True, preexec_fn=os.setsid)
    try:
        # Start the simulator, passing in the inter-process communication address.
        # Instantiate the Python-side model.
        # In this case, the model generates a random address and launches a new process for each simulation
        # The process will be terminated after each simulation and a new process is launched, by calling the
        # model_dispatcher function above.
        model = RemoteModel(random_server_address=True,
                            model_dispatcher=model_dispatcher,
                            restart_per_trace=True)
        # Run the simulator 10 times.
        samples = model.prior_results(num_traces=10)
    finally:
        print('Done, killing model process')
        model.kill_process()