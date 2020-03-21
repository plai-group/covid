import subprocess
import signal
import os
import pyprob
from pyprob import RemoteModel

model_address = 'ipc://@pyprob_cpp_test'
model_executable = './simulator'

if __name__ == '__main__':
    model_process = subprocess.Popen('{} {} > /dev/null &'.format(model_executable, model_address), shell=True, preexec_fn=os.setsid)
    model = RemoteModel(model_address)
    samples = model.prior_results(num_traces=2000)
    print(f'Mean = {samples.mean} (GT: 1)')
    print(f'Standard deviation = {samples.stddev} (GT: 2.236)')

    if model_process is not None:
        print('Done, killing model process: {}'.format(model_process.pid))
        os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)