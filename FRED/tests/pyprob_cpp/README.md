# pyprob and pyprob_cpp integration tests

This directory contains a simple test for running a simulator in C++ as a pyprob model. The simulator is implemented in [`main.cpp`](main.cpp) and is already in pyprob_cpp format. There are comments in the code briefly explaining how the conversion works.

To run the test the following steps should be done:

1. Compile the simulator and make an executable file via `cmake . && cmake --build .`
2. run the python file which internally spawns a process running the simulator's executable and connects to it via inter-process communication.

There are three python scripts provided

1. [`prior.py`](prior.py): It only runs the simulator 10 times. It is only to test that the simulator is properly launched and run multiple times.
2. [`prior_restart.py`](prior_restart.py): It is equivalent to `prior.py` except it launches a new process for each simulation, then terminates the process and launches a new one.
3. [`inference.py`](inference.py): It runs the simulator 5000 times and estimates the mean and standard deviation of the posterior given a fixed observation. The groundtruth and estimated values are printed to compare.

```bash
cd path_to_here
cmake . && cmake --build .
python prior.py # Only test that pyprob and the simulator are properly connected
python prior_restart.py # Test that pyprob can properly launch and terminate a process for each simulation
python main.py # A simple inference test
```