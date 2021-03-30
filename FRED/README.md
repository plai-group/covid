# FRED simulator

## Getting started
### Singularity images
We provide a singularity image that can be used for running the experiments. It has all the dependencies (pyprob, pyprob_cpp, PPX, and FRED simulator) installed. To create the singularity image from the recipe files, clone this repo and `cd` to `FRED/singularity`. Then,
```bash
sudo singularity build covid_base.sif covid_base.def
sudo singularity build covid.sif covid.def
```

Once the images are successfully created, start up the image:
```bash
singularity shell <path-to-covid.sif>
```

To do a quick test on your Singularity image, you can run [`FRED/tests/pyprob_cpp`](this test). To do so, run the following:
```bash
singularity shell -B <path-to-FRED/>:/workdir <path-to-covid.sif>
cd /workdir/tests/pyprob_cpp
cmake . && cmake --build .
python inference.py
```
It runs a test on a Gaussian with unknown mean model is in C++. It connects the C++ model to pyprob, samples from its prior and prints the empirical mean and standard deviation of the samples along with the analytically computed values for them.

### Dependencies
The the core ingredients of this experiment are the following:
- [FRED simulator](https://github.com/plai-group/FRED): An agent-based simulator of epidemiological dynamics in C++.
- [pyprob](https://github.com/plai-group/pyprob): A probabilistic programming library in python that can be coupled to the existing simulators in different platforms.
- [pyprob_cpp](https://github.com/plai-group/pyprob_cpp)
A C++ library of pyprob interface.
- [PPX](https://github.com/plai-group/ppx): The protocol over which pyprob and pyprob_cpp communicate.

## Running the experiments
Once you have the repo cloned and the singularity image created,
```
singularity shell -B <path-to-FRED/>:/workdir
cd /workdir
python main.py
```
The following list is the most important configurations for this experiment:
- `num_traces`: Number of simulator runs for the inference task (default: 10)
- `constraint_threshold`: Imposed threshold on the ratio of infectious population (default: 0.1)
- `days`: Number of days to simulate in each run (default: 150)
For a list of configurations, look at [main.py](main.py) file.

### Configuration files
FRED simulator has a large number of configurations. They should be stored in a text file. The parameters we used for our experiments can be found [here](https://github.com/plai-group/FRED/blob/FRED-v2.12.0/params). However, the five parameters we impose priors on will be overwritten be a value sampled from their prior. For more information, refer to the paper.

## Visualizing the results
Once the experiments are ran, it generates a zip file containing the results (simulator outputs, log files, inference-related files, etc.) The plotting script can then be used to plot an array of 2D histograms of marginal posterior distribution over the policy parameters, like [this](../figures/hist_allegheny_success.png).
```
python plot_histogram.py with exp_dir=<results_directory> county=<experiment_name>
```
where `<results_directory>/<experiment_name>` is the path of the directory containing the zip files.
