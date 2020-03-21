# COVID-19

## Singularity images
We have a singularity image that can be used for running the experiments. It has all the dependencies (pyprob, pyprob_cpp, PPX, and FRED simulator) installed.

To create the singularity image from the recipe files,
```
sudo singularity build covid_base.sif covid_base.def
sudo singularity build covid.sif covid.def
```
Once images are successfully created, run `covid.sif` image
```
singularity shell <path-to-covid.sif>
```
To test that your Singularity image, `cd` to [`tests/pyprob_cpp`](tests/pyprob_cpp) and run the following:
```
singularity shell <path-to-covid.sif>
cmake . && cmake --build .
python main.py
```
It runs a test on a Gaussian with unknown mean model is in C++. It connects the C++ model to pyprob, samples from its prior and prints the empirical mean and standard deviation of the samples.

## [pyprob_cpp](https://github.com/plai-group/pyprob_cpp)
It is a C++ interface to pyprob.

## [PPX](https://github.com/plai-group/ppx)
The protocol over which pyprob and pyprob_cpp communicate.

## [pyprob](https://github.com/plai-group/pyprob)

## [FRED simulator](https://github.com/plai-group/FRED)
An agent-based simulator of epidemics in C++.