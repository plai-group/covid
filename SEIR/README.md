# SEIR

Demonstration of using compartmental models for infection modelling and policy planning. 

Note we have had some people struggle running this code in virtual environments due to the multiprocessing libraries used. We are investigating and trying to remedy this. However, this experiment has minimal requirements all installable via `pip` and so a virtual environment is probably not required.

## Getting started
### Dependencies
- numpy
- torch
- colormap
- easydev
- matplotlib
- tqdm

## Running the Experiments
```
python seir_demo.py
```

To recreate the experiments in the paper, set the `experiment_*` variables to true. This will run each experiment from the paper and save the results figures into `./pdf`. 