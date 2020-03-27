import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

import plot_helpers as helpers

helpers.setup_matplotlib()

covid_textwidth = 469.75499

# `pip install colormap`
from colormap import hex2rgb
muted_colours_list = ["#4878D0", "#D65F5F", "#EE854A", "#6ACC64", "#956CB4",
                      "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
muted_colours_list = np.asarray([hex2rgb(_c) for _c in muted_colours_list]) / 256
muted_colours_dict = {'blue':   muted_colours_list[0],
                      'red':    muted_colours_list[1],
                      'orange': muted_colours_list[2],
                      'green':  muted_colours_list[3],
                      'purple': muted_colours_list[4],
                      'brown':  muted_colours_list[5],
                      'pink':   muted_colours_list[6],
                      'gray':   muted_colours_list[7],
                      'yellow': muted_colours_list[8],
                      'eggsh':  muted_colours_list[9]}


def read_output(path, offset):
    df = pd.read_csv(path, sep=' ', header=None)
    cols = df[df.columns[0::2]].iloc[0].tolist()
    df = df[df.columns[1::2]]
    df.columns = cols
    seed = int(str(path).split("out")[1].split(".txt")[0])
    df['percent_infected'] = df["I"] / df["N"] 
    df['seed'] = seed + offset
    return df.set_index(['seed','Day'])

# no_intervention = pd.concat([read_output(p) for p in Path('./notebooks/results/no_intervention_results/').glob("out*.txt")])
# hands_results = pd.concat([read_output(p) for p in Path('./notebooks/results/hands_results/').glob("out*.txt")])

#all_runs = pd.concat([read_output(p) for p in Path('./simulations/sim1000/').glob("out*.txt")])

#weights = json.load(open("simulations/weights.json", "r"))

def plot_seir(df, filename, seeds, legend=False):
    fig = plt.figure(figsize=helpers.set_size(covid_textwidth/2))

    colors = {"S": "green", "E": "blue", "I": "red", "R": "purple"}

    xmax = df.query("seed == 1").values.shape[0]
    def plot_index(index):
        xs = np.arange(1,xmax+1)
        ys = [df.query(f"seed == {s}")[index].values for s in seeds]

        helpers.plot_ensemble(xs, ys, label=index, color=muted_colours_dict[colors[index]], nstd=2)

    # plot_index("S")
    plot_index("E")
    plot_index("I")
    plot_index("R")

    POPULATION_COUNT = 1218695 # TODO parametrize
    ACCEPTABLE_PERCENTAGE = 0.1
    plt.hlines(ACCEPTABLE_PERCENTAGE*POPULATION_COUNT, 0, xmax, linestyles="dashed")

    #plt.ylim(0, 10000)
    plt.xlim(1, xmax)
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.tight_layout()
    if legend:
        plt.legend()
    plt.savefig(filename)
    plt.close()


#plot_seir(all_runs, "FRED_SEIR_uncontrolled.pdf", seeds=range(1000))

#accepted_seeds = [int(k) for k in weights.keys() if weights[k] == 1]

#plot_seir(all_runs, "FRED_SEIR_controlled.pdf", accepted_seeds)



def plot_experiment(experiment):
    run = pd.concat([read_output(p, offset=j*100)
                     for j in range(10)
                     for p in Path(f'/home/christian/allegheny/sim{j}/').glob("out*.txt")])

    seeds = []
    accepted_seeds = []
    offset = 0
    for j in range(10):
        weights = json.load(open(f'/home/christian/allegheny/sim{j}/weights.json', "r"))
        seeds += [int(k) + offset for k in weights.keys()]
        accepted_seeds += [int(k) + offset for k in weights.keys() if weights[k] == 1]
        offset += 100


    plot_seir(run, f"figures/{experiment}_uncontrolled.pdf", seeds)
    plot_seir(run, f"figures/{experiment}_controlled.pdf", accepted_seeds, legend=True)


plot_experiment("allegheny")

#plot_experiment("all_params_influenza")
#plot_experiment("all_params_new")
#plot_experiment("all_params_new_plus")
#plot_experiment("all_params_retry")



