import tarfile
import os
import numpy as np
from pathlib import Path
from pyprob.distributions import Empirical
from tqdm import tqdm
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from zipfile import ZipFile
import io
from colormap import hex2rgb
from types import SimpleNamespace
from sacred import Experiment
#from plot_threshold import plot
from matplotlib.colors import ListedColormap

ex = Experiment()

# Colors and matplotlib styles
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
mpl.rcParams['lines.markersize'] = 3
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
}
mpl.rcParams.update(nice_fonts)

params_order = ['hand_washing_compliance', 'school_closure_ar_threshold', 'isolation_rate', 'shelter_in_place_duration_mean', 'shelter_in_place_compliance']

plot_labels = {'school_closure_ar_threshold': 'School closure\nattack rate\nthreshold',
               'hand_washing_compliance': 'Hand washing\ncompliance',
               'isolation_rate': 'Isolation rate',
               'shelter_in_place_duration_mean': 'Shelter in place\nduration mean',
               'shelter_in_place_compliance': 'Shelter in place\ncompliance'}
plot_ticks = {'school_closure_ar_threshold': [0.01, 0.21],
              'hand_washing_compliance': [0, 1],
              'isolation_rate': [0, 1],
              'shelter_in_place_duration_mean': [0, 14],
              'shelter_in_place_compliance': [0, 1]}

@ex.config
def my_config():
    # paths
    exp_dir = '/scratch/covid'
    results_dir = '/workdir/plots'
    county = None
    assert county is not None

def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)
    
    args.exp_dir = Path(args.exp_dir) / args.county
    assert args.exp_dir.is_dir()
    args.out_dir = Path(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args


def set_size(width, fraction=1, subplots=(1, 1)):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'pnas':
        width_pt = 246.09686
    elif width == 'current':
        width_pt = 469.75499
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def param_data_from_traces(traces_file):
    traces = Empirical(file_name=traces_file)
    param_data = np.zeros((len(traces), len(params_order)+1))
    # Store the parameters along with the weight of the trace in each row
    for i, trace in enumerate(traces):
        params = {v.name: v.value.item() for v in trace.variables_controlled}
        weight = np.exp(trace.log_importance_weight)
        assert weight < 0.1 or weight > 0.9
        weight = int(weight > 0.5)
        param_data[i] = np.array([params[x] for x in params_order] + [weight])
    return param_data

def param_data_from_simulation_files(simulation_dir):
    def param_value_from_lines(lines, param_name):
        for line in lines:
            if line.startswith(param_name):
                return float(line.split('=')[1].split('#')[0].strip())
    # Read weights file
    with ZipFile(simulation_dir, 'r') as simulation_dir_h:
        weights = json.load(io.BytesIO(simulation_dir_h.read('weights.json')))

        # Read parameter files
        param_data = np.zeros((len(weights), len(params_order)+1))
        for row_cnt, simulation_idx in enumerate(weights):
            params_file = f'params{simulation_idx}'
            param_lines = io.StringIO(simulation_dir_h.read(params_file).decode("utf-8")).readlines()

            # Fill in the parameter in params_data
            for j, param_name in enumerate(params_order):
                param_data[row_cnt][j] = param_value_from_lines(param_lines, param_name)
            # Fill in the weight in params_data
            param_data[row_cnt][-1] = weights[simulation_idx]

    return param_data


def get_bins(data, discrete=False, num_bins=30):
    if discrete:
        left_of_first_bin = data.min() - 0.5
        right_of_last_bin = data.max() + 0.5
        return np.arange(left_of_first_bin, right_of_last_bin + 1, 1)
    else:
        return np.linspace(data.min(), data.max(), num_bins)


def get_gradient_colormap(color_rgb, alpha=False):
    if alpha:
        # Single color with changing alpha channel
        newcolors = np.zeros((256, 4))
        newcolors[:, :3] = color_rgb
        newcolors[:, 3] = np.linspace(0, 1, 256)
    else:
        # Single color gradually getting white with alpha always 1
        newcolors = np.ones((256, 4))
        newcolors[:, 0] = np.linspace(1, color_rgb[0], 256)
        newcolors[:, 1] = np.linspace(1, color_rgb[1], 256)
        newcolors[:, 2] = np.linspace(1, color_rgb[2], 256)
    newcmp = ListedColormap(newcolors)
    return newcmp


def plot_hist(dset, file_path, color, opacity=True, label=None, title=None):
    num_params = len(params_order)
    fig = plt.figure(figsize=set_size('current', fraction=1.2, subplots=(num_params, num_params)))

    # Compute vmin and vmax
    vmin_unit = len(dset)
    vmax_unit = 0
    for i in range(num_params):
        for j in range(i):
            x_vals = dset[:, j]
            y_vals = dset[:, i]
            xbins = get_bins(x_vals, discrete='duration' in params_order[j])
            ybins = get_bins(y_vals, discrete='duration' in params_order[i])
            bin_size = 1 / (len(xbins)-1) / (len(ybins)-1)

            H, _, _ = np.histogram2d(x_vals, y_vals, bins=[xbins, ybins])
            vmax_unit = max(vmax_unit, H.max() / bin_size)
            vmin_unit = min(vmin_unit, H.min() / bin_size)

    # Plot the 2d histograms
    for i in range(num_params):
        for j in range(i):
            x_vals = dset[:, j]
            y_vals = dset[:, i]
            ax = plt.subplot(num_params, num_params, 1 + (i-1)*num_params + j)

            xbins = get_bins(x_vals, discrete='duration' in params_order[j])
            ybins = get_bins(y_vals, discrete='duration' in params_order[i])
            bin_size = 1 / (len(xbins)-1) / (len(ybins)-1)
            counts, xedges, yedges, im = ax.hist2d(x_vals, y_vals, bins=[xbins, ybins],
                                                   cmap=get_gradient_colormap(muted_colours_dict[color]),
                                                   vmax=vmax_unit * bin_size, vmin=vmin_unit * bin_size)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(plot_labels[params_order[i]], labelpad=40, rotation=-30)
                ax.set_yticks(plot_ticks[params_order[i]])
            else:
                ax.set_yticks([])
    
    # Compute ymax and ymin (for the last row histograms)
    i = num_params
    ymin = len(dset)
    ymax = 0
    for j in range(num_params):
        x_vals = dset[:, j]
        bins = get_bins(x_vals, discrete='duration' in params_order[j], num_bins=16)
        counts, _ = np.histogram(x_vals, bins=bins)

        ymax = max(ymax, counts.max())
        ymin = min(ymin, counts.min())

    # Plot the last row
    for j in range(num_params):
        x_vals = dset[:, j]
        ax = plt.subplot(num_params, num_params, 1 + (num_params-1)*num_params + j)
        ax.set_xlabel(plot_labels[params_order[j]])
        ax.set_xticks(plot_ticks[params_order[j]])
        bins = None

        bins = get_bins(x_vals, discrete='duration' in params_order[j], num_bins=15)
        counts, bins, _ = ax.hist(x_vals, bins=bins, lw=0, color=muted_colours_dict[color],
                                  alpha=0.6 if opacity else 1, label=label if j == 0 else None)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_visible(False)
        _range = [np.min(x_vals), np.max(x_vals)]
        bin_width = bins[1] - bins[0]
        ax.plot(_range, [len(x_vals) / (_range[1] - _range[0]) * bin_width]*2,
                linestyle='--', color='black')
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()

    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.8, 0.9), shadow=False, frameon=False, ncol=1, fontsize=10)
    
    if title is not None:
        fig.suptitle(title)

    # fig.subplots_adjust(bottom=0.15, top=0.9, left=0.17, right=0.9,
    #                     wspace=0.2, hspace=0.2)
    # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    # cbar = fig.colorbar(im, cax=cb_ax)
    # cb_ax.yaxis.set_visible(False)

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95,
                        wspace=0.2, hspace=0.2)

    fig.savefig(file_path)


def run(args):
    if False:
        trace_files = list(args.exp_dir.glob('sim*/traces'))
        dset = None
        for trace_file in tqdm(trace_files):
            print()
            print(f'Loading {trace_file}')
            if dset is None:
                dset = param_data_from_traces(str(trace_file))
            else:
                dset = np.concatenate([dset, param_data_from_traces(str(trace_file))])
    else:
        dset_filename = args.exp_dir.parent / f"{args.exp_dir.name}_dset.npy"
        if dset_filename.exists():
            full_dset = np.load(dset_filename)
        else:
            simulation_dirs = list(args.exp_dir.glob('sim*.zip'))
            full_dset = None
            for simulation_dir in tqdm(simulation_dirs):
                # print()
                # print(f'Loading {simulation_dir}')
                if full_dset is None:
                    full_dset = param_data_from_simulation_files(str(simulation_dir))
                else:
                    full_dset = np.concatenate([full_dset, param_data_from_simulation_files(str(simulation_dir))])
            print(f'The whole dataset has {len(full_dset)} samples')
            full_dset[:, params_order.index('school_closure_ar_threshold')] /= 100 # Convert school closure from percentage to a ratio in [0-1]
            full_dset = full_dset[:1000000] # Filter out the samples beyond 1M (if any)
            np.save(dset_filename, full_dset)
            print(f"Extracted dataset saved at {dset_filename}")
    optimality = full_dset[:, -1]
    import pdb; pdb.set_trace()

    success_rate = f'{np.sum(optimality == 1) / len(full_dset) * 100:.2f}% satisfied'
    print(success_rate)
    plot_hist(full_dset[optimality==1], file_path=str(args.out_dir / f'hist_{args.exp_dir.name}_success.pdf'),
              color='green', label='Effective policies', title=None)
    plot_hist(full_dset[optimality==0], file_path=str(args.out_dir / f'hist_{args.exp_dir.name}_failure.pdf'),
              color='red', label='Ineffective policies', title=None)

    s = f'scatter_success_{args.exp_dir.name}.pdf'
    print(f'saved to {s}')

@ex.automain
def command_line_entry(_run,_config, _seed):
    args = init(_config)
    run(args)
