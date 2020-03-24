import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.ticker as mtick
import os

from copy import deepcopy as dc

# Import SEIR module.
import examples.seir as seir

# Limit the number of simulations we plot.
n_sims_to_plot = 100
_sims_to_plot = np.random.randint(0, 1000, n_sims_to_plot)

# `pip install colormap; pip install easydev`
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
mcd = muted_colours_dict

# Real misc shit.
fig_size_wide = (12, 3)
fig_size_small = (4, 4)
fig_size_short = (4, 2)
dpi=100


def get_statistics(_results):
    _populace = np.sum(_results[:, :, :-1].numpy(), axis=2)
    _s_n = _results.numpy()[:, :, 0]
    _e_n = _results.numpy()[:, :, 1]
    _i_n = _results.numpy()[:, :, 2] + _results.numpy()[:, :, 3] + _results.numpy()[:, :, 4]
    _r_n = _results.numpy()[:, :, 5]
    return _populace, _s_n, _e_n, _i_n, _r_n


def get_alphas(_valid_simulations, _plot_valid=None):
    _alpha_valid_base = 0.25
    _alpha_invalid_base = 0.25

    if _plot_valid is None:
        _alpha_valid = _alpha_valid_base
        _alpha_invalid = _alpha_invalid_base
    elif _plot_valid is True:
        _alpha_valid = _alpha_valid_base * 2
        _alpha_invalid = 0.0
    elif _plot_valid is False:
        _alpha_valid = 0.0
        _alpha_invalid = _alpha_invalid_base * 2
    else:
        _alpha_invalid = 2.0
        _alpha_valid = 2.0

    _alphas = _alpha_invalid + _alpha_valid * _valid_simulations.numpy().astype(np.int)
    return _alphas


def make_trajectory_plot(_axe, _params, _results_visited, _results_noise, _valid_simulations, _t,
                         _plot_valid=False, _ylim=None):
    """
    Plot the slightly crazy trajectory diagram
    :param axe:
    :return:_
    """
    # Get the alpha values.
    _alphas = get_alphas(_valid_simulations, _plot_valid)

    # Limit the number of plots.
    if len(_sims_to_plot) > len(_valid_simulations):
        __sims_to_plot = range(len(_valid_simulations))
    else:
        __sims_to_plot = _sims_to_plot

    if _results_visited is not None:
        _t_idx_current = len(_results_visited)
        _s_idx_current = np.int(np.round(1.0/_params.dt))
    else:
        _t_idx_current = 0
        _s_idx_current = 0

    # TODO - if we have supplied visited states, results noise is a projection so make those lines dashed.

    # _s_idx_current = np.round(((((__t - 1) * _params.dt) + 2) / _params.dt) + 1).astype(np.int)
    # _t_idx_current = np.round(((((__t - 1) * _params.dt) + 1) / _params.dt) + 1).astype(np.int)  # Different to above as we need to prune off the first.

    _axe.cla()
    if _t_idx_current > 0:
        _p_v, _s_v, _e_v, _i_v, _r_v = get_statistics(_results_visited)
        _axe.plot(_t[:_t_idx_current], _s_v, c=mcd['green'])
        _axe.plot(_t[:_t_idx_current], _e_v, c=mcd['blue'])
        _axe.plot(_t[:_t_idx_current], _r_v, c=mcd['purple'])
        _axe.plot(_t[:_t_idx_current], _i_v, c=mcd['red'])
        _axe.plot(_t[:_t_idx_current], _p_v, 'k:')

    # if _t_idx_current < (torch.max(torch.round(_t / _params.dt)) - 1):
    _p_n, _s_n, _e_n, _i_n, _r_n = get_statistics(_results_noise)
    [_axe.plot(_t[_t_idx_current:], _s_n[_s_idx_current:, _i], c=mcd['green'],     linestyle='-', alpha=np.min((_alphas[_i], 1.0)), label='$S_t$') for _i in __sims_to_plot]  # np.int(np.round(__t / _params.dt))
    [_axe.plot(_t[_t_idx_current:], _e_n[_s_idx_current:, _i], c=mcd['blue'],      linestyle='-', alpha=np.min((_alphas[_i], 1.0)), label='$E_t$') for _i in __sims_to_plot]
    [_axe.plot(_t[_t_idx_current:], _i_n[_s_idx_current:, _i], c=mcd['red'],       linestyle='-', alpha=np.min((2*_alphas[_i], 1.0)), label='$I_t$', zorder=10000) for _i in __sims_to_plot]
    [_axe.plot(_t[_t_idx_current:], _r_n[_s_idx_current:, _i], c=mcd['purple'],    linestyle='-', alpha=np.min((_alphas[_i], 1.0)), label='$R_t$') for _i in __sims_to_plot]
    [_axe.plot(_t[_t_idx_current:], _p_n[_s_idx_current:, _i], 'k:', alpha=np.min((_alphas[_i], 1.0)), label='$N_t$') for _i in __sims_to_plot]
    _axe.plot(_t.numpy(), (torch.ones_like(_t) * _params.policy['infection_threshold']).numpy(), 'k--', linewidth=2.0, label='$C$', zorder=10000-1)
    # FI.
    _axe.set_xlabel('Days')
    _axe.set_ylabel('Fraction of population')

    _axe.set_xlim(left=0.0, right=_params.T)
    _axe.set_ylim(bottom=-0.05, top=1.05)

    PLOT_ON_LOG = False
    if PLOT_ON_LOG:
        _axe.set_yscale('log')
        _axe.set_ylim((0.000001, 1.0))
    else:
        if _ylim is not None:
            _axe.set_ylim(_ylim)


def peak_infection_versus_deaths(_results, _params):
    _fig, _axe = plt.subplots()

    populace, s_n, e_n, i_n, r_n = get_statistics(_results)
    initial_pop = populace[0]
    final_pop = populace[-1]
    death_proportion = (initial_pop - final_pop) / initial_pop
    peak_infected = i_n.max(axis=0)

    # max_treatable = _params.log_max_treatable.exp().item()
    # _axe.scatter(peak_infected, death_proportion)
    #_axe.plot(max_treatable, max_treatable],
    _axe.scatter(peak_infected, death_proportion, color='k')
    _axe.set_xlabel('Peak number infected')
    _axe.set_ylabel('Proportion of population dead')
    _axe.set_xlim(0)
    _axe.set_ylim(0)
    plt.savefig('./png/infected_deaths.png')
    plt.savefig('./pdf/infected_deaths.pdf')


def make_parameter_plot(_axe, _new_parameters, _valid_simulations):
    """
    Plot the 1-D parameter histogram.
    :param _axe:
    :param _new_parameters:
    :param _valid_simulations:
    :return:
    """

    _param = _new_parameters.u * 100

    _axe.cla()

    num_bin = 11
    bin_lims = np.linspace(0, 1, num_bin + 1)
    bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    bin_widths = bin_lims[1:] - bin_lims[:-1]

    counts1, _ = np.histogram(1 - _new_parameters.u[torch.logical_not(_valid_simulations)].numpy(), bins=bin_lims)
    counts2, _ = np.histogram(1 - _new_parameters.u[_valid_simulations.type(torch.bool)].numpy(), bins=bin_lims)

    hist1b = counts1 / (np.sum(counts1) + np.sum(counts2))
    hist2b = counts2 / (np.sum(counts1) + np.sum(counts2))

    # 1- as we are reversing the axis.
    _axe.bar(bin_centers, hist1b, width=bin_widths, align='center', alpha=0.5, color=muted_colours_dict['red'])
    _axe.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.5, color=muted_colours_dict['green'])
    _axe.set_xlabel('$\\hat{R}_0$: Controlled exposure rate \n relative to uncontrolled exposure rate.')
    _axe.set_xlim((1.0, 0.0))

    _axe.set_ylim((0, 0.15))
    _y = plt.ylim()[1] * 0.8
    _axe.text(0.2, _y, s='$\\hat{R}_0 = (1 - u)R_0$', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))

    _xt = plt.xticks()
    _xt = ['$' + str(int(__xt * 100)) + '\\%R_0$' for __xt in list(_xt[0])]
    plt.xticks((0, 0.2, 0.4, 0.6, 0.8, 1.0), _xt)
    plt.pause(0.1)

    # _axe.xaxis.set_major_formatter(mtick.PercentFormatter())

    p = 0


def make_policy_plot(_axe, _params, _alpha, _beta, _valid_simulations, _typical_u, _typical_alpha, _typical_beta):

    _u = _params.u
    _color_list = [muted_colours_dict['red'], muted_colours_dict['green']]
    [_axe.plot(_alpha, _beta[_i], c=_color_list[_valid_simulations[_i].type(torch.int)]) for _i in range(len(_valid_simulations))]
    # _axe.plot(_typical_alpha, _typical_beta, 'k', alpha=0.8)
    # _axe.scatter((1.0, ), (1.0, ), c='k', marker='x')
    # _axe.axis('equal')
    _axe.set_xlim(left=0.0, right=1.0)
    _axe.set_ylim(bottom=0.0, top=1.0)
    _axe.grid(True)
    _axe.set_ylabel('$\\tau$: Transmission rate relative to normal (at 1.0).')
    _axe.set_xlabel('$\\eta$: Social contact relative to normal (at 1.0).')
    _axe.text(0.73, 0.92, s='$u=\\sqrt{\\tau \\times \\eta}$', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))


def do_family_of_plots(noised_parameters, results_noise, valid_simulations, t, _prepend, _visited_states=None, _t_now=0, _title=None, _num=""):
    alpha, beta, typical_u, typical_alpha, typical_beta = seir.policy_tradeoff(noised_parameters)

    _zoom_lims = (0.0, 0.2)

    os.mkdir('./pdf/{}'.format(_prepend))

    plt.figure(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=True)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}trajectory_full_valid{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}trajectory_full_valid{}.pdf'.format(_prepend, _prepend, _num))

    plt.figure(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=True, _ylim=_zoom_lims)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}trajectory_zoom_valid{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}trajectory_zoom_valid{}.pdf'.format(_prepend, _prepend, _num))

    plt.figure(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=None)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}trajectory_full_all{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}trajectory_full_all{}.pdf'.format(_prepend, _prepend, _num))

    plt.figure(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=None, _ylim=_zoom_lims)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}trajectory_zoom_all{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}trajectory_zoom_all{}.pdf'.format(_prepend, _prepend, _num))

    plt.figure(figsize=fig_size_short)
    make_parameter_plot(plt.gca(), noised_parameters, valid_simulations)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}parameters{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}parameters{}.pdf'.format(_prepend, _prepend, _num))

    plt.figure(figsize=fig_size_small)
    make_policy_plot(plt.gca(), noised_parameters, alpha, beta, valid_simulations, typical_u, typical_alpha, typical_beta)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}policy{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/{}policy{}.pdf'.format(_prepend, _prepend, _num))

    plt.pause(0.1)


def nmc_plot(outer_samples, _threshold):
    nmc_fig = plt.figure(10, fig_size_small)
    nmc_axe = plt.gca()
    plt.plot((0, 1), (0.9, 0.9), 'k:')
    plt.ylim((-0.05, 1.05))
    plt.xlim((0.0, 1.0))
    plt.grid(True)
    plt.ylabel('$p( Y=1 | u)$')
    plt.xlabel('$u$')

    _c_l = np.asarray([muted_colours_dict['red'], muted_colours_dict['green']])
    _c = _c_l[(np.asarray(outer_samples['p_valid']) > _threshold).astype(np.int)]

    nmc_axe.scatter(outer_samples['u'], np.asarray(outer_samples['p_valid']), c=_c)
    plt.savefig('./png/nmc.png')
    plt.pause(0.1)
    plt.close(10)


def det_plot(results_deterministic, valid_simulations, params, t, _append='', _legend=False):

    # Store and overwrite the sims to plot.
    global _sims_to_plot
    _sims_to_plot_store = dc(_sims_to_plot)
    _sims_to_plot = np.arange(np.alen(valid_simulations))

    n_survivor = 1.0 - results_deterministic[-1, 0, -1]
    peak_infected = float( (results_deterministic[:, 0, 2] + results_deterministic[:, 0, 3] + results_deterministic[:, 0, 4]).max( axis=0).values)

    fig, ax1 = plt.subplots(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), params, None, results_deterministic, valid_simulations, t, _plot_valid="full")
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([], [])
    ax2.set_yticks([n_survivor, peak_infected])
    ax2.set_yticklabels(['$N_T=$\n${:0.3f}$'.format(n_survivor), '$I_{max}=$\n' + '${:0.3f}$'.format(peak_infected)])
    plt.tight_layout()
    plt.savefig('./pdf/deterministic/deterministic_trajectory_full_{}.pdf'.format(_append))

    fig, ax1 = plt.subplots(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), params, None, results_deterministic, valid_simulations, t, _plot_valid="full", _ylim=(0.0, 0.2))
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([], [])
    ax2.set_yticks([peak_infected])
    ax2.set_yticklabels(['$I_{max}=$\n' + '${:0.3f}$'.format(peak_infected)])
    if _legend: fig.legend(loc=(0.5, 0.6), ncol=2, prop={'size': 8})
    plt.tight_layout()
    plt.savefig('./pdf/deterministic/deterministic_trajectory_zoom_nominal.pdf'.format())

    # Restore the global value.
    _sims_to_plot = dc(_sims_to_plot_store)
