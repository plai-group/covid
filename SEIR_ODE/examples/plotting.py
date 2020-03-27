import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
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
                         _plot_valid=False, _ylim=None, _shade=False):
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
        _s_idx_current = np.int(np.round(7.0/_params.dt))  # TODO - the number in here has to be equal to the planning horizon.
    else:
        _t_idx_current = 0
        _s_idx_current = 0

    if _results_visited is not None:
        _ls = '--'
        _ds = ':'
    else:
        _ls = '-'
        _ds = '-'

    # _s_idx_current = np.round(((((__t - 1) * _params.dt) + 2) / _params.dt) + 1).astype(np.int)
    # _t_idx_current = np.round(((((__t - 1) * _params.dt) + 1) / _params.dt) + 1).astype(np.int)  # Different to above as we need to prune off the first.

    _axe.cla()
    if _t_idx_current > 0:
        try:
            _p_v, _s_v, _e_v, _i_v, _r_v = get_statistics(_results_visited)
            _axe.plot(_t[:(_t_idx_current - _s_idx_current)], _s_v[:-_s_idx_current], c=mcd['green'])
            _axe.plot(_t[:(_t_idx_current - _s_idx_current)], _e_v[:-_s_idx_current], c=mcd['blue'])
            _axe.plot(_t[:(_t_idx_current - _s_idx_current)], _r_v[:-_s_idx_current], c=mcd['purple'])
            _axe.plot(_t[:(_t_idx_current - _s_idx_current)], _i_v[:-_s_idx_current], c=mcd['red'])
            _axe.plot(_t[:(_t_idx_current - _s_idx_current)], _p_v[:-_s_idx_current], 'k')
        except:
            pass

    # if _t_idx_current < (torch.max(torch.round(_t / _params.dt)) - 1):
    _p_n, _s_n, _e_n, _i_n, _r_n = get_statistics(_results_noise)
    if not _shade:
        [_axe.plot(_t[_t_idx_current - _s_idx_current:] - 1, _s_n[:, _i], c=mcd['green'],     linestyle=_ls, alpha=np.min((_alphas[_i], 1.0)), label='$S_t$') for _i in __sims_to_plot]  # np.int(np.round(__t / _params.dt))
        [_axe.plot(_t[_t_idx_current - _s_idx_current:] - 1, _e_n[:, _i], c=mcd['blue'],      linestyle=_ls, alpha=np.min((_alphas[_i], 1.0)), label='$E_t$') for _i in __sims_to_plot]
        [_axe.plot(_t[_t_idx_current - _s_idx_current:] - 1, _i_n[:, _i], c=mcd['red'],       linestyle=_ls, alpha=np.min((_alphas[_i], 1.0)), label='$I_t$', zorder=10000) for _i in __sims_to_plot]
        [_axe.plot(_t[_t_idx_current - _s_idx_current:] - 1, _r_n[:, _i], c=mcd['purple'],    linestyle=_ls, alpha=np.min((_alphas[_i], 1.0)), label='$R_t$') for _i in __sims_to_plot]
        [_axe.plot(_t[_t_idx_current - _s_idx_current:] - 1, _p_n[:, _i], 'k:', alpha=np.min((_alphas[_i], 1.0)), label='$N_t$') for _i in __sims_to_plot]
    else:
        _m = np.median(_p_n, axis=1)
        _uq = np.quantile(_p_n, 0.9, axis=1)
        _lq = np.quantile(_p_n, 0.1, axis=1)
        _axe.fill_between(_t[_t_idx_current - _s_idx_current:], _lq, _uq, color='k', alpha=0.5)
        _axe.plot(_t[_t_idx_current - _s_idx_current:], _m, color='k', zorder=-100)

        _m = np.median(_s_n, axis=1)
        _uq = np.quantile(_s_n, 0.9, axis=1)
        _lq = np.quantile(_s_n, 0.1, axis=1)
        _axe.fill_between(_t[_t_idx_current - _s_idx_current:], _lq, _uq, color=mcd['green'], alpha=0.5)
        _axe.plot(_t[_t_idx_current - _s_idx_current:], _m, color=mcd['green'], zorder=-100)

        _m = np.median(_r_n, axis=1)
        _uq = np.quantile(_r_n, 0.9, axis=1)
        _lq = np.quantile(_r_n, 0.1, axis=1)
        _axe.fill_between(_t[_t_idx_current - _s_idx_current:], _lq, _uq, color=mcd['purple'], alpha=0.5)
        _axe.plot(_t[_t_idx_current - _s_idx_current:], _m, color=mcd['purple'], zorder=-100)

        _m = np.median(_e_n, axis=1)
        _uq = np.quantile(_e_n, 0.9, axis=1)
        _lq = np.quantile(_e_n, 0.1, axis=1)
        _axe.fill_between(_t[_t_idx_current - _s_idx_current:], _lq, _uq, color=mcd['blue'], alpha=0.5)
        _axe.plot(_t[_t_idx_current - _s_idx_current:], _m, color=mcd['blue'], zorder=-100)

        _m = np.median(_i_n, axis=1)
        _uq = np.quantile(_i_n, 0.9, axis=1)
        _lq = np.quantile(_i_n, 0.1, axis=1)
        _axe.fill_between(_t[_t_idx_current - _s_idx_current:], _lq, _uq, color=mcd['red'], alpha=0.5)
        _axe.plot(_t[_t_idx_current - _s_idx_current:], _m, color=mcd['red'], zorder=-100)

    _axe.plot(_t.numpy(), (torch.ones_like(_t) * _params.policy['infection_threshold']).numpy(), 'k--', label='$C$', zorder=10000+1)
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


def peak_infection_versus_deaths(_axe, _results, _params, label=None, _prepend='', _c=None):
    # _fig, _axe = plt.subplots()

    try:
        os.mkdir('./pdf/' + _prepend)
    except:
        pass

    populace, s_n, e_n, i_n, r_n = get_statistics(_results)
    initial_pop = populace[0]
    final_pop = populace[-1]
    death_proportion = (initial_pop - final_pop) / initial_pop
    peak_infected = i_n.max(axis=0)

    # max_treatable = _params.log_max_treatable.exp().item()
    # _axe.scatter(peak_infected, death_proportion)
    #_axe.plot(max_treatable, max_treatable],
    _axe.plot(peak_infected, death_proportion, label=label, linewidth=2.0, c=_c)
    _axe.set_xlabel('Peak proportion infected')
    _axe.set_ylabel('Total mortality rate')
    _axe.set_xlim(0)
    _axe.set_ylim(0)
    # plt.savefig('./png/infected_deaths.png')
    plt.legend(loc=2, prop={'size': 8})
    plt.pause(0.1)
    plt.tight_layout()


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
    _axe.set_xlabel('$\\hat{R_0}$: Controlled exposure rate \n relative to uncontrolled exposure rate.')
    _axe.set_xlim((1.0, 0.0))

    _axe.set_ylim((0, 0.15))
    _y = plt.ylim()[1] * 0.8
    _axe.text(0.2, _y, s='$\\hat{R_0} = (1 - u)R_0$', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))

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
    _axe.set_xlabel('$\\rho$: Social contact relative to normal (at 1.0).')
    _axe.text(0.5, 0.92, s='$u=\\sqrt{(1-\\tau) \\times (1-\\rho)}$', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))


def do_family_of_plots(noised_parameters, results_noise, valid_simulations, t, _prepend, _visited_states=None, _t_now=0, _title=None, _num="", _shade=False):
    alpha, beta, typical_u, typical_alpha, typical_beta = seir.policy_tradeoff(noised_parameters)

    _zoom_lims = (0.0, 0.1)

    try: os.mkdir('./pdf/{}'.format(_prepend))
    except: pass
    try: os.mkdir('./pdf/{}/zoom'.format(_prepend))
    except: pass
    try: os.mkdir('./pdf/{}/full'.format(_prepend))
    except: pass
    try: os.mkdir('./pdf/{}/policy'.format(_prepend))
    except: pass
    #

    plt.figure(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=None, _ylim=_zoom_lims, _shade=_shade)
    if _title is not None:
        plt.title(_title)
    plt.tight_layout()
    # plt.savefig('./png/{}/{}trajectory_zoom_all{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
    plt.savefig('./pdf/{}/zoom/{}trajectory_zoom_all{}.pdf'.format(_prepend, _prepend, _num))

    # We dont want to plot valid or params if we have just done a round of control.
    if 'control' not in _num:

        plt.figure(figsize=fig_size_short)
        make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t,
                             _plot_valid=None, _shade=_shade)
        if _title is not None:
            plt.title(_title)
        plt.tight_layout()
        # plt.savefig('./png/{}/{}trajectory_full_all{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
        plt.savefig('./pdf/{}/full/{}trajectory_full_all{}.pdf'.format(_prepend, _prepend, _num))

        plt.figure(figsize=fig_size_short)
        make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=True, _shade=_shade)
        if _title is not None:
            plt.title(_title)
        plt.tight_layout()
        # plt.savefig('./png/{}/{}trajectory_full_valid{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
        plt.savefig('./pdf/{}/full/{}trajectory_full_valid{}.pdf'.format(_prepend, _prepend, _num))

        plt.figure(figsize=fig_size_short)
        make_trajectory_plot(plt.gca(), noised_parameters, _visited_states, results_noise, valid_simulations, t, _plot_valid=True, _ylim=_zoom_lims, _shade=_shade)
        if _title is not None:
            plt.title(_title)
        plt.tight_layout()
        # plt.savefig('./png/{}/{}trajectory_zoom_valid{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
        plt.savefig('./pdf/{}/zoom/{}trajectory_zoom_valid{}.pdf'.format(_prepend, _prepend, _num))

        plt.figure(figsize=fig_size_short)
        make_parameter_plot(plt.gca(), noised_parameters, valid_simulations)
        if _title is not None:
            plt.title(_title)
        plt.tight_layout()
        # plt.savefig('./png/{}/{}parameters{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
        plt.savefig('./pdf/{}/policy/{}parameters{}.pdf'.format(_prepend, _prepend, _num))

        plt.figure(figsize=fig_size_small)
        make_policy_plot(plt.gca(), noised_parameters, alpha, beta, valid_simulations, typical_u, typical_alpha, typical_beta)
        if _title is not None:
            plt.title(_title)
        plt.tight_layout()
        # plt.savefig('./png/{}/{}policy{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
        plt.savefig('./pdf/{}/policy/{}policy{}.pdf'.format(_prepend, _prepend, _num))

    plt.pause(0.1)


def nmc_plot(outer_samples, _threshold, _prepend='nmc_', _append=''):
    nmc_fig = plt.figure(10, fig_size_short)
    nmc_axe = plt.gca()
    nmc_axe.plot((0, 1), (0.9, 0.9), 'k:')
    # plt.ylim((-0.05, 1.05))
    # plt.xlim((0.0, 1.0))
    # plt.grid(True)
    # plt.ylabel('$p( Y=1 | u)$')
    # plt.xlabel('$u$')

    _c_l = np.asarray([muted_colours_dict['red'], muted_colours_dict['green']])
    _c = _c_l[(np.asarray(outer_samples['p_valid']) > _threshold).astype(np.int)]
    plt.scatter((1-outer_samples['u']), np.asarray(outer_samples['p_valid']), c=_c)

    plt.grid(True)
    plt.xlabel('$\\hat{R_0}$: Controlled exposure rate \n relative to uncontrolled exposure rate.')
    plt.text(0.2, 0.65, s='$\\hat{R_0} = (1 - u)R_0$', horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))
    _xt = plt.xticks()
    _xt = ['$' + str(int(__xt * 100)) + '\%R_0$' for __xt in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    plt.xticks((0, 0.2, 0.4, 0.6, 0.8, 1.0), _xt)
    plt.xlim((1.0, 0.0))
    plt.ylim((-0.02, 1.02))
    plt.ylabel('$p(\\forall_{t > 0} Y_t^{aux}=1 | \\theta)$')

    try: os.mkdir('./pdf/{}/policy'.format(_prepend))
    except: pass
    plt.tight_layout()
    plt.savefig('./pdf/{}/policy/{}nmc_parameters_{}.pdf'.format(_prepend, _prepend, _append))
    plt.pause(0.1)
    # plt.close(10)


def det_plot(results_deterministic, valid_simulations, params, t, _append='', _legend=False):

    # Store and overwrite the sims to plot.
    global _sims_to_plot
    _sims_to_plot_store = dc(_sims_to_plot)
    _sims_to_plot = np.arange(np.alen(valid_simulations))

    try:
        os.mkdir('./pdf/1_deterministic_')
    except:
        pass

    n_survivor = 1.0 - results_deterministic[-1, 0, -1]
    peak_infected = float( (results_deterministic[:, 0, 2] + results_deterministic[:, 0, 3] + results_deterministic[:, 0, 4]).max( axis=0).values)

    fig, ax1 = plt.subplots(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), params, None, results_deterministic, valid_simulations, t, _plot_valid="full")
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([], [])
    ax2.set_yticks([n_survivor, peak_infected])
    ax2.set_yticklabels(['$N_T$\n${:0.3f}$'.format(n_survivor), '$I_{max}$\n' + '${:0.3f}$'.format(peak_infected)])
    plt.tight_layout()
    plt.savefig('./pdf/1_deterministic_/deterministic_trajectory_full_{}.pdf'.format(_append))

    fig, ax1 = plt.subplots(figsize=fig_size_short)
    make_trajectory_plot(plt.gca(), params, None, results_deterministic, valid_simulations, t, _plot_valid="full", _ylim=(0.0, 0.21))
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([], [])
    ax2.set_yticks([peak_infected])
    ax2.set_yticklabels(['$I_{max}$\n' + '${:0.3f}$'.format(peak_infected)])
    if _legend: fig.legend(loc=(0.55, 0.6), ncol=2, prop={'size': 8})
    plt.tight_layout()
    plt.savefig('./pdf/1_deterministic_/deterministic_trajectory_zoom_{}.pdf'.format(_append))

    # Restore the global value.
    _sims_to_plot = dc(_sims_to_plot_store)


def do_controlled_plot(outer_samples, first_valid_simulation, params, N_simulation, current_state, t, _t,
                       valid_simulation, threshold, img_frame, visited_states):
    # Run simulation with the chosen control.
    controlled_parameter_values = dc({'u': torch.tensor([outer_samples['u'][first_valid_simulation]])})
    controlled_params = dc(params)
    controlled_params.u = controlled_parameter_values['u'][:] * torch.ones((N_simulation,))

    p_valid, results_noise, valid_simulations = \
        seir.nmc_estimate(current_state, params, _t * params.dt, controlled_parameter_values, valid_simulation)

    # TODO - check this code/
    tag = ''
    if torch.any(p_valid < threshold):
        if torch.any(torch.logical_not(valid_simulations)):
            print()
            print(warnings.warn('WARNING - CONSTRAINT VOLATION.'))
            print('img_frame:\t ' + str(img_frame))
            print('_t:\t\t\t ' + str(_t))
            print('p_valid:\t\t ' + str(p_valid))
            print('threshold:\t ' + str(threshold))
            print('valid_simulations: \n' + str(valid_simulations))
            # raise RuntimeError  # AAAHHHHH
            tag = '_WARNING'
            print()

    do_family_of_plots(controlled_params, results_noise, torch.ones((N_simulation,)), t,
                                _visited_states=visited_states,
                                _prepend='5_mpc_', _title='', _num='_controlled_{:05d}{}'.format(img_frame, tag))
