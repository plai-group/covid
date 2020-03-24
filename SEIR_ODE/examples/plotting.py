import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.ticker as mtick

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


def get_statistics(_results):
    _populace = np.sum(_results.numpy(), axis=2)
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
        raise NotImplementedError

    _alphas = _alpha_invalid + _alpha_valid * _valid_simulations.numpy().astype(np.int)
    return _alphas


def make_trajectory_plot(_axe, _params, _results_visited, _results_noise, _valid_simulations, _t, __t,
                         _plot_valid=False, _ylim=None):
    """
    Plot the slightly crazy trajectory diagram
    :param axe:
    :return:_
    """
    # Get the alpha values.
    _alphas = get_alphas(_valid_simulations, _plot_valid)

    _axe.cla()
    if __t > 0:
        _p_v, _s_v, _e_v, _i_v, _r_v = get_statistics(_results_visited)
        _axe.plot(_t[:__t + 1], _s_v, c=mcd['green'])
        _axe.plot(_t[:__t + 1], _e_v, c=mcd['blue'])
        _axe.plot(_t[:__t + 1], _r_v, c=mcd['purple'])
        _axe.plot(_t[:__t + 1], _i_v, c=mcd['red'])
        _axe.plot(_t[:__t + 1], _p_v, 'k-.')

    if __t < (torch.max(torch.round(_t / _params.dt)) - 1):
        _p_n, _s_n, _e_n, _i_n, _r_n = get_statistics(_results_noise)
        [_axe.plot(_t[__t:], _s_n[:, _i], c=mcd['green'],     linestyle='--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _e_n[:, _i], c=mcd['blue'],      linestyle='--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _r_n[:, _i], c=mcd['purple'],    linestyle='--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _i_n[:, _i], c=mcd['red'],       linestyle='--', alpha=2*_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _p_n[:, _i], 'k-.', alpha=_alphas[_i]) for _i in _sims_to_plot]
    _axe.plot(_t.numpy(), (torch.ones_like(_t) * _params.policy['infection_threshold']).numpy(), 'k--', linewidth=2.0)
    _axe.set_xlabel('Days')
    _axe.set_ylabel('Fraction of population')

    _axe.set_xlim(left=0.0, right=(_results_noise.size(0)-1) * _params.dt)
    _axe.set_ylim(bottom=-0.01, top=1.01)

    PLOT_ON_LOG = False
    if PLOT_ON_LOG:
        _axe.set_yscale('log')
        _axe.set_ylim((0.0001, 1.0))
    else:
        if _ylim is not None:
            _axe.set_ylim(_ylim)


def peak_infection_versus_deaths(_axe, _results, _params):

    populace, s_n, e_n, i_n, r_n = get_statistics(_results)
    initial_pop = populace[0]
    final_pop = populace[-1]
    death_proportion = (initial_pop - final_pop) / initial_pop
    peak_infected = i_n.max(axis=0)

    # max_treatable = _params.log_max_treatable.exp().item()
    # _axe.scatter(peak_infected, death_proportion)
    # _axe.plot([max_treatable, max_treatable],
    #           [0, death_proportion.max()],
    #           color='k', ls='--')
    _axe.set_xlabel('Peak number infected')
    _axe.set_ylabel('Proportion of population dead')
    _axe.set_xlim(0)
    _axe.set_ylim(0)

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
    counts2, _ = np.histogram(1 - _new_parameters.u[_valid_simulations].numpy(), bins=bin_lims)

    # 2* because we are going to plot on the same axis.
    hist1b = counts1 / (2 * np.sum(counts1))
    hist2b = counts2 / (2 * np.sum(counts2))

    # 1- as we are reversing the axis.
    _axe.bar(bin_centers, hist1b, width=bin_widths, align='center', alpha=0.5, color=muted_colours_dict['red'])
    _axe.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.5, color=muted_colours_dict['green'])
    _axe.set_xlabel('$\\hat{R}_0$: Controlled exposure rate \n relative to uncontrolled $R_0$.)')
    _axe.set_xlim((1.0, 0.0))

    _axe.text(0.73, 0.92, s='$\\hat{R}_0 = (1 - u)R_0$', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))

    _xt = plt.xticks()
    _xt = ['$' + str(int(__xt * 100)) + '\\%R_0$' for __xt in list(_xt[0])]
    plt.xticks((0, 0.2, 0.4, 0.6, 0.8, 1.0), _xt)
    plt.pause(0.1)

    # _axe.xaxis.set_major_formatter(mtick.PercentFormatter())

    p = 0


def make_policy_plot(_axe, _params, _alpha, _beta, _valid_simulations, _typical_u, _typical_alpha, _typical_beta):

    _u = _params.u
    _color_list = [muted_colours_dict['red'], muted_colours_dict['green']]
    [_axe.plot(_alpha, _beta[_i], c=_color_list[_valid_simulations[_i].type(torch.int)]) for _i in _sims_to_plot]
    _axe.plot(_typical_alpha, _typical_beta, 'k', alpha=0.8)
    # _axe.scatter((1.0, ), (1.0, ), c='k', marker='x')
    # _axe.axis('equal')
    _axe.set_xlim(left=0.0, right=1.0)
    _axe.set_ylim(bottom=0.0, top=1.0)
    _axe.grid(True)
    _axe.set_ylabel('$\\tau$: Transmission rate relative to normal (at 1.0).')
    _axe.set_xlabel('$\\eta$: Social contact relative to normal (at 1.0).')
    _axe.text(0.73, 0.92, s='$u=\\sqrt{\\tau \\times \\eta}$', bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))
