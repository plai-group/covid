import matplotlib.pyplot as plt
import numpy as np
import torch

# Limit the number of simulations we plot.
n_sims_to_plot = 50


# def _plot(_axe, _t, _results, _alphas):
#     # Limit the number of plots.
#     _sims_to_plot = np.random.randint(0, len(_results) - 1, n_sims_to_plot)
#
#     # Pull out key items for plotting.
#     _populace = np.sum(_results.numpy(), axis=2)
#     _t = _t.numpy()
#     _s = _results.numpy()[:, :, 0]
#     _e = _results.numpy()[:, :, 1] + _results.numpy()[:, :, 2]
#     _i = _results.numpy()[:, :, 3] + _results.numpy()[:, :, 4] + _results.numpy()[:, :, 5]
#     _r = _results.numpy()[:, :, 6]
#
#     # Plot those badboys.
#     [_axe.plot(_t, _s, 'c--', alpha=_alphas[_i]) for _i in _sims_to_plot]
#     [_axe.plot(_t, _e, 'm--', alpha=_alphas[_i]) for _i in _sims_to_plot]
#     [_axe.plot(_t, _r, 'y--', alpha=_alphas[_i]) for _i in _sims_to_plot]
#     [_axe.plot(_t, _i, 'k--', alpha=5 * _alphas[_i]) for _i in _sims_to_plot]
#     [_axe.plot(_t, _populace[:, _i], 'k:', label='N_t' if _i == 0 else None, alpha=_alphas[_i]) for _i in _sims_to_plot]


def get_statistics(_results):
    _populace = np.sum(_results.numpy(), axis=2)
    _s_n = _results.numpy()[:, :, 0]
    _e_n = _results.numpy()[:, :, 1] + _results.numpy()[:, :, 2]
    _i_n = _results.numpy()[:, :, 3] + _results.numpy()[:, :, 4] + _results.numpy()[:, :, 5]
    _r_n = _results.numpy()[:, :, 6]
    return _populace, _s_n, _e_n, _i_n, _r_n


def get_alphas(_valid_simulations, _plot_all=False):
    _alpha_valid = 0.1
    _alpha_invalid = 0.1

    if _plot_all:
        _alpha_invalid = _alpha_valid

    _alphas = _alpha_invalid + _alpha_valid * _valid_simulations.numpy().astype(np.int)
    return _alphas


def make_trajectory_plot(_axe, _params, _results_visited, _results_noise, _valid_simulations, _t, __t, _plot_all=False):
    """
    Plot the slightly crazy trajectory diagram
    :param axe:
    :return:_
    """
    # Limit the number of plots.
    try:
        _sims_to_plot = np.random.randint(0, _results_noise.size(1) - 1, n_sims_to_plot)
    except:
        _sims_to_plot = np.arange(_results_noise.size(1))

    # Get the alpha values.
    _alphas = get_alphas(_valid_simulations, _plot_all)

    _axe.cla()
    if __t > 0:
        _p_v, _s_v, _e_v, _i_v, _r_v = get_statistics(_results_visited)
        _axe.plot(_t[:__t + 1], _s_v, 'c')
        _axe.plot(_t[:__t + 1], _e_v, 'm')
        _axe.plot(_t[:__t + 1], _r_v, 'y')
        _axe.plot(_t[:__t + 1], _i_v, 'k')
        _axe.plot(_t[:__t + 1], _p_v, 'k:')

    if __t < (torch.max(torch.round(_t / _params.dt)) - 1):
        _p_n, _s_n, _e_n, _i_n, _r_n = get_statistics(_results_noise)
        [_axe.plot(_t[__t:], _s_n[:, _i], 'c--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _e_n[:, _i], 'm--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _r_n[:, _i], 'y--', alpha=_alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _i_n[:, _i], 'k--', alpha=2 * _alphas[_i]) for _i in _sims_to_plot]
        [_axe.plot(_t[__t:], _p_n[:, _i], 'k:', alpha=_alphas[_i]) for _i in _sims_to_plot]
    _axe.plot(_t.numpy(), (torch.ones_like(_t) * _params.policy['infection_threshold']).numpy(), 'k--', linewidth=3.0)
    _axe.set_xlabel('Time (~days)')
    _axe.set_ylabel('Fraction of populace')


def peak_infection_versus_deaths(_axe, _results, _params):

    populace, s_n, e_n, i_n, r_n = get_statistics(_results)
    initial_pop = populace[0]
    final_pop = populace[-1]
    death_proportion = (initial_pop - final_pop) / initial_pop
    peak_infected = i_n.max(axis=0)

    max_treatable = _params.log_max_treatable.exp().item()
    _axe.scatter(peak_infected, death_proportion)
    _axe.plot([max_treatable, max_treatable],
              [0, death_proportion.max()],
              color='k', ls='--')
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
    _axe.cla()
    _axe.hist([_new_parameters.u[torch.logical_not(_valid_simulations)].numpy(),
               _new_parameters.u[_valid_simulations].numpy()],
              100, histtype='bar', color=['red', 'green'], density=True)
    _axe.set_xlabel('U')
