import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
from types import SimpleNamespace
from copy import deepcopy as dc

import examples.ode_tools as ode

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
ode.setup_visual(args)


def get_diff(_state, _params):
    _d_s = - (np.exp(_params.log_beta)*_state[:, 0]*_state[:, 2])
    _d_e = (np.exp(_params.log_beta)*_state[:, 0]*_state[:, 2] - np.exp(_params.log_alpha)*_state[:, 1])
    _d_i = (np.exp(_params.log_alpha)*_state[:, 1] - np.exp(_params.log_gamma)*_state[:, 2])
    _d_r = (np.exp(_params.log_gamma)*_state[:, 2])
    return torch.stack((_d_s, _d_e, _d_i, _d_r)).T


def simulate_seir(_state, _params, _dt, _t, _noise_func):
    _state = dc(_state)
    _state_history = [dc(_state)]
    for _ in range(int(np.round(_t/_dt))):
        _grad = get_diff(_state, _noise_func(_params))
        _state += _grad * _dt
        _state_history.append(dc(_state))
    return torch.stack(_state_history)


if __name__ == '__main__':

    # CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

    # Define parameters
    N_sim = 100
    T = 100
    dt = .1
    N = 10000
    init_vals = torch.tensor([[1 - 1 / N, 1 / N, 0, 0]] * N_sim)
    t = torch.linspace(0, T, int(T / dt) + 1)
    dims = 4

    infection_threshold = 0.2

    log_alpha = np.log(torch.tensor((0.20, )))
    log_beta = np.log(torch.tensor((1.75, )))
    log_gamma = np.log(torch.tensor((0.25, )))

    params = SimpleNamespace(**{'log_alpha':    log_alpha,
                                'log_beta':     log_beta,
                                'log_gamma':    log_gamma})


    def prior_parameters(_params):
        """
        Draw the parameters from the prior to begin with.
        :param _params:
        :return:
        """
        _params = dc(_params)
        _params.log_alpha = log_alpha  # torch.tensor(np.log(np.random.rayleigh(np.exp(log_alpha[0]), (N_sim, ))))
        _params.log_beta = torch.tensor(np.log(np.random.rayleigh(np.exp(log_beta[0]), (N_sim, ))))
        _params.log_gamma = torch.tensor(np.log(np.random.rayleigh(np.exp(log_gamma[0]), (N_sim, ))))
        return _params


    def noise_parameters(_params):
        """
        Add some noise to the parameters as is done in Gillespies algorithm (i think).
        :param _params:
        :return:
        """
        _params = dc(_params)
        _params.log_alpha += np.random.normal(0, 0.5, (N_sim, ))
        _params.log_beta += np.random.normal(0, 0.2, (N_sim, ))
        _params.log_gamma += np.random.normal(0, 0.4, (N_sim, ))
        return _params


    def ident_parameters(_params):
        """
        Don't add any noise to the parameters.
        :param _params:
        :return:
        """
        return dc(_params)


    def valid_simulation(_state):
        _invalid = torch.logical_not(torch.any(_state[:, :, 2] > 0.2, dim=0))
        return _invalid

    def _plot(_results, _alphas):
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 0], 'c--', alpha=_alphas[_i]) for _i in range(N_sim)]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 1], 'm--', alpha=_alphas[_i]) for _i in range(N_sim)]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 3], 'y--', alpha=_alphas[_i]) for _i in range(N_sim)]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 2], 'k--', alpha=5 * _alphas[_i]) for _i in range(N_sim)]

    # noised_parameters = prior_parameters(params)
    # results_noise = simulate_seir(init_vals, noised_parameters, dt, T, ident_parameters)  # noise_parameters to use gil.
    # valid_simulations = valid_simulation(results_noise)
    # alphas = 0.0 + 0.1 * valid_simulations.numpy().astype(np.int)
    #
    #
    # plt.figure(1)
    # _plot(results_noise, alphas)
    # results_ident = simulate_seir(init_vals[0].unsqueeze(0), ident_parameters(params), dt, T, ident_parameters)
    # plt.plot(t.numpy(), results_ident.numpy()[:, 0, 0], 'c', label='S under prior', )
    # plt.plot(t.numpy(), results_ident.numpy()[:, 0, 1], 'm', label='E under prior', )
    # plt.plot(t.numpy(), results_ident.numpy()[:, 0, 2], 'k', label='I under prior', )
    # plt.plot(t.numpy(), results_ident.numpy()[:, 0, 3], 'y', label='R under prior', )
    # plt.plot(t.numpy(), (torch.ones_like(t) * infection_threshold).numpy(), 'k--', linewidth=3.0)
    # plt.ylabel('Fraction of populace.')
    # plt.xlabel('Time (~days).')
    # plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
    #           ncol=3, fancybox=True, shadow=True)
    # plt.savefig('./png/trajectory_example.png')
    #
    #
    # plt.figure(2)
    # plt.scatter(noised_parameters.log_beta, noised_parameters.log_gamma, c='r')
    # plt.scatter(noised_parameters.log_beta[valid_simulations], noised_parameters.log_gamma[valid_simulations], c='g')
    # plt.xlabel('log beta')
    # plt.ylabel('log gamma')
    #
    #
    # plt.close(1)
    # plt.close(2)


    # DO ITERATIVE REPLANNING ------------------------------------------------------------------------------------------

    ffmpeg_command = 'ffmpeg -y -r 25 -i ./png/mpc_%05d.png -c:v libx264 -vf fps=25 -tune stillimage ./mpc.mp4'
    print('ffmpeg command: ' + ffmpeg_command)

    current_state = dc(init_vals)
    parameter_traces = {'log_beta': [],
                        'log_gamma': [],
                        'valid_simulations': []}

    visited_states = [dc(current_state[0]).numpy()]

    # plt.switch_backend('agg')
    fig, axe = plt.subplots(2, 1, squeeze=True)
    sims_to_plot = np.random.randint(0, N_sim-1, 50)

    warnings.filterwarnings("ignore")

    for _t in tqdm(np.arange(int(T / dt))):


        new_parameters = prior_parameters(params)
        results_noise = simulate_seir(current_state, new_parameters, dt, T-(dt * _t), noise_parameters)
        valid_simulations = valid_simulation(results_noise)
        alphas = 0.0 + 0.1 * valid_simulations.numpy().astype(np.int)

        parameter_traces['log_beta'].append(new_parameters.log_beta)
        parameter_traces['log_gamma'].append(new_parameters.log_gamma)
        parameter_traces['valid_simulations'].append(valid_simulations)

        # The 1 is important being the simulate seir code also returns the initial state...
        next_state = results_noise[1, np.random.choice(len(valid_simulations), p=valid_simulations.numpy().astype(float)/np.sum(valid_simulations.numpy().astype(int)))]
        current_state[:] = dc(next_state)
        visited_states.append(dc(next_state).numpy())

        # Pull up the figure.
        # plt.figure()
        # plt.tight_layout()
        fig.suptitle('t={} / {} days'.format(_t / 10, T))

        # Trajectory plot.
        axe[0].cla()
        if _t > 0:
            axe[0].plot(t[:_t+1].numpy(), np.asarray(visited_states)[:-1, 0], 'c')  # -1 because we have appended next state already.
            axe[0].plot(t[:_t+1].numpy(), np.asarray(visited_states)[:-1, 1], 'm')  # -1 because we have appended next state already.
            axe[0].plot(t[:_t+1].numpy(), np.asarray(visited_states)[:-1, 3], 'y')  # -1 because we have appended next state already.
            axe[0].plot(t[:_t+1].numpy(), np.asarray(visited_states)[:-1, 2], 'k')  # -1 because we have appended next state already.
        if _t < (np.int(np.round(T / dt)) - 1):
            [axe[0].plot(t[_t:].numpy(), np.asarray(results_noise)[:, _i, 0], 'c--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[_t:].numpy(), np.asarray(results_noise)[:, _i, 1], 'm--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[_t:].numpy(), np.asarray(results_noise)[:, _i, 3], 'y--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[_t:].numpy(), np.asarray(results_noise)[:, _i, 2], 'k--', alpha=2 * alphas[_i]) for _i in sims_to_plot]
        axe[0].plot(t.numpy(), (torch.ones_like(t) * infection_threshold).numpy(), 'k--', linewidth=3.0)
        axe[0].set_xlabel('Time (~days)')
        axe[0].set_ylabel('Fraction of populace')

        # Parameter plot.
        axe[1].cla()
        axe[1].scatter(parameter_traces['log_beta'][-1], parameter_traces['log_gamma'][-1], c='r')
        axe[1].scatter(parameter_traces['log_beta'][-1][valid_simulations], parameter_traces['log_gamma'][-1][valid_simulations], c='g')
        axe[1].set_xlabel('log beta')
        axe[1].set_ylabel('log gamma')
        axe[1].set_xlim((-5, 2.5))
        axe[1].set_ylim((-5, 0.5))

        # Misc
        plt.pause(0.1)
        plt.savefig('./png/mpc_{:05}.png'.format(_t))
        p = 0

    os.system(ffmpeg_command)

    # GENERATE DATA ----------------------------------------------------------------------------------------------------

    raise NotImplementedError  # Probably don't want to go further than this.

    # Need to define an anonymous wrapper for the pytorch code.
    class ODELambda(nn.Module):
        def forward(self, t, y):
            return get_diff(y, params)

    # Go and get some data.
    with torch.no_grad():
        true_y = odeint(ODELambda(), init_vals, t, method='dopri5')
        plt.figure(1)
        plt.plot(t.numpy(), true_y.numpy()[:, 0], 'c--', label='True S',)
        plt.plot(t.numpy(), true_y.numpy()[:, 1], 'm--', label='True E',)
        plt.plot(t.numpy(), true_y.numpy()[:, 2], 'y--', label='True I',)
        plt.plot(t.numpy(), true_y.numpy()[:, 3], 'k--', label='True R',)
        plt.ylabel('Fraction of populace.')
        plt.xlabel('Time (~days).')
        plt.legend()
        plt.savefig('./png/trajectory_example.png')
        plt.close(1)

    # DO DIFF ODE LEARNING ---------------------------------------------------------------------------------------------

    ii = 0
    func = ode.ODEFunc(dims)
    optimizer = optim.Adam(func.parameters(), lr=1e-04)
    end = time.time()

    time_meter = ode.RunningAverageMeter(0.97)
    loss_meter = ode.RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = ode.get_batch(args, true_y, t)
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.pow(pred_y - batch_y, 2))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                try:
                    pred_y = odeint(func, torch.log(init_vals + ode.eps), t)
                    loss = torch.mean(torch.abs(pred_y - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    ode.visualize(args, t, true_y, torch.exp(pred_y) - ode.eps, func, itr)
                    ii += 1
                except:
                    pass

        end = time.time()
