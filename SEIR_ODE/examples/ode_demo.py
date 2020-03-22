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

# import examples.ode_tools as ode
#
# parser = argparse.ArgumentParser('ODE demo')
# parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
# parser.add_argument('--batch_time', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=5)
# parser.add_argument('--niters', type=int, default=10000)
# parser.add_argument('--test_freq', type=int, default=100)
# parser.add_argument('--viz', action='store_true', default=True)
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', action='store_true')
# args = parser.parse_args()
#
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
#     from torchdiffeq import odeint
#
# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# ode.setup_visual(args)

float_type = torch.float64

# Do we want to re-normalize the population online? Probably not.
DYNAMIC_NORMALIZATION = False

# Kill some matplotlib warnings.
warnings.filterwarnings("ignore")


def get_diff(_state, _params):

    mu = _params.mu
    r0 = np.exp(_params.log_r0)
    gamma = np.exp(_params.log_gamma)
    alpha = np.exp(_params.log_alpha)

    S, E1, E2, I1, I2, I3, R = tuple(_state[:, i] for i in range(_state.shape[1]))
    N = _state.sum(dim=1)

    # TODO - implement death and birth.

    s_to_e1 = ((1 - mu) * r0 * gamma * S * (I1 + I2 + I3) / N).type(float_type)
    e1_to_e2 = (2*alpha*E1).type(float_type)
    e2_to_i1 = (2*alpha*E2).type(float_type)
    i1_to_i2 = (3*gamma*I1).type(float_type)
    i2_to_i3 = (3*gamma*I2).type(float_type)
    i3_to_r = (3*gamma*I3).type(float_type)

    _d_s = -s_to_e1
    _d_e1 = s_to_e1 - e1_to_e2
    _d_e2 = e1_to_e2 - e2_to_i1
    _d_i1 = e2_to_i1 - i1_to_i2 - mu*I1
    _d_i2 = i1_to_i2 - i2_to_i3 - mu*I2
    _d_i3 = i2_to_i3 - i3_to_r - mu*I3
    _d_r = i3_to_r

    return torch.stack((_d_s, _d_e1, _d_e2, _d_i1, _d_i2, _d_i3, _d_r), dim=1)


def simulate_seir(_state, _params, _dt, _t, _noise_func):
    _state = dc(_state)
    _state_history = [dc(_state)]
    for _ in range(int(np.round(_t/_dt))):
        _n = torch.sum(_state, dim=1)
        _grad = get_diff(_state, _noise_func(_params))
        _state += _grad * _dt
        _new_n = torch.sum(_state, dim=1)

        if DYNAMIC_NORMALIZATION:
            _f_d_n = _n / _new_n
            _state *= _f_d_n.unsqueeze(1)

        _state_history.append(dc(_state))
    return torch.stack(_state_history)


if __name__ == '__main__':

    # CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

    # Define parameters
    N_sim = 100
    T = 200
    dt = .1
    N = 10000
    init_vals = torch.tensor([[1 - 1 / N, 1 / N, 0, 0, 0, 0, 0]] * N_sim)
    t = torch.linspace(0, T, int(T / dt) + 1)
    dims = 4

    infection_threshold = 0.1

    mu = 0.1
    log_alpha = np.log(torch.tensor((1/5.1, )))
    log_r0 = np.log(torch.tensor((2.6, )))
    log_gamma = np.log(torch.tensor((1/3.3, )))

    params = SimpleNamespace(**{'log_alpha':    log_alpha,
                                'log_r0':       log_r0,
                                'log_gamma':    log_gamma,
                                'mu':           mu})


    def sample_prior_parameters(_params):
        """
        Draw the parameters from the prior to begin with.
        :param _params:
        :return:
        """
        # TODO - WILL - make sure the prior is parameterized correct and has the correct conditional dependancies.
        _params = dc(_params)
        _params.mu = mu
        _params.log_alpha = torch.tensor(np.log(np.random.rayleigh(np.exp(log_alpha[0]), (N_sim, ))))
        _params.log_r0 =    torch.tensor(np.log(np.random.rayleigh(np.exp(log_r0[0]), (N_sim, ))))
        _params.log_gamma = torch.tensor(np.log(np.random.rayleigh(np.exp(log_gamma[0]), (N_sim, ))))
        return _params


    def sample_noise_parameters(_params):
        """
        Add some noise to the parameters as is done in Gillespies algorithm (i think).
        :param _params:
        :return:
        """
        _params = dc(_params)
        _params.log_alpha = torch.tensor(np.log(np.random.rayleigh(np.exp(log_alpha[0]), (N_sim, ))))
        _params.log_r0 += np.random.normal(0, 0.01, (N_sim, ))
        _params.log_gamma = torch.tensor(np.log(np.random.rayleigh(np.exp(log_gamma[0]), (N_sim, ))))
        return _params


    def sample_unkno_parameters(_params):
        """
        Sample the parameters we do not fix and hence wish to marginalize over.
        :param _params:
        :return:
        """
        _params = dc(_params)
        _params_prior = sample_prior_parameters(_params)
        _params.log_alpha = _params_prior.log_alpha
        _params.log_gamma = _params_prior.log_gamma
        return _params


    def sample_ident_parameters(_params):
        """
        Don't add any noise to the parameters.
        :param _params:
        :return:
        """
        return dc(_params)


    def valid_simulation(_state):
        _valid = torch.logical_not(torch.any((_state[:, :, 3] + _state[:, :, 4] + _state[:, :, 5]) > infection_threshold, dim=0))
        return _valid

    def _plot(_results, _alphas):
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 0], 'c--', alpha=_alphas[_i]) for _i in sims_to_plot]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 1] + _results.numpy()[:, _i, 2], 'm--', alpha=_alphas[_i]) for _i in sims_to_plot]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 6], 'y--', alpha=_alphas[_i]) for _i in sims_to_plot]
        [plt.plot(t.numpy(), _results.numpy()[:, _i, 3] + _results.numpy()[:, _i, 4] + _results.numpy()[:, _i, 5], 'k--', alpha=5 * _alphas[_i]) for _i in sims_to_plot]
        populace = _results.numpy()[:, :, 0] + _results.numpy()[:, :, 1] + _results.numpy()[:, :, 2] + \
                   _results.numpy()[:, :, 3] + _results.numpy()[:, :, 4] + _results.numpy()[:, :, 5] + \
                   _results.numpy()[:, :, 6]
        [plt.plot(t.numpy(), populace[:, _i], 'k:', label='N_t' if _i == 0 else None, alpha=_alphas[_i]) for _i in sims_to_plot]

    def make_trajectory_plot(_axe, _visited_states, _results_noise, _t, __t):
        """
        Plot the slightly crazy trajectory diagram
        :param axe:
        :return:_
        """
        _axe[0].cla()
        if __t > 0:
            _axe[0].plot(_t[:__t + 1].numpy(), np.asarray(_visited_states)[:-1, 0], 'c')
            _axe[0].plot(_t[:__t + 1].numpy(), np.asarray(_visited_states)[:-1, 1] + np.asarray(_visited_states)[:-1, 2], 'm')
            _axe[0].plot(_t[:__t + 1].numpy(), np.asarray(_visited_states)[:-1, 6], 'y')
            _axe[0].plot(_t[:__t + 1].numpy(), np.asarray(_visited_states)[:-1, 3] + np.asarray(_visited_states)[:-1, 4] + np.asarray(_visited_states)[:-1, 5], 'k')
            populace = np.asarray(_visited_states)[:-1, 0] + np.asarray(_visited_states)[:-1, 1] + \
                       np.asarray(_visited_states)[:-1, 2] + np.asarray(_visited_states)[:-1, 3] + \
                       np.asarray(_visited_states)[:-1, 4] + np.asarray(_visited_states)[:-1, 5] + \
                       np.asarray(_visited_states)[:-1, 6]
            axe[0].plot(_t[:__t + 1].numpy(), populace, 'k:', label='N_t')
        if __t < (np.int(np.round(T / dt)) - 1):
            [axe[0].plot(t[__t:].numpy(), np.asarray(_results_noise)[:, _i, 0], 'c--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[__t:].numpy(), np.asarray(_results_noise)[:, _i, 1] + np.asarray(_results_noise)[:, _i, 2], 'm--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[__t:].numpy(), np.asarray(_results_noise)[:, _i, 6], 'y--', alpha=alphas[_i]) for _i in sims_to_plot]
            [axe[0].plot(t[__t:].numpy(), np.asarray(_results_noise)[:, _i, 3] + np.asarray(_results_noise)[:, _i, 4] + np.asarray(_results_noise)[:, _i, 5], 'k--', alpha=2 * alphas[_i]) for _i in sims_to_plot]
            populace = np.asarray(_results_noise)[:, :, 0] + np.asarray(_results_noise)[:, :, 1] + np.asarray(_results_noise)[:, :, 2] + \
                       np.asarray(_results_noise)[:, :, 3] + np.asarray(_results_noise)[:, :, 4] + np.asarray(_results_noise)[:, :, 5] + \
                       np.asarray(_results_noise)[:, :, 6]
            [axe[0].plot(_t[__t:].numpy(), populace[:, _i], 'k:', alpha=alphas[_i]) for _i in sims_to_plot]
        axe[0].plot(_t.numpy(), (torch.ones_like(_t) * infection_threshold).numpy(), 'k--', linewidth=3.0)
        axe[0].set_xlabel('Time (~days)')
        axe[0].set_ylabel('Fraction of populace')

    def make_parameter_plot(_axe, _new_parameters, _valid_simulations):
        """
        Plot the 1-D parameter histogram.
        :param _axe:
        :param _new_parameters:
        :param _valid_simulations:
        :return:
        """
        _axe[1].cla()
        _axe[1].hist([np.exp(_new_parameters.log_r0[torch.logical_not(_valid_simulations)].numpy()),
                     np.exp(_new_parameters.log_r0[_valid_simulations].numpy())],
                    100, histtype='bar', color=['red', 'green'], density=True)
        _axe[1].set_xlabel('R0')


    # DO SINGLE ROLLOUT PLOT -------------------------------------------------------------------------------------------

    if False:
        noised_parameters = sample_prior_parameters(params)
        results_noise = simulate_seir(init_vals, noised_parameters, dt, T, sample_noise_parameters)  # noise_parameters to use gil.
        valid_simulations = valid_simulation(results_noise)
        alphas = 0.1 + 0.0 * valid_simulations.numpy().astype(np.int)
        sims_to_plot = np.random.randint(0, N_sim-1, 500)


        plt.figure(1)
        _plot(results_noise, alphas)

        results_ident = simulate_seir(init_vals[0].unsqueeze(0), sample_ident_parameters(params), dt, T, sample_ident_parameters)
        plt.plot(t.numpy(), results_ident.numpy()[:, 0, 0], 'c', label='S under prior', )
        plt.plot(t.numpy(), results_ident.numpy()[:, 0, 1] + results_ident.numpy()[:, 0, 2], 'm', label='E under prior', )
        plt.plot(t.numpy(), results_ident.numpy()[:, 0, 3] + results_ident.numpy()[:, 0, 4] + results_ident.numpy()[:, 0, 5], 'k', label='I under prior', )
        plt.plot(t.numpy(), results_ident.numpy()[:, 0, 6], 'y', label='R under prior', )
        populace = results_ident.numpy()[:, 0, 0] + results_ident.numpy()[:, 0, 1] + results_ident.numpy()[:, 0, 2] + \
                   results_ident.numpy()[:, 0, 3] + results_ident.numpy()[:, 0, 4] + results_ident.numpy()[:, 0, 5] + \
                   results_ident.numpy()[:, 0, 6]
        plt.plot(t.numpy(), populace, 'k:', label='N_t', )

        plt.plot(t.numpy(), (torch.ones_like(t) * infection_threshold).numpy(), 'k--', linewidth=3.0)

        plt.ylabel('Fraction of populace.')
        plt.xlabel('Time (~days).')
        plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=3, fancybox=True, shadow=True)
        plt.savefig('./png/trajectory_example.png')

        plt.figure(2)
        plt.gca().hist([np.exp(noised_parameters.log_r0[torch.logical_not(valid_simulations)].numpy()),
                        np.exp(noised_parameters.log_r0[valid_simulations].numpy())],
                       100, histtype='bar', color=['red', 'green'], density=True)
        plt.xlabel('R0')

        plt.close(1)
        plt.close(2)


    # DO SINGLE NMC ESTIMATE -------------------------------------------------------------------------------------------

    if True:

        time = 0.0
        current_state = dc(init_vals)
        parameter_traces = {'log_r0': [],
                            'p_valid': []}

        N_outer = 1000

        fig = plt.figure(1)
        for _i in range(N_outer):
            new_parameters = sample_prior_parameters(params)

            # Overwrite with the first set of parameters.
            new_parameters.log_r0[:] = new_parameters.log_r0[0]

            results_noise = simulate_seir(current_state, new_parameters, dt, T - time, sample_unkno_parameters)
            valid_simulations = valid_simulation(results_noise)
            p_valid = valid_simulations.type(torch.float).mean()

            parameter_traces['log_r0'].append(new_parameters.log_r0[0].numpy())
            parameter_traces['p_valid'].append(p_valid)

            plt.cla()
            plt.scatter(parameter_traces['log_r0'], np.asarray(parameter_traces['p_valid']))
            plt.plot((0, 1), (0.9, 0.9), 'k:')
            plt.ylim((-0.05, 1.05))
            # plt.xlim((-0.05, 1.05))
            plt.grid(True)
            plt.title('p( Y=1 | theta)')
            plt.pause(0.1)

        # Misc
        p = 0


    # DO ITERATIVE REPLANNING ------------------------------------------------------------------------------------------

    if False:

        ffmpeg_command = 'ffmpeg -y -r 25 -i ./png/mpc_%05d.png -c:v libx264 -vf fps=25 -tune stillimage ./mpc.mp4'
        print('ffmpeg command: ' + ffmpeg_command)

        current_state = dc(init_vals)
        parameter_traces = {'log_r0': [],
                            'valid_simulations': []}

        visited_states = [dc(current_state[0]).numpy()]

        # plt.switch_backend('agg')
        fig, axe = plt.subplots(2, 1, squeeze=True)
        sims_to_plot = np.random.randint(0, N_sim-1, 50)

        for _t in tqdm(np.arange(int(T / dt))):

            new_parameters = sample_prior_parameters(params)
            results_noise = simulate_seir(current_state, new_parameters, dt, T-(dt * _t), sample_noise_parameters)
            valid_simulations = valid_simulation(results_noise)
            alphas = 0.0 + 0.1 * valid_simulations.numpy().astype(np.int)

            parameter_traces['log_r0'].append(new_parameters.log_r0)
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
            make_trajectory_plot(axe)

            axe[1].cla()
            axe[1].hist([np.exp(new_parameters.log_r0[torch.logical_not(valid_simulations)].numpy()),
                         np.exp(new_parameters.log_r0[valid_simulations].numpy())],
                        100, histtype='bar', color=['red', 'green'], density=True)
            axe[1].set_xlabel('R0')

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
