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

# Import the SEIR module.
import examples.seir as seir
import examples.plotting as plotting

# Fix the type of the state tensors.
float_type = seir.float_type

# Kill some matplotlib warnings.
warnings.filterwarnings("ignore")


# CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

# Experiments to run.
experiment_single_rollout = False
experiment_nmc_example = True
experiment_mpc_example = True

# Define base parameters of SEIR model.
log_kappa =     torch.log(torch.tensor((0.057 / 3.3, )))
log_alpha = torch.log(torch.tensor((1 / 5.1, )))
log_r0 = torch.log(torch.tensor((2.6, )))
log_gamma = torch.log(torch.tensor((1 / 3.3,)))
log_lambda = torch.log(torch.tensor((0.00116 / 365, )))  # US birth rate from https://tinyurl.com/sezkqxc
log_mu = torch.log(torch.tensor((0.008678 / 365, )))     # non-covid death rate https://tinyurl.com/ybwdzmjs

# Make sure we are controlling the right things.
controlled_parameters = ['log_r0']  # We can select r0.
uncontrolled_parameters = ['log_kappa', 'log_alpha', 'log_gamma', 'log_lambda', 'log_mu']

# Define the simulation properties.
T = 200
dt = .1
initial_population = 10000

# Define the policy objectives.
infection_threshold = torch.scalar_tensor(0.1)

# Define inference settings.
N_simulation = 100

# Automatically define other required variables.
t = torch.linspace(0, T, int(T / dt) + 1)
params = SimpleNamespace(**{'log_alpha': log_alpha,
                            'log_r0': log_r0,
                            'log_gamma': log_gamma,
                            'log_mu': log_mu,
                            'log_kappa': log_kappa,
                            'log_lambda': log_lambda,
                            'controlled_parameters': controlled_parameters,
                            'uncontrolled_parameters': uncontrolled_parameters,
                            'policy': {'infection_threshold': infection_threshold},
                            'dt': dt})


def valid_simulation(_state, _params):
    """
    AW - valid_simulation - return a binary variable per simulation indicating whether or not that simulation
    satifies the desired policy outcome.
    :param _state: tensor (N x D):  tensor of the state trajectory.
    :return: tensor (N, ), bool:    tensor of whether each simulation was valid.
    """
    _n_infected = _state[:, :, 3] + _state[:, :, 4] + _state[:, :, 5]
    _valid = torch.logical_not(torch.any(_n_infected > _params.policy['infection_threshold'], dim=0))
    return _valid


if __name__ == '__main__':

    # DO SINGLE ROLLOUT PLOT -------------------------------------------------------------------------------------------

    if experiment_single_rollout:

        initial_state = seir.sample_x0(N_simulation, initial_population)
        noised_parameters = seir.sample_prior_parameters(params, N_simulation)
        results_noise = seir.simulate_seir(initial_state, noised_parameters, dt, T, seir.sample_unknown_parameters)  # noise_parameters to use gil.
        valid_simulations = valid_simulation(results_noise, noised_parameters)

        fig, axe = plt.subplots(2, 1, squeeze=True)
        plotting.make_trajectory_plot(axe[0], noised_parameters, None, results_noise, valid_simulations, t, 0, _plot_all=True)
        plotting.make_parameter_plot(axe[1], noised_parameters, valid_simulations)
        plt.savefig('./png/trajectory.png', dpi=500)
        plt.close()

    # DO SINGLE NMC EXPERIMENT -----------------------------------------------------------------------------------------

    if experiment_nmc_example:

        time = 0.0
        current_state = seir.sample_x0(N_simulation, initial_population)

        outer_samples = {'log_r0': [],
                         'p_valid': []}

        N_outer_samples = 1000

        fig_nmc = plt.figure(1)
        plt.plot((0, 1), (0.9, 0.9), 'k:')
        plt.ylim((-0.05, 1.05))
        # plt.xlim((-0.05, 1.05))
        plt.grid(True)
        plt.title('p( Y=1 | theta)')

        for _i in range(N_outer_samples):

            # Get a realization of parameters.
            _params = seir.sample_prior_parameters(params, 1)

            # Put the controlled parameter values into
            controlled_parameter_values = {'log_r0': _params.log_r0}

            def _nmc_estimate(_current_state, _controlled_parameters):
                """
                AW - _nmc_estimate - calculate the probability that the specified parameters will
                :param _current_state:
                :param _new_parameters:
                :return:
                """
                # Draw the parameters we wish to marginalize over.
                _new_parameters = seir.sample_prior_parameters(params, N_simulation)

                # Overwrite with the specified parameter value.
                _new_parameters.log_r0[:] = _controlled_parameters['log_r0']

                # Run the simulation with the controlled parameters, marginalizing over the others.
                results_noise = seir.simulate_seir(_current_state, _new_parameters, dt, T - time, seir.sample_unknown_parameters)
                valid_simulations = valid_simulation(results_noise, _new_parameters)
                p_valid = valid_simulations.type(torch.float).mean()
                return p_valid

            # Call the NMC subroutine using the parameters and current state.
            p_valid = _nmc_estimate(current_state, controlled_parameter_values)

            # Record and plot.
            outer_samples['log_r0'].append(controlled_parameter_values['log_r0'][0].numpy())
            outer_samples['p_valid'].append(p_valid)
            plt.scatter(outer_samples['log_r0'][-1], np.asarray(outer_samples['p_valid'][-1]))
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

        visited_states = [[dc(current_state[0]).numpy()]]

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
