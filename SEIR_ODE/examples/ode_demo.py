import os
import argparse
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

# Set up tex in matplotlib.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble='\\usepackage{amsmath,amssymb}')


# CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

# Experiments to run.
experiment_single_rollout = True
experiment_nmc_example = True
experiment_mpc_example = True

# Define base parameters of SEIR model.
log_kappa =     torch.log(torch.tensor((0.057 / 3.3, )))
log_alpha = torch.log(torch.tensor((1 / 5.1, )))
log_r0 = torch.log(torch.tensor((2.6, )))
log_gamma = torch.log(torch.tensor((1 / 3.3,)))
log_lambda = torch.log(torch.tensor((0.00116 / 365, )))  # US birth rate from https://tinyurl.com/sezkqxc
log_mu = torch.log(torch.tensor((0.008678 / 365, )))     # non-covid death rate https://tinyurl.com/ybwdzmjs
u = torch.tensor((0., ))

# Make sure we are controlling the right things.
controlled_parameters = ['log_u']  # We can select u.
uncontrolled_parameters = ['log_kappa', 'log_alpha', 'log_gamma', 'log_lambda', 'log_mu', 'log_r0']

# Define the simulation properties.
T = 250
dt = .1
initial_population = 10000

# Define the policy objectives.
infection_threshold = torch.scalar_tensor(0.014)

# Define inference settings.
N_simulation = 1000
plotting._sims_to_plot = np.random.randint(0, 1000, plotting.n_sims_to_plot)

# Automatically define other required variables.
t = torch.linspace(0, T, int(T / dt) + 1)
params = SimpleNamespace(**{'log_alpha': log_alpha,
                            'log_r0': log_r0,
                            'log_gamma': log_gamma,
                            'log_mu': log_mu,
                            'log_kappa': log_kappa,
                            'log_lambda': log_lambda,
                            'u': u,
                            'controlled_parameters': controlled_parameters,
                            'uncontrolled_parameters': uncontrolled_parameters,
                            'policy': {'infection_threshold': infection_threshold},
                            'dt': dt})
init_vals = seir.sample_x0(N_simulation, initial_population)

# Real misc shit.
fig_size_wide = (12, 3)
fig_size_small = (4, 4)


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

    p = 0

    # DO SINGLE ROLLOUT PLOT -------------------------------------------------------------------------------------------

    if experiment_single_rollout:

        initial_state = seir.sample_x0(N_simulation, initial_population)
        noised_parameters = seir.sample_prior_parameters(params, N_simulation)
        results_noise = seir.simulate_seir(initial_state, noised_parameters, dt, T, seir.sample_unknown_parameters)  # noise_parameters to use gil.
        valid_simulations = valid_simulation(results_noise, noised_parameters)

        alpha, beta, typical_u, typical_alpha, typical_beta = seir.policy_tradeoff(noised_parameters)

        plt.figure(figsize=fig_size_small)
        plotting.make_trajectory_plot(plt.gca(), noised_parameters, None, results_noise, valid_simulations, t, 0, _plot_valid=True)
        plt.tight_layout()
        plt.savefig('./png/simulation_trajectory_valid.png', dpi=500)

        plt.figure(figsize=fig_size_small)
        plotting.make_trajectory_plot(plt.gca(), noised_parameters, None, results_noise, valid_simulations, t, 0, _plot_valid=True, _ylim=(0.0, 0.2))
        plt.tight_layout()
        plt.savefig('./png/simulation_traj_zoom_valid.png', dpi=500)

        plt.figure(figsize=fig_size_small)
        plotting.make_trajectory_plot(plt.gca(), noised_parameters, None, results_noise, valid_simulations, t, 0, _plot_valid=None)
        plt.tight_layout()
        plt.savefig('./png/simulation_trajectory_invalid.png', dpi=500)

        plt.figure(figsize=fig_size_small)
        plotting.make_trajectory_plot(plt.gca(), noised_parameters, None, results_noise, valid_simulations, t, 0, _plot_valid=None, _ylim=(0.0, 0.2))
        plt.tight_layout()
        plt.savefig('./png/simulation_traj_zoom_invalid.png', dpi=500)

        plt.figure(figsize=fig_size_small)
        plotting.make_parameter_plot(plt.gca(), noised_parameters, valid_simulations)
        plt.tight_layout()
        plt.savefig('./png/simulation_parameters.png', dpi=500)

        plt.figure(figsize=fig_size_small)
        # plt.gca().set_aspect('equal')
        plotting.make_policy_plot(plt.gca(), noised_parameters, alpha, beta, valid_simulations, typical_u, typical_alpha, typical_beta)
        plt.tight_layout()
        plt.savefig('./png/simulation_policy.png', dpi=500)

        # plt.tight_layout()
        # plt.savefig('./png/trajectory.png', dpi=500)
        plt.close('all')

    # DO SINGLE NMC EXPERIMENT -----------------------------------------------------------------------------------------


    def _nmc_estimate(_current_state, _controlled_parameters, _time_now):
        """
        AW - _nmc_estimate - calculate the probability that the specified parameters will
        :param _current_state:          state to condition on.
        :param _controlled_parameters:  dictionary of parameters to condition on.
        :param _time_now:               reduce the length of the sim.
        :return:
        """
        # Draw the parameters we wish to marginalize over.
        _new_parameters = seir.sample_prior_parameters(params, N_simulation)

        # Overwrite with the specified parameter value.
        _new_parameters.u[:] = _controlled_parameters['u']

        # Run the simulation with the controlled parameters, marginalizing over the others.
        _results_noised = seir.simulate_seir(_current_state, _new_parameters, dt, T - _time_now,
                                             seir.sample_unknown_parameters)
        _valid_simulations = valid_simulation(_results_noised, _new_parameters)
        _p_valid = _valid_simulations.type(torch.float).mean()
        return _p_valid, _results_noised, _valid_simulations


    if experiment_nmc_example:

        time_now = 0.0
        current_state = seir.sample_x0(N_simulation, initial_population)

        outer_samples = {'u': [],
                         'p_valid': []}

        N_outer_samples = 1000

        fig_nmc = plt.figure(1)
        plt.plot((0, 1), (0.9, 0.9), 'k:')
        plt.ylim((-0.05, 1.05))
        # plt.xlim((-0.05, 1.05))
        plt.grid(True)
        plt.title('$p( Y=1 | \\theta)$')

        for _i in range(N_outer_samples):

            # Get a realization of parameters.
            _params = seir.sample_prior_parameters(params, 1)

            # Put the controlled parameter values into
            controlled_parameter_values = {'u': _params.u}

            # Call the NMC subroutine using the parameters and current state.
            p_valid, _, _ = _nmc_estimate(current_state, controlled_parameter_values, time_now)

            # Record and plot.
            outer_samples['u'].append(controlled_parameter_values['u'][0].numpy())
            outer_samples['p_valid'].append(p_valid)
            plt.scatter(outer_samples['u'][-1], np.asarray(outer_samples['p_valid'][-1]))
            plt.pause(0.1)

        # Misc
        p = 0

    # DO ITERATIVE REPLANNING: MODEL PREDICTIVE CONTROL ----------------------------------------------------------------

    if experiment_mpc_example:

        ffmpeg_command = 'ffmpeg -y -r 25 -i ./png/mpc_%05d.png -c:v libx264 -vf fps=25 -tune stillimage ./mpc.mp4'
        print('ffmpeg command: ' + ffmpeg_command)

        current_state = dc(init_vals)
        visited_states = [[dc(current_state[0]).numpy()]]

        parameter_traces = {'log_r0': [],
                            'valid_simulations': []}

        fig, axe = plt.subplots(2, 1, squeeze=True)

        for _t in tqdm(np.arange(int(T / dt))):

            N_outer_samples = 5
            controlled_parameter_values = []
            p_valid = []
            results_noise = []

            for _i in range(N_outer_samples):
                _params = seir.sample_prior_parameters(params, 1)               # Get a realization of parameters.
                controlled_parameter_values.append({'log_r0': _params.log_r0})  # Put the controlled parameter values into.
                _p_valid, _results_noise, _valid_simulations = _nmc_estimate(dc(current_state),
                                                                             controlled_parameter_values[-1],
                                                                             _time_now=_t)
                p_valid.append(_p_valid)
                results_noise.append(_results_noise)


            parameter_traces['log_r0'].append(new_parameters.log_r0)
            parameter_traces['valid_simulations'].append(valid_simulations)

            # The 1 is important being the simulate seir code also returns the initial state...
            next_state = results_noise[1, np.random.choice(len(valid_simulations), p=valid_simulations.numpy().astype(float)/np.sum(valid_simulations.numpy().astype(int)))]
            current_state[:] = dc(next_state)
            visited_states.append([dc(next_state).numpy()])

            # Do plotting.
            fig.suptitle('t={} / {} days'.format(_t / 10, T))

            # Trajectory plot.
            plotting.make_trajectory_plot(axe[0], new_parameters, visited_states, results_noise, valid_simulations, t, _t)
            plotting.make_parameter_plot(axe[1], new_parameters, valid_simulations)

            # Misc
            plt.pause(0.1)
            plt.savefig('./png/mpc_{:05}.png'.format(_t))
            p = 0

        os.system(ffmpeg_command)
