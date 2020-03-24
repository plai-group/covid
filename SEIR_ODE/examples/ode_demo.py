import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import multiprocessing as proc

from tqdm import tqdm
from types import SimpleNamespace
from copy import deepcopy as dc

# Import the SEIR module.
import examples.seir as seir
import examples.plotting as plotting

# Import custom istarmap.
import examples.istarmap

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
experiment_single_rollout = False
experiment_nmc_example = False
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
T = 500
dt = 1.0
initial_population = 10000

# Define the policy objectives.
infection_threshold = torch.scalar_tensor(0.014)

# Define inference settings.
N_simulation = 100
N_parameter_sweep = 101

plotting._sims_to_plot = np.random.randint(0, N_simulation, plotting.n_sims_to_plot)

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
                            'dt': dt,
                            'T': T})
init_vals = seir.sample_x0(N_simulation, initial_population)


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

        plotting.do_family_of_plots(noised_parameters, results_noise, valid_simulations, t, _prepend='simulation')
        plt.close('all')

    # DO SINGLE NMC EXPERIMENT -----------------------------------------------------------------------------------------


    def _nmc_estimate(_current_state, _params, _controlled_parameters, _time_now):
        """
        AW - _nmc_estimate - calculate the probability that the specified parameters will
        :param _current_state:          state to condition on.
        :param _params:                 need to pass in the params dictionary to make sure the prior has the right shape
        :param _controlled_parameters:  dictionary of parameters to condition on.
        :param _time_now:               reduce the length of the sim.
        :return:
        """
        # Draw the parameters we wish to marginalize over.
        _new_parameters = seir.sample_prior_parameters(_params, N_simulation)

        # Overwrite with the specified parameter value.
        _new_parameters.u[:] = _controlled_parameters['u']

        # Run the simulation with the controlled parameters, marginalizing over the others.
        _results_noised = seir.simulate_seir(_current_state, _new_parameters, _params.dt, _params.T - _time_now,
                                             seir.sample_unknown_parameters)
        _valid_simulations = valid_simulation(_results_noised, _new_parameters)
        _p_valid = _valid_simulations.type(torch.float).mean()
        return _p_valid, _results_noised, _valid_simulations


    def _parallel_nmc_estimate(_pool, _current_state, _params, _time_now, _controlled_parameter_values):
        # Create args dictionary.
        _args = [(_current_state, _params, _c, _time_now) for _c in _controlled_parameter_values]

        # Do the sweep
        _results = _pool.starmap(_nmc_estimate, _args)

        # Pull out the results for plotting.
        _p_valid = [_r[0] for _r in _results]
        _results_noise = [_r[1] for _r in _results]
        _valid_simulations = [_r[2] for _r in _results]
        return _p_valid, _results_noise, _valid_simulations


    if experiment_nmc_example:

        time_now = 0.0
        current_state = seir.sample_x0(N_simulation, initial_population)

        n_sweep = 11
        u_sweep = torch.linspace(0, 1, n_sweep)

        # Put the controlled parameter values into and args array.
        controlled_parameter_values = [dc({'u': _u}) for _u in u_sweep]

        # Push these into a param dict to make sure.
        controlled_params = dc(params)
        controlled_params.u = u_sweep

        outer_samples = {'u': u_sweep,
                         'p_valid': [],
                         'results_noise': [],
                         'valid_simulations': []}

        # Do we want to do this in serial or parallel?
        n_worker = proc.cpu_count()

        # Go and do the sweep.
        if n_worker > 1:
            pool = proc.Pool(processes=n_worker)
            outer_samples['p_valid'], outer_samples['results_noise'], outer_samples['valid_simulations'] = \
                _parallel_nmc_estimate(pool, current_state, params, time_now, controlled_parameter_values)
            pool.close()
        else:
            for _u in range(len(u_sweep)):
                # Call the NMC subroutine using the parameters and current state.
                results = _nmc_estimate(current_state, controlled_params, controlled_parameter_values[_u], time_now)

                # Record and plot.
                outer_samples['p_valid'].append(results[0])
                outer_samples['results_noise'].append(results[1])
                outer_samples['valid_simulations'].append(results[2])

        # Prepare the simulated traces for plotting by taking the expectation.
        expect_results_noised = torch.mean(torch.stack(outer_samples['results_noise']), dim=2).transpose(0, 1)

        # Work out which simulations are probabilistically valid.
        threshold = 0.9
        prob_valid_simulations = torch.tensor([(_p > 0.9) for _p in outer_samples['p_valid']]).type(torch.int)

        # Do some plotting.
        plotting._sims_to_plot = np.random.randint(0, len(u_sweep), plotting.n_sims_to_plot)
        plotting.nmc_plot(outer_samples, threshold)
        plotting.do_family_of_plots(controlled_params, expect_results_noised, prob_valid_simulations, t, _prepend='simulation')
        plt.pause(0.1)

    # DO ITERATIVE REPLANNING: MODEL PREDICTIVE CONTROL ----------------------------------------------------------------
    if experiment_mpc_example:

        ffmpeg_command = 'ffmpeg -y -r 25 -i ./png/mpc_%05d.png -c:v libx264 -vf fps=25 -tune stillimage ./mpc.mp4'
        print('ffmpeg command: ' + ffmpeg_command)

        current_state = dc(init_vals)
        visited_states = torch.empty((0, 0, 7))  # torch.tensor(dc(current_state[0])).unsqueeze(0).unsqueeze(0)

        parameter_traces = {'u': [],
                            'valid_simulations': []}

        pool = proc.Pool(proc.cpu_count())

        # We don't want to re-plan at every step.
        planning_step = np.int(np.round(1 / dt))

        # Counter for labelling images.
        img_frame = 0

        for _t in range(1):  # tqdm(np.arange(0, int(T / dt), step=planning_step)):

            u_sweep = torch.linspace(0, 1, N_parameter_sweep)
            outer_samples = {'u': u_sweep,
                             'p_valid': [],
                             'results_noise': [],
                             'valid_simulations': []}

            controlled_parameter_values = [dc({'u': _u}) for _u in u_sweep]

            # Push these into a param dict to make sure.
            controlled_params = dc(params)
            controlled_params.u = u_sweep

            # Run simulation.
            outer_samples['p_valid'], outer_samples['results_noise'], outer_samples['valid_simulations'] = \
                _parallel_nmc_estimate(pool, current_state, params, _t * params.dt, controlled_parameter_values)

            # Work out which simulations are probabilistically valid.
            threshold = 0.9
            prob_valid_simulations = torch.tensor([(_p > 0.9) for _p in outer_samples['p_valid']]).type(torch.int)
            first_valid_simulation = np.searchsorted(prob_valid_simulations, threshold)

            # Prepare the simulated traces for plotting by taking the expectation.
            expect_results_noised = torch.mean(torch.stack(outer_samples['results_noise']), dim=2).transpose(0, 1)

            # The 1 is important being the simulate seir code also returns the initial state...
            _n = np.random.randint(0, N_simulation)
            _visited_states = outer_samples['results_noise'][first_valid_simulation][1:(planning_step+1), _n]
            current_state[:] = dc(_visited_states[-1])
            if len(visited_states) > 0:
                visited_states = torch.cat((visited_states, dc(_visited_states).unsqueeze(1)), dim=0)
            else:
                visited_states = _visited_states.unsqueeze(1)

            # Do plotting.
            do_plot = True
            _title = 't={} / {} days'.format(int(_t) / planning_step, T)
            if do_plot:
                _plot_frequency = 10  # Have to plot at the frequency of planning right now.
                if _t % _plot_frequency == 0:

                    # Trajectory plot.
                    plotting.do_family_of_plots(controlled_params, expect_results_noised, prob_valid_simulations, t,
                                                _visited_states=visited_states,
                                                _prepend='mpc_', _title=None, _num='_{:05d}'.format(img_frame))
                    plt.close('all')
                    plt.pause(0.1)

                    _do_controled_plot = True
                    if _do_controled_plot:
                        # Run simulation with the chosen control.
                        controlled_parameter_values = dc({'u': torch.tensor([outer_samples['u'][first_valid_simulation]])})
                        controlled_params = dc(params)
                        controlled_params.u = controlled_parameter_values['u'][:] * torch.ones((N_simulation, ))

                        p_valid, results_noise, valid_simulations = \
                            _nmc_estimate(current_state, params, controlled_parameter_values, _t * params.dt)

                        plotting.do_family_of_plots(controlled_params, results_noise, torch.ones((N_simulation, )), t,
                                                    _visited_states=visited_states,
                                                    _prepend='mpc_', _title=None, _num='_controlled_{:05d}'.format(img_frame))

                    img_frame += 1

                    plt.pause(0.1)


        os.system(ffmpeg_command)
