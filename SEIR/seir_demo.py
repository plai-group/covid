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
import seir
import plotting

# Fix the type of the state tensors.
float_type = seir.float_type

# Kill some matplotlib warnings.
warnings.filterwarnings("ignore")

# Set up tex in matplotlib.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble='\\usepackage{amsmath,amssymb}')

try:
    os.mkdir('./pdf')
except:
    pass


# CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

# Experiments to run.
experiment_do_single_sim =  False
experiment_single_rollout = False
# experiment_icu_capacity =   False
experiment_nmc_example =    False
experiment_stoch_vs_det =   False
experiment_mpc_example =    True

# Make sure we are controlling the right things.
controlled_parameters = ['u']  # We can select u.
uncontrolled_parameters = ['log_kappa', 'log_a', 'log_p1', 'log_p2', 'log_p1',
                           'log_p2', 'log_g1', 'log_g2', 'log_g3']  # , 'log_icu_capacity']

# Define the simulation properties.
T = 1000
dt = 1.0
initial_population = 10000

# Define inference settings.
N_simulation = 100
N_parameter_sweep = 150

plotting._sims_to_plot = np.random.randint(0, N_simulation, plotting.n_sims_to_plot)

# Automatically define other required variables.
t = torch.linspace(0, T, int(T / dt) + 1)

params = seir.sample_prior_parameters(SimpleNamespace(), _n=1, get_map=True)
params.policy = {'infection_threshold': 0.0145, }  #'icu_capacity': torch.tensor(.259e-3)}
params.controlled_parameters = controlled_parameters
params.uncontrolled_parameters = uncontrolled_parameters
params.dt = dt
params.T = T

# Sample the initial state.
init_vals = seir.sample_x0(N_simulation, initial_population)


def valid_simulation(_state, _params):
    """
    AW - valid_simulation - return a binary variable per simulation indicating whether or not that simulation
    satifies the desired policy outcome.
    :param _state: tensor (N x D):  tensor of the state trajectory.
    :return: tensor (N, ), bool:    tensor of whether each simulation was valid.
    """
    _n_infected = _state[:, :, 2] + _state[:, :, 3] + _state[:, :, 4]
    _valid = torch.logical_not(torch.any(_n_infected > _params.policy['infection_threshold'], dim=0))
    return _valid


if __name__ == '__main__':

    # DO DETERMINISTIC SIMULATION PLOT ---------------------------------------------------------------------------------

    if experiment_do_single_sim:

        # Run some deterministic simulations with and without control.

        os.system('\\rm -rf ./pdf/1_deterministic_')
        os.mkdir('./pdf/1_deterministic_')

        # Do single determinstic trajectories. ----------------------
        initial_state = seir.sample_x0(1, initial_population)
        params_nominal = seir.sample_prior_parameters(params, 1, get_map=True)

        results_deterministic = seir.simulate_seir(initial_state, params_nominal, dt, T, seir.sample_identity_parameters)
        valid_simulations = valid_simulation(results_deterministic, params_nominal)
        plotting.det_plot(results_deterministic, valid_simulations, params_nominal, t, _append='nominal', _legend=True)

        # Do single plot with controls. -----------------------------
        initial_state = seir.sample_x0(1, initial_population)
        params_controlled = seir.sample_prior_parameters(params, 1, get_map=True)
        params_controlled.u = 0.37

        results_deterministic = seir.simulate_seir(initial_state, params_controlled, dt, T, seir.sample_identity_parameters)
        valid_simulations = valid_simulation(results_deterministic, params)
        plotting.det_plot(results_deterministic, valid_simulations, params_controlled, t, _append='controlled')

        plt.close('all')

        # # ICU CAPACITY EXPERIMENT ------------------------------------------------------------------------------------
        #
        # # Dev code - developing ICU capacity.
        #
        # if experiment_icu_capacity:
        #
        #     print('\nICU capacity sweep')
        #
        #     capacities = {
        #         'zero': 0.0,  # params_controlled.log_icu_capacity - float('inf'),
        #         'nominal': 0.0,  # params_controlled.log_icu_capacity,
        #         'infinite': 0.0,  # params_controlled.log_icu_capacity + float('inf'),
        #     }
        #
        #     fig = plt.figure(figsize=(6, 2))
        #     axe = plt.gca()
        #     for _i, capacity_name in enumerate(['zero', 'nominal', 'infinite']):
        #
        #         initial_state = seir.sample_x0(N_simulation, initial_population)
        #         params_controlled = seir.sample_prior_parameters(params, N_simulation, get_map=True)
        #         params_controlled.u = torch.linspace(0, 1, N_simulation)
        #         # params_controlled.log_icu_capacity = capacities[capacity_name]
        #         results_deterministic = seir.simulate_seir(initial_state, params_controlled, dt,
        #                                                    T, seir.sample_identity_parameters)
        #
        #         plotting.peak_infection_versus_deaths(axe, results_deterministic, params_controlled,
        #                                               label=f'{capacity_name}',
        #                                               _c=[plotting.mcd['red'],
        #                                                   plotting.mcd['blue'],
        #                                                   plotting.mcd['green']][_i])
        #
        #     plt.savefig('./pdf/infected_deaths.pdf')
        #     plt.close('all')

    # DO SINGLE ROLLOUT PLOT -------------------------------------------------------------------------------------------

    if experiment_single_rollout:

        # Run a stochastic experiment with zero control (or some amount of control).

        os.system('\\rm -rf ./pdf/2_simulation_')
        os.mkdir('./pdf/2_simulation_')

        initial_state = seir.sample_x0(N_simulation, initial_population)
        noised_parameters = seir.sample_prior_parameters(params, N_simulation)
        noised_parameters.u[:] = 0.0
        results_noise = seir.simulate_seir(initial_state, noised_parameters, dt, T, seir.sample_unknown_parameters)
        valid_simulations = valid_simulation(results_noise, noised_parameters)

        plotting._sims_to_plot = np.arange(np.alen(valid_simulations))
        plotting.do_family_of_plots(noised_parameters, results_noise, valid_simulations, t, _prepend='2_simulation_', _num='')
        plt.close('all')

        plt.pause(0.1)
        plt.close('all')

    # DO SINGLE NMC EXPERIMENT -----------------------------------------------------------------------------------------

    if experiment_nmc_example:

        # This experiment isn't particularly interesting, it is pretty much subsumed into MPC below.

        os.system('\\rm -rf ./pdf/3_nmc_')
        os.mkdir('./pdf/3_nmc_')

        time_now = 0.0
        current_state = seir.sample_x0(N_simulation, initial_population)
        u_sweep = torch.linspace(0, 1, N_parameter_sweep)

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
                seir.parallel_nmc_estimate(pool, current_state, params, time_now, controlled_parameter_values, valid_simulation)
            pool.close()
        else:
            for _u in range(len(u_sweep)):
                # Call the NMC subroutine using the parameters and current state.
                results = seir.nmc_estimate(current_state, controlled_params, time_now, controlled_parameter_values[_u], valid_simulation)

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
        _sims_to_plot_store = dc(plotting._sims_to_plot)
        plotting._sims_to_plot = np.arange(0, len(u_sweep))
        plotting.nmc_plot(outer_samples, threshold, _prepend='3_nmc_', _append='0')
        plotting.do_family_of_plots(controlled_params, expect_results_noised, prob_valid_simulations, t, _prepend='3_nmc_')
        plotting._sims_to_plot = dc(_sims_to_plot_store)

        plt.pause(0.1)
        plt.close('all')

    # DO STOCHASTIC vs DETERMINISTIC COMPARISON ------------------------------------------------------------------------

    if experiment_stoch_vs_det:

        os.system('\\rm -rf ./pdf/4_stoch_det_')
        os.mkdir('./pdf/4_stoch_det_')

        time_now = 0.0
        current_state = seir.sample_x0(N_parameter_sweep, initial_population)
        u_sweep = torch.linspace(0, 1, N_parameter_sweep)

        # Do deterministic 'sweep.'
        controlled_parameter_values = dc({'u': u_sweep})
        controlled_params = seir.sample_prior_parameters(params, _n=N_parameter_sweep, get_map=True)
        controlled_params.u = u_sweep
        _, _, valid_simulations_deterministic = seir.nmc_estimate(current_state, controlled_params,
                                                                  time_now, controlled_parameter_values,
                                                                  valid_simulation,
                                                                  _proposal=seir.sample_identity_parameters)

        # Do stochastic 'sweep.'
        current_state = seir.sample_x0(N_simulation, initial_population)
        controlled_parameter_values = [dc({'u': _u}) for _u in u_sweep]
        controlled_params = dc(params)
        controlled_params.u = u_sweep
        pool = proc.Pool(processes=proc.cpu_count())
        valid_simulations_stochastic, _, _ = \
        seir.parallel_nmc_estimate(pool, current_state, params, time_now, controlled_parameter_values, valid_simulation)
        pool.close()

        def _plot(_valid_simulations_deterministic, _valid_simulations_stochastic=None):
            # Analyse the results.
            plt.figure(figsize=plotting.fig_size_short)
            axe = plt.gca()
            if _valid_simulations_stochastic is not None:
                plt.scatter((1 - u_sweep), torch.stack(valid_simulations_stochastic), c=plotting.mcd['green'], marker='.', label='Stochastic')
            plt.plot((1 - u_sweep), valid_simulations_deterministic.type(torch.float), c=plotting.mcd['red'], label='Deterministic')
            plt.plot((0, 1), (0.9, 0.9), 'k:', label='$90\\%$ conf.')
            plt.legend(loc='lower right', prop={'size': 8})
            plt.xlim((0, 1))
            plt.ylim((-0.02, 1.02))
            plt.grid(True)
            axe.set_xlabel('$\\hat{R_0}$: Controlled exposure rate \n relative to uncontrolled exposure rate.')
            axe.text(0.2, 0.67, s='$\\hat{R_0} = (1 - u)R_0$', horizontalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.9, linestyle='-'))
            _xt = plt.xticks()
            _xt = ['$' + str(int(__xt * 100)) + '\\%R_0$' for __xt in list(_xt[0])]
            plt.xticks((0, 0.2, 0.4, 0.6, 0.8, 1.0), _xt)
            axe.set_xlim((1.0, 0.0))
            axe.set_ylim((-0.02, 1.02))
            plt.ylabel('$p(\\forall_{t > 0} Y_t^{aux}=1 | \\theta)$')
            plt.tight_layout()
            plt.pause(0.1)
            if _valid_simulations_stochastic is None:
                plt.savefig('./pdf/4_stoch_det_/det_parameters.pdf')
            else:
                plt.savefig('./pdf/4_stoch_det_/stoch_parameters.pdf')
            p = 0

        # Now run some sweeps using the estimated values.

        # Do deterministic 'sweep' and get the results in.
        current_state = seir.sample_x0(3, initial_population)
        u_sweep = torch.tensor([0.300, 0.37, 0.450])
        controlled_parameter_values = dc({'u': u_sweep})
        controlled_params = seir.sample_prior_parameters(params, _n=N_parameter_sweep, get_map=True)
        controlled_params.u = u_sweep
        _, results_deterministic, _ = seir.nmc_estimate(current_state, controlled_params,
                                                        time_now, controlled_parameter_values,
                                                        valid_simulation,
                                                        _proposal=seir.sample_identity_parameters)
        fig = plt.figure(figsize=plotting.fig_size_short)
        axe = plt.gca()
        plotting.make_trajectory_plot(axe, controlled_params, None, results_deterministic, torch.ones((len(u_sweep), )), t,
                                      _plot_valid=None, _ylim=(0.0, 0.10))
        plt.tight_layout()
        plt.savefig('./pdf/4_stoch_det_/det_traj.pdf')

        pool = proc.Pool(processes=proc.cpu_count())

        for _val, _tag in zip((0.37, 0.5, 0.6), ('under', 'borderline', 'safe')):
            # Now run some sweeps using the estimated values.
            current_state = seir.sample_x0(N_simulation, initial_population)
            u_sweep = torch.tensor([0.37])
            controlled_parameter_values = [dc({'u': _u}) for _u in u_sweep]
            controlled_params = dc(params)
            controlled_params.u = u_sweep
            _, results_stochastic, _ = \
            seir.parallel_nmc_estimate(pool, current_state, controlled_params, time_now, controlled_parameter_values, valid_simulation)
            fig = plt.figure(figsize=plotting.fig_size_short)
            axe = plt.gca()
            plotting.make_trajectory_plot(axe, controlled_params, None, results_stochastic[0], torch.ones((N_simulation, )), t,
                                          _plot_valid=None, _ylim=(0.0, 0.10))
            plt.tight_layout()
            plt.savefig('./pdf/4_stoch_det_/sto_traj_{}.pdf'.format(_tag))

        pool.close()
        plt.pause(0.1)
        plt.close('all')

    # DO ITERATIVE REPLANNING: MODEL PREDICTIVE CONTROL ----------------------------------------------------------------

    if experiment_mpc_example:

        # Set up folders for saving results..
        os.system('\\rm -rf ./pdf/5_mpc_')
        os.mkdir('./pdf/5_mpc_')
        os.mkdir('./pdf/5_mpc_/zoom')
        os.mkdir('./pdf/5_mpc_/full')
        os.mkdir('./pdf/5_mpc_/polocy')

        current_state = dc(init_vals)
        visited_states = torch.empty((0, 0, ))

        parameter_traces = {'u': [],
                            'valid_simulations': []}

        # Set what validity function we want to use.
        _valid_func = valid_simulation

        # Pool of workers must be used for this to be reasonably fast.
        pool = proc.Pool(proc.cpu_count())

        # We don't want to re-plan at every step.  7 = replan every week.  # TODO must match value in plotting.
        planning_step = np.int(np.round(7.0 / dt))

        # Counter for labelling images.
        img_frame = 0

        # Hide output to stop you machine frying.
        plt.switch_backend('agg')

        for _t in tqdm(np.arange(0, int(T / dt), step=planning_step)):

            # Set the parameter values we are going to test.
            u_sweep = torch.normal(0.5, 0.2, (N_parameter_sweep, )).clamp(0.0, 1.0).sort().values
            # u_sweep = torch.linspace(0.0, 1.0, N_parameter_sweep)

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
                seir.parallel_nmc_estimate(pool, current_state, params, _t * params.dt, controlled_parameter_values, _valid_func)

            # Work out which simulations are probabilistically valid.
            threshold = 0.9
            prob_valid_simulations = torch.tensor([(_p > 0.9) for _p in outer_samples['p_valid']]).type(torch.int)
            first_valid_simulation = np.min((np.searchsorted(prob_valid_simulations, threshold), N_parameter_sweep - 1))

            # If we are above t=200 sample random continuations to test the MPC.
            if _t > 200:
                _n = np.random.randint(0, N_simulation - 1)
                _visited_states = outer_samples['results_noise'][first_valid_simulation][1:(planning_step + 1), _n]
            else:
                # Otherwise, sample an ``adversarial'' example to generate some baseline infection.
                _tr = outer_samples['results_noise'][first_valid_simulation][planning_step + 1]
                _I = _tr[:, 2] + _tr[:, 3] + _tr[:, 4]
                _n = _I.argmax()
                _visited_states = outer_samples['results_noise'][first_valid_simulation][1:(planning_step + 1), _n]

            # Update the current state and append it to the history.
            current_state[:] = dc(_visited_states[-1])
            if len(visited_states) > 0:
                visited_states = torch.cat((visited_states, dc(_visited_states).unsqueeze(1)), dim=0)
            else:
                visited_states = _visited_states.unsqueeze(1)

            # Do plotting.
            _title = None  # 't={} / {} days'.format(int(_t) / planning_step, T)
            _plot_frequency = 1
            if _t % _plot_frequency == 0:

                # Do NMC plot.
                plotting.nmc_plot(outer_samples, threshold, _prepend='5_mpc_', _append='_{:05d}'.format(img_frame))

                # Trajectory plot.
                plotting._sims_to_plot = np.random.randint(0, N_parameter_sweep-1, 50)
                controlled_params.u = torch.ones((N_simulation,)) * u_sweep[first_valid_simulation]
                plotting.do_family_of_plots(controlled_params, outer_samples['results_noise'][first_valid_simulation],
                                            torch.ones((N_simulation,)), t, _visited_states=visited_states,
                                            _prepend='5_mpc_', _title='', _num='_control_{:05d}'.format(img_frame),
                                            _shade=True)

                # Below is a big-old hotch potch of plotting scripts that i still need to tidy up.

                # plt.figure(figsize=(2, 2))
                # plotting.make_trajectory_plot(plt.gca(), controlled_params, None, outer_samples['results_noise'][first_valid_simulation],
                #                                 torch.ones((N_simulation,)), t, _plot_valid=None, _ylim=(0.0, 0.025), _shade=True)
                # if _title is not None:
                #     plt.title(_title)
                # plt.tight_layout()
                # # plt.savefig('./png/{}/{}trajectory_zoom_all{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
                # plt.savefig('./pdf/{}/zoom/{}trajectory_zoom_all{}.pdf'.format('5_mpc_', '5_mpc_', '_control{}_{:05d}'.format(_tag, img_frame)))

                # # Do policy plot.
                # plt.figure(figsize=plotting.fig_size_small)
                # alpha, beta, typical_u, typical_alpha, typical_beta = seir.policy_tradeoff(controlled_params)
                # plotting.make_policy_plot(plt.gca(), controlled_params, alpha, beta, prob_valid_simulations,
                #                           typical_u, typical_alpha, typical_beta)
                # if _title is not None:
                #     plt.title(_title)
                # plt.tight_layout()
                # # plt.savefig('./png/{}/{}policy{}.png'.format(_prepend, _prepend, _num), dpi=dpi)
                # plt.savefig('./pdf/{}/policy/{}policy{}.pdf'.format('5_mpc_', '5_mpc_full_', '_{:05d}'.format(img_frame)))

                # # Plot the means of trajectories.
                # expect_results_noised = torch.stack(outer_samples['results_noise']).transpose(0, 1).mean(dim=2)
                # plotting.do_family_of_plots(controlled_params, expect_results_noised, torch.ones((N_parameter_sweep, )),
                #                             t, _visited_states=visited_states,
                #                             _prepend='5_mpc_mean_', _title='', _num='_{:05d}'.format(img_frame))

                img_frame += 1
                plt.pause(0.1)
                plt.close('all')
