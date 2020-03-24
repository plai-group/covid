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
experiment_peak_versus_deaths = True
experiment_nmc_example = False
experiment_mpc_example = False

# Define base parameters of SEIR model.
log_alpha = torch.log(torch.tensor((1 / 5.1, )))
log_r0 = torch.log(torch.tensor((2.6, )))
log_gamma = torch.log(torch.tensor((1 / 3.3,)))

# mortality rate stuff dependent from Imperial paper
age_categories = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
us_ages_millions = [10.13+9.68+10.32+9.88, 10.66+10.22+10.77+10.32, 11.2+10.67+12.02+11.54, 11.19+10.94+10.79+10.77, 9.9+9.92+10.26+10.48, 10.28+10.61+10.67+11.27, 9.73+10.6+8.03+9.05, 6.21+7.19+4.14+5.12, 2.59+3.54+2.33+4.33]  # from https://www.statista.com/statistics/241488/population-of-the-us-by-sex-and-age/
us_ages_proportions = [mil/sum(us_ages_millions) for mil in us_ages_millions]
mortality_rates = [2e-5, 6e-5, 3e-4, 8e-4, 1.5e-3, 6e-3, 2.2e-2, 5.1e-2, 9.3e-2]
icu_rates = [1e-3*5e-2, 3e-3*5e-2, 1.2e-2*5e-2, 3.2e-2*5e-2, 4.9*6.3e-4, 10.2*12.2e-4, 16.6*27.4e-4, 24.3*43.2e-4, 27.3*70.9e-4]

covid_mortality_rate = sum(rate*prop for rate, prop in zip(mortality_rates, us_ages_proportions))

# Imperial paper assumes 30 % ICU mortality rate (plus small non-ICU rate) - so if everyone in ICU dies due to lack of beds:
untreated_extra_mortality_rate_per_age = [r*0.7 for r in icu_rates]
untreated_extra_mortality_rate = sum(rate*prop for rate, prop in
                                     zip(untreated_extra_mortality_rate_per_age, us_ages_proportions))

# # AW.
# covid_total_mortality_rate = 0.0108
#
# ICU_admission_rate = 0.05
#
# population = 1000000
# ICU_beds = population * 0.0028
#
# f_I_ICU =     lambda _I: np.min((_I, ICU_beds))
# f_I_NICU =    lambda _I: np.max((_I - ICU_beds, 0))
#
# I = 10000
# __I_fraction = I / population
#
# I_ICU_candidates = I * ICU_admission_rate
#
# I_H =       I - I_ICU_candidates           # Death rate = 0.0
# I_ICU =     f_I_ICU(I_ICU_candidates)      # Death rate = 50%
# I_NICU =    f_I_NICU(I_ICU_candidates)     # Death rate = 100%
#
# death_rate_H =      0.0
# death_rate_ICU =    0.50
# death_rate_NICU =   1.0
#
# D_H = death_rate_H * I_H
# D_ICU = death_rate_ICU * I_ICU
# D_NICU = death_rate_NICU * I_NICU
#
#
# # AW simple.
#
# cov_total_mortality_rate =    0.0108
#
# hos_admission_rate =          0.19
# hos_bed_fraction =            0.0028  # Hospital bed rate.
#
# icu_progression_rate =        0.26
# icu_bed_fraction =            0.000347
#
# dr_icu =        0.5     # Needs to be in ICU, gets into ICU.
# dr_nicu_h =     0.75    # Needs to be in ICU, gets into hospital.
# dr_nicu_nh =    1.0     # Needs to be in ICU, doesn't get into hospital.
#
# P = 1000000
# I = 10000
# icu_beds = icu_bed_fraction * P
# hos_beds = hos_bed_fraction * P
#
# f_I_ICU =       lambda _I: np.min((_I, ICU_beds))
# f_I_NICU_H =    lambda _I: np.max((_I - ICU_beds, 0))
# # f_I_NICH_NH =
#
# __I_fraction = I / population
#
# I_ICU = I * hos_admission_rate * icu_progression_rate
# # N_ICU =


# # AW super simple.
# P = 1000000
# I = 50000
# __I_fraction = I / P

# dr_icu =    0.5
# dr_nicu =   1.0
# dr_hosp =   0.01

# icu_bed_fraction = 0.000347
# icu_beds = icu_bed_fraction * P

# hos_admission_rate =          0.19
# icu_progression_rate =        0.26

# f_ICU_candidate =   lambda _I: _I * hos_admission_rate * icu_progression_rate

# f_I_ICU =           lambda _I: np.min((f_ICU_candidate(_I), icu_beds))
# f_I_NICU =          lambda _I: np.max((f_ICU_candidate(_I) - icu_beds, 0))
# f_I_hosp =          lambda _I: _I * hos_admission_rate * (1 - icu_progression_rate)

# n_ICU = f_I_ICU(I)
# n_NICU = f_I_NICU(I)
# n_hosp = f_I_hosp(I)

# deaths = (n_ICU * dr_icu) + (n_NICU * dr_nicu) + (n_hosp * dr_hosp)
# dr_effective = (deaths / I) * log_gamma.exp()  # This number should be returned by finite capacity death rate.

# kappa_at_I = deaths / P
# breakdown = (n_ICU * dr_icu), (n_NICU * dr_nicu), (n_hosp * dr_hosp)

# # dr_effective =

# # intensive care capacities:
# intensive_care_beds_per_infected = 0.044 * 0.3 * 10. / (1/log_gamma.exp())   # 4.4% hospitalised / infection * 30% ICU / hospitalised * 10 days in ICU / assumed infected time of 1/gamma  source: the Imperial paper
# intensive_care_capacity = 2.8e-3  # beds per population   https://data.oecd.org/healtheqt/hospital-beds.htm
# max_treatable = intensive_care_capacity / intensive_care_beds_per_infected

log_kappa =     torch.log(covid_mortality_rate * log_gamma.exp())
# log_untreated_extra_kappa = torch.log(untreated_extra_mortality_rate * log_gamma.exp())
# log_max_treatable = torch.log(max_treatable)
log_lambda = torch.log(torch.tensor((0.00116 / 365, )))  # US birth rate from https://tinyurl.com/sezkqxc
log_mu = torch.log(torch.tensor((0.008678 / 365, )))     # non-covid death rate https://tinyurl.com/ybwdzmjs
u = torch.tensor((0., ))

# Make sure we are controlling the right things.
controlled_parameters = ['u']  # We can select u.
uncontrolled_parameters = ['log_kappa', 'log_a', 'log_p1', 'log_p2', 'log_p1',
                           'log_p2', 'log_g1', 'log_g2', 'log_g3']

# Define the simulation properties.
T = 250
dt = .1
initial_population = 10000

# Define the policy objectives.
infection_threshold = torch.tensor(0.15)  # log_max_treatable.exp().item()

# Define inference settings.
N_simulation = 1000
plotting._sims_to_plot = np.random.randint(0, 1000, plotting.n_sims_to_plot)

# Automatically define other required variables.
t = torch.linspace(0, T, int(T / dt) + 1)

params = seir.sample_prior_parameters(SimpleNamespace(), _n=1, get_map=True)
params.policy = {'infection_threshold': torch.tensor((0.05,))}
params.controlled_parameters = controlled_parameters
params.uncontrolled_parameters = uncontrolled_parameters
params.dt = dt
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

    if experiment_peak_versus_deaths:

        assert experiment_single_rollout  # we will reuse simulation

        fig, axe = plt.subplots()
        plotting.peak_infection_versus_deaths(axe, results_noise, params)
        plt.savefig('./png/infected_deaths.pdf')
        plt.close()

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
