import torch
import numpy as np
from copy import deepcopy as dc

# Fix the type of the state tensors.
float_type = torch.float64

# Do we want to re-normalize the population online? Probably not.
DYNAMIC_NORMALIZATION = False


def simple_death_rate(_state, _params):

    return _params.log_kappa.exp()


def finite_capacity_death_rate(_state, _params):

    n_infected = _state[:, 3:6].sum(dim=1)
    max_treatable = _params.log_max_treatable.exp()
    standard_rate = _params.log_kappa.exp()
    higher_rate = standard_rate + _params.log_untreated_extra_kappa.exp()
    is_too_many = (n_infected > max_treatable).type(_state.dtype)
    proportion_treated = max_treatable/torch.max(n_infected, max_treatable)
    too_many_rate = proportion_treated*standard_rate + \
        (1-proportion_treated)*higher_rate
    return is_too_many * too_many_rate + (1 - is_too_many) * standard_rate


def get_diff(_state, _params):

    _lambda =    torch.exp(_params.log_lambda)
    death_rate = finite_capacity_death_rate(_state, _params)
    _mu =        torch.exp(_params.log_mu)
    _u =                   _params.u
    _r0 =        torch.exp(_params.log_r0)
    _gamma =     torch.exp(_params.log_gamma)
    _alpha =     torch.exp(_params.log_alpha)

    _s, _e1, _e2, _i1, _i2, _i3, _r = tuple(_state[:, i] for i in range(_state.shape[1]))
    _n = _state.sum(dim=1)

    s_to_e1 = ((1 - _u) * _r0 * _gamma * _s * (_i1 + _i2 + _i3) / _n).type(float_type)
    e1_to_e2 = (2*_alpha*_e1).type(float_type)
    e2_to_i1 = (2*_alpha*_e2).type(float_type)
    i1_to_i2 = (3*_gamma*_i1).type(float_type)
    i2_to_i3 = (3*_gamma*_i2).type(float_type)
    i3_to_r = (3*_gamma*_i3).type(float_type)

    _d_s = _lambda*_n - s_to_e1 - _mu*_s
    _d_e1 = s_to_e1 - e1_to_e2 - _mu*_e1
    _d_e2 = e1_to_e2 - e2_to_i1 - _mu*_e2
    _d_i1 = e2_to_i1 - i1_to_i2 - (_mu+death_rate)*_i1
    _d_i2 = i1_to_i2 - i2_to_i3 - (_mu+death_rate)*_i2
    _d_i3 = i2_to_i3 - i3_to_r - (_mu+death_rate)*_i3
    _d_r = i3_to_r - _mu*_r

    return torch.stack((_d_s, _d_e1, _d_e2, _d_i1, _d_i2, _d_i3, _d_r), dim=1)


def simulate_seir(_state, _params, _dt, _t, _noise_func):
    """
    AW - simulate_seir - simulate entire trajectories of the SEIR model.
    :param _state:      tensor (N x D):     the initial state of the simulation, where N independent experiments
    :param _params:     SimpNameSp:         dot accessible simulation parameters.
    :param _dt:         float:              Euler integration timestep.
    :param _t:          float:              Time to simulate over.
    :param _noise_func: function handle:    Function handle linking to a function that controls the perturbation of the
                                             simulation parameters at each time step. Valid objects are
                                             sample_prior_parameters, sample_noised_parameters,
                                             sample_unknown_parameters, sample_ident_parameters
    :return:
    """
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


def sample_x0(_n, _N):
    """
    AW - sample_x0 - sample from the initial distribution over state.
    :param _n: int: number of samples to draw.
    :param _N: int: initial population size.
    :return: SE2I3R state tensor.
    """
    return torch.tensor([[1 - 1 / _N, 1 / _N, 0, 0, 0, 0, 0]] * _n)


def sample_prior_parameters(_params, _n=None):
    """
    WH - sample_prior_parameters - Draw the parameters from the prior to begin with.

    :param _params: SimpNameSp: dot accessible simple name space of simulation parameters.
    :param _n       int:        number of parameters to draw.
    :return:        SimpNameSp: dot accessible simple name space of simulation parameters where all parameters have been
                                 redrawn from the prior over parameters, including those that are controllable.

    prior for incubation period from https://www.ncbi.nlm.nih.gov/pubmed/32150748
    prior for death rate from https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30195-X/fulltext#coronavirus-linkback-header
    all other numbers from `Risk Assessment of Novel Coronavirus COVID-19 Outbreaks Outside China`
    """

    if _n is None:
        _n = len(_params.log_alpha)

    def _sample_from_confidence_interval(_low, _high):
        return _low + torch.rand((_n, )) * (_high-_low)

    _params = dc(_params)

    _incubation_period = _sample_from_confidence_interval(4.5, 5.8)
    _params.log_alpha = torch.tensor(1/_incubation_period).log()

    _infectious_period = _sample_from_confidence_interval(1.7, 5.6)
    _params.log_gamma = (1/_infectious_period).log()

    _r0 = _sample_from_confidence_interval(2.1, 3.1)
    _params.log_r0 = _r0.log()

    _death_rate = _sample_from_confidence_interval(0.055, 0.059)
    _params.log_kappa = torch.log(_death_rate / _infectious_period)

    u = torch.rand((_n, ))
    _params.u = u

    return _params


def sample_unknown_parameters(_params, _n=None):
    """
    AW - sample_unknown_parameters - Sample the parameters we do not fix and hence wish to marginalize over.
    :param _params: SimpNameSp: dot accessible simple name space of simulation parameters.
    :return:        SimpNameSp: dot accessible simple name space of simulation parameters, where those parameters
                                 that are not fixed have been re-drawn from the prior.
    """

    if _n is None:
        _n = len(_params.log_alpha)

    _params_from_unknown = dc(_params)
    _params_from_prior = sample_prior_parameters(_params_from_unknown, _n)
    for _k in _params.uncontrolled_parameters:
        setattr(_params_from_unknown, _k, getattr(_params_from_prior, _k))
    return _params_from_unknown


def sample_perturb_parameters(_params, _n=None):
    """
    AW - sample_perturb_parameters - sample a small perturbation to all parameters.
    :param _params:
    :return:
    """
    _params_perturbed = dc(_params)
    _params_perturbed.log_alpha   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_alpha), ))
    _params_perturbed.log_r0      += torch.normal(0.0, 0.01, (len(_params_perturbed.log_r0),    ))
    _params_perturbed.log_gamma   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_gamma), ))
    _params_perturbed.log_mu      += torch.normal(0.0, 0.01, (len(_params_perturbed.log_mu),    ))
    _params_perturbed.log_kappa   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_kappa), ))
    _params_perturbed.log_lambda  += torch.normal(0.0, 0.01, (len(_params_perturbed.log_lambda),))
    return _params_perturbed


def sample_identity_parameters(_params, _n=None):
    """
    AW - sample_identity_parameters - Don't add any noise to the parameters.
    :param _params: SimpNameSp: dot accessible simple name space of simulation parameters.
    :return: copy of the simulation parameters.
    """
    return dc(_params)
