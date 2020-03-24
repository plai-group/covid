import torch
import numpy as np
from copy import deepcopy as dc

# Fix the type of the state tensors.
float_type = torch.float64

# Do we want to re-normalize the population online? Probably not.
DYNAMIC_NORMALIZATION = False


# def simple_death_rate(_state, _params):

#     return _params.log_kappa.exp()


# def finite_capacity_death_rate(_state, _params):

#     n_infected = _state[:, 3:6].sum(dim=1)

#     max_treatable = _params.log_max_treatable.exp()
#     standard_rate = _params.log_kappa.exp()

#     higher_rate = standard_rate + _params.log_untreated_extra_kappa.exp()
#     is_too_many = (n_infected > max_treatable).type(_state.dtype)
#     proportion_treated = max_treatable/torch.max(n_infected, max_treatable)
#     too_many_rate = proportion_treated*standard_rate + (1-proportion_treated)*higher_rate
#     return is_too_many * too_many_rate + (1 - is_too_many) * standard_rate


def get_diff(_state, _params):

    a = torch.exp(_params.log_a)
    b1 = torch.exp(_params.log_b1)
    b2 = torch.exp(_params.log_b2)
    b3 = torch.exp(_params.log_b3)
    g1 = torch.exp(_params.log_g1)
    g2 = torch.exp(_params.log_g2)
    g3 = torch.exp(_params.log_g3)
    p1 = torch.exp(_params.log_p1)
    p2 = torch.exp(_params.log_p2)
    kappa = torch.exp(_params.log_kappa)
    u = _params.u

    _s, _e, _i1, _i2, _i3, _r, _d = tuple(_state[:, i] for i in range(_state.shape[1]))
    _n = _state.sum(dim=1)

    s_to_e = (1-u) * (b1*_i1 + b2*_i2 + b3*_i3)/_n*_s
    e_to_i1 = a*_e
    i1_to_r = g1*_i1
    i1_to_i2 = p1*_i1
    i2_to_r = g2*_i2
    i2_to_i3 = p2*_i2
    i3_to_r = g3*_i3
    i3_to_d = kappa*_i3

    _d_s = -s_to_e
    _d_e = s_to_e - e_to_i1
    _d_i1 = e_to_i1 - i1_to_i2 - i1_to_r
    _d_i2 = i1_to_i2 - i2_to_i3 - i2_to_r
    _d_i3 = i2_to_i3 - i3_to_r - i3_to_d
    _d_r = i1_to_r + i2_to_r + i3_to_r
    _d_d = i3_to_d

    return torch.stack((_d_s, _d_e, _d_i1, _d_i2, _d_i3, _d_r, _d_d), dim=1)


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


def sample_prior_parameters(_params, _n=None, get_map=False):
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
        _n = len(_params.log_a)

    def _sample_from_confidence_interval(_low, _nominal, _high):
        if get_map:
            return _nominal
        return _low + torch.rand((_n, )) * (_high-_low)

    _params = dc(_params)

    # code from https://alhill.shinyapps.io/COVID19seir/
    IncubPeriod = _sample_from_confidence_interval(4.5, 5.1, 5.8)
    DurMildInf = _sample_from_confidence_interval(5.5, 6., 6.5)
    FracMild = 0.81
    FracSevere = 0.14
    FracCritical = 0.05
    CFR = 0.02
    DurHosp = _sample_from_confidence_interval(8.7, 11.2, 14.9)  # https://arxiv.org/pdf/2002.03268.pdf
    TimeICUDeath = 7.

    a=1/IncubPeriod
    g1=(1/DurMildInf)*FracMild
    p1=(1/DurMildInf)-g1
    p2=(1/DurHosp)*(FracCritical/(FracSevere+FracCritical))
    g2=(1/DurHosp)-p2
    kappa=(1/TimeICUDeath)*(CFR/FracCritical)
    g3=(1/TimeICUDeath)-kappa

    b1 = _sample_from_confidence_interval(0.23, 0.33, 0.43)
    b2 = _sample_from_confidence_interval(0., 0., 0.05)
    b3 = _sample_from_confidence_interval(0., 0., 0.025)

    def tensor_log(x):
        return torch.tensor(x).expand(_n).log()

    _params.log_a = tensor_log(a)
    _params.log_b1 = tensor_log(b1)
    _params.log_b2 = tensor_log(b2)
    _params.log_b3 = tensor_log(b3)
    _params.log_g1 = tensor_log(g1)
    _params.log_g2 = tensor_log(g2)
    _params.log_g3 = tensor_log(g3)
    _params.log_p1 = tensor_log(p1)
    _params.log_p2 = tensor_log(p2)
    _params.log_kappa = tensor_log(kappa)
    _params.u = torch.randn((_n, ))

    return _params


def sample_unknown_parameters(_params, _n=None):
    """
    AW - sample_unknown_parameters - Sample the parameters we do not fix and hence wish to marginalize over.
    :param _params: SimpNameSp: dot accessible simple name space of simulation parameters.
    :return:        SimpNameSp: dot accessible simple name space of simulation parameters, where those parameters
                                 that are not fixed have been re-drawn from the prior.
    """

    if _n is None:
        _n = len(_params.log_a)

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
    for key in dir(_params_perturbed):
        if key[:3] != 'log':
            pass
        old_val = getattr(_params_perturbed, key,)
        new_val = old_val + torch.normal(0.0, 0.01, old_val.shape)
        setattr(_params_perturbed, key, new_val)
        print('noising', key)
    # _params_perturbed.log_alpha   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_alpha), ))
    # _params_perturbed.log_r0      += torch.normal(0.0, 0.01, (len(_params_perturbed.log_r0),    ))
    # _params_perturbed.log_gamma   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_gamma), ))
    # _params_perturbed.log_mu      += torch.normal(0.0, 0.01, (len(_params_perturbed.log_mu),    ))
    # _params_perturbed.log_kappa   += torch.normal(0.0, 0.01, (len(_params_perturbed.log_kappa), ))
    # _params_perturbed.log_lambda  += torch.normal(0.0, 0.01, (len(_params_perturbed.log_lambda),))
    return _params_perturbed


def sample_identity_parameters(_params, _n=None):
    """
    AW - sample_identity_parameters - Don't add any noise to the parameters.
    :param _params: SimpNameSp: dot accessible simple name space of simulation parameters.
    :return: copy of the simulation parameters.
    """
    return dc(_params)


def policy_tradeoff(_params):
    # Do u / R0 plotting.
    n_sweep = 1001
    u = np.square(1 - _params.u)  # 1-u because u is a _reduction_.
    alpha = np.linspace(0, 2.0, num=n_sweep)
    beta = np.zeros((len(u), n_sweep))
    for _u in range(len(u)):
        for _a in range(len(alpha)):
            beta[_u, _a] = u[_u] / alpha[_a]

    typical_u = 1.0
    typical_alpha = alpha
    typical_beta = typical_u / typical_alpha

    return alpha, beta, typical_u, typical_alpha, typical_beta
