import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from types import SimpleNamespace
from copy import deepcopy as dc

import examples.ode_tools as ode

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--test_freq', type=int, default=50)
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
    _d_s = - (_params.beta*_state[0]*_state[2])
    _d_e = (_params.beta*_state[0]*_state[2] - _params.alpha*_state[1])
    _d_i = (_params.alpha*_state[1] - _params.gamma*_state[2])
    _d_r = (_params.gamma*_state[2])
    return torch.tensor((_d_s, _d_e, _d_i, _d_r))


def simulate_seir(_state, _params, _dt, _t):
    _state_history = [dc(_state)]
    for _ in range(int(_t/_dt)):
        _grad = get_diff(_state, _params)
        _state += _grad * _dt
        _state_history.append(dc(_state))
    return torch.stack(_state_history)


if __name__ == '__main__':

    # CONFIGURE SIMULATION ---------------------------------------------------------------------------------------------

    # Define parameters
    T = 100
    dt = .1
    N = 10000
    init_vals = torch.tensor((1 - 1 / N, 1 / N, 0, 0))
    t = torch.linspace(0, T, int(T / dt) + 1)
    dims = 4

    params = SimpleNamespace(**{'alpha':    0.20,
                                'beta':     1.75,
                                'gamma':    0.50})

    # GENERATE DATA ----------------------------------------------------------------------------------------------------

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
        plt.savefig('./png/trajectory_example.png')
        plt.close(1)

    # DO DIFF ODE LEARNING ---------------------------------------------------------------------------------------------
    ii = 0
    func = ode.ODEFunc(dims)
    optimizer = optim.Adam(func.parameters(), lr=1e-4)
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
                pred_y = odeint(func, init_vals, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ode.visualize(args, t, true_y, pred_y, func, itr)
                ii += 1

        end = time.time()
