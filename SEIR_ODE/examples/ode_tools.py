import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

fig, ax_traj, ax_phase, ax_vecfield = None, None, None, None

eps = 0.001


# def logit(_x):
#     _z = ( ( (_x - 0.0) / (1.0 - 0.0) ) * 0.98 ) + 0.01  # (_x * (1.0 - 2*eps)) + eps
#     _zz = torch.log(_z / (1.0 - _z))
#     return _zz
#
#
# def sigmoid(_z):
#     _x = ( ( (_z - 0.01) / (0.99 - 0.01) ) * 1.0 ) + 0.0
#     _xx = torch.scalar_tensor(1.0) / (1.0 + torch.exp(_x))
#     return _xx


def setup_visual(args):
    if args.viz:
        global fig, ax_traj, ax_phase, ax_vecfield
        makedirs('png')
        fig = plt.figure(figsize=(12, 4), facecolor='white', num=999)
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)


def get_batch(args, true_y, t):
    s = torch.from_numpy(np.random.choice(np.arange(len(true_y) - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return torch.log(batch_y0 + eps), batch_t, torch.log(batch_y + eps)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize(args, t, true_y, pred_y, odefunc, itr):

    if args.viz:

        plt.figure(999)

        true_y = torch.log(true_y + eps)
        pred_y = torch.log(pred_y + eps)

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0], 'c--',
                     t.numpy(), true_y.numpy()[:, 1], 'm--',
                     t.numpy(), true_y.numpy()[:, 2], 'y--',
                     t.numpy(), true_y.numpy()[:, 3], 'k--')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0], 'c',
                     t.numpy(), pred_y.numpy()[:, 1], 'm',
                     t.numpy(), pred_y.numpy()[:, 2], 'y',
                     t.numpy(), pred_y.numpy()[:, 3], 'k')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-10.1, 1.1)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait ( dims = (0,1) = (S,E) )')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0], true_y.numpy()[:, 1], 'g--')
        ax_phase.plot(pred_y.numpy()[:, 0], pred_y.numpy()[:, 1], 'b')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)
        #
        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self, _state_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(_state_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, _state_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
