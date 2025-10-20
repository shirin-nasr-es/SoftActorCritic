import math
import torch
import utils
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
"""
This module defines the actor component, implementing a diagonal Gaussian policy with tanh-squashed outputs 
to generate bounded continuous actions for training with the Soft Actor-Critic algorithm.
"""


class TanhTransform(pyd.transforms.Transform):
    # applies a numerically stable tanh transformation to map values from the real domain to the interval (-1, 1).

    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
           return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
           return 2. * (math.log(2.) -x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    # defines a Gaussian (Normal) distribution transformed by a tanh function to produce bounded actions between -1 and 1.

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    # "Implements a diagonal Gaussian policy network that outputs a squashed Normal action distribution for continuous control tasks.

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):

        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        mu, log_std = self.trunk(obs.float()).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = map(float, self.log_std_bounds)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp().clamp(min=1e-6)

        self.outputs['mu'] = mu
        self.outputs['log_std'] = log_std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)



