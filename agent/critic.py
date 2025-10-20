import utils
import torch
from torch import nn

"""
This module defines the critic network using twin Q-functions for stable value estimation in Soft Actor-Critic.
"""

class DoubleQCritic(nn.Module):
    # computes two independent Q-values (Q1, Q2) from (obs, action) to reduce overestimation.
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # return (Q1, Q2) for the given observation and action.
        obs = obs.float()
        action = action.float()

        if obs.ndim == 1: obs = obs.unsqueeze(0)
        if action.ndim == 1: action = action.unsqueeze(0)

        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1).contiguous()

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
