import gym
import utils
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple, Union

from agent import Agent
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf, DictConfig

"""
This module implements the Soft Actor-Critic (SAC) algorithm that coordinates the actor and critic networks for 
training in continuous control tasks.
"""


class SACAgent(Agent):
    """SAC Algorithm"""

    def _instantiate_net(self, cfg):

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            if "class" in cfg:  # Hydra 0.11 style
                cls = get_class(cfg["class"])
                params = cfg.get("params", {}) or {}
                return cls(**params)
            if "_target_" in cfg:  # Hydra 1.x style
                cls = get_class(cfg["_target_"])
                params = {k: v for k, v in cfg.items() if k != "_target_"}
                return cls(**params)

        return cfg

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_range: Union[Tuple[float, float], gym.spaces.Box],
        device: torch.device,
        critic_cfg,
        actor_cfg,
        discount: float,
        init_temperature: float,
        alpha_lr: float,
        alpha_betas: Tuple[float, float],
        actor_lr: float,
        actor_betas: Tuple[float, float],
        actor_update_frequency: int,
        critic_lr: float,
        critic_betas: Tuple[float, float],
        critic_tau: float,
        critic_target_update_frequency: int,
        batch_size: int,
        learnable_temperature: bool,
    ):
        super().__init__()

        self.training = True
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = float(discount)
        self.critic_tau = float(critic_tau)
        self.actor_update_frequency = int(actor_update_frequency)
        self.critic_target_update_frequency = int(critic_target_update_frequency)
        self.batch_size = int(batch_size)
        self.learnable_temperature = bool(learnable_temperature)

        # cache action range
        if hasattr(action_range, "low"):
            self.action_low = torch.as_tensor(action_range.low, device=self.device, dtype=torch.float32)
            self.action_high = torch.as_tensor(action_range.high, device=self.device, dtype=torch.float32)
        else:
            lo, hi = action_range
            self.action_low = torch.full((action_dim,), float(lo), device=self.device)
            self.action_high = torch.full((action_dim,), float(hi), device=self.device)

        # networks
        self.critic = self._instantiate_net(critic_cfg).to(self.device)

        self.critic_target = self._instantiate_net(critic_cfg).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.actor = self._instantiate_net(actor_cfg).to(self.device)

        if self.device.type == 'cuda'and False:
            try:
                self.critic = torch.compile(self.critic)
                self.critic_target = torch.compile(self.critic_target)
                self.actor = torch.compile(self.actor)
            except Exception:
                pass  # safe fallback if not supported

        # optimizers
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)

        # entropy temperature
        self.target_entropy = -float(action_dim)
        init_temperature = max(1e-8, float(init_temperature))
        log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32, device=self.device)

        if self.learnable_temperature:
            self.log_alpha = torch.nn.Parameter(log_alpha)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)
        else:
            self.log_alpha = log_alpha
            self.alpha_opt = None  # no optimizer when fixed

        # start in training mode
        self.train()

    def _scale_action(self, u: torch.Tensor) -> torch.Tensor:
        return self.action_low + (u + 1.0) * 0.5 * (self.action_high - self.action_low)

    def train(self, training: bool = True):
        self.training = bool(training)
        self.actor.train(self.training)
        self.critic.train(self.training)
        self.critic_target.eval()

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, obs, sample: bool = False):
        # Convert observation to tensor
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Get distribution from actor
            dist = self.actor(obs)
            u = dist.sample() if sample else dist.mean

            if (u > 1.001).any() or (u < -1.001).any():
                u = u.tanh()

            action = self._scale_action(u)


        assert action.ndim == 2 and action.shape[0] == 1, f"Unexpected action shape: {action.shape}"

        return action.squeeze(0).cpu().numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):

        obs = obs.to(self.device)
        action = action.to(self.device)  # actions from buffer are env-scaled already
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            dist = self.actor(next_obs)
            u_next = dist.rsample()

            if (u_next > 1.001).any() or (u_next < -1.001).any():
                u_next = u_next.tanh()


            log_prob = dist.log_prob(u_next).sum(-1, keepdim=True)
            log_prob = torch.nan_to_num(log_prob, neginf=-30.0, posinf=0.0)

            next_action = self._scale_action(u_next)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = reward + not_done * self.discount * target_V


        current_Q1, current_Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        if (step + 1) % 100 == 0:
            print(f"[upd {step + 1}] critic_loss={critic_loss.item():.4f}")

        if logger is not None:
            logger.log('train/critic_loss', float(critic_loss.item()), step)
            logger.log('train/q1_mean', float(current_Q1.mean().item()), step)
            logger.log('train/q2_mean', float(current_Q2.mean().item()), step)

    def update_actor_and_alpha(self, obs, logger=None, step=0):

        obs = obs.to(self.device)

        dist = self.actor(obs)
        u = dist.rsample()

        if (u > 1.001).any() or (u < -1.001).any():
            u = u.tanh()

        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        log_prob = torch.nan_to_num(log_prob, neginf=-30.0, posinf=0.0)

        a_pi = self._scale_action(u)

        Q1, Q2 = self.critic(obs, a_pi)
        Q_min = torch.min(Q1, Q2)

        actor_loss = (self.alpha * log_prob - Q_min).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        if (step + 1) % 100 == 0:
            print(f"[upd {step + 1}] actor_loss={actor_loss.item():.4f}, alpha={self.alpha:.4f}")

        alpha_loss = None
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        if logger is not None:
            logger.log('train/actor_loss', float(actor_loss.item()), step)
            logger.log('train/entropy', float(-log_prob.mean().item()), step)
            logger.log('train/target_entropy', float(self.target_entropy), step)
            if alpha_loss is not None:
                logger.log('train/alpha_loss', float(alpha_loss.item()), step)
                logger.log('train/alpha', float(self.alpha), step)


    def update(self, replay_buffer, logger, step):

        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

        if logger is not None:
            logger.log('train/batch_reward', float(reward.mean().item()), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if (step + 1) % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if (step + 1) % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


