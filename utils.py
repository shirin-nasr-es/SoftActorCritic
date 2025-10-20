"""Utility module providing environment setup, training/evaluation mode managers, MLP builders, and general helper functions for the SAC framework."""
import os
import random
import numpy as np

import torch
from torch import nn

from contextlib import ExitStack
from typing import Optional, Type
from calibrate_env import CalibrateEnv

# ============================================================
# Minimal time-limit wrapper for non-Gym environments
# ============================================================

class TimeLimitLite:
    """Enforce a max number of steps even if env is not a gymnasium.Env."""
    def __init__(self, env, max_episode_steps: int):
        self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        # best-effort passthroughs
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.spec = getattr(env, "spec", None)

    def reset(self, *args, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        out = self.env.step(action)
        self._elapsed_steps += 1
        time_up = self._elapsed_steps >= self.max_episode_steps

        # gymnasium >=0.26 tuple: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            truncated = bool(truncated or time_up)
            return obs, reward, terminated, truncated, info

        # classic gym tuple: (obs, reward, done, info)
        obs, reward, done, info = out
        done = bool(done or time_up)
        return obs, reward, done, info


# ============================================================
# Environment setup
# ============================================================
def make_env(cfg, dataset, init_K):
    # builds the CalibrateEnv environment from config.
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 images to form a pair.")

    # sequential pairs (i, i+1)
    pair_indices = [(i, i + 1) for i in range(len(dataset) - 1)]

    # ---- safe internal horizon ----
    episode_len_raw = getattr(cfg, "episode_len", None)
    episode_len_val = int(episode_len_raw) if isinstance(episode_len_raw, (int, float)) else 50

    cxb = getattr(cfg, "cx_bound", None)
    cyb = getattr(cfg, "cy_bound", None)

    env = CalibrateEnv(
        dataset=dataset,
        init_K=init_K,
        pair_indices=pair_indices,
        mode=getattr(cfg, "mode", "rmse"),
        ransac_thresh=float(getattr(cfg, "ransac_thresh", 1.2)),
        fx_bound=tuple(getattr(cfg, "fx_bound", (100.0, 3000.0))),
        fy_bound=tuple(getattr(cfg, "fy_bound", (100.0, 3000.0))),
        cx_bound=(tuple(cxb) if cxb is not None else None),
        cy_bound=(tuple(cyb) if cyb is not None else None),
        episode_len=episode_len_val,                  # internal horizon
        failure_penalty=float(getattr(cfg, "failure_penalty", 50.0)),
        seed=int(getattr(cfg, "seed", 0)),
        pairs_per_step=int(getattr(cfg, "pairs_per_step", 5)),
        # Wire-throughs from YAML:
        step_scale=float(getattr(cfg, "step_scale", 0.01)),
        err_norm_scale=(None if getattr(cfg, "err_norm_scale", None) is None
                        else float(getattr(cfg, "err_norm_scale"))),
        ema_beta=float(getattr(cfg, "ema_beta", 0.8)),
        early_termination=bool(getattr(cfg, "early_termination", False)),
        early_thresh=float(getattr(cfg, "early_thresh", 1.0)),
        early_patience=int(getattr(cfg, "early_patience", 8)),
    )

    # ---- enforce episode cap even without gym wrappers ----
    max_steps_raw = getattr(cfg, "max_episode_steps", None)
    if not isinstance(max_steps_raw, (int, float)) or max_steps_raw <= 0:
        max_steps_raw = getattr(cfg, "episode_len", episode_len_val)
    max_steps = int(max_steps_raw)

    class TimeLimitLite:
        def __init__(self, env_, max_episode_steps_):
            self.env = env_
            self.max_episode_steps = int(max_episode_steps_)
            self._elapsed_steps = 0
        def reset(self, *a, **k):
            self._elapsed_steps = 0
            return self.env.reset(*a, **k)
        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            self._elapsed_steps += 1
            if self._elapsed_steps >= self.max_episode_steps:
                trunc = True
            return obs, rew, term, trunc, info
        @property
        def observation_space(self): return self.env.observation_space
        @property
        def action_space(self): return self.env.action_space
        @property
        def spec(self): return getattr(self.env, "spec", None)

    env = TimeLimitLite(env, max_steps)
    return env


# ============================================================
# Context managers for training/eval modes
# ============================================================

class eval_mode:
    def __init__(self, *models, use_no_grad=True):
        self.models = models
        self.use_no_grad = use_no_grad
        self.prev_states = None
        self._stack = None

    def __enter__(self):
        self.prev_states = [m.training for m in self.models]
        for m in self.models:
            m.train(False)

        self._stack = ExitStack()
        if self.use_no_grad:
            self._stack.enter_context(torch.no_grad())
        return self

    def __exit__(self, exc_type, exc, tb):
        for m, state in zip(self.models, self.prev_states):
            m.train(state)
        if self._stack is not None:
            self._stack.close()
        return False


class train_mode:
    def __init__(self, *models):
        self.models = models
        self.prev_states = None

    def __enter__(self):
        self.prev_states = [m.training for m in self.models]
        for m in self.models:
            m.train(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        for m, state in zip(self.models, self.prev_states):
            m.train(state)
        return False


# ============================================================
# Utility functions
# ============================================================

def soft_update_params(net, target_net, tau: float):
    # performs Polyak averaging update for target network.
    with torch.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)


def set_seed_everywhere(seed: int):
    # set random seeds for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dir(*path_parts):
    # create directory if missing.
    dir_path = os.path.join(*path_parts)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def weight_init(m):
    # orthogonal weight initialization for Linear and Conv2D.
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ============================================================
# MLP utilities
# ============================================================

class MLP(nn.Module):
    """Configurable MLP used in critic and actor networks."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        mods = []

        if hidden_depth <= 0:
            last_in = input_dim
        else:
            mods.append(nn.Linear(input_dim, hidden_dim))
            if layer_norm:
                mods.append(nn.LayerNorm(hidden_dim))
            mods.append(activation())

            for _ in range(hidden_depth - 1):
                mods.append(nn.Linear(hidden_dim, hidden_dim))
                if layer_norm:
                    mods.append(nn.LayerNorm(hidden_dim))
                mods.append(activation())
            last_in = hidden_dim

        mods.append(nn.Linear(last_in, output_dim))
        if output_activation is not None:
            mods.append(output_activation())

        self.trunk = nn.Sequential(*mods)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    # lightweight MLP constructor for compatibility.
    act = nn.ReLU
    if hidden_depth <= 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    net = nn.Sequential(*mods)
    net.apply(weight_init)
    return net


def to_np(t: Optional[torch.Tensor]):
    if t is None:
        return None
    if t.nelement() == 0:
        return np.array([])
    return t.detach().cpu().numpy()
