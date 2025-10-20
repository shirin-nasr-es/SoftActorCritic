""" Fixed-size replay buffer that stores past experiences and samples random batches for off-policy RL training."""

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        obs_dtype = np.uint8 if len(obs_shape) == 3 else np.float32

        self.obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)

        self.obs_is_uint8 = (obs_dtype == np.uint8)
        self.idx = 0
        self.full = False

    def __len__(self):

        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 1.0 - float(done))
        np.copyto(self.not_dones_no_max[self.idx], 1.0 - float(done_no_max))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):

        assert self.full or self.idx >= batch_size, \
            f"ReplayBuffer: not enough samples ({self.idx}) to draw {batch_size}"

        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device).float()

        if self.obs_is_uint8:
            obses = obses.float().div_(255.0)
            next_obses = next_obses.float().div_(255.0)
        else:
            obses = obses.float()
            next_obses = next_obses.float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
