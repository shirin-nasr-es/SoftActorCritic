"""Entry point for training: builds env/agent/logger/buffer from Hydra config and runs SAC training & evaluation loops."""

import os
import time
import numpy as np

import torch
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

from utils import make_env, set_seed_everywhere, eval_mode
from agent.sac import SACAgent


class Workspace(object):
    # orchestrates setup (device, env, agent, logger, buffer) and manages the full train/eval lifecycle.
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if isinstance(cfg.device, str):
            self.device = torch.device(cfg.device)
        else:
            # fallback if cfg.device is missing/None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # (optional) CUDA speed knobs
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 2) LOGGER
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
        )

        # 3) SEED, ENV, AGENT, etc.
        set_seed_everywhere(cfg.seed)

        # make env
        self.env = make_env(cfg)

        # fill agent dims
        cfg.agent.params.obs_dim    = int(np.prod(self.env.observation_space.shape))
        cfg.agent.params.action_dim = int(np.prod(self.env.action_space.shape))
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        # build agent
        self.agent = SACAgent(**cfg.agent.params)

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        # runs evaluation episodes (no exploration), logs metrics/videos, and reports averaged return.
        avg_ep_ret = 0.0
        for ep in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(ep == 0))
            done = False
            ep_ret = 0.0

            while not done:
                # with utils.eval_mode(self.agent):
                with eval_mode(self.agent), torch.no_grad():
                    action = self.agent.act(obs, sample=False)

                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                ep_ret += float(reward)

            self.logger.log('eval/episode_reward_single', ep_ret, self.step)

            avg_ep_ret += ep_ret
            self.video_recorder.save(f'eval_step{self.step}_ep{ep}.mp4')

        avg_ep_ret /= max(1, self.cfg.num_eval_episodes)
        self.logger.log('eval/episode_reward', avg_ep_ret, self.step)
        self.logger.dump(self.step)

    def run(self):
        # main training loop: collects experience, updates SAC after seed steps, and periodically evaluates/logs.
        episode, episode_reward, done = 0, 0.0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0.0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # exploration vs policy action
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self.agent), torch.no_grad():
                    action = self.agent.act(obs, sample=True)

            next_obs, reward, done, _ = self.env.step(action)

            # convert to float (for buffer conventions)
            done_float = float(done)
            spec = getattr(self.env, "spec", None)
            max_steps = getattr(spec, "max_episode_steps", None)
            done_no_max = 0.0 if (max_steps is not None and episode_step + 1 == max_steps) else done_float

            episode_reward += float(reward)

            self.replay_buffer.add(obs, action, float(reward), next_obs, done_float, done_no_max)

            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            obs = next_obs
            episode_step += 1
            self.step += 1

@hydra.main(config_path="config/train.yaml")
def main(cfg):
    # Hydra entry point: creates a Workspace and launches training
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()


