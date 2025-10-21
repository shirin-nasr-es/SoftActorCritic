# Main training script for the Soft Actor-Critic (SAC) framework.
# This file runs the end-to-end self-calibration of UAV camera intrinsics 
# using visual reprojection error without ground-truth metadata.

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# ---------- project imports ----------
from agent.sac import SACAgent
from replay_buffer import ReplayBuffer
from utils import make_env, set_seed_everywhere, eval_mode
from data_utils.data_loader import UAVDataset
from data_utils.Intrinsic_matrix import get_intrinsic_matrix

# ---------- optional tensorboard ----------
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _register_resolvers():
    try:
        OmegaConf.register_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    except Exception:
        pass


def _unwrap_step_out(step_out):
    if isinstance(step_out, (tuple, list)):
        if len(step_out) == 5:  # gymnasium
            nobs, rew, terminated, truncated, info = step_out
            return nobs, rew, bool(terminated or truncated), info
        if len(step_out) == 4:  # classic gym
            return step_out
    raise RuntimeError(f"Unexpected env.step output format: {type(step_out)}")


def _unwrap_reset_out(reset_out):
    # return obs from reset(); handle gymnasium's (obs, info).
    if isinstance(reset_out, (tuple, list)) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        return reset_out[0]
    return reset_out


def _make_run_dir(cfg) -> Path:
    ts_date = datetime.now().strftime("%Y.%m.%d")
    ts_time = datetime.now().strftime("%H%M")
    agent_name = cfg["agent"].get("name", "agent")
    exp_name = cfg.get("experiment", "exp")
    run_dir = Path("runs") / ts_date / f"{ts_time}_{agent_name}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    print(f"workspace: {run_dir}")
    return run_dir


def _maybe_set(d: dict, k: str, v):
    # set d[k]=v if key missing OR value is None.
    if (k not in d) or (d[k] is None):
        d[k] = v


def _inject_dims_into_net_cfg(net_cfg: dict, obs_dim: int, action_dim: int):
    # insert obs_dim/action_dim at the right level for either Hydra 0.11 ('class' + 'params')
    # or Hydra 1.x ('_target_') configs. Overwrite if present but None.
    if not isinstance(net_cfg, dict):
        return net_cfg

    if "_target_" in net_cfg:
        _maybe_set(net_cfg, "obs_dim", obs_dim)
        _maybe_set(net_cfg, "action_dim", action_dim)
        _maybe_set(net_cfg, "hidden_dim", 256)
        _maybe_set(net_cfg, "hidden_depth", 2)

        # drop accidental nested params (don’t forward)
        if "params" in net_cfg and isinstance(net_cfg["params"], dict):
            net_cfg.pop("params", None)

    elif "class" in net_cfg:
        params = dict(net_cfg.get("params", {}) or {})
        if ("obs_dim" not in params) or (params.get("obs_dim") is None):
            params["obs_dim"] = obs_dim
        if ("action_dim" not in params) or (params.get("action_dim") is None):
            params["action_dim"] = action_dim
        if ("hidden_dim" not in params) or (params.get("hidden_dim") is None):
            params["hidden_dim"] = 256
        if ("hidden_depth" not in params) or (params.get("hidden_depth") is None):
            params["hidden_depth"] = 2
        # if params itself wrongly contains a nested 'params', drop it
        params.pop("params", None)
        net_cfg["params"] = params

    else:
        _maybe_set(net_cfg, "obs_dim", obs_dim)
        _maybe_set(net_cfg, "action_dim", action_dim)
        _maybe_set(net_cfg, "hidden_dim", 256)
        _maybe_set(net_cfg, "hidden_depth", 2)

    return net_cfg


def _force_dims(cfg_dict: dict, obs_dim: int, action_dim: int):
    # final safety override: force dims even if upstream left nulls.
    # this prevents None flowing into DoubleQCritic(obs_dim + action_dim).
    if not isinstance(cfg_dict, dict):
        return cfg_dict

    if "_target_" in cfg_dict:
        cfg_dict["obs_dim"] = obs_dim
        cfg_dict["action_dim"] = action_dim
        if "params" in cfg_dict and isinstance(cfg_dict["params"], dict):
            cfg_dict["params"].pop("obs_dim", None)
            cfg_dict["params"].pop("action_dim", None)

    elif "class" in cfg_dict:
        p = dict(cfg_dict.get("params", {}) or {})
        p["obs_dim"] = obs_dim
        p["action_dim"] = action_dim
        p.pop("params", None)
        cfg_dict["params"] = p

    else:
        cfg_dict["obs_dim"] = obs_dim
        cfg_dict["action_dim"] = action_dim

    return cfg_dict


def main():
    # ---------- system info ----------
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("NumPy:", np.__version__)

    # ---------- config ----------
    _register_resolvers()
    cfg = OmegaConf.load("config/train.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)  # plain dict with interpolations resolved

    # ---------- device & seeds ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(int(cfg.get("seed", 0)))

    # ---------- dataset ----------
    images_path = Path(cfg["dataset"]["images_path"])
    csv_path    = Path(cfg["dataset"]["csv_path"])
    resize_factor = float(cfg["dataset"].get("resize_factor", 1.0))
    normalization = cfg["dataset"].get("normalization", "none")
    fail_sentinel = tuple(cfg["dataset"].get("fail_sentinel_size", (128, 128)))

    dataset = UAVDataset(
        images_path=images_path,
        data_path=csv_path,                 # class expects data_path (CSV)
        resize_factor=resize_factor,
        normalization=normalization,
        fail_sentinel_size=fail_sentinel,
    )

    print(f"[dataset] n_images={len(dataset)}  resize_factor={resize_factor}  norm={normalization}")
    img0, _ = dataset[0]
    if img0.ndim == 3:
        _, H, W = img0.shape
    else:
        H, W = img0.shape[-2:]
    print(f"[sample] tensor shape: {tuple(img0.shape)}  (W={W}, H={H})")

    # ---------- intrinsics (get_intrinsic_matrix returns 5 values) ----------
    # match dataset resizing: pass the SAME resize_factor used above.
    K, fx, fy, cx, cy = get_intrinsic_matrix(
        scale_factor=resize_factor,
        image_width=W,
        image_height=H,
        focal_length_x=640,
        focal_length_y=540,
    )
    K = np.asarray(K, dtype=np.float64)
    assert K.shape == (3, 3), f"bad K shape from get_intrinsic_matrix: {K.shape}"
    print("[K]\n", K)

    # ---------- environment  ----------
    env = make_env(cfg, dataset, K)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # ---------- agent  ----------
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(np.prod(action_shape))

    # build a clean params dict from config and drop fields
    params = dict(cfg["agent"]["params"])  # shallow copy
    for k in ("device", "obs_dim", "action_dim", "action_range"):
        params.pop(k, None)

    # get net cfgs and inject dims depending on style
    critic_cfg = dict(params.get("critic_cfg", {}))
    actor_cfg  = dict(params.get("actor_cfg", {}))

    critic_cfg = _inject_dims_into_net_cfg(critic_cfg, obs_dim, action_dim)
    actor_cfg  = _inject_dims_into_net_cfg(actor_cfg,  obs_dim, action_dim)

    # final safety override (handles YAML nulls that survived)
    critic_cfg = _force_dims(critic_cfg, obs_dim, action_dim)
    actor_cfg  = _force_dims(actor_cfg,  obs_dim, action_dim)

    # drop any stray nested 'params' keys in top-level cfgs
    if "_target_" in critic_cfg and "params" in critic_cfg:
        critic_cfg.pop("params", None)
    if "_target_" in actor_cfg and "params" in actor_cfg:
        actor_cfg.pop("params", None)

    params["critic_cfg"] = critic_cfg
    params["actor_cfg"]  = actor_cfg

    # action_range: SACAgent accepts a gym Box directly
    action_range = env.action_space if hasattr(env.action_space, "low") else (-1.0, 1.0)

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=action_range,
        device=device,
        **params,
    )

    # ---------- replay buffer ----------
    rb = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=int(cfg["replay_buffer_capacity"]),
        device=device,
    )

    # ---------- run dir + logging ----------
    run_dir = _make_run_dir(cfg)
    tb = SummaryWriter(str(run_dir / "tb")) if cfg.get("log_save_tb", True) and SummaryWriter else None

    # save a frozen copy of the config
    with open(run_dir / "config_frozen.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # CSV log & meta
    csv_path_out = run_dir / "train_log.csv"
    meta_path    = run_dir / "run_meta.json"
    csv_file = open(csv_path_out, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["episode", "step", "reward"])
    csv_writer.writeheader()
    meta = {
        "device": str(device),
        "dataset_size": len(dataset),
        "obs_shape": tuple(obs_shape),
        "action_shape": tuple(action_shape),
        "resize_factor": resize_factor,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---------- training loop ----------
    print("[train] start")
    num_train_steps = int(cfg["num_train_steps"])
    num_seed_steps  = int(cfg["num_seed_steps"])
    log_frequency   = int(cfg["log_frequency"])

    # unwrap reset for gymnasium (obs, info)
    obs = _unwrap_reset_out(env.reset())
    episode = 0
    step = 0
    episode_reward = 0.0
    t0 = time.time()
    done = False

    while step < num_train_steps:
        # episode rollover
        if done or getattr(env, "done", False):
            episode += 1
            dt = time.time() - t0
            print(f"[ep {episode:05d}] step={step}  reward={episode_reward:.3f}  dt={dt:.2f}s")
            csv_writer.writerow({"episode": episode, "step": step, "reward": episode_reward})
            csv_file.flush()
            if tb:
                tb.add_scalar("train/episode_reward", episode_reward, global_step=step)
            episode_reward = 0.0
            t0 = time.time()
            obs = _unwrap_reset_out(env.reset())
            done = False

        # act
        if step < num_seed_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent), torch.no_grad():
                action = agent.act(obs, sample=True)

        # env step
        s_out = env.step(action)
        next_obs, reward, done, info = _unwrap_step_out(s_out)

        # store + update
        rb.add(obs, action, reward, next_obs, done, done)
        if step >= num_seed_steps:
            agent.update(rb, logger=None, step=step)

        episode_reward += float(reward)
        obs = next_obs
        step += 1

        if (step % log_frequency) == 0:
            print(f"[upd {step}] training...")

    print("[train] done")
    csv_file.close()
    if tb:
        tb.close()


if __name__ == "__main__":
    main()
