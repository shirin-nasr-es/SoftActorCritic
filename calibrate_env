import gym
import numpy as np
from gym import spaces

from geometry_utils.pairwise_reprojection_error import pairwise_reprojection_error


class CalibrateEnv(gym.Env):
    """
    Calibrate intrinsics (fx, fy, cx, cy) with RL.
    Key changes vs your original:
      • Longer episodes (default 25) so Δ-error matters across time.
      • Cached image pairs per episode to reduce stochastic noise.
      • Exponential smoothing of error (EMA) for stability.
      • Normalized error and Δ-error–centric reward shaping.
      • Gentle regularizers and bounded, scaled action updates.
      • Patient early-termination (needs several consecutive good steps).
    """

    def __init__(
            self,
            dataset,
            init_K,
            pair_indices,
            *,
            mode="rmse",
            ransac_thresh=1.2,
            fx_bound=(100.0, 3000.0),
            fy_bound=(100.0, 3000.0),
            cx_bound=None,
            cy_bound=None,
            episode_len=25,  # keep longer episodes so Δ-error matters
            failure_penalty=50.0,  # align with YAML; large enough to matter, not extreme
            seed=0,
            pairs_per_step=5,
            step_scale=0.01,  # fraction of each bound range per step
            err_norm_scale=None,  # bootstrap from data in reset() for stability
            ema_beta=0.8,  # EMA smoothing of error (higher = smoother)
            early_termination=True,
            early_thresh=1.0,  # realistic “good” RMSE threshold (in pixels)
            early_patience=3,      # require N consecutive “good” steps to terminate early
            geom_model="H",
    ):
        super().__init__()
        self.dataset = dataset
        self.init_K = np.array(init_K, dtype=np.float64)
        self.pair_indices = list(pair_indices)
        self.mode = mode
        self.ransac_thresh = float(ransac_thresh)
        self.fx_bound = tuple(fx_bound)
        self.fy_bound = tuple(fy_bound)

        # Infer image size from principal point if bounds not given
        H_guess = int(round(self.init_K[1, 2] * 2.0))
        W_guess = int(round(self.init_K[0, 2] * 2.0))
        if cx_bound is None:
            cx_bound = (0.0, float(max(1, W_guess)))
        if cy_bound is None:
            cy_bound = (0.0, float(max(1, H_guess)))
        self.cx_bound = tuple(cx_bound)
        self.cy_bound = tuple(cy_bound)

        # Core knobs
        self.episode_len = int(episode_len)
        self.failure_penalty = float(failure_penalty)
        self.rng = np.random.default_rng(int(seed))
        self.step_scale = float(step_scale)
        self.pairs_per_step = int(pairs_per_step)

        # Error handling / smoothing
        # self.err_norm_scale = float(err_norm_scale)  # pixels -> unit scale
        self.err_norm_scale = None if err_norm_scale is None else float(err_norm_scale)
        self.ema_beta = float(ema_beta)

        # Early termination settings
        self.early_termination = bool(early_termination)
        self.early_thresh = float(early_thresh) if mode == "rmse" else float(early_thresh ** 2)
        self.early_patience = int(early_patience)

        assert len(self.pair_indices) > 0, "pair_indices must be non-empty"

        # Action space: adjustments to (fx, fy, cx, cy) in normalized [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space: normalized (fx, fy, cx, cy, err_ema_norm)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # State (filled in reset)
        self.K = None
        self.t = 0
        self.prev_err_ema = None   # EMA’ed (smoothed) error (pixels)
        self.err_ema = None
        self._good_streak = 0

        # Per-episode pair caching to reduce noise
        self.active_pairs = None
        self.pair_cursor = 0

    # --------------- Gym API -----------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.K = self.init_K.copy()
        self.t = 0
        self.prev_err_ema = None
        self.err_ema = None
        self._good_streak = 0

        # Cache a deterministic sequence of pairs for this episode
        # Sample without replacement if possible, then cycle.
        n_pairs = max(self.pairs_per_step * max(1, self.episode_len // 2), self.pairs_per_step)
        if len(self.pair_indices) >= n_pairs:
            idxs = self.rng.choice(len(self.pair_indices), size=n_pairs, replace=False)
        else:
            idxs = self.rng.choice(len(self.pair_indices), size=n_pairs, replace=True)
        self.active_pairs = [self.pair_indices[i] for i in idxs.tolist()]
        self.pair_cursor = 0

        # --- Bootstrap normalization scale from data if requested ---
        if self.err_norm_scale is None:
            probe = []
            # sample up to 5 cached pairs (no action yet → current self.K)
            for k in range(min(5, len(self.active_pairs))):
                i, j = self.active_pairs[k]
                im1, _ = self.dataset[i]
                im2, _ = self.dataset[j]

                try:
                    e = float(
                        pairwise_reprojection_error(
                            im1, im2, self.K,
                            ransac_thresh=self.ransac_thresh,
                            mode=self.mode,
                            return_details=False,
                            model=self.geom_model,
                        )
                    )
                    if np.isfinite(e):
                        probe.append(e)
                except Exception:
                    pass
            # robust default if empty
            self.err_norm_scale = float(np.median(probe)) if len(probe) else 1.0
            # keep it within sane bounds
            self.err_norm_scale = float(np.clip(self.err_norm_scale, 0.5, 5.0))

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # ----- action processing -----
        a = np.asarray(action, dtype=np.float32).flatten()
        if a.shape[0] == 2:  # legacy support: (fx, fy) only
            a = np.concatenate([a, np.zeros(2, dtype=np.float32)])
        if a.shape[0] != 4:
            raise ValueError(f"Expected action of length 2 or 4, got shape {a.shape}")
        a = np.clip(a, -1.0, 1.0)

        # Scale actions to parameter deltas
        d_fx = a[0] * (self.fx_bound[1] - self.fx_bound[0]) * self.step_scale
        d_fy = a[1] * (self.fy_bound[1] - self.fy_bound[0]) * self.step_scale
        d_cx = a[2] * (self.cx_bound[1] - self.cx_bound[0]) * self.step_scale
        d_cy = a[3] * (self.cy_bound[1] - self.cy_bound[0]) * self.step_scale

        # Update K with clamping
        self.K[0, 0] = float(np.clip(self.K[0, 0] + d_fx, *self.fx_bound))
        self.K[1, 1] = float(np.clip(self.K[1, 1] + d_fy, *self.fy_bound))
        self.K[0, 2] = float(np.clip(self.K[0, 2] + d_cx, *self.cx_bound))
        self.K[1, 2] = float(np.clip(self.K[1, 2] + d_cy, *self.cy_bound))

        # ----- evaluate M cached pairs (robust median) -----
        errs = []
        fails = 0
        used_pairs = []
        for _ in range(self.pairs_per_step):
            i, j = self._next_pair()
            used_pairs.append((int(i), int(j)))
            im1, _ = self.dataset[i]
            im2, _ = self.dataset[j]
            try:
                e = float(
                    pairwise_reprojection_error(
                        im1,
                        im2,
                        self.K,
                        ransac_thresh=self.ransac_thresh,
                        mode=self.mode,
                        return_details=False,
                        model=self.geom_model,
                    )
                )
                if not np.isfinite(e):
                    e = self.failure_penalty
            except Exception:
                e = self.failure_penalty

            errs.append(e)
            if e >= self.failure_penalty:
                fails += 1

        # robust aggregate (median)
        err_px = float(np.median(errs)) if len(errs) > 0 else float(self.failure_penalty)
        if not np.isfinite(err_px):
            err_px = float(self.failure_penalty)

        # ----- EMA smoothing -----
        if (self.err_ema is None) or (not np.isfinite(self.err_ema)):
            self.err_ema = err_px
        else:
            self.err_ema = self.ema_beta * self.err_ema + (1.0 - self.ema_beta) * err_px

        # ----- Reward shaping (normalize + emphasize improvement) -----
        # Normalize errors roughly to [0, 1]
        err_norm = float(np.clip(err_px / self.err_norm_scale, 0.0, 5.0))  # keep a cap for safety
        err_ema_norm = float(np.clip(self.err_ema / self.err_norm_scale, 0.0, 5.0))

        if (self.prev_err_ema is None) or (not np.isfinite(self.prev_err_ema)):
            delta_norm = 0.0
        else:
            delta_norm = float(np.clip((self.prev_err_ema - self.err_ema) / self.err_norm_scale,
                                       -0.05, 0.05))  # cap Δ to reduce noise spikes

        # Reward = (improvement weight)*Δ(err)  –  (absolute error weight)*err
        w_improve = 10.0
        w_abs = 1.0
        reward = w_improve * delta_norm - w_abs * err_ema_norm

        if not np.isfinite(reward):
            reward = -1.0  # safe fallback

        # Update prev for next step
        self.prev_err_ema = float(self.err_ema)

        # ----- Soft regularizers -----
        # discourage hovering near the bounds
        margin = 30.0
        near_fx = (self.K[0, 0] - self.fx_bound[0] < margin) or (self.fx_bound[1] - self.K[0, 0] < margin)
        near_fy = (self.K[1, 1] - self.fy_bound[0] < margin) or (self.fy_bound[1] - self.K[1, 1] < margin)
        reward -= 0.02 * float(near_fx) + 0.02 * float(near_fy)

        # small penalty for large anisotropy between fx and fy
        ratio = self.K[0, 0] / (self.K[1, 1] + 1e-8)
        reward -= 3e-4 * (np.log(ratio) ** 2)

        # ----- Termination logic -----
        # “Good” if raw (unsmoothed) error is under the threshold
        is_good = (err_px <= self.early_thresh)
        self._good_streak = self._good_streak + 1 if is_good else 0
        converged = self.early_termination and (self._good_streak >= self.early_patience)
        terminated = bool(converged)

        self.t += 1
        truncated = self.t >= self.episode_len

        obs = self._get_obs(err_ema_norm=err_ema_norm)

        info = {
            "error": float(err_px),              # raw median error (px)
            "error_ema": float(self.err_ema),    # smoothed error (px)
            "error_norm": err_norm,              # raw / scale
            "error_ema_norm": err_ema_norm,      # smoothed / scale
            "delta_norm": float(delta_norm),     # normalized improvement
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "pairs_used": used_pairs,
            "pairs_per_step": int(self.pairs_per_step),
            "n_fail": int(fails),
            "reward": float(reward),
            "mode": self.mode,
        }
        return obs, float(reward), terminated, truncated, info

    # --------------- Helpers -----------------

    def _next_pair(self):
        # Cycle through cached pairs deterministically
        if self.active_pairs is None or len(self.active_pairs) == 0:
            # fallback to random if something went wrong
            return self.pair_indices[self.rng.integers(0, len(self.pair_indices))]
        p = self.active_pairs[self.pair_cursor % len(self.active_pairs)]
        self.pair_cursor += 1
        return p

    def _get_obs(self, err_ema_norm=None):
        fx_n = (self.K[0, 0] - self.fx_bound[0]) / (self.fx_bound[1] - self.fx_bound[0])
        fy_n = (self.K[1, 1] - self.fy_bound[0]) / (self.fy_bound[1] - self.fy_bound[0])
        cx_n = (self.K[0, 2] - self.cx_bound[0]) / (self.cx_bound[1] - self.cx_bound[0])
        cy_n = (self.K[1, 2] - self.cy_bound[0]) / (self.cy_bound[1] - self.cy_bound[0])

        if err_ema_norm is None:
            if (self.err_ema is None) or (not np.isfinite(self.err_ema)):
                e_n = 0.0
            else:
                e_n = float(np.clip(self.err_ema / self.err_norm_scale, 0.0, 1.0))
        else:
            e_n = float(np.clip(err_ema_norm, 0.0, 1.0))

        obs = np.array([fx_n, fy_n, cx_n, cy_n, e_n], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
