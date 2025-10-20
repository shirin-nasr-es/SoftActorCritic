# === episode_return_like_sample.py ===
# Faint per-episode line + thick orange moving average, cropped to the
# most improving episode segment (auto). Set AUTO_SEGMENT=False to choose
# START_EP/END_EP manually.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Plots per-episode returns with a moving average, automatically highlighting
the most improving segment of training (portable paths; auto-select latest run)."""


# -------- PORTABLE PATHS --------
# This file lives in: <repo_root>/data_utils/
# Base project directory = parent of this script's folder
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
OUT_DIR  = BASE_DIR / "data_utils" / "plots"


def latest_run_dir(runs_dir: Path) -> Path:
    # return the most recently modified subdirectory in runs_dir.
    if not runs_dir.exists():
        raise FileNotFoundError(f"'runs' folder not found at: {runs_dir}")
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run subfolders found in: {runs_dir}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


# Auto-pick CSV from the latest run folder
LATEST = latest_run_dir(RUNS_DIR)
CSV_PATH = LATEST / "train_log.csv"
if not CSV_PATH.exists():
    # fall back: try to find any csv inside the latest run dir
    candidates = list(LATEST.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in latest run folder: {LATEST}")
    CSV_PATH = candidates[0]


# -------- PLOT CONFIG --------
MA_WIN        = 25     # moving average window (episodes)
AUTO_SEGMENT  = True   # auto-pick best improving segment
MIN_EP_SPAN   = 80     # min length (episodes) for auto segment
# If AUTO_SEGMENT=False, use these:
START_EP, END_EP = 525, 670
FIG_W, FIG_H, DPI = 7.10, 2.20, 300
# -----------------------------


def movavg(y, w):
    s = pd.Series(y, dtype=float)
    w = max(3, int(w))
    return s.rolling(w, center=True, min_periods=max(3, w // 3)).mean().to_numpy()


# --- load returns per episode ---
df = pd.read_csv(CSV_PATH)

# Validate presence of required columns
required = {"episode", "return"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV must contain columns {sorted(required)}; missing {sorted(missing)} in: {CSV_PATH}")

# aggregate: one return per episode (use max to be robust to multiple rows per episode)
ep_ret = df.groupby("episode", as_index=False)["return"].max().sort_values("episode")

ep  = ep_ret["episode"].to_numpy()
ret = ep_ret["return"].to_numpy()

# If the run is very short, shrink the smoothing window
if len(ep) < MA_WIN:
    print(f"[warn] MA_WIN={MA_WIN} > n_episodes={len(ep)}; reducing window to ~{max(3, len(ep)//5)}.")
    MA_WIN = max(3, len(ep) // 5)

ret_ma = movavg(ret, MA_WIN)

# --- choose improving segment ---
if AUTO_SEGMENT:
    if len(ep) < max(MIN_EP_SPAN, 5):
        # Not enough episodes to segment—use full range
        sl = slice(0, len(ep))
    else:
        best_i = 0
        best_j = len(ep) - 1
        best_gain = -np.inf
        left = 0
        max_val = -np.inf
        max_idx = 0

        for right in range(len(ep)):
            while ep[right] - ep[left] >= MIN_EP_SPAN:
                if ret_ma[left] > max_val:
                    max_val, max_idx = ret_ma[left], left
                left += 1
            if ep[right] - ep[max_idx] >= MIN_EP_SPAN:
                gain = ret_ma[right] - max_val
                if gain > best_gain:
                    best_gain = gain
                    best_i = max_idx
                    best_j = right
        sl = slice(best_i, best_j + 1)
else:
    sl = (ep >= START_EP) & (ep <= END_EP)

ep_c     = ep[sl]
ret_c    = ret[sl]
ret_ma_c = ret_ma[sl]

# --- plot ---
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.plot(ep_c, ret_c,  color="#1f77b4", alpha=0.25, linewidth=2.0, label="Per-episode return")
ax.plot(ep_c, ret_ma_c, color="#ff7f0e", linewidth=3.0, label=f"Moving avg (win={MA_WIN} ep)")

ax.set_title("Episode return (improving segment)", pad=6)
ax.set_xlabel("Episode")
ax.set_ylabel("Return (higher is better)")
ax.grid(True, linestyle="--", alpha=0.35)
ax.margins(x=0.01)
ax.legend(loc="best", frameon=False)

# tidy y-limits
ymin = np.nanmin(np.concatenate([ret_c, ret_ma_c]))
ymax = np.nanmax(np.concatenate([ret_c, ret_ma_c]))
rng  = (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 1e-6 else 1.0
ax.set_ylim(ymin - 0.05 * rng, ymax + 0.05 * rng)

# save + show
OUT_DIR.mkdir(parents=True, exist_ok=True)
png_path = OUT_DIR / "episode_return_like_sample.png"
pdf_path = OUT_DIR / "episode_return_like_sample.pdf"
fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()

print("Saved to:", png_path)
