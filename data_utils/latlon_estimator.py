"""Estimates a 2D UAV trajectory from consecutive AIRPAI images via SIFT + homography decomposition, aligns it to GPS (lat/lon), and plots GT vs. estimate."""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from data_loader import UAVDataset
from sklearn.preprocessing import StandardScaler
from Intrinsic_matrix import get_intrinsic_matrix


def to_cv_gray(img_torch):
    # img_torch: torch tensor [1,H,W] in [0,1] float32
    if isinstance(img_torch, torch.Tensor):
        g = img_torch.squeeze(0).detach().cpu().numpy()  # [H,W], float32
    else:
        g = img_torch  # already numpy
    g = (g * 255.0).clip(0, 255).astype(np.uint8)  # [0,1] -> [0,255] uint8
    return np.ascontiguousarray(g)


def normalize_and_align(estimate, gt):
    """
    z-score normalize `estimate` then map it to the distribution of `gt`
    (so mean/std match gt). Returns (estimate_mapped, gt_norm).
    """
    estimate = np.asarray(estimate).reshape(-1, 1)
    gt = np.asarray(gt).reshape(-1, 1)

    est_scaler = StandardScaler().fit(estimate)
    estimate_norm = est_scaler.transform(estimate)

    gt_scaler = StandardScaler().fit(gt)
    estimate_final = gt_scaler.inverse_transform(estimate_norm)
    gt_norm = gt_scaler.transform(gt)
    return estimate_final, gt_norm


# ---------- Dataset Initialization --------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

image_dir = PROJECT_ROOT / "Data" / "AIRPAI" / "Images" / "grass"
csv_path  = PROJECT_ROOT / "Data" / "AIRPAI" / "CSV Files" / "grass.csv"

if not image_dir.exists():
    raise FileNotFoundError(f"Dataset path does not exist: {image_dir}")
if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

resize_factor = 0.125
ds = UAVDataset(
    images_path=image_dir,
    data_path=csv_path,
    resize_factor=resize_factor,
    normalization='none',
    fail_sentinel_size=(128, 128),
)

print(f"Dataset size: {len(ds)} images")
num_images = len(ds)

# --------------- Camera Parameters -------------
fx, fy = 600.0, 550.0  # Example SAC-estimated focal lengths

im0, _ = ds[0]
_, H, W = im0.shape
print("Image size:", W, "x", H)
image_width, image_height = W, H

# IMPORTANT: unpack the tuple; use only K (np.ndarray)
K_tuple = get_intrinsic_matrix(
    scale_factor=resize_factor,
    image_width=image_width,
    image_height=image_height,
    focal_length_x=fx,
    focal_length_y=fy
)
if isinstance(K_tuple, tuple):
    K, fx_eff, fy_eff, cx_eff, cy_eff = K_tuple
else:
    K = K_tuple
K = np.asarray(K, dtype=np.float64)
print("Intrinsic Matrix K:\n", K)

# ---------- Load Ground Truth GPS Coordinates ----------
Xr, Yr, Zr = [], [], []
for _, meta in ds:
    lon, lat, alt = meta["coordinates"].tolist()
    # Xr: latitude, Yr: longitude (kept as in your original)
    Xr.append(lat)
    Yr.append(lon)
    Zr.append(alt)
Xr, Yr, Zr = np.array(Xr), np.array(Yr), np.array(Zr)

print("Trajectory arrays built:")
print("Latitudes (Xr):", Xr[:5])
print("Longitudes (Yr):", Yr[:5])
print("Altitudes (Zr):", Zr[:5])

# ---------- Feature Matching and Pose Estimation ----------
translation_vectors = []
num_inliers = []

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # use KNN + ratio test

for pair_idx in range(num_images - 1):
    print(f'\n---- Processing Image Pair {pair_idx + 1}/{num_images - 1} ----')

    I1_t, meta1 = ds[pair_idx]
    I2_t, meta2 = ds[pair_idx + 1]

    I1 = to_cv_gray(I1_t)
    I2 = to_cv_gray(I2_t)

    # SIFT
    kps1, des1 = sift.detectAndCompute(I1, None)
    kps2, des2 = sift.detectAndCompute(I2, None)
    print(f'Number of keypoints: {len(kps1)} | {len(kps2)}')

    if des1 is None or des2 is None or len(kps1) < 4 or len(kps2) < 4:
        print("Insufficient keypoints/descriptors; appending zero translation.")
        translation_vectors.append(np.zeros(3, dtype=float))
        continue

    # KNN match + ratio test (Lowe’s ratio)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        print(f"Too few good matches ({len(good)}); appending zero translation.")
        translation_vectors.append(np.zeros(3, dtype=float))
        continue

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    # Homography (RANSAC)
    Hmat, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.2)
    if Hmat is None:
        print("Homography failed; appending zero translation.")
        translation_vectors.append(np.zeros(3, dtype=float))
        continue

    inlier_count = int(mask.sum()) if mask is not None else 0
    num_inliers.append(inlier_count)
    print(f'Number of inliers: {inlier_count}')

    # Decompose Homography
    try:
        num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(Hmat, K)
    except cv2.error as e:
        print("decomposeHomographyMat error:", e)
        translation_vectors.append(np.zeros(3, dtype=float))
        continue

    if num_solutions <= 0:
        print("No homography decomposition solutions; appending zero translation.")
        translation_vectors.append(np.zeros(3, dtype=float))
        continue

    # Pick a plausible solution.
    # Simple heuristic: choose solution with positive forward component (tz) and largest inlier support (mask already applied to H).
    # If multiple tz>0, pick the one with the largest tz magnitude; otherwise fall back to first.
    valid_idxs = [i for i in range(num_solutions) if float(ts[i][2]) > 0]
    if valid_idxs:
        # choose the one with largest tz
        valid = max(valid_idxs, key=lambda i: float(ts[i][2]))
        t_chosen = np.array(ts[valid], dtype=float).reshape(3)
        R_chosen = np.array(Rs[valid], dtype=float)
    else:
        # fall back: pick solution with max |tz|
        valid = max(range(num_solutions), key=lambda i: abs(float(ts[i][2])))
        t_chosen = np.array(ts[valid], dtype=float).reshape(3)
        R_chosen = np.array(Rs[valid], dtype=float)
        print("No tz>0; falling back to solution with max |tz|.")

    translation_vectors.append(t_chosen)

translation_vectors = np.array(translation_vectors, dtype=float)

# Extract X, Y components (image-plane proxy)
translation_X = translation_vectors[:, 0]
translation_Y = translation_vectors[:, 1]

# Integrate to build a relative 2D path (start at zero)
X_sift = np.concatenate([[0.0], np.cumsum(translation_X)])
Y_sift = np.concatenate([[0.0], np.cumsum(translation_Y)])

# ---------------------- Align to GT -------------------------
X_sift_final, Xr_normalized = normalize_and_align(X_sift, Xr)
Y_sift_final, Yr_normalized = normalize_and_align(Y_sift, Yr)

# ---------------------- Plots ---------------------
# Use 0..N-1 like the figure you attached
idx = np.arange(num_images)

# Styling
GT_COLOR = "#1f77b4"   # blue
EST_COLOR = "#ff7f0e"  # orange
LINE_W   = 4.0
MS_GT    = 7
MS_EST   = 8

plt.rcParams.update({
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})

fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
fig.suptitle("AIRPAI-Grass Dataset", fontsize=28, fontweight="bold")

# ----- Latitude (X) -----
axs[0].plot(idx, Xr, marker="o", markersize=MS_GT, linewidth=LINE_W,
            color=GT_COLOR, label="Ground Truth")
axs[0].plot(idx, X_sift_final.squeeze(), marker="x", markersize=MS_EST,
            linewidth=LINE_W, linestyle="--", color=EST_COLOR, label="Estimated")

axs[0].set_title("Latitude Comparison", fontweight="bold", color="navy")
axs[0].set_ylabel("Latitude (degrees)", color="navy", fontweight="bold")
axs[0].grid(True, linestyle="--", color="#444444", linewidth=0.9, alpha=1.0)
axs[0].legend(
    loc="upper left",
    prop={'weight': 'bold', 'size': 18}  # bold + larger font size
)


# ----- Longitude (Y) -----
axs[1].plot(idx, Yr, marker="o", markersize=MS_GT, linewidth=LINE_W,
            color=GT_COLOR, label="Ground Truth")
axs[1].plot(idx, Y_sift_final.squeeze(), marker="x", markersize=MS_EST,
            linewidth=LINE_W, linestyle="--", color=EST_COLOR, label="Estimated")

axs[1].set_title("Longitude Comparison", fontweight="bold", color="navy")
axs[1].set_xlabel("Frame Number", fontweight="bold")
axs[1].set_ylabel("Longitude (degrees)", color="navy", fontweight="bold")
axs[1].grid(True, linestyle="--", color="#444444", linewidth=0.9, alpha=1.0)
axs[1].legend(
    loc="upper left",
    prop={'weight': 'bold', 'size': 18}  # bold + larger font size
)


# make x ticks a bit sparser like your figure
axs[1].set_xticks(np.arange(0, num_images, max(1, num_images // 10)))

for ax in axs:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16)


# Save the figure in high resolution (HD) PDF
# fig.savefig("AIRPAI_grass_HD.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1)


fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
