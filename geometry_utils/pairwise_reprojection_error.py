"""
Compute pairwise reprojection error between two images via two-view geometry (SIFT→RANSAC→pose→triangulation),
returning RMSE or MSE.
"""

import cv2
import numpy as np

from geometry_utils.pair_geometry import TwoViewGeometry
from data_utils.reprojection_error import Camera, compute_reprojection_error


def pairwise_reprojection_error(image1, image2, K, *, ransac_thresh=1.2, mode="rmse",
                                return_details=False, failure_value=None, model = "H"):
    """
    compute reprojection error between two views.

    pipeline:
        1) detect & match SIFT features.
        2) RANSAC (Homography by default) to get inliers.
        3) recover (R, t) and triangulate 3D points.
        4) project 3D points back to both images and measure residuals.

    args:
        image1, image2: Grayscale images (np.uint8 H×W) or tensors [1,H,W] in [0,1].
        K (np.ndarray): 3×3 intrinsic matrix.
        ransac_thresh (float): Pixel threshold for RANSAC (default 1.2 px).
        mode (str): "rmse" → root mean squared error in pixels,
                    "mse"  → mean squared error in pixels².
        return_details (bool): If True, also return a dict with metrics.
        failure_value (float or None): If set, return this value on failure.

    returns:
        float or (float, dict):
            - if return_details=False: scalar error (RMSE in px or MSE in px²).
            - if return_details=True: (scalar, details) where details includes:
                * N_obs   : number of 2D observations
                * L_proj  : sum of squared residuals (pixels²)
                * mean_sq : mean squared residual (pixels²)
                * rmse    : root mean squared residual (pixels)
                * mean    : mean per-observation residual (pixels)
                * n_points: number of triangulated 3D points
    """
    try:
        assert mode in {"rmse", "mse"}

        K = np.asarray(K, dtype=np.float64)

        # two-view pipeline
        # tv = TwoViewGeometry(image1, image2, K)
        tv = TwoViewGeometry(image1, image2, K, model=model)
        tv.match_features()
        tv.find_inliers(ransac_thresh=ransac_thresh)
        R, t, pts1_inliers, pts2_inliers, _ = tv.select_best_pose()

        # triangulate + initial error
        X, p1k, p2k, total_cost, _rmse_tri = tv.triangulate_evaluate(R, t, pts1_inliers, pts2_inliers, verbose=False)

        # build obs for detailed metrics
        cameras = {0: Camera(K, np.eye(3), np.zeros(3)),
                   1: Camera(K, R, t)}

        points3d, obs = {}, []
        for pid in range(len(X)):
            points3d[pid] = X[pid]
            u0, v0 = p1k[pid]; obs.append((0, pid, float(u0), float(v0)))
            u1, v1 = p2k[pid]; obs.append((1, pid, float(u1), float(v1)))

        total_cost_check, per_obs = compute_reprojection_error(obs, cameras, points3d, return_per_obs=True)

        if (not np.isfinite(total_cost)) or (not np.isclose(total_cost, total_cost_check, rtol=1e-6, atol=1e-3)):
            total_cost = total_cost_check

        # metrics
        N = max(1, len(per_obs))       # number of observations
        L_proj = 2.0 * total_cost      # sum of squared residuals (px^2)
        mean_sq = L_proj / N
        rmse = float(np.sqrt(mean_sq))

        per_obs_px = np.sqrt(2.0 * np.asarray(per_obs, dtype=np.float64))
        mean_px = float(np.mean(per_obs_px)) if per_obs_px.size > 0 else 0.0
        details = {
                    "N_obs": int(N),
                    "L_proj": float(L_proj),
                    "mean_sq": float(mean_sq),
                    "rmse": float(rmse),
                    "mean": float(mean_px),
                    "n_points": int(len(X)),
                    }

        # select scalar to return
        scalar = mean_sq if mode == "mse" else rmse
        return (scalar, details) if return_details else scalar

    except Exception:
        # stable failure handling (SAC-friendly if failure_value is set)
        if failure_value is not None:
            return (failure_value, {"N_obs": 0, "L_proj": None, "mean_sq": None, "rmse": None,
                                    "mean": None, "n_points": 0}) if return_details else failure_value
        if return_details:
            return float("inf"), {"N_obs": 0, "L_proj": None, "mean_sq": None, "rmse": None,
                                  "mean": None, "n_points": 0}
        return float("inf")
