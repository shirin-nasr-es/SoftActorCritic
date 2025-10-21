"""
data_utils package

Reusable utilities for UAV image processing and basic camera geometry.

Modules
-------
- data_loader
    * UAVDataset: loads UAV grayscale images + metadata from CSV, with optional
      resizing/normalization; returns (1xHxW) tensors and coordinates.

- Intrinsic_matrix
    * get_intrinsic_matrix(scale_factor, image_width, image_height, fx, fy):
      builds a 3×3 intrinsic matrix (fx, fy, cx, cy) adjusted for image scaling.

- reprojection_error
    * Camera: minimal pinhole camera (K, R, t) with 3D->2D projection.
    * compute_reprojection_error: 0.5 * sum of squared reprojection residuals across all observations.
    * reprojection_rmse: reprojection error RMSE (pixels).

- episode_return_like_sample
    * Script to plot per-episode returns and a moving average, auto-selecting
      the most improving training segment (for reports/figures).

- latlon_estimator
    * Script that estimates a 2D UAV trajectory from consecutive UAV images
      via SIFT + homography decomposition, aligns it to GPS (lat/lon), and plots GT vs estimate.
"""

from .data_loader import UAVDataset
from .Intrinsic_matrix import get_intrinsic_matrix
from .reprojection_error import (
    Camera,
    compute_reprojection_error,
    reprojection_rmse,
)

__all__ = [
    "UAVDataset",
    "get_intrinsic_matrix",
    "Camera",
    "compute_reprojection_error",
    "reprojection_rmse",
]

__version__ = "0.1.0"
