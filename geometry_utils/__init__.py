"""
Geometry utilities package.

This package provides modular tools for two-view geometry estimation and evaluation.

Modules:
- pair_geometry:
    - TwoViewGeometry: performs SIFT feature matching, robust RANSAC estimation,
      and 3D→2D triangulation-based reprojection error analysis.
- pairwise_reprojection_error:
    - pairwise_reprojection_error(): high-level wrapper that computes reprojection
      RMSE or MSE between a pair of UAV images using either Homography ("H") or
      Essential ("E") geometry models.
"""

from .pair_geometry import TwoViewGeometry
from .pairwise_reprojection_error import pairwise_reprojection_error

__all__ = ["TwoViewGeometry", "pairwise_reprojection_error"]
__version__ = "0.1.0"
