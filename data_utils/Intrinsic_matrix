import numpy as np

def get_intrinsic_matrix(scale_factor, image_width, image_height,
                         focal_length_x, focal_length_y):
    """
    Computes a 3x3 camera intrinsic matrix (fx, fy, cx, cy) adjusted for image scaling.

    Parameters
    ----------
    scale_factor : float
        Factor by which images are scaled down (e.g., 0.5 or 0.125)
    image_width : int
        Original image width in pixels
    image_height : int
        Original image height in pixels
    focal_length_x, focal_length_y : float
        Focal lengths in pixels (from calibration or estimation)

    Returns
    -------
    intr_mx : np.ndarray
        3×3 intrinsic matrix corresponding to the resized image
    fx, fy, cx, cy : tuple of floats
        Intrinsic parameters for reference
    """

    # Scale focal lengths for the resized image
    fx = focal_length_x * scale_factor
    fy = focal_length_y * scale_factor

    # Scale principal point as well
    cx = 0.5 * image_width * scale_factor
    cy = 0.5 * image_height * scale_factor

    intr_mx = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return intr_mx, fx, fy, cx, cy
