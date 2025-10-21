"""Pinhole camera utilities to project 3D points and compute reprojection error (total cost and RMSE)."""

import numpy as np

class Camera:
    """
    simple pinhole camera with intrinsic (K) and extrinsic (R, t) parameters.
      - K: 3x3 intrinsic matrix
      - R: 3x3 rotation matrix
      - t: 3x1 translation vector
    """

    def __init__(self, K, R, t):
        """
        Initialize the camera with intrinsic and extrinsic parameters.

        args:
            K (np.ndarray): 3x3 camera intrinsic matrix
            R (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3-element translation vector
        """
        self.K = np.asarray(K, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float).reshape(3)

    def projection_matrix(self):
        """
        compute the 3x4 projection matrix P = K [R | t].

        note:
            P maps 3D world points X (as homogeneous [X, Y, Z, 1]^T)
            into homogeneous image coordinates x = P X.
            to obtain pixel coordinates (u, v), divide by the third component:
                u = x0 / x2, v = x1 / x2
        """
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])  # [R|t], shape (3,4)
        return np.dot(self.K, Rt)  # shape (3,4)


    def project_points(self, X):
        """
        project 3D world points into 2D pixel coordinates.

        args:
            X (np.ndarray): Nx3 array of 3D world points.

        returns:
           np.ndarray: Nx2 array of 2D pixel coordinates (u,v).
        """

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, 3)

        # homogeneous coordinates
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_h = np.hstack([X, ones])  # (N,4)

        # apply projection
        P = self.projection_matrix()    # (3,4)
        x = np.dot(P, X_h.T).T          # (N,3)

        # normalize to get (u, v); guard against divide-by-zero
        w = np.clip(x[:, 2:3], 1e-12, None) # (N,1)
        uv = x[:, :2] / w                                # (N,2)
        return uv


def compute_reprojection_error(observations, cameras, points3d, return_per_obs=False):
    """
    compute total reprojection error over all observations.

    args:
        observations (list of 4-tuples): [(cam_id, pt_id, u, v), ...]
            cam_id: int key into `cameras`
            pt_id : int key into `points3d`
            u, v  : observed pixel coordinates (floats)
        cameras (dict): cam_id -> camera instance (with K, R, t)
        points3d (dict): pt_id -> np.array([X, Y, Z]) in world coordinates
        return_per_obs (bool): if True, also return per-observation residual norms (px)

    returns:
        total_cost (float): sum over all observations of 1/2 * ||r||^2 (pixels^2)
         per_obs_norm (np.ndarray or None): (M,) residual lengths in pixels, if requested
    """

    obs = list(observations)
    if not obs:
        return (0.0, None) if return_per_obs else 0.0

    # group obs by camera for efficient projection
    cam_to_idx = {}
    for i, (cid, pid, u, v) in enumerate(obs):
        cam_to_idx.setdefault(cid, []).append(i)

    residuals = np.zeros((len(obs), 2), dtype=float)
    valid_rows = []

    # for each camera, project its points in a batch
    for cid, idxs in cam_to_idx.items():
        cam = cameras.get(cid, None)
        if cam is None:
            continue  # skip unknown camera id

        pts, uvs, valid = [], [], []
        for i in idxs:
            _, pid, u, v = obs[i]

            X = points3d.get(pid, None)
            if X is None:
                continue    # skip missing 3D points

            pts.append(X)
            uvs.append([u, v])
            valid.append(i)

        if not pts:
            continue

        pts = np.asarray(pts, dtype=float)
        uvs = np.asarray(uvs, dtype=float)

        # project with this camera
        uv_pred = cam.project_points(pts)   # (N,2)

        residuals[valid, :] = (uvs - uv_pred)
        valid_rows.extend(valid)

    if not valid_rows:
        return (0.0, np.array([])) if return_per_obs else 0.0

    valid_rows = np.asarray(valid_rows, dtype=int)
    costs = 0.5 * np.sum(residuals[valid_rows] ** 2, axis=1)
    total_cost = float(np.sum(costs))

    if return_per_obs:
        per_norm = np.linalg.norm(residuals[valid_rows], axis=1)    # (M,)
        return total_cost, per_norm

    return total_cost

def reprojection_rmse(observations, cameras, points3d, return_per_obs=False):
    """
    root-mean-square reprojection error (pixels) over all observations.

    args:
        observations (list of 4-tuples): [(cam_id, pt_id, u, v), ...]
        cameras (dict): cam_id -> Camera
        points3d (dict): pt_id -> np.array([X, Y, Z])

    returns:
        float: RMSE in pixels
    """
    total, per = compute_reprojection_error(observations, cameras, points3d, return_per_obs=True)

    if per is None or len(per) == 0:
        return 0.0
    return float (np.sqrt(np.mean(per ** 2)))

