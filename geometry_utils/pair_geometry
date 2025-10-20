"""Two-view geometry utilities: SIFT matching, RANSAC (H/E), pose selection via triangulation + reprojection error, and evaluation helpers."""

import cv2
import numpy as np

from data_utils.reprojection_error import Camera, compute_reprojection_error, reprojection_rmse


class TwoViewGeometry:
    def __init__(self, image1, image2, K, model="H"):
        """
        handles feature matching, inlier estimation (Homography/Essential), pose recovery, triangulation, and
         reprojection-error scoring for an image pair.

        image1, image2: np.uint8 grayscale (H, W) or torch tensors [1,H,W] in [0,1]
        K: 3x3 intrinsics
        """
        self.im1 = self._convert_to_uint8(image1)
        self.im2 = self._convert_to_uint8(image2)
        self.K = np.asarray(K, float)
        self.model = model   # "H" or "E"

        self.matches = None
        self.kpts1_matched = None
        self.kpts2_matched = None

        self.H = None
        self.inliers = None
        self.kpts1_inliers = None
        self.kpts2_inliers = None

        # print(f"[TwoViewGeometry] model={model}")

    @staticmethod
    def _convert_to_uint8(im):
        # accept numpy or torch [1,H,W]; return uint8 (H,W)
        if hasattr(im, "numpy"):
            im = im.squeeze(0).detach().cpu().numpy()
            if im.max() <= 1.0: im = im * 255.0

        im = np.asarray(im).astype(np.uint8)
        return im

    def match_features(self):
        # detect SIFT keypoints + descriptors and return matched keypoints between the two images..
        sift = cv2.SIFT_create()
        kpts1, desc1 = sift.detectAndCompute(self.im1, None)
        kpts2, desc2 = sift.detectAndCompute(self.im2, None)

        if desc1 is None or desc2 is None or len(kpts1) < 8 or len(kpts2) < 8:
            raise RuntimeError("\n Not enough features for pose estimation.")

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)

        if len(matches) < 8:
            raise RuntimeError("\n Not enough matches for pose estimation.")

        kpts1_matched = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=float)
        kpts2_matched = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=float)

        self.matches = matches
        self.kpts1_matched = kpts1_matched
        self.kpts2_matched = kpts2_matched


        return kpts1_matched, kpts2_matched, matches

    def find_inliers(self, ransac_thresh=1.2):
        # run RANSAC for Homography ('H') or Essential ('E') to get inlier correspondences and the estimated H/E.
        if self.kpts1_matched is None or self.kpts2_matched is None:
            raise RuntimeError("Call match_features() first.")

        # Ensure float32 Nx2
        pts1 = np.asarray(self.kpts1_matched, dtype=np.float32).reshape(-1, 2)
        pts2 = np.asarray(self.kpts2_matched, dtype=np.float32).reshape(-1, 2)

        if self.model == "E":
            # Essential-matrix RANSAC (threshold is in pixels when cameraMatrix is given)
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=float(ransac_thresh)
            )
            if E is None or mask is None:
                raise RuntimeError("Compute Essential matrix failed.")
            inliers = mask.ravel().astype(bool)
            if inliers.sum() < 5:
                raise RuntimeError("Too few inliers after Essential RANSAC.")

            # If multiple solutions are stacked, pick the first 3x3 block
            if E.ndim == 2 and E.shape == (3, 3):
                self.E = E
            else:
                self.E = E.reshape(-1, 3, 3)[0]

            self.H = None  # clear other model
        else:
            # Homography RANSAC (original path)
            H, mask = cv2.findHomography(
                pts1, pts2, cv2.RANSAC, float(ransac_thresh)
            )
            if H is None or mask is None:
                raise RuntimeError("Compute Homography matrix failed.")
            inliers = mask.ravel().astype(bool)
            if inliers.sum() < 8:
                raise RuntimeError("Too few inliers after Homography RANSAC.")

            self.H = H
            self.E = None  # clear other model

        # Save inliers
        self.inliers = inliers
        self.kpts1_inliers = pts1[inliers]
        self.kpts2_inliers = pts2[inliers]

        return (self.E if self.model == "E" else self.H), self.kpts1_inliers, self.kpts2_inliers, inliers

    def rmse_for_solution(self, R, t, p1i, p2i):
        # triangulate with candidate (R, t), enforce cheirality, and return reprojection RMSE on surviving points.

        K = self.K.astype(np.float64)
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3, )

        Pj0 = K @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3,1), dtype=np.float64)])
        Pj1 = K @ np.hstack([R, t.reshape(3, 1)])

        X_h = cv2.triangulatePoints(Pj0.astype(np.float64), Pj1.astype(np.float64),
                                    p1i.T.astype(np.float64), p2i.T.astype(np.float64))
        X = cv2.convertPointsFromHomogeneous(X_h.T).reshape(-1, 3)

        # cheirality
        front0 = X[:, 2] > 0
        front1 = ((X @ R.T) + t.reshape(1, 3))[:, 2] > 0
        keep = front0 & front1
        if not np.any(keep):
            return np.inf, None, None, None

        X = X[keep]
        p1k = p1i[keep]
        p2k = p2i[keep]

        cams = {0: Camera(K, np.eye(3), np.zeros(3)),
                1: Camera(K, R, t)}
        pts3d, obs = {}, []
        for pid in range(len(X)):
            pts3d[pid] = X[pid]
            u0, v0 = p1k[pid];
            u1, v1 = p2k[pid]
            obs += [(0, pid, float(u0), float(v0)),
                    (1, pid, float(u1), float(v1))]

        rmse = reprojection_rmse(obs, cams, pts3d)
        return rmse, X, p1k, p2k

    def select_best_pose(self):
        """
        choose (R,t) by evaluating candidates on triangulation/reprojection RMSE.
        returns:
            R, t, pts1_out, pts2_out, rmse_best
        where pts1_out/pts2_out are the inlier correspondences used for the best pose
        (after the pose-specific inlier mask), and rmse_best is the evaluation score.
        """
        if self.kpts1_inliers is None or self.kpts2_inliers is None:
            raise RuntimeError("Call find_inliers() first.")

        best = (np.inf, None, None, None, None)  # (rmse, R, t, p1_out, p2_out)

        if getattr(self, "model", "H") == "E":
            if getattr(self, "E", None) is None:
                raise RuntimeError("Essential matrix not set. Call find_inliers() with model='E' first.")

            # handle single E (3x3) or multiple Es (N x 3 x 3)
            if self.E.ndim == 2:
                E_list = [self.E]
            else:
                # expect shape (N,3,3)
                E_list = [self.E[i] for i in range(self.E.shape[0])]

            for Ei in E_list:
                Ei = np.asarray(Ei, dtype=np.float64)
                # recoverPose returns an inlier mask specific to this pose
                n_inl, R, t, mask_pose = cv2.recoverPose(
                    Ei,
                    np.asarray(self.kpts1_inliers, dtype=np.float64),
                    np.asarray(self.kpts2_inliers, dtype=np.float64),
                    np.asarray(self.K, dtype=np.float64)
                )
                if n_inl is None or int(n_inl) < 8 or mask_pose is None:
                    continue

                # filter to pose-consistent inliers
                m = mask_pose.ravel().astype(bool)
                p1 = self.kpts1_inliers[m]
                p2 = self.kpts2_inliers[m]
                if len(p1) < 8:
                    continue

                # evaluate this (R,t)
                rmse_i, Xi, p1k, p2k = self.rmse_for_solution(
                    np.asarray(R, dtype=np.float64),
                    np.asarray(t, dtype=np.float64).reshape(3),
                    p1, p2
                )
                if np.isfinite(rmse_i) and rmse_i < best[0]:
                    best = (rmse_i, R, t.reshape(3), p1k, p2k)

            rmse_best, Rb, tb, pts1_out, pts2_out = best
            if Rb is None:
                raise RuntimeError("No valid (R,t) solution from Essential matrix.")
            return Rb, tb, pts1_out, pts2_out, float(rmse_best)

        # --------- Homography fallback (legacy path) ----------
        if getattr(self, "H", None) is None:
            raise RuntimeError("Homography not set. Call find_inliers() first.")

        num_solutions, Rs, ts, _ = cv2.decomposeHomographyMat(
            np.asarray(self.H, dtype=np.float64),
            np.asarray(self.K, dtype=np.float64)
        )
        if num_solutions == 0:
            raise RuntimeError("Decompose Homography matrix failed.")

        for Ri, ti in zip(Rs, ts):
            rmse_i, Xi, p1k, p2k = self.rmse_for_solution(
                np.asarray(Ri, dtype=np.float64),
                np.asarray(ti, dtype=np.float64).reshape(3),
                self.kpts1_inliers, self.kpts2_inliers
            )
            if np.isfinite(rmse_i) and rmse_i < best[0]:
                best = (rmse_i, Ri, ti.reshape(3), p1k, p2k)

        rmse_best, Rb, tb, pts1_out, pts2_out = best
        if Rb is None:
            raise RuntimeError("No valid (R,t) solution after homography decomposition.")
        return Rb, tb, pts1_out, pts2_out, float(rmse_best)

    def triangulate_evaluate(self, R, t, pts1_inliers, pts2_inliers, *, verbose=False):
        """
            triangulate 3D points from inlier 2D correspondences and compute reprojection error.

            args:
                R (np.ndarray): 3x3 rotation (second camera wrt first).
                t (np.ndarray): 3-vector translation.
                pts1_inliers (np.ndarray): Nx2 inlier pixels from image 1.
                pts2_inliers (np.ndarray): Nx2 inlier pixels from image 2.
                verbose (bool): if True, print intermediate arrays/sizes.

            returns:
                X (np.ndarray): Mx3 triangulated 3D points after cheirality.
                pts1_kept (np.ndarray): Mx2 surviving inliers in image 1.
                pts2_kept (np.ndarray): Mx2 surviving inliers in image 2.
                total_cost (float): sum of 0.5*||residual||^2 over all observations.
                rmse (float): reprojection RMSE in pixels.
        """

        K = self.K.astype(np.float64)

        pts1_inliers = np.asarray(pts1_inliers, float)
        pts2_inliers = np.asarray(pts2_inliers, float)
        assert pts1_inliers.shape == pts2_inliers.shape and pts1_inliers.shape[1] == 2, \
            "Inliers must be Nx2 and of equal length."

        # build projection matrices P = K [R|t] for the two views
        Pj0 = K @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3,1), dtype=np.float64)])
        Pj1 = K @ np.hstack([R, t.reshape(3, 1)])

        # inlier 2D point correspondences (2, N)
        pts1 = pts1_inliers.astype(np.float64).T
        pts2 = pts2_inliers.astype(np.float64).T

        # triangulate and convert from homogeneous (4xN) to Euclidean (Nx3)
        X_h = cv2.triangulatePoints(Pj0, Pj1, pts1, pts2)
        X = cv2.convertPointsFromHomogeneous(X_h.T).reshape(-1, 3)

        if verbose:
            print("Triangulated (before cheirality):", X.shape[0])

        # cheirality: keep points in front of both cameras
        front0 = X[:, 2] > 0
        front1 = ((X @ R.T) + t.reshape(1, 3))[:, 2] > 0
        keep = front0 & front1

        X = X[keep]
        pts1_kept = pts1_inliers[keep]
        pts2_kept = pts2_inliers[keep]

        if X.size == 0:
            raise RuntimeError("No valid triangulated points after cheirality filtering.")

        if verbose:
            print("Triangulated (kept):", X.shape[0])

        # build cameras
        cameras = {
            0: Camera(K, np.eye(3), np.zeros(3)),
            1: Camera(K, R, t),
        }
        # pack 3D points + observations
        points3d, obs = {}, []
        for pid in range(len(X)):
            points3d[pid] = X[pid]
            u0, v0 = pts1_kept[pid]
            u1, v1 = pts2_kept[pid]
            obs.append((0, pid, float(u0), float(v0)))
            obs.append((1, pid, float(u1), float(v1)))

        total_cost, _ = compute_reprojection_error(obs, cameras, points3d, return_per_obs=True)
        rmse = reprojection_rmse(obs, cameras, points3d)

        if verbose:
            print(f"Total BA cost: {total_cost:.4f} px^2")
            print(f"Reprojection RMSE: {rmse:.4f} px")

        return X, pts1_kept, pts2_kept, total_cost, rmse



