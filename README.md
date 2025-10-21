# SoftActorCritic — Self-Calibrating UAV Camera Intrinsics

This repository provides the implementation of a Soft Actor-Critic (SAC) framework for **self-calibration of UAV camera intrinsics** using only visual information.  
The learning process relies on minimizing reprojection error from image pairs and does **not** use any labeled data or metadata during training.

**Dataset Description:**  
This work used two UAV datasets — [AU-AIR](https://arxiv.org/abs/2001.11737) [1] and [AIRPAI](https://github.com/dededust/UAV-AIRPAI) [2] — to evaluate the proposed self-calibration framework under varying visual and texture conditions.  
The AIRPAI dataset includes *building* and *grass* subsets with high-resolution UAV images captured from diverse viewpoints, while the AU-AIR dataset contributes additional evaluation data by using one frame out of every five from its first video sequence to test generalization.  
All associated metadata (camera intrinsics, GPS, and IMU information) is employed **only for evaluation and comparison**, not for training or reward computation.  
The SAC agent learns camera intrinsics purely from **visual reprojection error** between image pairs, without using any ground-truth parameters.

**Data Access:**
All data used for this work are shared [here](https://drive.google.com/drive/folders/1YmYroIzXu0mSyFk7hFsi238S2I22Nibd?usp=drive_link).
