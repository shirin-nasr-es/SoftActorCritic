# SoftActorCritic — Self-Calibrating UAV Camera Intrinsics

This repository provides the implementation of a Soft Actor-Critic (SAC) framework for **self-calibration of UAV camera intrinsics** using only visual information.  
The learning process relies on minimizing reprojection error from image pairs and does **not** use any labeled data or metadata during training.

Dataset Description:
This work used  two UAV datasets — AU-AIR[1] and AIRPAI[2] - to evaluate the proposed self-calibration framework under varying visual and texture conditions.
The AIRPAI[2] dataset includes Building and Grass subsets with high-resolution UAV images captured from diverse viewpoints, 
while the AU-AIR[1] dataset contributes additional evaluation data by using one frame out of every five from its first video sequence to test generalization.
All associated metadata (camera intrinsics, GPS, and IMU information) is employed only for evaluation and comparison, not for training or reward computation. 
The SAC agent learns camera intrinsics purely from visual reprojection error between image pairs, without using any ground-truth parameters.

Please click on the following link for images used in this wok:
https://drive.google.com/drive/folders/1YmYroIzXu0mSyFk7hFsi238S2I22Nibd?usp=drive_link

[1] Bozcan, Ilker, and Erdal Kayaan. "AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance." IEEE International Conference on Robotics and Automation (ICRA), 2020, to appear.
[2] Moxuan Ren, ; Jianan Li; Liqiang Song; Hui Li; Tingfa Xu.  "MLP-Based Efficient Stitching Method for UAV Images," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 2503305, doi: 10.1109/LGRS.2022.3141890.


