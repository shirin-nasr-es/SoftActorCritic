import cv2
import numpy as np
import pandas as pd

from pathlib import Path

import torch
from torch.utils.data import Dataset

"""
UAVDataset class for loading and preprocessing UAV images with metadata. 
It converts images to grayscale, resizes them, and optionally applies normalization. 
Designed for efficient batch-wise loading in deep learning pipelines.
"""
class UAVDataset(Dataset):

    def __init__(
        self,
        images_path,
        data_path,
        resize_factor=1.0,
        normalization= "none",
        fail_sentinel_size=(128,128),       # used if an image can't be read
    ):
        self.images_path = Path(images_path)
        self.data_path = Path(data_path)
        self.resize_factor = float(resize_factor)
        self.normalization = normalization
        self.fail_sentinel_size = tuple(fail_sentinel_size)

        if not self.images_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.images_path}")
        if not self.data_path.exists():
            raise ValueError(f"CSV path does not exist: {self.data_path}")

        df = pd.read_csv(self.data_path)
        required = {"image_name","longitude","latitude","altitude"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")


        # store rows as tuples for speed
        self.rows = [
            (row["image_name"], float(row["longitude"]), float(row["latitude"]), float(row["altitude"]))
            for _, row in df.iterrows()
        ]

        # pre-validate global normalization tuple if provided
        if isinstance(self.normalization, tuple):
            if len(self.normalization) != 3 or self.normalization[0] != "global":
                raise ValueError(f"For global normalization, use ('global', mean, std).")
            _, self.global_mean, self.global_std = self.normalization
            self.global_mean = float(self.global_mean)
            self.global_std = float(self.global_std if self.global_std!=0 else 1.0)

    def __len__(self):
        return len(self.rows)

    def _read_and_process(self, im_path):
        im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None

        if self.resize_factor != 1.0:
            h, w = im.shape[:2]
            new_w = max(1, int(w * self.resize_factor))
            new_h = max(1, int(h * self.resize_factor))
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # to float32 and scale to [0,1]
        im = im.astype(np.float32) / 255.0

        # normalization
        if self.normalization == "zscore_per_image":
            m = float(im.mean())
            s = float(im.std())
            s = s if s > 1e-12 else 1.0
            im = (im - m) / s
        elif isinstance(self.normalization, tuple) and self.normalization[0] == "global":
            im = (im - self.global_mean) / (self.global_std if self.global_std != 0 else 1.0)
        # else : "none" -> keep [0,1]

        # to tensor [1, H, W]
        im = torch.from_numpy(im).unsqueeze(0)
        return im

    def __getitem__(self, idx):
        filename, lon, lat, alt = self.rows[idx]
        im_path = self.images_path / filename

        image = self._read_and_process(im_path)
        if image is None:

            h, w = self.fail_sentinel_size
            image = torch.zeros((1, h, w), dtype=torch.float32)

        meta = {
            "filename": filename,
            "coordinates": torch.tensor([lon, lat, alt], dtype=torch.float32),
        }

        return image, meta








