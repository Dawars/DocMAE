import os
import logging
from pathlib import Path

import h5py
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class Doc3D(Dataset):
    def __init__(self, data_root: Path, split: str, transforms=None, image_transform=None):
        """
        Args:
            data_root: Directory where the doc3d dataset is extracted
            split: split name of subset of images
            transforms: optional transforms for data augmentation
            image_transforms: optional transforms for data augmentation only applied to rgb image
        """

        self.data_root = data_root
        self.filenames = (data_root / f"{split}.txt").read_text().strip().split("\n")
        self.prefix_img = "img/"
        self.prefix_bm = "bm/"
        self.prefix_uv = "uv/"

        self.transforms = transforms
        self.image_transform = image_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(self.data_root / self.prefix_img / f"{filename}.png").convert("RGB")
        image = datapoints.Image(image)

        # backwards mapping
        h5file = h5py.File(self.data_root / self.prefix_bm / f"{filename}.mat", "r")
        bm = np.array(h5file.get("bm"))
        bm = bm.transpose((2, 1, 0))

        bm = datapoints.Image((bm / image.shape[1:]).transpose((2, 0, 1)))  # absolute back mapping [0, 1]

        # mask from uv
        # Decode the EXR data using OpenCV
        uv_mask = cv2.imread(str(self.data_root / self.prefix_uv / f"{filename}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv_mask = cv2.cvtColor(uv_mask, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # forward mapping
        uv = datapoints.Mask(uv_mask[:2])
        mask = datapoints.Mask(uv_mask[2:3].astype(bool))

        if self.transforms:
            image, bm, uv, mask = self.transforms(image, bm, uv, mask)

        if self.image_transform:
            image = self.image_transform(image.to(torch.uint8))

        return {"image": image, "bm": bm, "uv": uv, "mask": mask}
