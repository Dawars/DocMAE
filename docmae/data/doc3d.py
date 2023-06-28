import os
import logging
import warnings
from io import BytesIO
from pathlib import Path

import h5py
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from minio import Minio
from PIL import Image
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
torchvision.disable_beta_transforms_warning()


class Doc3D(Dataset):
    def __init__(self, data_root: Path, split: str, transforms=None):
        """
        Args:
            data_root: Directory where the doc3d dataset is extracted
            split: split name of subset of images
            transforms: optional transforms for data augmentation
        """

        self.data_root = data_root
        self.filenames = (data_root / f"{split}.txt").read_text().split()
        self.prefix_img = "img/"
        self.prefix_bm = "bm/"
        self.prefix_uv = "uv/"

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(self.data_root / self.prefix_img / f"{filename}.png").convert("RGB")
        image = datapoints.Image(image)

        # backwards mapping
        h5file = h5py.File(self.data_root / self.prefix_bm / f"{filename}.mat", "r")
        flow = np.array(h5file.get("bm"))
        flow = datapoints.Image(flow)

        # mask from uv
        # Decode the EXR data using OpenCV
        uv = cv2.imread(str(self.data_root / self.prefix_uv / f"{filename}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv = cv2.cvtColor(uv, cv2.COLOR_BGR2RGB)
        mask = datapoints.Mask(uv[..., 2])

        if self.transforms:
            image, flow, mask = self.transforms(image, flow, mask)

        return {"image": image, "bm": flow, "mask": mask}
