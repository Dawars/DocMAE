import os
import logging
import warnings
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
from urllib3.exceptions import InsecureRequestWarning
from PIL import Image
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
torchvision.disable_beta_transforms_warning()
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from scipy.interpolate import griddata


def fm2bm(fm, msk, s):
    """
    fm: the forward mapping in range [0,1]
    s: shape of the image required to sample from, defines the range of image coordinates in
    """
    fm = fm.numpy() * s

    # msk = fm[..., 0] > 0
    s2d = fm[msk]
    tx, ty = np.where(msk)
    # s2d = fm[:, msk]
    # tx, ty = np.where(msk)
    grid = np.meshgrid(np.linspace(1, s, s), np.linspace(1, s, s))
    vx = griddata(s2d, tx, tuple(grid), method="linear")
    vy = griddata(s2d, ty, tuple(grid), method="linear")
    bm = np.stack([vy, vx], axis=-1)
    return bm


def uv2mp(uv, msk, scale):
    # uv is a k*k*3 numpy array
    # s is the size of the output map
    s = scale
    # Rescale the 1.0 in uv to s which is the size of the output map
    uv = uv * s
    sx = uv[0, msk]
    sy = uv[1, msk]
    # Valid point
    # sx sy are the value in the forward mapping but the coord in the backward mapping
    # sx = sx[msk]
    # sy = sy[msk]
    # tx ty are the coord in the forward mapping but the value in the backward mapping
    ty, tx = np.where(msk)
    Fx = griddata((sx, sy), tx, (sx, sy), method="linear")
    Fy = griddata((sx, sy), ty, (sx, sy), method="linear")
    # Sampling coord on the output mapping
    xq, yq = np.meshgrid(np.arange(1, s + 1), np.arange(1, s + 1))
    # Get the value based on the scattered interpolation
    vx = griddata((sx, sy), Fx, (xq, yq), method="linear")
    vy = griddata((sx, sy), Fy, (xq, yq), method="linear")
    # Concatenate together
    invmap = np.stack((vx, vy), axis=2)
    return invmap


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

        self.grid_input = self.coords_grid(448, 448).transpose(2, 1, 0)
        self.grid_output = self.coords_grid(288, 288)

    @staticmethod
    def coords_grid(ht, wd):
        coords = np.meshgrid(np.arange(ht), np.arange(wd))
        coords = np.stack(coords[::-1], axis=0)
        return coords

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(self.data_root / self.prefix_img / f"{filename}.png").convert("RGB")
        image = datapoints.Image(image)

        # backwards mapping
        h5file = h5py.File(self.data_root / self.prefix_bm / f"{filename}.mat", "r")
        bm = np.array(h5file.get("bm"))
        bm = bm.transpose(2, 1, 0)

        bm = datapoints.Image(bm.transpose(2, 0, 1))  # absolute back mapping

        # mask from uv
        # Decode the EXR data using OpenCV
        uv = cv2.imread(str(self.data_root / self.prefix_uv / f"{filename}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv = cv2.cvtColor(uv, cv2.COLOR_BGR2RGB)  # forward mapping
        uv_mask = datapoints.Mask(uv.transpose(2, 0, 1))

        if self.transforms:
            image, bm, uv_mask = self.transforms(image, bm, uv_mask)

        return {"image": image, "bm": bm, "mask": uv_mask}
