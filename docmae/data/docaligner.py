"""
DocAligner https://github.com/ZZZHANG-jx/DocAligner
@article{zhang2023docaligner,
title={DocAligner: Annotating Real-world Photographic Document Images by Simply Taking Pictures},
author={Zhang, Jiaxin and Chen, Bangdong and Cheng, Hiuyi and Guo, Fengjun and Ding, Kai and Jin, Lianwen},
journal={arXiv preprint arXiv:2306.05749},
year={2023}}
"""
import logging
from pathlib import Path

import cv2
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datapoints
from torchvision.transforms import InterpolationMode, functional

LOGGER = logging.getLogger(__name__)

import scipy.interpolate as spin
import numpy as np


def bm_to_uv(backward_mapping, s):
    """
    Convert a backward mapping to UV coordinates in the range (0, 1).

    Parameters:
    - backward_mapping: A backward mapping with coordinates in the range (0, 1).
    - image_shape: The shape of the original image.

    Returns:
    - uv_mapping: UV coordinates in the range (0, 1).
    """

    # Create a grid of UV coordinates in the range (0, 1)
    uv_x, uv_y = np.mgrid[0:1:1024j, 0:1:1024j] #np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s))
    uv_y = 1 - uv_y

    # Rescale the backward_mapping to match the image_shape
    bm_x = backward_mapping[0] * s
    bm_y = backward_mapping[1] * s
    # bm_x = cv2.blur(bm_x.numpy(), (3, 3))
    # bm_y = cv2.blur(bm_y.numpy(), (3, 3))

    grid = np.meshgrid(np.linspace(1, s, s), np.linspace(1, s, s))

    # Interpolate to get UV coordinates
    uv_mapping_x = spin.griddata((bm_x.flatten(), bm_y.flatten()), uv_x.flatten(), tuple(grid), method="cubic")
    uv_mapping_y = spin.griddata((bm_x.flatten(), bm_y.flatten()), uv_y.flatten(), tuple(grid), method="cubic")
    uv_mapping = np.stack([uv_mapping_x, uv_mapping_y], axis=-1).transpose(2, 0, 1)

    mask = ~np.isnan(uv_mapping[0])[None]
    np.nan_to_num(uv_mapping, copy=False, nan=0)

    # uv_mapping[0] = cv2.blur(uv_mapping[0], (3, 3))
    # uv_mapping[1] = cv2.blur(uv_mapping[1], (3, 3))

    return uv_mapping, mask


# Example usage:
# backward_mapping = Your backward mapping data (shape: [height, width, 2], range: (-1, 1))
# image_shape = Shape of the original image (e.g., [height, width])
# uv_mapping = bm_to_uv(backward_mapping, image_shape)


class DocAligner(Dataset):
    def __init__(self, data_root: Path, split: str, transforms=None):
        """
        Args:
            data_root: Directory where the doc3d dataset is extracted
            split: split name of subset of images
            transforms: optional transforms for data augmentation
        """

        self.data_root = data_root
        self.filenames = (data_root / f"{split}.txt").read_text().strip().split("\n")

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = Path(self.filenames[idx])

        image = Image.open(self.data_root / filename).convert("RGB")
        image = datapoints.Image(image.resize((1024, 1024)))

        # backwards mapping
        bm_raw = np.load(str(self.data_root / filename.with_suffix(".npy")).replace("origin", "grid3"))
        # bm_raw = bm_raw.astype(np.float32)
        bm = (bm_raw + 1) / 2
        bm = datapoints.Image(bm.transpose((2, 0, 1)))  # absolute back mapping [0, 1]

        from matplotlib import pyplot as plt
        plt.imshow(bm[0])
        plt.show()
        # plt.imshow(bm[1])
        # plt.show()
        # bm = functional.resize(bm[None], [448, 448])[0]

        # mask = Image.open(str(self.data_root / filename).replace("origin", "mask_new")).convert("1").resize((1024, 1024))

        uv, mask = bm_to_uv(bm, 1024)
        uv = datapoints.Mask(uv)

        mask = datapoints.Mask(mask)

        import torch

        # plt.imshow(mask[0], cmap="gray")
        # plt.show()
        # plt.imshow(torch.cat([uv, torch.ones_like(uv[0:1])], dim=0).permute(1, 2, 0))
        # plt.show()
        if self.transforms:
            image, bm, uv, mask = self.transforms(image, bm, uv, mask)

        return {"image": image, "bm": bm, "uv": uv, "mask": mask}
