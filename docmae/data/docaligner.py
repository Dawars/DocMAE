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

import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)


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

        image = Image.open(self.data_root / filename).convert("RGB").resize((1024, 1024))
        image = datapoints.Image(image)

        # backwards mapping
        bm_raw = np.load(str(self.data_root / filename.with_suffix(".npy")).replace("origin", "grid3")).astype(float)
        bm = (bm_raw + 1) / 2
        bm = datapoints.Image(bm.transpose((2, 0, 1)))  # absolute back mapping [0, 1]

        uv = None

        mask = Image.open(str(self.data_root / filename).replace("origin", "mask_new")).convert("1").resize((1024, 1024))
        mask = datapoints.Mask(mask)

        if self.transforms:
            image, bm, uv, mask = self.transforms(image, bm, uv, mask)

        item = {"image": image, "bm": bm, "mask": mask}
        if uv is not None:
            item["uv"] = uv
        return item
