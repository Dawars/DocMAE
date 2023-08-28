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
        bm_raw = np.load(str(self.data_root / filename.with_suffix(".npy")).replace("origin", "grid3"))
        bm = (bm_raw + 1) / 2
        bm = datapoints.Image(bm.transpose((2, 0, 1)))  # absolute back mapping [0, 1]

        shape = (1, 3, 1024, 1024)
        horizontal = (
            torch.linspace(0, 1.0, shape[3], dtype=torch.float).view(1, 1, 1, shape[3]).expand(1, 1, shape[2], shape[3])
        )
        vertical = torch.linspace(0, 1.0, shape[2], dtype=torch.float).view(1, 1, shape[2], 1).expand(1, 1, shape[2], shape[3])
        grid = torch.cat([horizontal, vertical], 1)
        uv = F.grid_sample(grid, torch.from_numpy(bm_raw).float()[None])[0]
        uv = datapoints.Mask(uv)

        mask = Image.open(str(self.data_root / filename).replace("origin", "mask_new")).convert("1").resize((1024, 1024))
        mask = datapoints.Mask(mask)

        if self.transforms:
            image, bm, uv, mask = self.transforms(image, bm, uv, mask)

        return {"image": image, "bm": bm, "uv": uv, "mask": mask}
