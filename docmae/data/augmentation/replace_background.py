import logging
import random
from pathlib import Path

import cv2
import kornia
import torch
from PIL import Image
from torchvision import datapoints
from torchvision.transforms.v2 import functional as F

LOGGER = logging.getLogger(__name__)


def match_brightness(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match brightness of source to target
    Args:
        source:
        target:
    """
    source_mean = torch.mean(source.float())
    target_mean = torch.mean(target.float())

    ratio = source_mean / target_mean
    return source / ratio


class ReplaceBackground(object):
    def __init__(self, data_root: Path, split: str, match_brightness=True):
        """
        Replace the background of the image (where mask is 0) by a random image
        Args:
            data_root: Directory where the dtd dataset is extracted
            split: split name of subset of images (train1-10, val1-10, test1-10)
            match_brightness: whether to match the brightness of the background to the image
        """
        self.data_root = data_root
        self.filenames = (data_root / "labels" / f"{split}.txt").read_text().strip().split("\n")
        self.match_brightness = match_brightness

    def __call__(self, sample):
        image, bm, uv, mask = sample
        shape = image.shape

        filename = random.choice(self.filenames)
        background = Image.open(self.data_root / "images" / filename).convert("RGB").resize(shape[1:])
        background = F.to_image_tensor(background)
        if self.match_brightness:
            background = match_brightness(background, image)

        smooth_mask = kornia.filters.box_blur(mask[None].float(), 3)
        image = ((1 - smooth_mask) * background + smooth_mask * image)[0]

        return datapoints.Image(image), bm, uv, mask
