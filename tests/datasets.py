import os
from pathlib import Path

import torch

from docmae.data.doc3d import Doc3D

import torchvision.transforms.v2 as transforms

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def test_raw_output():
    dataset = Doc3D(Path("/home/dawars/datasets/doc3d/"), "tiny")

    sample = dataset[0]

    assert sample["image"].shape == (3, 448, 448)
    assert sample["bm"].shape == (2, 448, 448)
    assert sample["mask"].shape == (448, 448)


def test_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize((288, 288), antialias=True),
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )

    dataset = Doc3D(Path("/home/dawars/datasets/doc3d/"), "tiny", transform)

    sample = dataset[0]

    assert sample["image"].shape == (3, 288, 288)
    assert sample["bm"].shape == (2, 288, 288)
    assert sample["mask"].shape == (288, 288)
