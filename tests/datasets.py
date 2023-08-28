import os
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as transforms

from docmae.data.doc3d import Doc3D
from docmae.utils.transforms import RandomResizedCropWithUV

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def test_raw_output():
    dataset = Doc3D(Path("./tests/doc3d/"), "tiny")

    sample = dataset[0]

    # size
    assert sample["image"].shape == (3, 448, 448)
    assert sample["bm"].shape == (2, 448, 448)
    assert sample["uv"].shape == (2, 448, 448)
    assert sample["mask"].shape == (1, 448, 448)

    # range
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 255
    assert 0 <= sample["bm"].min() <= sample["bm"].max() < 1  # [0, 1]
    assert 0 <= sample["uv"].min() <= sample["uv"].max() <= 1
    assert set(sample["mask"].numpy().flatten()) == {1.0, 0.0}


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
    assert sample["uv"].shape == (2, 288, 288)
    assert sample["mask"].shape == (1, 288, 288)

    # range
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 255
    assert 0 <= sample["bm"].min() <= sample["bm"].max() < 1  # [0, 1]
    assert 0 <= sample["uv"].min() <= sample["uv"].max() <= 1
    assert set(sample["mask"].numpy().flatten()) == {1.0, 0.0}


def test_crop():
    transform = RandomResizedCropWithUV((288, 288), interpolation=InterpolationMode.BICUBIC, antialias=True)

    image = torch.randint(0, 255, (3, 448, 448))
    bm = torch.rand((2, 448, 448))
    uv = torch.rand((2, 448, 448))
    mask = torch.randint(0, 2, (1, 448, 448))  # 0 or 1

    image, bm, uv, mask = transform((image, bm, uv, mask))

    # size
    assert image.shape == (3, 288, 288)
    assert bm.shape == (2, 288, 288)
    assert uv.shape == (2, 288, 288)
    assert mask.shape == (1, 288, 288)

    # range
    assert 0 <= image.min() <= image.max() <= 255
    assert -0.9 <= bm.min() <= bm.max() <= 1.9  # [0, 1] but crop reaches outside
    assert 0 <= uv.min() <= uv.max() <= 1
    assert set(mask.numpy().flatten()) == {1.0, 0.0}
