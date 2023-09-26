"""This datamodule is responsible for loading and setting up dataloaders for DocAligner dataset"""
from pathlib import Path

import gin
import lightning as L
import torch
# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision
from torch.utils.data import DataLoader

from docmae.datamodule.utils import get_image_transforms

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms

from docmae.data.docaligner import DocAligner
from docmae.data.augmentation.random_resized_crop import RandomResizedCropWithUV


@gin.configurable
class DocAlignerDataModule(L.LightningDataModule):
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, crop: bool):
        """
        Datamodule to set up DocAligner dataset
        Args:
            data_dir: DocAligner path
            batch_size: batch size
            num_workers: number of workers in dataloader
            crop: whether to crop document images using UV
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = max(batch_size, 4)  # use 4 when searching batch size
        self.num_workers = num_workers
        self.crop = crop

        self.train_transform = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(0.08, 1.0) if self.crop else (1.0, 1.0), antialias=True),
                # ReplaceBackground(Path(config["background_path"]), "train1"),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )
        self.image_transforms = get_image_transforms(gin.REQUIRED)
        self.val_transform = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(0.08, 1.0) if self.crop else (1.0, 1.0), antialias=True),
                # ReplaceBackground(Path(config["background_path"]), "val1"),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )

    def setup(self, stage: str):
        self.train_dataset = DocAligner(Path(self.data_dir), "train", self.train_transform)
        self.val_dataset = DocAligner(Path(self.data_dir), "val", self.val_transform)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if isinstance(batch, dict):  # not example tensor
            with torch.no_grad():
                batch["image"] = self.image_transforms(batch["image"])
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=min(self.batch_size, self.num_workers),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=min(self.batch_size, self.num_workers),
            pin_memory=True,
        )
