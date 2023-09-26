"""This datamodule is responsible for loading and setting up dataloaders for all available dataset"""
from pathlib import Path

import gin
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision
import torchvision.transforms as T

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms

from docmae.data.augmentation.replace_background import ReplaceBackground
from docmae.data.doc3d import Doc3D
from docmae.datamodule.utils import get_image_transforms
from docmae.data.docaligner import DocAligner
from docmae.data.augmentation.random_resized_crop import RandomResizedCropWithUV


@gin.configurable
class MixedDataModule(L.LightningDataModule):
    train_docaligner: torch.utils.data.Dataset
    val_docaligner: torch.utils.data.Dataset
    train_doc3d: torch.utils.data.Dataset
    val_doc3d: torch.utils.data.Dataset

    def __init__(self, docaligner_dir: str, doc3d_dir: str, background_dir: str, batch_size: int, num_workers: int, crop: bool):
        """
        Datamodule to set up DocAligner dataset
        Args:
            docaligner_dir: DocAligner path
            doc3d_dir: Doc3D path
            background_dir: Path for background images
            batch_size: batch size
            num_workers: number of workers in dataloader
            crop: whether to crop document images using UV
        """
        super().__init__()
        self.docaligner_dir = Path(docaligner_dir)
        self.doc3d_dir = Path(doc3d_dir)
        self.background_dir = Path(background_dir)
        self.batch_size = batch_size
        self.num_workers = min(batch_size, num_workers)
        self.crop = crop

        self.train_transform = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(0.08, 1.0) if self.crop else (1.0, 1.0), antialias=True),
                T.RandomApply([ReplaceBackground(self.background_dir, "train1")], p=0.25),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(0.08, 1.0) if self.crop else (1.0, 1.0), antialias=True),
                T.RandomApply([ReplaceBackground(self.background_dir, "val1")], p=0.25),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )
        self.train_transform_nocrop = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(1.0, 1.0), antialias=True),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )
        self.val_transform_nocrop = transforms.Compose(
            [
                RandomResizedCropWithUV((288, 288), scale=(1.0, 1.0), antialias=True),
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.float32),
            ]
        )
        self.image_transforms = get_image_transforms(gin.REQUIRED)

    def setup(self, stage: str):
        self.train_docaligner = DocAligner(Path(self.docaligner_dir), "train", self.train_transform_nocrop)
        self.val_docaligner = DocAligner(Path(self.docaligner_dir), "val", self.val_transform_nocrop)
        self.train_doc3d = Doc3D(Path(self.doc3d_dir), "train", self.train_transform)
        self.val_doc3d = Doc3D(Path(self.doc3d_dir), "val", self.val_transform)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if isinstance(batch, dict):  # not example tensor
            with torch.no_grad():
                batch["image"] = self.image_transforms(batch["image"])
        return batch

    def train_dataloader(self):
        return DataLoader(
            ConcatDataset([self.train_doc3d, self.train_docaligner]),
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ConcatDataset([self.val_doc3d, self.val_docaligner]),
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
