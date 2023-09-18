import json
import logging

import argparse
import shutil
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import kornia.augmentation as ka

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms
import torchvision.transforms as T

from docmae.data.doc3d import Doc3D
from docmae.data.docaligner import DocAligner

from docmae import setup_logging
from docmae.models.transformer import BasicEncoder
from docmae.models.upscale import UpscaleRAFT, UpscaleTransposeConv, UpscaleInterpolate
from docmae.models.doctr import DocTr
from docmae.models.doctr_custom import DocTrOrig
from docmae.models.doctr_plus import DocTrPlus
from docmae.models.rectification import Rectification
from docmae.data.augmentation.random_resized_crop import RandomResizedCropWithUV

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file for training parameters")
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="config file for training parameters",
    )
    parser.add_argument("-l", "--log-dir", type=str, default="", help="folder to store log files")
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="", help="folder to store tensorboard logs")
    parser.add_argument("-m", "--model-output-dir", type=str, default="model", help="folder to store trained models")
    return parser.parse_args()


def train(args, config: dict):
    L.seed_everything(config["training"]["seed"])

    train_transform = transforms.Compose(
        [
            RandomResizedCropWithUV(
                (288, 288), scale=(0.08, 1.0) if config["training"]["crop"] else (1.0, 1.0), antialias=True
            ),
            # ReplaceBackground(Path(config["background_path"]), "train1"),
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )
    image_transforms = T.Compose([
        T.ConvertImageDtype(torch.float),
        T.RandomChoice(
            transforms=[
                # change color
                ka.RandomPlanckianJitter(keepdim=True),

                ka.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-0.25, 0), shade_quantity=(0, 0.5), p=1.0, keepdim=True),
                ka.RandomPlasmaBrightness(roughness=(0.1, 0.7), intensity=(0.1, 0.5), p=1.0, keepdim=True),

                ka.RandomInvert(p=1., keepdim=True),
                ka.RandomPosterize(bits=4, p=1., keepdim=True),
                # ka.RandomSharpness(p=1., keepdim=True),
                ka.RandomAutoContrast(p=1., keepdim=True),
                ka.RandomEqualize(p=1., keepdim=True),

                ka.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1), p=1., keepdim=True),
                ka.RandomMotionBlur(3, 35., 0.5, p=1., keepdim=True),
            ],
            p=[0.5, 0.25, 0.2,   0.05, 0.1, 0.05, 0.05, 0.1,  0.1, ],
        ),
        T.ConvertImageDtype(torch.uint8),
    ])
    val_transform = transforms.Compose(
        [
            RandomResizedCropWithUV(
                (288, 288), scale=(0.08, 1.0) if config["training"]["crop"] else (1.0, 1.0), antialias=True
            ),
            # ReplaceBackground(Path(config["background_path"]), "val1"),
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )
    train_dataset = DocAligner(Path(config["dataset_path"]), "train", train_transform, image_transforms)
    val_dataset = DocAligner(Path(config["dataset_path"]), "val", val_transform)
    train_loader = DataLoader(
        train_dataset,
        config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    callback_list = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=args.model_output_dir,
            filename="epoch_{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    tb_logger = TensorBoardLogger(save_dir=args.tensorboard_dir, log_graph=False, default_hp_metric=False)

    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=callback_list,
        accelerator="cuda",
        devices=max(torch.cuda.device_count(), config["training"]["num_devices"]),  # use all gpus if config is -1
        max_epochs=config["training"].get("epochs", None),
        max_steps=config["training"].get("steps", -1),
        num_sanity_val_steps=1,
        enable_progress_bar=config["progress_bar"],
        limit_train_batches=20
    )

    hidden_dim = config["model"]["hidden_dim"]
    backbone = BasicEncoder(output_dim=hidden_dim, norm_fn="instance")
    model = DocTrPlus(config["model"])
    upscale_type = config["model"]["upscale_type"]
    if upscale_type == "raft":
        upscale_module = UpscaleRAFT(8, hidden_dim)
    elif upscale_type == "transpose_conv":
        upscale_module = UpscaleTransposeConv(hidden_dim, hidden_dim // 2)
    elif upscale_type == "interpolate":
        upscale_module = UpscaleInterpolate(hidden_dim, hidden_dim // 2)
    else:
        raise NotImplementedError
    model = Rectification(backbone, model, upscale_module, config).cuda()

    # test export
    print(model.to_torchscript(method="trace"))

    trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.endswith(".json")

    # Save config for training traceability and load config parameters
    config_file = Path(args.model_output_dir) / "config.json"
    config = json.loads(Path(args.config).read_text())
    shutil.copyfile(args.config, config_file)
    train(args, config)


if __name__ == "__main__":
    main()
