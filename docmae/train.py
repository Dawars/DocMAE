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
# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms

from docmae.data.doc3d import Doc3D

from docmae import setup_logging
from docmae.models.doctr import DocTr
from docmae.models.doctr_custom import DocTrOrig
from docmae.models.rectification import Rectification
from docmae.utils.transforms import RandomResizedCropWithUV

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
            # transforms.RandomRotation((-10, 10)),
            RandomResizedCropWithUV((288, 288), scale=(0.08, 1.0) if config["training"]["crop"] else (1.0, 1.0), antialias=True),
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )
    train_dataset = Doc3D(Path(config["dataset_path"]), "train", train_transform)
    val_dataset = Doc3D(Path(config["dataset_path"]), "val", train_transform)
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
            save_top_k=2,
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
    )

    model = DocTrOrig(config["model"])
    model = Rectification(model, config)

    # test export
    print(model.cuda().to_torchscript(method="trace"))

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
