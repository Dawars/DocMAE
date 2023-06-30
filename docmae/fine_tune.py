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
import torchvision.transforms.v2 as transforms

from transformers import (
    ViTMAEConfig,
    AutoImageProcessor,
)
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEModel

from docmae.models.docmae import DocMAE

from docmae import setup_logging

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
    train_transform = transforms.Compose(
        [
            transforms.Resize((288, 288), antialias=True),
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )

    if config["use_minio"]:
        from docmae.data.doc3d_minio import Doc3D

        train_files = (Path(config["dataset_path"]) / "train.txt").read_text().split()
        val_files = (Path(config["dataset_path"]) / "val.txt").read_text().split()
        train_dataset = Doc3D(train_files, train_transform)
        val_dataset = Doc3D(val_files, train_transform)
    else:
        from docmae.data.doc3d import Doc3D

        train_dataset = Doc3D(Path(config["dataset_path"]), "tiny", train_transform)
        val_dataset = Doc3D(Path(config["dataset_path"]), "tiny", train_transform)
    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    callback_list = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=args.model_output_dir,
            filename="epoch_{epoch:d}",
            monitor="val/loss",
            mode="max",
            save_top_k=1,
        ),
    ]

    logger = TensorBoardLogger(save_dir=args.tensorboard_dir, log_graph=False, default_hp_metric=False)

    trainer = L.Trainer(
        logger=logger,
        callbacks=callback_list,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=1,
    )

    pretrained_config = ViTMAEConfig.from_pretrained(config["mae_path"])
    pretrained_config.mask_ratio = 0
    image_processor = AutoImageProcessor.from_pretrained(config["mae_path"], size={"height": 288, "width": 288})
    mae_encoder = ViTMAEModel(pretrained_config)
    mae_decoder = ViTMAEDecoder(pretrained_config, mae_encoder.embeddings.num_patches)

    model = DocMAE(image_processor, mae_encoder, mae_decoder, config)

    trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.endswith(".json")

    # Save config for training traceability and load config parameters
    config_file = Path(args.model_output_dir) / "fine_tune_config.json"
    config = json.loads(Path(args.config).read_text())
    shutil.copyfile(args.config, config_file)
    train(args, config)


if __name__ == "__main__":
    main()
