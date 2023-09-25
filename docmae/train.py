import json
import logging

import argparse
import shutil
from pathlib import Path

import gin
import gin.torch.external_configurables
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision

torchvision.disable_beta_transforms_warning()

from docmae import setup_logging
from docmae.models.transformer import BasicEncoder
from docmae.models.upscale import UpscaleRAFT, UpscaleTransposeConv, UpscaleInterpolate
from docmae.models.doctr import DocTr
from docmae.models.doctr_custom import DocTrOrig
from docmae.models.doctr_plus import DocTrPlus
from docmae.models.rectification import Rectification
from docmae.datamodule.utils import init_external_gin_configurables

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=Path, help="json config file for training parameters")
    parser.add_argument("-d", "--data-config", required=True, type=Path, help="gin config file for the dataloader")
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="config file for training parameters",
    )
    parser.add_argument("-l", "--log-dir", type=Path, default="", help="folder to store log files")
    parser.add_argument("-t", "--tensorboard-dir", type=Path, default="", help="folder to store tensorboard logs")
    parser.add_argument("-m", "--model-output-dir", type=Path, default="model", help="folder to store trained models")
    return parser.parse_args()


@gin.configurable
def train(args, config: dict, datamodule: L.LightningDataModule):
    L.seed_everything(config["training"]["seed"])

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

    trainer.fit(model, datamodule=datamodule)

    print(callback_list[1].best_model_path)


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.suffix == ".json"

    # Save config for training traceability and load config parameters
    config_file = args.model_output_dir / "config.json"
    config = json.loads(args.config.read_text())
    shutil.copyfile(args.config, config_file)

    init_external_gin_configurables()
    gin.parse_config_file(args.data_config)
    Path(args.model_output_dir / "data_config.gin").write_text(gin.operative_config_str())

    train(args, config, datamodule=gin.REQUIRED)


if __name__ == "__main__":
    main()
