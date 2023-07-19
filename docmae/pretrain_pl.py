import json
import logging
import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms as T
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import ViTMAEConfig, ViTMAEModel, ViTImageProcessor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder

from docmae import setup_logging
from docmae.data.list_dataset import ListDataset
from docmae.models.mae import MAE
from docmae.pretrain import parse_arguments

""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377."""

logger = logging.getLogger(__name__)


def train(args, config_file: str):
    config = json.loads(Path(config_file).read_text())
    L.seed_everything(config["seed"])

    image_processor = ViTImageProcessor(do_rescale=False, do_resize=False, do_normalize=False, size={"height": 288, "width": 288})

    pretrained_config = ViTMAEConfig.from_pretrained(config_file)
    pretrained_config.image_size = 288
    encoder = ViTMAEModel(pretrained_config)
    decoder = ViTMAEDecoder(pretrained_config, encoder.embeddings.num_patches)
    encoder.mask_ratio = 0.75
    decoder.mask_ratio = 0.75

    callback_list = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=args.model_output_dir,
            filename="epoch_{epoch:d}",
            monitor="val/loss",
            mode="min",
            save_top_k=config["save_total_limit"],
        ),
    ]

    tb_logger = TensorBoardLogger(save_dir=args.tensorboard_dir, log_graph=False, default_hp_metric=False)
    num_epochs = config["num_train_epochs"]
    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=callback_list,
        accelerator="cuda",
        max_epochs=num_epochs,
        enable_progress_bar=config["disable_tqdm"],

        limit_train_batches=200_000 // num_epochs,
        limit_val_batches=10_000 // num_epochs,
        val_check_interval=10_000 // num_epochs,
    )

    model = MAE(image_processor, encoder, decoder, config, training=True)
    transforms = T.Compose([T.Resize(size=(288, 288)), T.ToTensor()])

    dataset_train = ListDataset(Path(config["train_dir"]), "train", transforms)
    dataset_val = ListDataset(Path(config["validation_dir"]), "val", transforms)

    loader_train = DataLoader(
        dataset_train,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=config["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )

    trainer.fit(model, loader_train, loader_val)


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.endswith(".json")

    # Save config for training traceability and load config parameters
    config_file = Path(args.model_output_dir) / "config.json"
    config = json.loads(Path(args.config).read_text())

    config["logging_dir"] = args.tensorboard_dir
    config["output_dir"] = os.path.join(args.model_output_dir, "checkpoints")

    config_file.write_text(json.dumps(config))
    train(args, str(config_file))


if __name__ == "__main__":
    main()
