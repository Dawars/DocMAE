import argparse
import json
import logging
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import flow_to_image
import torchvision.transforms.v2 as transforms
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from docmae import setup_logging
from docmae.data.list_dataset import ListDataset
from docmae.models.doctr import DocTr
from docmae.models.rectification import Rectification


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path, save_bm: bool, save_mask: bool, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.save_bm = save_bm
        self.save_mask = save_mask

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        rectified, bm, mask = prediction
        for i, idx in enumerate(batch_indices):
            rectified_ = rectified[i]
            bm_ = bm[i]
            mask_ = mask[i]

            plt.imsave(self.output_dir / f"{idx}_rect.jpg", rectified_.permute(1, 2, 0).clip(0, 255).cpu().numpy() / 255)
            if self.save_bm:
                plt.imsave(self.output_dir / f"{idx}_bm.jpg", flow_to_image(bm_).permute(1, 2, 0).numpy())
            if self.save_mask and mask_ is not None:
                plt.imsave(self.output_dir / f"{idx}_mask.png", mask_[0].cpu().numpy(), cmap="gray")


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Model config file")
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="config file for training parameters",
    )
    parser.add_argument("-l", "--log-dir", type=Path, default="", help="folder to store log files")
    parser.add_argument("-d", "--data-path", type=Path, default="", help="Dataset directory")
    parser.add_argument("-s", "--split", type=str, default="test", help="Dataset split")
    parser.add_argument("-o", "--output-path", type=Path, default="", help="Directory to save inference results")
    parser.add_argument("-m", "--ckpt-path", type=Path, default="", help="Checkpoint path")
    return parser.parse_args()


def inference(args, config):
    model = DocTr(config["model"])
    model = Rectification.load_from_checkpoint(args.ckpt_path, "cuda", model=model, config=config)

    inference_transform = transforms.Compose(
        [
            transforms.ToImageTensor(),
            transforms.ToDtype(torch.float32),
        ]
    )
    dataset = ListDataset(args.data_path, args.split, inference_transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)  # todo implement collate for arbitrary images

    writer_callback = CustomWriter(args.output_path, save_bm=True, save_mask=True)

    trainer = Trainer(
        callbacks=[writer_callback],
        accelerator="cuda",
        # limit_predict_batches=160,
    )
    trainer.predict(model, dataloader, return_predictions=False)


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.suffix == ".json"
    config = json.loads(args.config.read_text())
    inference(args, config)


if __name__ == "__main__":
    main()
