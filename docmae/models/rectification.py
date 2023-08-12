import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from docmae.models.upscale import UpscaleRAFT, UpscaleTransposeConv, UpscaleInterpolate, coords_grid


class Rectification(L.LightningModule):
    tb_log: SummaryWriter

    def __init__(
            self,
            model: nn.Module,
            hparams,
    ):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 288, 288)

        self.model = model

        H, W = self.example_input_array.shape[2:]
        self.coodslar = coords_grid(1, H, W).to(self.example_input_array.device)

        self.upscale_type = hparams["model"]["upscale_type"]
        self.segment_background = hparams["model"]["segment_background"]
        self.hidden_dim = hdim = 256  # todo add model config
        if self.upscale_type == "raft":
            self.upscale_module = UpscaleRAFT(8, self.hidden_dim)  # todo add config hparams
        elif self.upscale_type == "transpose_conv":
            self.upscale_module = UpscaleTransposeConv(self.hidden_dim)
        elif self.upscale_type == "interpolate":
            self.upscale_module = UpscaleInterpolate()
        else:
            raise NotImplementedError

        self.loss = L1Loss()
        self.save_hyperparameters(hparams["model"])

    def on_fit_start(self):
        self.tb_log = self.logger.experiment
        self.coodslar = self.coodslar.to(self.device)
        additional_metrics = ["val/loss"]
        self.logger.log_hyperparams(self.hparams, {**{key: 0 for key in additional_metrics}})

    def configure_optimizers(self):
        """
        calling setup_optimizers for which all missing parameters must be registered with gin

        Returns:
            dictionary defining optimizer, learning rate scheduler and value to monitor as expected by pytorch lightning
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)
        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            "monitor": "train/loss",
        }

    def forward(self, inputs):
        """
        Runs inference: image_processing, encoder, decoder, layer norm, flow head
        Args:
            inputs: image tensor of shape [B, C, H, W]
        Returns: flow displacement
        """
        outputs = self.model(inputs)
        flow_up = self.upscale_module(**outputs)
        return flow_up

    def training_step(self, batch):
        image = batch["image"] / 255
        if self.segment_background:
            image = image * batch["mask"][:, 2:3]

        bm_target = batch["bm"] * 288
        batch_size = len(image)

        # training image sanity check
        if self.global_step == 0:
            ones = torch.ones((batch_size, 1, 288, 288), device=self.device)
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
            self.tb_log.add_images("val/flow", torch.cat((bm_target / 288, ones), dim=1), global_step=self.global_step)

        flow_pred = self.forward(image)
        bm_pred = flow_pred + self.coodslar

        # log metrics
        loss = self.loss(bm_target, bm_pred)

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def on_after_backward(self):
        global_step = self.global_step
        # if self.global_step % 100 == 0:
        #     for name, param in self.model.named_parameters():
        #         self.tb_log.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.tb_log.add_histogram(f"{name}_grad", param.grad, global_step)

    def on_validation_start(self):
        self.tb_log = self.logger.experiment
        self.coodslar = self.coodslar.to(self.device)

    def validation_step(self, val_batch, batch_idx):
        image = val_batch["image"] / 255
        if self.segment_background:
            image = image * val_batch["mask"][:, 2:3]
        bm_target = val_batch["bm"] * 288
        batch_size = len(image)

        # training image sanity check
        if self.global_step == 0:
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
        flow_pred = self.forward(image)
        bm_pred = self.coodslar + flow_pred

        # log metrics
        loss = self.loss(bm_target, bm_pred)

        self.log("val/loss", loss, on_epoch=True, batch_size=batch_size)

        ones = torch.ones((batch_size, 1, 288, 288), device=self.device)

        if batch_idx == 0 and self.global_step == 0:
            self.tb_log.add_images("val/image", image, global_step=self.global_step)
            self.tb_log.add_images("val/flow", torch.cat((bm_target / 288, ones), dim=1), global_step=self.global_step)

        if batch_idx == 0:
            self.tb_log.add_images(
                "val/flow_pred", torch.cat((bm_pred / 288, ones), dim=1), global_step=self.global_step
            )

            bm_ = (bm_pred / 288 - 0.5) * 2
            bm_ = bm_.permute((0, 2, 3, 1))
            img_ = image
            uw = F.grid_sample(img_, bm_)

            self.tb_log.add_images("val/unwarped", uw, global_step=self.global_step)

    def on_test_start(self):
        self.tb_log = self.logger.experiment
