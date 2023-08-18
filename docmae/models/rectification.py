import numpy as np
import cv2
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import flow_to_image

from docmae.models.upscale import UpscaleRAFT, UpscaleTransposeConv, UpscaleInterpolate, coords_grid


class Rectification(L.LightningModule):
    tb_log: SummaryWriter

    def __init__(
        self,
        model: nn.Module,
        config,
    ):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 288, 288)

        self.model = model
        self.config = config
        hparams = config["model"]

        H, W = self.example_input_array.shape[2:]
        self.coodslar = coords_grid(1, H, W).to(self.example_input_array.device)

        self.upscale_type = hparams["upscale_type"]
        self.segment_background = hparams["segment_background"]
        self.hidden_dim = hparams["hidden_dim"]
        if self.upscale_type == "raft":
            self.upscale_module = UpscaleRAFT(8, self.hidden_dim)
        elif self.upscale_type == "transpose_conv":
            self.upscale_module = UpscaleTransposeConv(self.hidden_dim, self.hidden_dim // 2)
        elif self.upscale_type == "interpolate":
            self.upscale_module = UpscaleInterpolate(self.hidden_dim, self.hidden_dim // 2)
        else:
            raise NotImplementedError

        self.loss = L1Loss()
        self.save_hyperparameters(hparams)

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

        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = {
            "scheduler": OneCycleLR(optimizer, max_lr=1e-4, pct_start=0.0014, total_steps=self.config["training"]["steps"]),
            "interval": "step",
        }
        return [optimizer], [scheduler]

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

        flow_pred = self.forward(image)
        bm_pred = flow_pred + self.coodslar

        # training image sanity check
        if self.global_step == 0:
            ones = torch.ones((batch_size, 1, 288, 288))
            self.tb_log.add_images("train/image", image.detach().cpu(), global_step=self.global_step)
            self.tb_log.add_images(
                "train/bm_target", torch.cat((bm_target.detach().cpu() / 288, ones), dim=1), global_step=self.global_step
            )
            self.tb_log.add_images(
                "train/flow_target",
                flow_to_image(bm_target.detach().cpu() - self.coodslar.detach().cpu()),
                global_step=self.global_step,
            )

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

        ones = torch.ones((batch_size, 1, 288, 288))

        if batch_idx == 0 and self.global_step == 0:
            self.tb_log.add_images("val/image", image.detach().cpu(), global_step=self.global_step)
            self.tb_log.add_images(
                "val/bm", torch.cat((bm_target.detach().cpu() / 288, ones), dim=1), global_step=self.global_step
            )
            self.tb_log.add_images(
                "val/flow",
                flow_to_image((bm_target.detach().cpu() - self.coodslar.detach().cpu())),
                global_step=self.global_step,
            )

        if batch_idx == 0:
            self.tb_log.add_images(
                "val/bm_pred", torch.cat((bm_pred.detach().cpu() / 288, ones), dim=1), global_step=self.global_step
            )
            self.tb_log.add_images("val/flow_pred", flow_to_image(flow_pred.detach().cpu()), global_step=self.global_step)

            self.tb_log.add_images("val/bm_diff", flow_to_image(bm_target.cpu() - bm_pred.cpu()), global_step=self.global_step)

            bm_ = (bm_pred / 288 - 0.5) * 2
            bm_ = bm_.permute((0, 2, 3, 1))
            img_ = image
            uw = F.grid_sample(img_, bm_, align_corners=False).detach().cpu()

            self.tb_log.add_images("val/unwarped", uw, global_step=self.global_step)

    def on_predict_start(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.segmenter = torch.jit.load(self.config["segmenter_ckpt"], map_location=self.device)
        self.segmenter = torch.jit.freeze(self.segmenter)
        self.segmenter = torch.jit.optimize_for_inference(self.segmenter)
        self.resize = transforms.Resize((288, 288), antialias=True)
        self.coodslar = self.coodslar.to(self.device)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        image_orig = batch["image"]
        b, c, h, w = image_orig.shape
        image = self.resize(image_orig)
        # resize to 288
        image /= 255
        if self.segment_background:
            mask = (self.segmenter(self.normalize(image)) > 0.5).to(torch.bool)
            image = image * mask
        else:
            mask = None

        flow_pred = self.forward(image)
        bm_pred = self.coodslar + flow_pred  # rescale to original

        bm = (bm_pred / 288 - 0.5) * 2

        # pytorch reimplementation
        # import kornia
        # bm = torch.nn.functional.interpolate(bm, image_orig.shape[2:])
        # bm = kornia.filters.box_blur(bm, 3)
        # rectified = F.grid_sample(image_orig, bm.permute((0, 2, 3, 1)), align_corners=False, mode="bilinear")

        # doctr implementation
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).to(self.device).unsqueeze(0)  # h * w * 2

        rectified = F.grid_sample(image_orig, lbl, align_corners=True)
        return rectified, bm, mask

    def on_test_start(self):
        self.tb_log = self.logger.experiment
        self.coodslar = self.coodslar.to(self.device)
