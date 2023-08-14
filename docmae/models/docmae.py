import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEModel

from docmae.models.upscale import UpscaleRAFT, UpscaleTransposeConv, UpscaleInterpolate, coords_grid

PATCH_SIZE = 16


class DocMAE(L.LightningModule):
    tb_log: SummaryWriter

    def __init__(
        self,
        encoder: ViTMAEModel,
        decoder: ViTMAEDecoder,
        hparams,
    ):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 288, 288)
        self.coodslar = self.initialize_flow(self.example_input_array)

        self.encoder = encoder
        self.decoder = decoder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.P = PATCH_SIZE
        self.hidden_dim = hparams["hidden_dim"]
        self.upscale_type = hparams["upscale_type"]
        self.freeze_backbone = hparams["freeze_backbone"]

        if self.upscale_type == "raft":
            self.upscale_module = UpscaleRAFT(self.hidden_dim, hidden_dim=256)
        elif self.upscale_type == "transpose_conv":
            self.upscale_module = UpscaleTransposeConv(self.hidden_dim)
        elif self.upscale_type == "interpolate":
            self.upscale_module = UpscaleInterpolate()
        else:
            raise NotImplementedError
        self.loss = L1Loss()
        self.save_hyperparameters(hparams)
        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def on_fit_start(self):
        self.coodslar = self.coodslar.to(self.device)

        self.tb_log = self.logger.experiment
        additional_metrics = ["val/loss"]
        self.logger.log_hyperparams(self.hparams, {**{key: 0 for key in additional_metrics}})

    def configure_optimizers(self):
        """
        calling setup_optimizers for which all missing parameters must be registered with gin

        Returns:
            dictionary defining optimizer, learning rate scheduler and value to monitor as expected by pytorch lightning
        """

        optimizer = torch.optim.Adam(self.parameters())
        scheduler = OneCycleLR(
            optimizer, 1e-4, epochs=self.hparams["epochs"], steps_per_epoch=200_000 // self.hparams["batch_size"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }

    def forward(self, inputs):
        """
        Runs inference: image_processing, encoder, decoder, layer norm, flow head
        Args:
            inputs: image tensor of shape [B, C, H, W]
        Returns: flow displacement
        """
        inputs = self.normalize(inputs)
        bottleneck = self.encoder.forward(inputs)
        fmap = self.decoder(
            bottleneck.last_hidden_state,
            bottleneck.ids_restore,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = fmap.hidden_states[-1][:, 1:, :]  # remove CLS token
        fmap = last_hidden_state  # layer norm
        # B x 18*18 x 512
        # -> B x 512 x 18 x 18 (B x 256 x 36 x 36)
        fmap = fmap.permute(0, 2, 1)
        fmap = fmap.reshape(-1, self.hidden_dim, 18, 18)
        upflow = self.flow(fmap)
        return upflow

    def training_step(self, batch):
        image = batch["image"] * batch["mask"].unsqueeze(1) / 255
        flow = batch["bm"]
        batch_size = len(image)

        # training image sanity check
        if self.global_step == 0:
            zeros = torch.zeros((batch_size, 1, 288, 288), device=self.device)
            def viz_flow(img): return (img / 448 - 0.5) * 2
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
            self.tb_log.add_images("val/flow", torch.cat((viz_flow(flow), zeros), dim=1), global_step=self.global_step)

        dflow = self.forward(image)
        flow_pred = self.coodslar + dflow

        # log metrics
        loss = self.loss(flow, flow_pred)

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
        self.coodslar = self.coodslar.to(self.device)
        self.tb_log = self.logger.experiment

    def validation_step(self, val_batch, batch_idx):
        image = val_batch["image"] * val_batch["mask"].unsqueeze(1) / 255
        flow_target = val_batch["bm"]
        batch_size = len(image)

        # training image sanity check
        if self.global_step == 0:
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
        dflow = self.forward(image)
        flow_pred = self.coodslar + dflow

        # log metrics
        loss = self.loss(flow_target, flow_pred)

        self.log("val/loss", loss, on_epoch=True, batch_size=batch_size)

        zeros = torch.zeros((batch_size, 1, 288, 288), device=self.device)
        def viz_flow(img): return (img / 448 - 0.5) * 2
        if batch_idx == 0 and self.global_step == 0:
            self.tb_log.add_images("val/image", image, global_step=self.global_step)
            self.tb_log.add_images("val/flow", torch.cat((viz_flow(flow_target), zeros), dim=1), global_step=self.global_step)

        if batch_idx == 0:
            self.tb_log.add_images("val/flow_pred", torch.cat((viz_flow(flow_pred), zeros), dim=1), global_step=self.global_step)

            bm_ = viz_flow(flow_pred)
            bm_ = bm_.permute((0, 2, 3, 1))
            img_ = image
            uw = F.grid_sample(img_, bm_, align_corners=False)

            self.tb_log.add_images("val/unwarped", uw, global_step=self.global_step)

    def on_test_start(self):
        self.tb_log = self.logger.experiment

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        # coords0 = coords_grid(N, H // self.P, W // self.P).to(img.device)
        # coords1 = coords_grid(N, H // self.P, W // self.P).to(img.device)

        return coodslar  # , coords0, coords1

    def flow(self, fmap):
        # convex upsample based on fmap
        flow_up = self.upscale_module(fmap)
        return flow_up
