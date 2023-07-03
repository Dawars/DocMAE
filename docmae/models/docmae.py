import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTImageProcessor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEModel

from docmae.models.modules import expansion_block

PATCH_SIZE = 16


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# Flow related code taken from https://github.com/fh2019ustc/DocTr/blob/main/GeoTr.py
class UpscaleRAFT(nn.Module):
    """
    Infers conv mask to upscale flow
    """

    def __init__(self, input_dim=512, hidden_dim=256):
        super(UpscaleRAFT, self).__init__()
        self.P = PATCH_SIZE

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.mask = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, PATCH_SIZE**2 * 9, 1, padding=0)
        )

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.P, self.P, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.P * flow, (3, 3), padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, self.P * H, self.P * W)

    def forward(self, imgf):
        mask = 0.25 * self.mask(imgf)  # scale mask to balance gradients
        flow = self.conv2(self.relu(self.conv1(imgf)))
        upflow = self.upsample_flow(flow, mask)
        return upflow


class UpscaleTransposeConv(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, mode="bilinear"):
        super().__init__()
        self.layers = [
            expansion_block(input_dim, hidden_dim, hidden_dim // 2),
            expansion_block(hidden_dim // 2, hidden_dim // 4, hidden_dim // 8),
            expansion_block(hidden_dim // 8, hidden_dim // 16, 2, relu=False),

            nn.Upsample(scale_factor=2, mode=mode),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, imgf):
        return self.layers(imgf)


class UpscaleInterpolate(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgf):
        flow = self.conv2(self.relu(self.conv1(imgf)))

        new_size = (16 * flow.shape[2], 16 * flow.shape[3])
        return 16 * F.interpolate(flow, size=new_size, mode=self.mode, align_corners=True)


class DocMAE(L.LightningModule):
    tb_log: SummaryWriter

    def __init__(
            self,
            image_processor: ViTImageProcessor,
            encoder: ViTMAEModel,
            decoder: ViTMAEDecoder,
            hparams,
    ):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 288, 288)
        self.coodslar = self.initialize_flow(self.example_input_array)

        self.image_processor = image_processor
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_norm = nn.LayerNorm(decoder.config.decoder_hidden_size, eps=decoder.config.layer_norm_eps)

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
            for p in self.decoder_norm.parameters():
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

    def forward(self, x):
        """
        Runs inference: image_processing, encoder, decoder, layer norm, flow head
        Args:
            x: image
        Returns: flow displacement
        """
        inputs = self.image_processor(images=x, return_tensors="pt")

        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        bottleneck = self.encoder.forward(**inputs)
        fmap = self.decoder(
            bottleneck.last_hidden_state,
            bottleneck.ids_restore,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = fmap.hidden_states[-1][:, 1:, :]  # remove CLS token
        fmap = self.decoder_norm(last_hidden_state)  # layer norm
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
        self.tb_log.add_images("val/flow_pred", torch.cat((viz_flow(flow_pred), zeros), dim=1), global_step=self.global_step)

        bm_ = viz_flow(flow_pred)
        bm_ = bm_.permute((0, 2, 3, 1))
        img_ = image
        uw = F.grid_sample(img_, bm_)

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
