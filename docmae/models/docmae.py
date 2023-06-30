import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTImageProcessor
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEModel

PATCH_SIZE = 16


# Flow related code taken from https://github.com/fh2019ustc/DocTr/blob/main/GeoTr.py
class FlowHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, PATCH_SIZE ** 2 * 9, 1, padding=0)
        )

    def forward(self, imgf, coords1):
        dflow = self.flow_head(imgf)
        mask = 0.25 * self.mask(imgf)  # scale mask to balance gradients
        coords1 = coords1 + dflow

        return mask, coords1


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow16(flow, mode="bilinear"):
    new_size = (16 * flow.shape[2], 16 * flow.shape[3])
    return 16 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


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

        self.image_processor = image_processor
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_norm = nn.LayerNorm(decoder.config.decoder_hidden_size, eps=decoder.config.layer_norm_eps)

        self.P = PATCH_SIZE
        self.hidden_dim = hparams["hidden_dim"]
        self.upscale_type = hparams["upscale_type"]
        self.freeze_backbone = hparams["freeze_backbone"]

        self.update_block = UpdateBlock(self.hidden_dim)  # todo check paper for hidden_dim
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
        self.tb_log = self.logger.experiment

    def configure_optimizers(self):
        """
        calling setup_optimizers for which all missing parameters must be registered with gin

        Returns:
            dictionary defining optimizer, learning rate scheduler and value to monitor as expected by pytorch lightning
        """

        optimizer = torch.optim.Adam(self.parameters())
        scheduler = OneCycleLR(optimizer, 1e-4, epochs=self.hparams["epochs"], steps_per_epoch=200_000 // self.hparams["batch_size"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }

    def forward(self, x):
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
        bm_up = self.flow_head(fmap, x)
        return bm_up

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

        outputs = self.forward(image)

        # log metrics
        loss = self.loss(flow, outputs)

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

    def validation_step(self, val_batch, batch_idx):
        image = val_batch["image"] * val_batch["mask"].unsqueeze(1) / 255
        flow_target = val_batch["bm"]
        batch_size = len(image)

        # training image sanity check
        if self.global_step == 0:
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
        flow_pred = self.forward(image)

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
        coords0 = coords_grid(N, H // self.P, W // self.P).to(img.device)
        coords1 = coords_grid(N, H // self.P, W // self.P).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.P, self.P, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.P * flow, (3, 3), padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, self.P * H, self.P * W)

    def flow_head(self, fmap, image1):
        # convex upsample based on fmap
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()

        if self.upscale_type == "raft":
            mask, coords1 = self.update_block(fmap, coords1)
            flow_up = self.upsample_flow(coords1 - coords0, mask)

        elif self.upscale_type == "interpolate":
            mask, coords1 = self.update_block(fmap, coords1)
            flow_up = upflow16(coords1 - coords0)
        else:
            raise NotImplementedError

        bm_up = coodslar + flow_up
        return bm_up
