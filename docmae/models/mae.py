"""
Wrapper class for Masked Auto-encoder taken from the huggingface library
"""
from typing import Optional

import torch
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class MAE(L.LightningModule):
    tb_log: SummaryWriter

    def __init__(self, encoder, decoder, hparams, training: bool):
        super().__init__()
        self.example_input_array = torch.rand(1, 3, 288, 288)

        self.segmenter = torch.jit.load(hparams["segmenter_ckpt"])
        self.segmenter = torch.jit.freeze(self.segmenter)
        self.segmenter = torch.jit.optimize_for_inference(self.segmenter)
        self.encoder = encoder
        self.decoder = decoder
        self.is_training = training

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.save_hyperparameters(hparams)

    def on_fit_start(self):
        self.tb_log = self.logger.experiment
        additional_metrics = ["val/loss"]
        self.logger.log_hyperparams(self.hparams, {**{key: 0 for key in additional_metrics}})

    def configure_optimizers(self):
        """
        calling setup_optimizers for which all missing parameters must be registered with gin

        Returns:
            dictionary defining optimizer, learning rate scheduler and value to monitor as expected by pytorch lightning
        """
        parameters = self.segmenter.named_parameters()
        for name, param in parameters:
            param.requires_grad = False

        num_epochs = self.hparams["num_train_epochs"]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["base_learning_rate"], weight_decay=0)
        # scheduler = ReduceLROnPlateau(optimizer)
        scheduler = OneCycleLR(
            optimizer, self.hparams["base_learning_rate"], epochs=num_epochs, steps_per_epoch=200_000 // num_epochs
        )
        # scheduler = CosineAnnealingLR(optimizer, self.hparams["base_learning_rate"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        outputs = self.encoder(
            pixel_values.float(),
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        latent, ids_restore, mask = outputs

        if not self.is_training:
            return outputs

        decoder_outputs = self.decoder(latent, ids_restore.long())
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        output = (logits, mask, ids_restore)
        return output

    """Taken from transformers TODO"""

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.encoder.config.patch_size, self.encoder.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.encoder.config.patch_size, self.encoder.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        # if self.encoder.config.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def training_step(self, batch):
        self.encoder.train()
        self.decoder.train()

        image = batch["image"]
        batch_size = len(image)

        with torch.no_grad():
            seg_image = self.normalize(image)
            seg_mask = self.segmenter(seg_image)
            seg_mask = (seg_mask > 0.5).double()

        seg_image = image * seg_mask

        # training image sanity check
        if self.global_step == 0:
            self.tb_log.add_images("train/image", image, global_step=self.global_step)
            self.tb_log.add_images("train/seg_mask", seg_mask, global_step=self.global_step)

        output = self.forward(seg_image)
        (logits, ids_restore, mask) = output
        loss = self.forward_loss(seg_image, logits, mask)

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

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        self.encoder.eval()
        self.decoder.eval()

        image = val_batch["image"]
        batch_size = len(image)

        with torch.no_grad():
            seg_image = self.normalize(image)
            seg_mask = self.segmenter(seg_image)
            seg_mask = (seg_mask > 0.5).double()

        with torch.device(self.device):
            seg_image = image * seg_mask
            inputs = self.image_processor(images=seg_image, return_tensors="pt")

        # val image sanity check
        if self.global_step == 0 and batch_idx == 0:
            self.tb_log.add_images("val/image", image, global_step=self.global_step)
            self.tb_log.add_images("val/seg_mask", seg_mask, global_step=self.global_step)

        output = self.forward(**inputs)
        (logits, ids_restore, mask) = output

        loss = self.forward_loss(inputs["pixel_values"], logits, mask)

        self.log(f"val/loss", loss, batch_size=batch_size, prog_bar=True)

        if batch_idx == 0:
            y = self.unpatchify(logits)

            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.encoder.config.patch_size**2 * 3)  # (N, H*W, p*p*3)
            mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping

            # masked image
            im_masked = seg_image * (1 - mask) + mask * 0.5

            # MAE reconstruction pasted with visible patches
            im_paste = seg_image * (1 - mask) + y * mask

            self.tb_log.add_images(f"val/masked", im_masked, global_step=self.global_step)
            self.tb_log.add_images(f"val/prediction", y, global_step=self.global_step)
            self.tb_log.add_images(f"val/reconstruction", im_paste, global_step=self.global_step)

    def on_test_start(self):
        self.tb_log = self.logger.experiment
