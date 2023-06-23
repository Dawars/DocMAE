import torch

from transformers import (
    ViTMAEConfig,
    AutoImageProcessor,
)
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEModel

from docmae.models.docmae import DocMAE

pretrained_config = ViTMAEConfig.from_pretrained("dawars/docmae_pretrain/")
pretrained_config.mask_ratio = 0
image_processor = AutoImageProcessor.from_pretrained("dawars/docmae_pretrain/", size={"height": 288, "width": 288})
mae_encoder = ViTMAEModel(pretrained_config)
mae_decoder = ViTMAEDecoder(pretrained_config, mae_encoder.embeddings.num_patches)

model = DocMAE(image_processor, mae_encoder, mae_decoder, hidden_dim=512, upscale_type="raft")

# testing forward pass
flow = model(torch.rand(1, 3, 288, 288))
print(flow)
