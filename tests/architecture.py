import json
from pathlib import Path

import torch
import torchvision
from transformers import ViTMAEConfig, AutoImageProcessor, ViTImageProcessor, ViTMAEModel
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder

from docmae.models.docmae import DocMAE


def test_upscaling():
    config_file = "./config/finetune.json"
    config = json.loads(Path(config_file).read_text())

    pretrained_config = ViTMAEConfig.from_pretrained(config["mae_path"])
    pretrained_config.mask_ratio = 0

    with torch.device("cuda"):
        mae_encoder = ViTMAEModel(pretrained_config)
        mae_decoder = ViTMAEDecoder(pretrained_config, mae_encoder.embeddings.num_patches)

        model = DocMAE(mae_encoder, mae_decoder, config).cuda()
        model.eval()

        x = torch.rand([4, 3, 288, 288])
        x.requires_grad = True
        out = model.forward(x)  # fmap

        i = 2
        loss = out[i].sum()

        loss.backward()

        mask = torch.zeros([4])
        mask[i] = 1
        mask = mask.bool()

        assert torch.count_nonzero(x.grad[~mask]) == 0
        assert torch.count_nonzero(x.grad[i]) > 0

