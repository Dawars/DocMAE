from torch import nn
import torch
import torch.nn.functional as F
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
            nn.Conv2d(hidden_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, PATCH_SIZE**2 * 9, 1, padding=0)
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


class DocMAE(nn.Module):
    def __init__(
        self,
        image_processor: ViTImageProcessor,
        encoder: ViTMAEModel,
        decoder: ViTMAEDecoder,
        hidden_dim: int,
        upscale_type: str = "raft",
    ):
        super().__init__()
        self.image_processor = image_processor
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_norm = nn.LayerNorm(decoder.config.decoder_hidden_size, eps=decoder.config.layer_norm_eps)

        self.P = PATCH_SIZE
        self.hidden_dim = hidden_dim
        self.upscale_type = upscale_type
        self.update_block = UpdateBlock(self.hidden_dim)  # todo check paper for hidden_dim

    def forward(self, x):
        inputs = self.image_processor(images=x, return_tensors="pt")

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

        if self.upscale_type == "raft":
            bm_up = self.flow_head(fmap, x)
        elif self.upscale_type == "interpolate":
            bm_up = upflow16(fmap)
        else:
            raise NotImplementedError
        return bm_up

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
        # convex upsample baesd on fmap
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up
        return bm_up
