import torch
from torch import nn
import torch.nn.functional as F


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# Flow related code taken from https://github.com/fh2019ustc/DocTr/blob/main/GeoTr.py
class UpscaleRAFT(nn.Module):
    """
    Infers conv mask to upscale flow
    """

    def __init__(self, patch_size: int, input_dim=512):
        super(UpscaleRAFT, self).__init__()
        self.P = patch_size

        self.mask = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, patch_size**2 * 9, 1, padding=0)
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

    def forward(self, feature_map, flow):
        mask = 0.25 * self.mask(feature_map)  # scale mask to balance gradients
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

    def forward(self, feature_map, **kwargs):
        return self.layers(feature_map)


class UpscaleInterpolate(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map):
        flow = self.conv2(self.relu(self.conv1(feature_map)))

        new_size = (16 * flow.shape[2], 16 * flow.shape[3])  # to scale 8/16
        return 16 * F.interpolate(flow, size=new_size, mode=self.mode, align_corners=False)


def expansion_block(in_channels, mid_channel, out_channels, relu=True):
    """Build block of two consecutive convolutions followed by upsampling.

    The following chain of layers is applied:
    [Conv2D -> ReLU -> BatchNorm -> Conv2D -> ReLU -> BatchNorm -> ConvTranspose2d -> ReLU]

    This block doubles the dimensions of input tensor, i.e. input is of size
    (rows, cols, in_channels) and output is of size (rows*2, cols*2, out_channels).

    Args:
        in_channels (int): Number of channels of input tensor.
        mid_channel (int): Number of channels of middle channel.
        out_channels (int): Number of channels of output tensor, i.e., number of filters.
        relu (bool): Indicates whether to apply ReLU after transposed convolution.

    Returns:
        block (nn.Sequential): Built expansive block.

    """
    block = nn.Sequential(
        conv_relu_bn(in_channels, mid_channel),
        nn.ConvTranspose2d(
            in_channels=mid_channel,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        ),
    )
    if relu is True:
        block = nn.Sequential(block, nn.ReLU())
    return block


def conv_relu_bn(in_channels, out_channels, kernel_size=3, padding=1):
    """Build [Conv2D -> ReLu -> BatchNorm] block.

    Args:
        in_channels (int): Number of channels of input tensor.
        out_channels (int): Number of channels of output tensor, i.e., number of filters.
        kernel_size (int): Size of convolution filters, squared filters are assumed.
        padding (int): Amount of **zero** padding around input tensor.

    Returns:
        block (nn.Sequential): [Conv2D -> ReLu -> BatchNorm] block.

    """
    block = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block
