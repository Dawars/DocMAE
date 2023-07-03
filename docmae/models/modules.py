from torch import nn


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
