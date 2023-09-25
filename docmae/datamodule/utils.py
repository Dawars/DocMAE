from enum import EnumMeta

import gin
import torchvision.transforms as T
from kornia import augmentation
from torch import nn
from torchvision import transforms


def init_external_gin_configurables():
    # Set torchvision transforms as gin configurable
    for name in transforms.transforms.__all__:
        attribute = getattr(transforms, name)
        if isinstance(attribute, EnumMeta):
            # enums like InterpolationMode don't work because external registration of enums is not possible
            continue
        gin.external_configurable(attribute, module="torchvision.transforms")

    # Set torchvision transforms as gin configurable
    for name in augmentation.__all__:
        attribute = getattr(augmentation, name)
        if isinstance(attribute, EnumMeta):
            # enums like InterpolationMode don't work because external registration of enums is not possible
            continue
        if not hasattr(attribute, "__call__"):
            continue
        gin.external_configurable(attribute, module="kornia.augmentation")


@gin.configurable
def get_image_transforms(transform_list: list[nn.Module]):
    return T.Compose(transform_list)
