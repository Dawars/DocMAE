import logging
import math
import warnings
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torchvision import datapoints
from torchvision.transforms import InterpolationMode, functional
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2._utils import _setup_size
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.utils import query_spatial_size

logger = logging.getLogger(__name__)


class RandomResizedCropWithUV(object):
    """Crop a random portion of the input and resize it to a given size.
    This version correctly handles forward and backward mapping. Originally from torchvision.transforms.v2

    If the input is a :class:`torch.Tensor` or a ``Datapoint`` (e.g. :class:`~torchvision.datapoints.Image`,
    :class:`~torchvision.datapoints.Video`, :class:`~torchvision.datapoints.BoundingBox` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    A crop of the original input is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float, optional): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float, optional): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        scale = cast(Tuple[float, float], scale)
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        ratio = cast(Tuple[float, float], ratio)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.interpolation = _check_interpolation(interpolation)
        self.antialias = antialias

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = query_spatial_size(flat_inputs)
        area = height * width

        log_ratio = self._log_ratio
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def __call__(self, sample) -> Any:
        image, bm, uv, mask = sample
        orig_size = image.shape[1:]

        if uv is None:
            params = {"top": 0, "left": 0, "height": orig_size[0], "width": orig_size[1]}
            mask_crop = TF.resized_crop(
                mask, **params, size=self.size, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False
            ).squeeze()

            image_crop = TF.resized_crop(
                image[None], **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
            )[0].clip(0, 255)
            bm_crop = TF.resized_crop(
                bm[None], **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
            )[0]
            return (
                datapoints.Image(image_crop),
                datapoints.Image((((bm_crop.permute(1, 2, 0) - 0.5) * 2).float().permute(2, 0, 1) + 1) / 2),
                uv,  # None
                datapoints.Mask(mask_crop[None]),
            )
        params = self._get_params([image, bm, uv, mask])
        crop = True
        while crop:
            params = self._get_params([image, bm, uv, mask])
            uv_crop = TF.resized_crop(
                uv, **params, size=self.size, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False
            )
            mask_crop = TF.resized_crop(
                mask, **params, size=self.size, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False
            ).squeeze()
            # more than half of the image is filled
            if mask_crop.sum() > 10:
                crop = False
            else:
                logging.warning("Crop contains little content, recropping")
        # test values
        # params = {"top": 14, "left": 39, "height": 419, "width": 331}  # full page crop
        # params = {"top": 200, "left": 0, "height": 201, "width": 446}  # bottom half crop
        # params = {"top": 0, "left": 113, "height": 326, "width": 309}  # top right corner
        # params = {"top": 200, "left": 200, "height": 248, "width": 248}  # bottom right corner
        # params = {"top": 2, "left": 2, "height": 366, "width": 366}  # top left corner
        # params = {"top": 27, "left": 67, "height": 100, "width": 100}  # test

        image_crop = TF.resized_crop(
            image[None], **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
        )[0].clip(0, 255)

        # flip uv Y
        uv_crop[1, mask_crop.bool()] = 1 - uv_crop[1, mask_crop.bool()]
        min_uv_w, min_uv_h = uv_crop[0, mask_crop.bool()].min(), uv_crop[1, mask_crop.bool()].min()
        max_uv_w, max_uv_h = uv_crop[0, mask_crop.bool()].max(), uv_crop[1, mask_crop.bool()].max()

        min_uv_h = min_uv_h * orig_size[0]
        max_uv_h = max_uv_h * orig_size[0]
        min_uv_w = min_uv_w * orig_size[1]
        max_uv_w = max_uv_w * orig_size[1]

        bm_crop = bm[:, min_uv_h.long() : max_uv_h.long() + 1, min_uv_w.long() : max_uv_w.long() + 1]
        bm_crop = functional.resize(bm_crop[None], self.size, interpolation=self.interpolation, antialias=self.antialias)[0]

        # normalized relative displacement for sampling
        bm_crop_norm = (bm_crop.permute(1, 2, 0) - 0.5) * 2
        # extend crop to include background
        min_crop_w = params["left"]
        min_crop_h = params["top"]
        max_crop_w = params["left"] + params["width"]
        max_crop_h = params["top"] + params["height"]

        # get center of crop in normalized coords [-1, 1]
        center_x = min_crop_w + (max_crop_w - min_crop_w) / 2
        center_y = min_crop_h + (max_crop_h - min_crop_h) / 2
        center_x_norm = 2 * center_x / orig_size[1] - 1
        center_y_norm = 2 * center_y / orig_size[0] - 1

        bm_crop_norm[..., 1] = bm_crop_norm[..., 1] - center_y_norm  # h
        bm_crop_norm[..., 0] = bm_crop_norm[..., 0] - center_x_norm  # w

        # rescale to [-1, 1] for crop
        bm_crop_norm[..., 1] = (bm_crop_norm[..., 1]) * orig_size[1] / (max_crop_h - min_crop_h)
        bm_crop_norm[..., 0] = (bm_crop_norm[..., 0]) * orig_size[0] / (max_crop_w - min_crop_w)

        """
        import torch.nn.functional as F
        from matplotlib import pyplot as plt
        import matplotlib.patches as patches
        from copy import copy
        
        align_corners = False
        bm_crop_norm = bm_crop_norm.float()[None]

        image_crop_manual = image[:, min_crop_h : max_crop_h + 1, min_crop_w : max_crop_w + 1]
        image_crop_manual = functional.resize(
            image_crop_manual[None], self.size, interpolation=self.interpolation, antialias=self.antialias
        )[0]

        mask_crop_manual = mask[0, min_crop_h : max_crop_h + 1, min_crop_w : max_crop_w + 1]
        mask_crop_manual = functional.resize(
            mask_crop_manual[None], self.size, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False
        )[0]

        zeros = torch.ones((448, 448, 1))

        f, axrr = plt.subplots(3, 5)
        for ax in axrr:
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

        # scale bm to -1.0 to 1.0
        bm_norm = (bm - 0.5) * 2
        bm_norm = bm_norm.permute(1, 2, 0)[None].float()

        axrr[0][0].imshow(image.permute(1, 2, 0) / 255)
        axrr[0][0].title.set_text("full image")
        axrr[0][0].scatter((center_x), (center_y), c="b", s=1)
        axrr[0][0].scatter((((center_x_norm + 1) / 2) * orig_size[1]), (((center_y_norm + 1) / 2) * orig_size[0]), c="r", s=1)
        axrr[0][1].imshow(mask[0], cmap="gray")
        axrr[0][1].title.set_text("mask")
        axrr[0][2].imshow(torch.cat([uv.permute(1, 2, 0), zeros], dim=-1))
        axrr[0][2].title.set_text("uv")
        axrr[0][3].imshow(torch.cat([bm_norm[0] * 0.5 + 0.5, zeros], dim=-1), cmap="gray")
        axrr[0][3].title.set_text("bm")
        axrr[0][4].imshow(F.grid_sample(image[None] / 255, bm_norm, align_corners=align_corners)[0].permute(1, 2, 0))
        axrr[0][4].title.set_text("unwarped full doc")

        rect_patch_crop = patches.Rectangle(
            (min_crop_w, min_crop_h),
            max_crop_w - min_crop_w,
            max_crop_h - min_crop_h,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        axrr[0][0].add_patch(copy(rect_patch_crop))
        axrr[0][1].add_patch(copy(rect_patch_crop))
        axrr[0][2].add_patch(copy(rect_patch_crop))
        rect_patch_uv = patches.Rectangle(
            (min_uv_w, min_uv_h), (max_uv_w - min_uv_w), (max_uv_h - min_uv_h), linewidth=1, edgecolor="g", facecolor="none"
        )
        axrr[0][4].add_patch(copy(rect_patch_uv))
        axrr[0][3].add_patch(copy(rect_patch_uv))

        zeros = torch.ones_like(uv_crop.permute(1, 2, 0))

        axrr[1][0].imshow(image_crop.permute(1, 2, 0) / 255)
        axrr[1][0].title.set_text("image crop")
        axrr[1][1].imshow(mask_crop, cmap="gray")
        axrr[1][1].title.set_text("mask crop")
        axrr[1][2].imshow(torch.cat([uv_crop.permute(1, 2, 0), zeros], dim=-1))
        axrr[1][2].title.set_text("uv crop")
        axrr[1][3].title.set_text("bm crop manual")
        axrr[1][4].imshow(
            F.grid_sample(image[None] / 255, (bm_crop.permute(1, 2, 0).float() - 0.5)[None] * 2, align_corners=align_corners,)[
                0
            ].permute(1, 2, 0)
        )
        axrr[1][4].title.set_text("unwarped crop from orig")

        axrr[2][0].imshow(image_crop_manual.permute(1, 2, 0) / 255)
        axrr[2][0].title.set_text("image crop manual")
        axrr[2][1].imshow(mask_crop_manual, cmap="gray")
        axrr[2][1].title.set_text("crop mask manual")
        axrr[2][2].imshow(
            F.grid_sample(mask_crop_manual[None][None], bm_crop_norm, mode="nearest", align_corners=align_corners)[0].permute(
                1, 2, 0
            ),
            cmap="gray",
        )
        axrr[2][2].title.set_text("mask unwarped manual")
        axrr[2][3].imshow(
            torch.cat([bm_crop_norm[0] * 0.5 + 0.5, torch.ones_like(bm_crop_norm)[0, ..., 0:1]], dim=-1).clip(0, 1), cmap="gray"
        )
        axrr[2][3].title.set_text("bm crop manual")
        axrr[2][4].imshow(
            F.grid_sample(image_crop_manual[None] / 255, bm_crop_norm, align_corners=align_corners)[0].permute(1, 2, 0)
        )
        axrr[2][4].title.set_text("unwarped crop manual")

        plt.tight_layout()
        plt.show()
        """
        return (
            datapoints.Image(image_crop),
            datapoints.Image((bm_crop_norm.float().permute(2, 0, 1) + 1) / 2),
            # datapoints.Mask(uv_crop),
            None,
            datapoints.Mask(mask_crop[None]),
        )
