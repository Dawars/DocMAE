import os
import logging
import warnings
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
from urllib3.exceptions import InsecureRequestWarning
from PIL import Image
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
torchvision.disable_beta_transforms_warning()
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from scipy.interpolate import griddata


def fm2bm(fm, msk, s):
    """
    fm: the forward mapping in range [0,1]
    s: shape of the image required to sample from, defines the range of image coordinates in
    """
    fm = fm.numpy() * s

    msk = fm[..., 0] > 0
    s2d = fm[msk]
    tx, ty = np.nonzero(msk)
    # s2d = fm[:, msk]
    # tx, ty = np.where(msk)
    grid = np.meshgrid(np.linspace(1, s, s), np.linspace(1, s, s))
    vx = griddata(s2d, tx, tuple(grid), method="nearest") / float(s)
    vy = griddata(s2d, ty, tuple(grid), method="nearest") / float(s)
    bm = np.stack([vy, vx], axis=-1)
    return bm


def uv2mp(uv, msk, scale):
    # uv is a k*k*3 numpy array
    # s is the size of the output map
    s = scale
    # Rescale the 1.0 in uv to s which is the size of the output map
    uv = uv * s
    sx = uv[0, msk]
    sy = uv[1, msk]
    # Valid point
    # sx sy are the value in the forward mapping but the coord in the backward mapping
    # sx = sx[msk]
    # sy = sy[msk]
    # tx ty are the coord in the forward mapping but the value in the backward mapping
    ty, tx = np.where(msk)
    Fx = griddata((sx, sy), tx, (sx, sy), method="linear")
    Fy = griddata((sx, sy), ty, (sx, sy), method="linear")
    # Sampling coord on the output mapping
    xq, yq = np.meshgrid(np.arange(1, s + 1), np.arange(1, s + 1))
    # Get the value based on the scattered interpolation
    vx = griddata((sx, sy), Fx, (xq, yq), method="linear")
    vy = griddata((sx, sy), Fy, (xq, yq), method="linear")
    # Concatenate together
    invmap = np.stack((vx, vy), axis=2)
    return invmap


class Doc3D(Dataset):
    def __init__(self, data_root: Path, split: str, transforms=None):
        """
        Args:
            data_root: Directory where the doc3d dataset is extracted
            split: split name of subset of images
            transforms: optional transforms for data augmentation
        """

        self.data_root = data_root
        self.filenames = (data_root / f"{split}.txt").read_text().split()
        self.prefix_img = "img/"
        self.prefix_bm = "bm/"
        self.prefix_uv = "uv/"

        self.transforms = transforms

        self.grid_input = self.coords_grid(448, 448).transpose(2, 1, 0)
        self.grid_output = self.coords_grid(288, 288)

    @staticmethod
    def coords_grid(ht, wd):
        coords = np.meshgrid(np.arange(ht), np.arange(wd))
        coords = np.stack(coords[::-1], axis=0)
        return coords

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(self.data_root / self.prefix_img / f"{filename}.png").convert("RGB")
        image = datapoints.Image(image)
        full_image = image
        # backwards mapping
        h5file = h5py.File(self.data_root / self.prefix_bm / f"{filename}.mat", "r")
        bm = np.array(h5file.get("bm"))
        bm = bm.transpose(2, 1, 0)
        full_flow = bm

        bm = datapoints.Image(bm.transpose(2, 0, 1))  # absolute back mapping
        grid = datapoints.Image(self.grid_input.transpose(2, 0, 1))  # pixel coordinates in a grid

        zeros = np.ones((448, 448, 1))
        # plt.imshow(np.concatenate((bm / 448, zeros), axis=-1))
        # plt.colorbar()
        # plt.show()

        # plt.imshow(np.concatenate((grid/448, zeros), axis=-1))
        # large_abs = np.concatenate(((bm - grid) / 448 + 0.5, zeros), axis=-1)
        # plt.imshow(large_abs)
        # plt.colorbar()
        # plt.show()

        # mask from uv
        # Decode the EXR data using OpenCV
        uv = cv2.imread(str(self.data_root / self.prefix_uv / f"{filename}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv = cv2.cvtColor(uv, cv2.COLOR_BGR2RGB)  # forward mapping
        full_uv = uv
        mask = datapoints.Mask(uv[..., 2])
        uv = datapoints.Image(uv[..., :2].transpose(2, 0, 1))

        import torch.nn.functional as F
        import torch
        import hdf5storage as h5

        # bm = h5.loadmat(str(self.data_root / self.prefix_bm / f"{filename}.mat"))["bm"]
        #
        # f, axrr = plt.subplots(4, 2)
        # for ax in axrr:
        #     for a in ax:
        #         a.set_xticks([])
        #         a.set_yticks([])
        #
        # axrr[0][0].imshow(image.data.permute(1, 2, 0))
        # axrr[0][0].title.set_text("image")
        # axrr[0][1].imshow(mask)  # .transpose(2, 0, 1))
        # axrr[0][1].title.set_text("uv map")
        #
        # scale bm to -1.0 to 1.0
        # flow_ = bm / 448
        # flow_ = (flow_ - 0.5) * 2
        # flow_ = flow_[None]
        # flow_ = torch.from_numpy(flow_).float()
        #
        # flow_ = flow_.permute(1, 2, 0)
        #
        # axrr[1][0].imshow(flow_[0, ..., 0], cmap="gray")
        # axrr[1][0].title.set_text("backward map x")
        # axrr[1][1].imshow(flow_[0, ..., 1], cmap="gray")
        # axrr[1][1].title.set_text("backward map y")
        #
        # uw = F.grid_sample(image[None] / 255, flow_)
        # axrr[2][0].imshow(uw[0].permute(1, 2, 0))
        # axrr[2][0].title.set_text("bm resample")
        # axrr[2][1].imshow(np.concatenate((uv.permute(1, 2, 0), zeros), axis=-1))
        # axrr[2][1].title.set_text("uv")
        # plt.tight_layout()
        # plt.show()

        if self.transforms:
            image, bm, mask, uv, grid = self.transforms(image, bm, mask, uv, grid)

        zeros = np.ones((288, 288, 1))

        displacement = bm - grid  # between [0, 447]  displacement

        # bounds of crop of distorted image
        min_x, min_y = grid[0].min(), grid[1].min()
        max_x, max_y = grid[0].max(), grid[1].max()
        # region of full undistorted image that is undistorted TO the cropped region, not useful
        # min_flow_x, min_flow_y = bm[0].min(), bm[1].min()
        # max_flow_x, max_flow_y = bm[0].max(), bm[1].max()

        print(min_x, max_x, min_y, max_y)
        # print(min_flow_x, max_flow_x, min_flow_y, max_flow_y)

        uv_ = uv.data.detach().clone()
        uv_[1, mask.bool()] = 1 - uv_[1, mask.bool()]
        min_uv_x, min_uv_y = uv_[0, mask.bool()].min(), uv_[1, mask.bool()].min()
        max_uv_x, max_uv_y = uv_[0, mask.bool()].max(), uv_[1, mask.bool()].max()
        print(min_uv_x, max_uv_x, min_uv_y, max_uv_y)

        # transforming backward mapping so that it works for crop (288)
        # bm.data[0] = (bm.data[0] - min_x) / (max_x - min_x)
        # bm.data[1] = (bm.data[1] - min_y) / (max_y - min_y)
        # bm.data = (bm.data / 448) * 288

        import matplotlib.patches as patches

        f, axrr = plt.subplots(5, 2)
        for ax in axrr:
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

        axrr[0][0].imshow(image.data.permute(1, 2, 0) / 255)
        axrr[0][0].title.set_text("image")
        axrr[0][1].imshow(mask, cmap="gray")
        axrr[0][1].title.set_text("mask")

        # scale cropped uv to [0, 1] range

        # bw_crop = np.nan_to_num(uv2mp(uv, mask.bool(), 288))
        # scale bm to -1.0 to 1.0
        flow_ = full_flow / 448
        flow_ = (flow_ - 0.5) * 2
        flow_ = flow_[None]
        flow_ = torch.from_numpy(flow_).float()

        # this doesn't work
        # bw_crop = flow_[
        #     :,
        #     int(min_uv_x.item() * 488) : int(max_uv_x.item() * 488) + 1,
        #     int(min_uv_y.item() * 488) : int(max_uv_y.item() * 488) + 1,
        # ]
        # bw_crop = torch.nn.functional.interpolate(bw_crop.permute(0, 3, 1, 2), (288, 288)).permute(0, 2, 3, 1)

        axrr[1][0].imshow(flow_[0, ..., 0], cmap="gray")
        axrr[1][0].title.set_text("backward map x")
        axrr[1][1].imshow(flow_[0, ..., 1], cmap="gray")
        axrr[1][1].title.set_text("backward map y")

        uv[0] = (uv[0] - min_uv_x) / (max_uv_x - min_uv_x)
        uv[1] = (uv[1] - min_uv_y) / (max_uv_y - min_uv_y)  # todo how can uv be -4?

        bm_manual = torch.from_numpy(fm2bm(uv.permute(1, 2, 0), mask.bool(), 288)).float()
        axrr[2][0].imshow(np.concatenate((bm_manual / 288, np.ones((288, 288, 1))), axis=-1))
        axrr[2][0].title.set_text("bm")
        # axrr[2][0].imshow(np.concatenate((flow_[0] / 2 + 0.5, torch.ones(448, 448, 1)), axis=-1))
        bm_manual_ = (bm_manual[None] - 0.5) * 2
        uw_ = F.grid_sample(image[None] / 255, bm_manual_, padding_mode="zeros")
        axrr[2][1].imshow(uw_[0].permute(1, 2, 0))
        axrr[2][1].title.set_text("bm resample")

        axrr[3][0].imshow(full_image.permute(1, 2, 0))
        # Create a rectangle patch
        rect_patch = patches.Rectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor="r", facecolor="none"
        )
        # Add the rectangle to the axes
        axrr[3][0].add_patch(rect_patch)
        # axrr[3][0].add_patch(rect_patch_flow)

        # scale bm to -1.0 to 1.0
        flow_ = full_flow / 448
        flow_ = (flow_ - 0.5) * 2
        flow_ = flow_[None]
        flow_ = torch.from_numpy(flow_).float()

        # flow_ = flow_.permute(1, 2, 0)
        axrr[3][1].title.set_text("unwarped full doc")
        axrr[3][1].imshow(F.grid_sample(full_image[None] / 255, flow_)[0].permute(1, 2, 0))

        # get full BM in this area (+ unwarp border mask with the result)
        rect_patch_uv = patches.Rectangle(
            (min_uv_x * 448, min_uv_y * 448),
            (max_uv_x - min_uv_x) * 448,
            (max_uv_y - min_uv_y) * 448,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        axrr[3][1].add_patch(rect_patch_uv)

        axrr[4][0].title.set_text("uv")
        axrr[4][0].imshow(np.concatenate((uv.permute(1, 2, 0), zeros), axis=-1))

        mask_unwarped = F.grid_sample(mask[None][None], bm_manual_, padding_mode="zeros", mode="nearest")
        axrr[4][1].title.set_text("unwarped mask")
        axrr[4][1].imshow(mask_unwarped[0, 0], cmap="gray", vmin=0, vmax=1)

        plt.tight_layout()
        plt.show()

        # plt.imshow(np.concatenate((bm.data.permute(1, 2, 0) / 288, torch.ones(288, 288, 1)), axis=-1))
        # plt.colorbar()
        # plt.show()
        return {"image": image, "bm": bm, "mask": mask}
