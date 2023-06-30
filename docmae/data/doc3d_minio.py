import os
import logging
import warnings
from io import BytesIO

import h5py
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from minio import Minio
from PIL import Image
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datapoints

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
torchvision.disable_beta_transforms_warning()


class Doc3D(Dataset):
    def __init__(self, filenames, transforms=None):
        self.client = self.create_gini_data_minio_client()

        self.prefix_img = "computer_vision/rectification/doc3d/img/"
        self.prefix_bm = "computer_vision/rectification/doc3d/bm/"
        self.prefix_uv = "computer_vision/rectification/doc3d/uv/"
        self.filenames = filenames

        self.transforms = transforms

    @staticmethod
    def create_gini_data_minio_client(minio_url: str = None, access_key: str = None, access_secret: str = None):
        """
        To create a minio client with values from following ENV VARs.
            url:   PROD_MINIO_URL
            key:   CVIE_MINIO_USER
            secret:  CVIE_MINIO_PASSWORD

        Note: ENV Vars override the given values.
        Client REF: https://min.io/docs/minio/linux/developers/python/API.html

        Returns:
            a configured minio client
        """
        minio_url = os.getenv("PROD_MINIO_URL", minio_url)
        minio_key = os.getenv("CVIE_MINIO_USER", access_key)
        minio_secret = os.getenv("CVIE_MINIO_PASSWORD", access_secret)
        if not minio_key or not minio_secret:
            LOGGER.warning("No minio credential is set. ")
            return
        timeout = 60

        http_client = urllib3.PoolManager(
            timeout=urllib3.util.Timeout(connect=timeout, read=timeout),
            maxsize=10,
            cert_reqs="CERT_NONE",
            retries=urllib3.Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504]),
        )
        # Note: we have to set up a client that uses ssl but no cert check involved.
        # with mc command it's mc --insecure,
        # but with python client, it's combination of secure=True (ensure https) and cert_req=CERT_NONE.
        return Minio(minio_url, access_key=minio_key, secret_key=minio_secret, secure=True, http_client=http_client)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        obj_img = self.client.get_object("cvie", self.prefix_img + filename + ".png")
        image = Image.open(obj_img).convert("RGB")
        image = datapoints.Image(image)

        # backwards mapping
        obj_bm = self.client.get_object("cvie", self.prefix_bm + filename + ".mat")
        h5file = h5py.File(BytesIO(obj_bm.data), "r")
        flow = np.array(h5file.get("bm"))
        flow = np.flip(flow, 0).copy()
        flow = datapoints.Image(flow)

        # mask from uv
        obj_uv = self.client.get_object("cvie", self.prefix_uv + filename + ".exr")
        exr_data = obj_uv.read()
        exr_array = np.asarray(bytearray(exr_data), dtype=np.uint8)

        # Decode the EXR data using OpenCV
        uv = cv2.imdecode(exr_array, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv = cv2.cvtColor(uv, cv2.COLOR_BGR2RGB)
        mask = datapoints.Mask(uv[..., 2])

        if self.transforms:
            image, flow, mask = self.transforms(image, flow, mask)

        return {"image": image, "bm": flow, "mask": mask}
