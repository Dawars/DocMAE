from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


class ListDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transforms=None):
        """
        Args:
            data_root: Directory where the split files are located
            split: split name of subset of images
            transforms: optional transforms for data augmentation
        """

        self.data_root = data_root

        self.filenames = natural_sort((data_root / f"{split}.txt").read_text().split())

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(self.data_root / filename).convert("RGB")
        image = ImageOps.exif_transpose(image)

        if self.transforms:
            image = self.transforms(image)

        return {"image": image}
