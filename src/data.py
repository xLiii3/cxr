"""Dataset over the pre-resized uint8 memmap plus inference helpers."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config as C

_MEAN = torch.tensor(C.MEAN).view(3, 1, 1)
_STD = torch.tensor(C.STD).view(3, 1, 1)


def read_split():
    df = pd.read_csv(C.SPLIT_CSV)
    return df


def to_tensor(images):
    """List of PIL images -> normalised [B,3,H,W] float tensor."""
    if isinstance(images, Image.Image):
        images = [images]
    batch = []
    for im in images:
        im = im.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BICUBIC)
        batch.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return (torch.stack(batch) - _MEAN) / _STD


class ChestXray(Dataset):
    """Reads rows of the memmap; label vector is the 14 binary columns."""

    def __init__(self, df, npy=None):
        self.df = df.reset_index(drop=True)
        self.npy = str(npy or C.IMAGE_NPY)
        self.labels = self.df[C.CLASSES].to_numpy("float32")
        self._arr = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self._arr is None:
            self._arr = np.load(self.npy, mmap_mode="r")
        img = torch.from_numpy(np.asarray(self._arr[int(self.df["idx"][i])], dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        return (img - _MEAN) / _STD, torch.from_numpy(self.labels[i])


def loader(df, batch_size=64, shuffle=False, workers=0, npy=None):
    return DataLoader(
        ChestXray(df, npy), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=False,
        persistent_workers=workers > 0,
    )
