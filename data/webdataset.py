from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

try:
    import webdataset as wds
except ImportError as e:
    raise ImportError("webdataset not installed – add it to requirements.txt") from e


def make_train_loader(cfg):
    """Returns a torch DataLoader that reads \*.tar shards via WebDataset.

    Assumes `cfg.data.dir` contains files like 00000.tar, 00001.tar, ...
    Each tar must have at least an `img` key (jpg/png)."""
    
    shards = str(Path(cfg.data.dir) / "{000000..999999}.tar")  # brace expands

    preprocess = T.Compose([
        T.RandomResizedCrop(cfg.model.img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = (
        wds.WebDataset(shards, resampled=True)
        .decode("pil")
        .to_tuple("img")
        .map_tuple(preprocess)
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,  # handled by resampled=True
    )
    return loader



class _CatFolder(Dataset):
    """Treat *every* image file under root as a sample, ignore labels."""
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths: List[Path] = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img,


def make_val_loader(cfg):
    """Returns a DataLoader for the Kaggle cat dataset or a fallback folder.

    If `cfg.data.val_dir` is a path that *doesn’t* exist or is the sentinel
    string "cats", we auto‑download `crawford/cat-dataset` via kagglehub.
    """

    # 1. Ensure dataset is present ------------------------------------------------
    root = Path(cfg.data.val_dir)
    if cfg.data.val_dir.lower() in {"cats", "cat_dataset", "kaggle_cats"} or not root.exists():
        try:
            from kagglehub import dataset_download
        except ImportError as e:
            raise ImportError("kagglehub not installed – `pip install kagglehub`.") from e
        print("Downloading crawford/cat-dataset via KaggleHub… (first time only)")
        root = Path(dataset_download("crawford/cat-dataset"))
        cfg.data.val_dir = str(root)

    # 2. Build transforms ---------------------------------------------------------
    tfm = T.Compose([
        T.Resize((cfg.model.img_size, cfg.model.img_size)),
        T.ToTensor(),
    ])

    # 3. Dataset ------------------------------------------------------------------
    if any(root.joinpath(cls).is_dir() for cls in os.listdir(root)):
        # likely already ImageFolder structure (cat/ sub‑dir). Keep labels.
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root, transform=tfm)
    else:
        dataset = _CatFolder(root, transform=tfm)

    # 4. DataLoader ---------------------------------------------------------------
    return DataLoader(dataset, batch_size=1, shuffle=False)