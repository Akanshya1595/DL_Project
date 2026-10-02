"""Shared preprocessing + augmentation pipeline (identical for all four models)."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD, SPLIT_CSV, DATA_DIR


def get_transforms(img_size: int, augment: bool):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])


class WasteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, class_to_idx: dict, transform):
        self.paths = df["path"].tolist()
        self.labels = [class_to_idx[l] for l in df["label"]]
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def resolve_root(data_root=None) -> Path:
    if data_root:
        return Path(data_root)
    f = DATA_DIR / "data_root.txt"
    if f.exists():
        return Path(f.read_text().strip())
    raise SystemExit("Dataset location unknown. Pass --data_root or run src.data_prep first.")


def build_loaders(img_size, batch_size, augment=True, workers=2, data_root=None):
    df = pd.read_csv(SPLIT_CSV)
    classes = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    root = resolve_root(data_root)
    loaders = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        ds = WasteDataset(sub, root, c2i, get_transforms(img_size, augment and split == "train"))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=workers, pin_memory=True,
                                    drop_last=(split == "train"))
    train_counts = df[df["split"] == "train"]["label"].value_counts()
    return loaders, classes, [int(train_counts[c]) for c in classes]
