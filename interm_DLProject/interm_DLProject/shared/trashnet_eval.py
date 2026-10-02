"""
Optional cross-dataset generalisation check (synopsis: secondary dataset TrashNet, unseen test set).

  python -m shared.trashnet_eval --model resnet50 --trashnet_root /path/to/trashnet

TrashNet has 6 classes (cardboard, glass, metal, paper, plastic, trash). The 12-class prediction is mapped
down: brown/green/white-glass -> glass. Predictions in classes TrashNet does not contain
(battery, biological, clothes, shoes) are counted as errors.
Needs the checkpoint saved by training: results/checkpoints/<tag>.pt
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from . import registry
from .config import RESULTS_DIR, SPLIT_CSV
from .data_prep import find_class_root
from .dataset import get_transforms

GLASS = {"brown-glass", "green-glass", "white-glass"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tag, e.g. mlp, cnn, cnn_noaug, resnet50, vit, swin")
    ap.add_argument("--trashnet_root", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    classes12 = sorted(pd.read_csv(SPLIT_CSV)["label"].unique())
    build_fn, img_size = registry.get(a.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_fn(len(classes12), pretrained=False)
    model.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / f"{a.model}.pt", map_location=device))
    model.to(device).eval()

    root = find_class_root(Path(a.trashnet_root), min_classes=6)
    ds = ImageFolder(str(root), transform=get_transforms(img_size, augment=False))
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu().tolist()
            y_pred += ["glass" if classes12[i] in GLASS else classes12[i] for i in p]
            y_true += [ds.classes[i] for i in y.tolist()]

    res = {"model": a.model, "trashnet_images": len(y_true),
           "accuracy": float(accuracy_score(y_true, y_pred)),
           "macro_f1_6class": float(f1_score(y_true, y_pred, labels=ds.classes,
                                             average="macro", zero_division=0)),
           "per_class_f1": dict(zip(ds.classes, map(float, f1_score(
               y_true, y_pred, labels=ds.classes, average=None, zero_division=0))))}
    out = RESULTS_DIR / f"{a.model}_trashnet.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
