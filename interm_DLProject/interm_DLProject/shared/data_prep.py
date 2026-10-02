"""
Dataset acquisition, integrity check, EDA and the FROZEN stratified 70/15/15 split.
Owner: Akanshya Chakraborty.

Usage
-----
# automatic download from Kaggle (needs Kaggle credentials configured):
python -m src.data_prep --download
# or point at a folder you already downloaded/unzipped:
python -m src.data_prep --data_root /path/to/garbage_classification

The split is written ONCE to data/splits.csv and then committed. Re-running is
refused unless --force is given, so every member trains on the identical split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RESULTS_DIR, SEED, SPLIT_CSV, KAGGLE_ID

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download():
    import kagglehub  # imported lazily
    return Path(kagglehub.dataset_download(KAGGLE_ID))


def find_class_root(base: Path, min_classes: int = 10) -> Path:
    """Locate the directory whose sub-folders are the class folders."""
    candidates = [base] + sorted((p for p in base.rglob("*") if p.is_dir()),
                                 key=lambda p: len(p.parts))
    for d in candidates:
        subs = [s for s in d.iterdir()
                if s.is_dir() and any(f.suffix.lower() in EXTS for f in s.iterdir() if f.is_file())]
        if len(subs) >= min_classes:
            return d
    raise FileNotFoundError(f"No folder with >= {min_classes} class sub-folders under {base}")


def scan(root: Path) -> pd.DataFrame:
    rows, corrupt = [], []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:  # reopen: verify() invalidates the handle
                    w, h = im.size
                rows.append({"path": f.relative_to(root).as_posix(), "label": cls_dir.name,
                             "width": w, "height": h,
                             "md5": hashlib.md5(f.read_bytes()).hexdigest()})
            except Exception:
                corrupt.append(str(f))
    if corrupt:
        print(f"[warn] {len(corrupt)} unreadable images skipped, e.g. {corrupt[:3]}")
    return pd.DataFrame(rows)


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    return (pd.concat([train.assign(split="train"), val.assign(split="val"),
                       test.assign(split="test")])
            .sort_values("path").reset_index(drop=True))


def eda(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    counts.to_csv(out_dir / "class_counts.csv")

    ax = counts["total"].plot(kind="bar", figsize=(9, 4), color="#3b7dd8")
    ax.set_title("Class distribution (full dataset)")
    ax.set_ylabel("images")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()

    stats = {
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "imbalance_ratio_max_over_min": round(float(counts["total"].max() / counts["total"].min()), 2),
        "width": df["width"].describe().round(1).to_dict(),
        "height": df["height"].describe().round(1).to_dict(),
        "split_sizes": df["split"].value_counts().to_dict(),
        # data-leakage check: byte-identical files, and those landing in different splits
        "exact_duplicate_files": int(df["md5"].duplicated().sum()),
        "duplicate_groups_spanning_splits": int((df.groupby("md5")["split"].nunique() > 1).sum()),
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch from Kaggle via kagglehub")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min_classes", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen split")
    a = ap.parse_args()

    if SPLIT_CSV.exists() and not a.force:
        raise SystemExit(f"{SPLIT_CSV} already exists (split is frozen). Use --force only if the whole team agrees.")

    base = download() if a.download else Path(a.data_root or "")
    if not base.exists():
        raise SystemExit("Provide --data_root or use --download")
    root = find_class_root(base, a.min_classes)
    print(f"Class root: {root}")

    df = make_split(scan(root), a.seed)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(SPLIT_CSV, index=False)
    (DATA_DIR / "data_root.txt").write_text(str(root))
    eda(df, RESULTS_DIR / "eda")
    print(f"Saved frozen split -> {SPLIT_CSV}")


if __name__ == "__main__":
    main()
