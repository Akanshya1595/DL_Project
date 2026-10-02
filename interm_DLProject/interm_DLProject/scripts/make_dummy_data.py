"""Create a tiny synthetic image dataset to smoke-test the pipeline without downloading Kaggle data.
   python scripts/make_dummy_data.py /tmp/dummy && python -m shared.data_prep --data_root /tmp/dummy --min_classes 12 --force
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ["battery","biological","brown-glass","cardboard","clothes","green-glass",
           "metal","paper","plastic","shoes","trash","white-glass"]
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dummy")
rng = np.random.default_rng(0)
for i, c in enumerate(CLASSES):
    (root / c).mkdir(parents=True, exist_ok=True)
    n = 40 + 10 * (i % 4)
    for k in range(n):
        base = np.full((96, 128, 3), (i * 20) % 255, dtype=np.uint8)
        noise = rng.integers(0, 60, base.shape, dtype=np.uint8)
        Image.fromarray(base + noise).save(root / c / f"{k}.jpg")
print("dummy data at", root)
