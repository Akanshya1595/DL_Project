"""
Train Model 2 (CNN from scratch):
  python -m model2_cnn.train --epochs 40
  python -m model2_cnn.train --epochs 40 --no_aug     # augmentation ablation
"""
import argparse

from shared.trainer import add_common_args, run
from .model import IMG_SIZE, build


def main():
    ap = argparse.ArgumentParser(description="Model 2 - CNN from scratch")
    add_common_args(ap, epochs=40, batch_size=64)
    run("cnn", build, IMG_SIZE, ap.parse_args())


if __name__ == "__main__":
    main()
