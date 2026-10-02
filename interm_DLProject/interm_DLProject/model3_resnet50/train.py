"""Train Model 3 (ResNet-50 transfer learning):  python -m model3_resnet50.train --epochs 15"""
import argparse

from shared.trainer import add_common_args, run
from .model import IMG_SIZE, build


def main():
    ap = argparse.ArgumentParser(description="Model 3 - ResNet-50 transfer learning")
    add_common_args(ap, epochs=15, batch_size=64)
    run("resnet50", build, IMG_SIZE, ap.parse_args())


if __name__ == "__main__":
    main()
