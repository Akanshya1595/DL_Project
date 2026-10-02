"""Train Model 1 (MLP):  python -m model1_mlp.train --epochs 30"""
import argparse

from shared.trainer import add_common_args, run
from .model import IMG_SIZE, build


def main():
    ap = argparse.ArgumentParser(description="Model 1 - MLP baseline")
    add_common_args(ap, epochs=30, batch_size=64)
    run("mlp", build, IMG_SIZE, ap.parse_args())


if __name__ == "__main__":
    main()
