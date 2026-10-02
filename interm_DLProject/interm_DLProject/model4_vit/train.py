"""
Train Model 4 (Vision Transformer):
  python -m model4_vit.train --epochs 10 --batch_size 32
  python -m model4_vit.train --epochs 10 --variant swin      # Swin-Tiny fallback
"""
import argparse

from shared.trainer import add_common_args, run
from .model import IMG_SIZE, build_swin, build_vit


def main():
    ap = argparse.ArgumentParser(description="Model 4 - Vision Transformer (ViT-B/16, Swin-T fallback)")
    add_common_args(ap, epochs=10, batch_size=32)
    ap.add_argument("--variant", choices=["vit", "swin"], default="vit")
    a = ap.parse_args()
    run(a.variant, build_vit if a.variant == "vit" else build_swin, IMG_SIZE, a)


if __name__ == "__main__":
    main()
