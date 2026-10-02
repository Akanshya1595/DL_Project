"""
Model 4 - Vision Transformer ViT-B/16 (attention-based), fine-tuned from an ImageNet checkpoint.
Owner: Varun (230953538)

Stage 1 trains the new head only; stage 2 fine-tunes the whole network at a small learning rate.
Swin-Tiny is retained as the fallback should ViT-B/16 prove too heavy for the available compute.
"""
import torch.nn as nn
from torchvision import models as tvm

IMG_SIZE = 224


def build_vit(n_classes, pretrained=True):
    m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    m.heads = nn.Linear(m.heads.head.in_features, n_classes)
    return m, [
        dict(name="stage1_head", params=list(m.heads.parameters()), lr=1e-3, epoch_frac=0.2),
        dict(name="stage2_full", params=list(m.parameters()), lr=2e-5, epoch_frac=0.8),
    ]


def build_swin(n_classes, pretrained=True):
    m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    m.head = nn.Linear(m.head.in_features, n_classes)
    return m, [
        dict(name="stage1_head", params=list(m.head.parameters()), lr=1e-3, epoch_frac=0.2),
        dict(name="stage2_full", params=list(m.parameters()), lr=5e-5, epoch_frac=0.8),
    ]
