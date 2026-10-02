"""
Model 3 - ResNet-50 with transfer learning (ImageNet pre-trained).
Owner: Krish Uppal (230953478)

Two-stage fine-tuning: (1) backbone frozen, classification head trained;
(2) final residual stages (layer3, layer4) unfrozen together with the head at a reduced learning rate.
"""
import torch.nn as nn
from torchvision import models as tvm

IMG_SIZE = 224


def build(n_classes, pretrained=True):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_classes))
    stages = [
        dict(name="stage1_head", params=list(m.fc.parameters()), lr=1e-3, epoch_frac=0.3),
        dict(name="stage2_layer3-4+head", lr=1e-4, epoch_frac=0.7,
             params=list(m.layer3.parameters()) + list(m.layer4.parameters()) + list(m.fc.parameters())),
    ]
    return m, stages
