"""
Model 2 - Convolutional Neural Network trained from scratch (no pre-training).
Owner: Arushi Pandey (230911146)

VGG-style stack of five convolutional blocks (2 convs each) with batch normalisation,
max pooling and global average pooling; random initialisation, trained only on the project data.
The with-vs-without-augmentation ablation is run with the --no_aug flag.
"""
import torch.nn as nn

IMG_SIZE = 128


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class VGGStyleCNN(nn.Module):
    def __init__(self, n_classes, widths=(32, 64, 128, 256, 512), p=0.4):
        super().__init__()
        blocks, c = [], 3
        for w in widths:
            blocks.append(_block(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p),
                                  nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(n_classes, pretrained=True):
    """Returns (model, training stages). `pretrained` is ignored (random init)."""
    m = VGGStyleCNN(n_classes)
    return m, [dict(name="full", params=list(m.parameters()), lr=1e-3, epoch_frac=1.0)]
