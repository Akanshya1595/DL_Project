"""
Model 1 - Multilayer Perceptron (classical / fully-connected baseline).
Owner: Akanshya Chakraborty (230953042)

Images resized to 64x64 and flattened; three dense hidden layers with ReLU,
batch normalisation and dropout. Establishes the floor when spatial structure is discarded.
"""
import torch.nn as nn

IMG_SIZE = 64


class MLP(nn.Module):
    def __init__(self, n_classes, img_size=IMG_SIZE, hidden=(1024, 512, 256), p=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], 3 * img_size * img_size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build(n_classes, pretrained=True):
    """Returns (model, training stages). `pretrained` is ignored (random init)."""
    m = MLP(n_classes)
    return m, [dict(name="full", params=list(m.parameters()), lr=1e-3, epoch_frac=1.0)]
