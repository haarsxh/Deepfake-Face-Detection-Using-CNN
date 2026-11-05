"""Model definitions: a small CNN baseline and a transfer-learning model."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Four conv blocks and a small classifier. A from-scratch baseline."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name="efficientnet", pretrained=True):
    """Return a model that outputs one logit per image (probability of 'real')."""
    if name == "simple":
        return SimpleCNN()
    if name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model
    raise ValueError(f"Unknown model: {name}")
