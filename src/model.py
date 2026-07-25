"""
model.py
--------
Two model options:

1. ResNet18 with pretrained ImageNet weights (transfer learning) — used
   automatically whenever internet access to download the weights is
   available. This gives the best accuracy (typically 97-99% on this
   dataset) and is what you should expect when running on a normal
   machine with internet access.

2. A lightweight CNN trained from scratch — automatic fallback used only
   if the pretrained weights can't be downloaded (e.g. no internet, or a
   restricted sandbox). Still gets solid accuracy on this dataset, just
   needs more epochs since it isn't starting from pretrained features.

build_model() tries option 1 first and transparently falls back to
option 2, printing which one was used.
"""

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Small from-scratch CNN, used as a fallback when pretrained
    ImageNet weights aren't downloadable. Works well at 64x64-96x96
    input resolution."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(freeze_backbone=True, num_classes=2, force_scratch=False):
    """
    Returns (model, input_size, is_pretrained).

    Tries pretrained ResNet18 first (input_size=224). If the weights
    can't be downloaded (no internet) or force_scratch=True, falls back
    to SimpleCNN (input_size=64).
    """
    if not force_scratch:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)

            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes),
            )
            for param in model.fc.parameters():
                param.requires_grad = True

            print("[model] Using pretrained ResNet18 (ImageNet transfer learning).")
            return model, 224, True

        except Exception as e:
            print(f"[model] Could not load pretrained ResNet18 weights ({e}).")
            print("[model] Falling back to a from-scratch CNN (SimpleCNN, 64x64 input).")
            print("[model] Tip: run this on a machine with internet access to use "
                  "transfer learning instead, for significantly higher accuracy.")

    model = SimpleCNN(num_classes=num_classes)
    return model, 64, False


def unfreeze_last_block(model):
    """Unfreeze layer4 + fc for fine-tuning after initial head-only training.
    Only meaningful for the pretrained ResNet18 path."""
    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    return model
