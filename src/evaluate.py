"""
evaluate.py
-----------
Loads the best saved checkpoint and reports accuracy on the held-out test
split, plus a confusion matrix and classification report (precision,
recall, f1).

Usage:
    python src/evaluate.py
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

from dataset import CatDogDataset
from model import build_model

CLASSES = ["cat", "dog"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_path(relative):
    return os.path.join(PROJECT_ROOT, relative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=_default_path("data/dataset_info.csv"))
    parser.add_argument("--images-root", default=_default_path("data/images"))
    parser.add_argument("--checkpoint", default=_default_path("checkpoints/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    is_pretrained = checkpoint.get("is_pretrained", True)
    input_size = checkpoint.get("input_size", 224)

    model, _, _ = build_model(freeze_backbone=False, force_scratch=not is_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = CatDogDataset(args.csv, args.images_root, "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\nTest accuracy: {accuracy:.4f}\n")

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (acc={accuracy:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    outputs_dir = _default_path("outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix plot saved to {cm_path}")


if __name__ == "__main__":
    main()
