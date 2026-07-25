"""
train.py
--------
Trains a cat-vs-dog classifier.

Automatically uses transfer learning (pretrained ResNet18) if internet
access is available to download ImageNet weights — this gives the best
accuracy (typically 97-99%). If that download fails (e.g. no internet),
it automatically falls back to training a lightweight CNN from scratch.

Two-phase training when using the pretrained model:
  Phase 1: freeze backbone, train only the new classifier head (fast).
  Phase 2: unfreeze the last residual block, fine-tune with a low LR.

When falling back to the from-scratch CNN, all training happens in a
single phase (there's no pretrained backbone to unfreeze).

Usage:
    python src/train.py
    python src/train.py --epochs1 5 --epochs2 5 --batch-size 32
    python src/train.py --force-scratch   # skip the pretrained attempt entirely

Outputs:
    checkpoints/best_model.pt   -- best weights by validation accuracy
    outputs/training_curves.png -- loss/accuracy plot
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import CatDogDataset
from model import build_model, unfreeze_last_block

CLASSES = ["cat", "dog"]

# Resolve paths relative to the project root (parent of this file's folder)
# so the script behaves the same whether you run it from the project root
# or from inside src/ (e.g. VS Code's "Run Python File" button).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_path(relative):
    return os.path.join(PROJECT_ROOT, relative)


def get_transforms(input_size):
    train_tf = transforms.Compose([
        transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=_default_path("data/dataset_info.csv"))
    parser.add_argument("--images-root", default=_default_path("data/images"))
    parser.add_argument("--epochs1", type=int, default=5, help="head-only epochs (pretrained) or all epochs (scratch)")
    parser.add_argument("--epochs2", type=int, default=5, help="fine-tune epochs (pretrained only)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-scratch", action="store_true",
                         help="skip the pretrained-weights attempt and train the lightweight CNN from scratch")
    parser.add_argument("--resume", action="store_true",
                         help="resume from checkpoints/best_model.pt if it exists (continue training in chunks)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, input_size, is_pretrained = build_model(freeze_backbone=True, force_scratch=args.force_scratch)
    model = model.to(device)

    best_val_acc = 0.0
    best_model_path = _default_path("checkpoints/best_model.pt")
    if args.resume and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"[resume] Loaded existing checkpoint, val_acc so far: {best_val_acc:.4f}")
    train_tf, eval_tf = get_transforms(input_size)

    train_ds = CatDogDataset(args.csv, args.images_root, "train", transform=train_tf)
    val_ds = CatDogDataset(args.csv, args.images_root, "validation", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss()
    checkpoints_dir = _default_path("checkpoints")
    outputs_dir = _default_path("outputs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    history_path = os.path.join(checkpoints_dir, "history.json")

    if args.resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"[resume] Loaded history with {len(history['train_loss'])} prior epochs.")
    else:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_phase(epochs, lr, phase_name):
        nonlocal best_val_acc
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"[{phase_name}] Epoch {epoch}/{epochs} "
                  f"- train_loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"- val_loss {val_loss:.4f} acc {val_acc:.4f} "
                  f"- {elapsed:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "val_acc": val_acc,
                            "classes": CLASSES,
                            "input_size": input_size,
                            "is_pretrained": is_pretrained}, best_model_path)
                print(f"  -> new best model saved (val_acc={val_acc:.4f})")

            with open(history_path, "w") as f:
                json.dump(history, f)

    if is_pretrained:
        # Phase 1: train the classifier head only
        train_phase(args.epochs1, args.lr_head, "head")
        # Phase 2: unfreeze last block and fine-tune with a lower LR
        model = unfreeze_last_block(model)
        train_phase(args.epochs2, args.lr_finetune, "fine-tune")
    else:
        # No pretrained backbone to freeze/unfreeze -- just train everything
        train_phase(args.epochs1 + args.epochs2, args.lr_head, "scratch")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to {best_model_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    curves_path = os.path.join(outputs_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    print(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()
