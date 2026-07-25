"""
predict.py
----------
Run the trained model on a single image and print cat/dog with confidence.
This is the script to use for "does my model detect any picture correctly" —
point it at any photo, not just ones from the training/test set.

Usage:
    python src/predict.py --image path/to/some_photo.jpg
"""

import argparse
import os

import torch
from torchvision import transforms
from PIL import Image

from model import build_model

CLASSES = ["cat", "dog"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    is_pretrained = checkpoint.get("is_pretrained", True)
    input_size = checkpoint.get("input_size", 224)

    model, _, _ = build_model(freeze_backbone=False, force_scratch=not is_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    print(f"Prediction: {CLASSES[pred_idx]} (confidence {probs[pred_idx]:.2%})")
    print(f"  cat: {probs[0]:.2%} | dog: {probs[1]:.2%}")


if __name__ == "__main__":
    main()
