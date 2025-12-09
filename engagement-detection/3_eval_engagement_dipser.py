#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 3_eval_engagement_dipser.py
#
# Generate a 6x2 figure showing correct predictions:
#   - Top row: Engagement (label=1) correctly predicted
#   - Bottom row: No Engagement (label=0) correctly predicted

import os
import csv
import argparse

import torch
import timm
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

import numpy as np


def load_csv(csv_path):
    items = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            items.append(r)
    return items


def get_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@torch.no_grad()
def predict_one(model, img, device):
    img = img.unsqueeze(0).to(device)
    logits = model(img)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1).item()
    return pred, prob.max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--save", type=str, default="correct_examples.png")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Load model (same backbone used)
    # ------------------------------
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=args.n_classes
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    # ------------------------------
    # Load validation items
    # ------------------------------
    items = load_csv(args.val_csv)
    tfm = get_transform(args.img_size)

    correct_engaged = []
    correct_not = []

    # ------------------------------
    # Collect correct predictions
    # ------------------------------
    for r in items:
        img_path = r["image_path"]
        label = int(r["label"])

        img = Image.open(img_path).convert("RGB")
        pred, conf = predict_one(model, tfm(img), device)

        if pred == label and label == 1 and len(correct_engaged) < 6:
            correct_engaged.append((img, conf))
        if pred == label and label == 0 and len(correct_not) < 6:
            correct_not.append((img, conf))

        if len(correct_engaged) == 6 and len(correct_not) == 6:
            break

    # ------------------------------
    # Plot 6x2 Figure
    # ------------------------------
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # row 0: engaged
    for i, (img, conf) in enumerate(correct_engaged):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Engaged ({conf:.2f})")
        axes[0, i].axis("off")

    # row 1: not engaged
    for i, (img, conf) in enumerate(correct_not):
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Not Engaged ({conf:.2f})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"[DONE] Figure saved to: {args.save}")


if __name__ == "__main__":
    main()
