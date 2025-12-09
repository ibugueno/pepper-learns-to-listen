#!/usr/bin/env python3
# 3_eval_engagement_dipser.py
#
# Generate a 6x2 figure (6 columns, 2 rows) with examples from both classes:
#   - Row 0: label = 1 (high engagement)
#   - Row 1: label = 0 (low engagement)
#
# Uses ONLY ground-truth labels from the CSV, no model prediction.

import os
import csv
import argparse
from typing import List, Dict

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models as tvm  # kept in case you want to extend later


# ---------------- Dataset (same logic as train) ----------------
class DipserSingleFrameDataset(Dataset):
    """
    CSV must have columns:
      image_path,label,clip_id,datetime,x1,y1,x2,y2

    Here label is already binarized in the CSV as:
      0 = low engagement
      1 = high engagement
    """
    def __init__(self, csv_path: str, img_size: int, train: bool,
                 n_classes: int, augment: bool = True):
        self.samples = self._read_csv(csv_path)
        self.n_classes = n_classes

        normalize = T.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))

        if train and augment:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=10,
                               translate=(0.05, 0.05),
                               scale=(0.95, 1.05)),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                normalize,
            ])

    def _read_csv(self, path: str) -> List[Dict]:
        out = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["image_path"]

                # bbox columns
                try:
                    x1 = int(row["x1"])
                    y1 = int(row["y1"])
                    x2 = int(row["x2"])
                    y2 = int(row["y2"])
                except KeyError:
                    raise ValueError(
                        f"CSV {path} must have columns x1,y1,x2,y2. "
                        f"Found: {reader.fieldnames}"
                    )

                # label already binarized (0/1) in *_bbox.csv
                lbl = int(row["label"])

                if not os.path.isfile(img):
                    continue

                out.append({
                    "image_path": img,
                    "label": lbl,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
        if not out:
            raise RuntimeError(f"No samples in {path}")
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")

        # face crop
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        img = img.crop((x1, y1, x2, y2))

        x = self.tf(img)
        y = torch.tensor(s["label"], dtype=torch.long)
        return x, y


# (Model class kept for compatibility / future use, but NOT used here)
class VisualClassifier(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            feat_dim = m.fc.in_features
            m.fc = nn.Linear(feat_dim, n_classes)
            self.net = m

        elif backbone == "vit_b16":
            m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
            feat_dim = m.heads.head.in_features
            m.heads.head = nn.Linear(feat_dim, n_classes)
            self.net = m
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--save", type=str, default="engagement_examples.png")
    args = parser.parse_args()

    # ---------------- Build dataset ----------------
    ds = DipserSingleFrameDataset(
        csv_path=args.val_csv,
        img_size=args.img_size,
        train=False,
        n_classes=args.n_classes,
        augment=False
    )

    print(f"[INFO] Loaded {len(ds.samples)} samples from {args.val_csv}")

    # ---------------- Split indices by class label ----------------
    engaged_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 1]
    not_engaged_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 0]

    print(f"[INFO] Found {len(engaged_indices)} engaged samples (label=1)")
    print(f"[INFO] Found {len(not_engaged_indices)} not engaged samples (label=0)")

    if len(engaged_indices) == 0 or len(not_engaged_indices) == 0:
        raise RuntimeError("Need at least one sample of each class to create the figure.")

    # Take first up to 6 from each class
    engaged_indices = engaged_indices[:6]
    not_engaged_indices = not_engaged_indices[:6]

    # ---------------- Create 6x2 figure ----------------
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # Row 0: engaged
    for col in range(6):
        ax = axes[0, col]
        if col < len(engaged_indices):
            idx = engaged_indices[col]
            s = ds.samples[idx]
            img_path = s["image_path"]
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
            img_full = Image.open(img_path).convert("RGB")
            face = img_full.crop((x1, y1, x2, y2))
            ax.imshow(face)
            ax.set_title("Engaged (GT=1)")
        ax.axis("off")

    # Row 1: not engaged
    for col in range(6):
        ax = axes[1, col]
        if col < len(not_engaged_indices):
            idx = not_engaged_indices[col]
            s = ds.samples[idx]
            img_path = s["image_path"]
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
            img_full = Image.open(img_path).convert("RGB")
            face = img_full.crop((x1, y1, x2, y2))
            ax.imshow(face)
            ax.set_title("Not engaged (GT=0)")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=200)
    print(f"[DONE] Figure saved to: {args.save}")


if __name__ == "__main__":
    main()
