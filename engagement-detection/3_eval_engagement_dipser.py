#!/usr/bin/env python3
# 3_eval_engagement_dipser.py
#
# Generate a 6x2 figure (6 columns, 2 rows) with correct predictions:
#   - Row 0: high engagement (label=1) correctly predicted
#   - Row 1: low engagement (label=0) correctly predicted
#
# Uses the SAME model definition and normalization as in 2_train_engagement_dipser.py
# (VisualClassifier with torchvision.resnet18 / torchvision.vit_b_16)

import os
import csv
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import models as tvm

import matplotlib.pyplot as plt


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


# ---------------- Model (same as train) ----------------
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


@torch.no_grad()
def predict_one(model: nn.Module, x: torch.Tensor, device: torch.device):
    x = x.unsqueeze(0).to(device)   # [1, C, H, W]
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1).item()
    conf = prob.max().item()
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--backbone", type=str,
                        choices=["resnet18", "vit_b16"],
                        default=None,
                        help="If None, will be taken from checkpoint args.")
    parser.add_argument("--save", type=str, default="engagement_examples.png")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---------------- Load checkpoint ----------------
    ckpt = torch.load(args.checkpoint, map_location=device)

    # If backbone not given, infer from saved args
    if args.backbone is None:
        ckpt_args = ckpt.get("args", {})
        backbone = ckpt_args.get("backbone", "vit_b16")
    else:
        backbone = args.backbone

    print(f"[INFO] Using backbone: {backbone}")

    model = VisualClassifier(n_classes=args.n_classes, backbone=backbone)
    model.load_state_dict(ckpt["model"])   # <- this now matches
    model.to(device)
    model.eval()

    # ---------------- Build dataset ----------------
    ds = DipserSingleFrameDataset(
        csv_path=args.val_csv,
        img_size=args.img_size,
        train=False,
        n_classes=args.n_classes,
        augment=False
    )

    # We'll reuse ds.tf for preprocessing
    tf = ds.tf

    correct_engaged = []   # list of (idx, conf)
    correct_not = []       # list of (idx, conf)

    # ---------------- Iterate over samples ----------------
    print(f"[INFO] Evaluating over {len(ds.samples)} samples...")
    for idx, s in enumerate(ds.samples):
        img_path = s["image_path"]
        label = s["label"]
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]

        # load + crop
        img_full = Image.open(img_path).convert("RGB")
        face = img_full.crop((x1, y1, x2, y2))

        x = tf(face)   # normalized tensor
        pred, conf = predict_one(model, x, device)

        if pred == label:
            if label == 1:
                correct_engaged.append((idx, conf))
            else:
                correct_not.append((idx, conf))

    print(f"[INFO] Correct engaged: {len(correct_engaged)}")
    print(f"[INFO] Correct not engaged: {len(correct_not)}")

    if not correct_engaged and not correct_not:
        raise RuntimeError("No correct predictions found, cannot create figure.")

    # ---------------- Pick top-6 per class by confidence ----------------
    correct_engaged.sort(key=lambda t: t[1], reverse=True)
    correct_not.sort(key=lambda t: t[1], reverse=True)

    top_engaged = correct_engaged[:6]
    top_not = correct_not[:6]

    # ---------------- Create 6x2 figure ----------------
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # Row 0: engaged
    for col in range(6):
        ax = axes[0, col]
        if col < len(top_engaged):
            idx, conf = top_engaged[col]
            s = ds.samples[idx]
            img_path = s["image_path"]
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
            img_full = Image.open(img_path).convert("RGB")
            face = img_full.crop((x1, y1, x2, y2))
            ax.imshow(face)
            ax.set_title(f"Engaged (p={conf:.2f})")
        ax.axis("off")

    # Row 1: not engaged
    for col in range(6):
        ax = axes[1, col]
        if col < len(top_not):
            idx, conf = top_not[col]
            s = ds.samples[idx]
            img_path = s["image_path"]
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
            img_full = Image.open(img_path).convert("RGB")
            face = img_full.crop((x1, y1, x2, y2))
            ax.imshow(face)
            ax.set_title(f"Not engaged (p={conf:.2f})")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=200)
    print(f"[DONE] Figure saved to: {args.save}")


if __name__ == "__main__":
    main()
