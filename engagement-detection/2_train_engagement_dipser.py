#!/usr/bin/env python3
# train_engagement_dipser_single.py
#
# Single-frame engagement classifier sobre DIPSER.
# Lee CSV con columnas: image_path,label,clip_id,datetime
# Backbone: resnet18 o vit_b16 (solo imagen).

import os
import csv
import time
import json
import random
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision import models as tvm


# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------- Dataset ----------------
class DipserSingleFrameDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int, train: bool, n_classes: int, augment: bool = True):
        self.samples = self._read_csv(csv_path)
        self.n_classes = n_classes

        normalize = T.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
        if train and augment:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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
                lbl = int(row["label"]) - 1
                if not os.path.isfile(img):
                    continue
                out.append({"image_path": img, "label": lbl})
        if not out:
            raise RuntimeError(f"No samples in {path}")
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(s["label"], dtype=torch.long)
        return x, y


# ---------------- Model ----------------
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
            raise ValueError(f"Backbone no soportado: {backbone}")

    def forward(self, x):
        return self.net(x)


# ---------------- Train / Eval ----------------
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_acc += (preds == y).float().sum().item()
        n += x.size(0)

    return {
        "loss": total_loss / n,
        "acc": total_acc / n
    }


@torch.no_grad()
def validate(model, loader, device, n_classes: int):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    # opcional: macro-F1
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_acc += (preds == y).float().sum().item()
        n += x.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # macro-F1 simple
    f1s = []
    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))

    macro_f1 = float(np.mean(f1s))

    return {
        "loss": total_loss / n,
        "acc": total_acc / n,
        "macro_f1": macro_f1
    }


# ---------------- Main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="DIPSER single-frame engagement classifier")

    p.add_argument("--gpu", type=int, default=0,
                   help="Índice de GPU a usar (por ejemplo 0, 1, 2...).")

    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-classes", type=int, default=3)

    p.add_argument("--backbone", type=str,
                   choices=["resnet18", "vit_b16"],
                   default="vit_b16")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--opt", type=str, choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="./runs/dipser_single")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] Usando GPU {args.gpu} → {device}")
    else:
        device = torch.device("cpu")
        print("[WARN] No hay CUDA disponible, usando CPU.")

    # ViT necesita 224x224 típicamente
    if args.backbone == "vit_b16" and args.img_size < 224:
        print("[WARN] Ajustando img_size a 224 para vit_b16")
        args.img_size = 224

    train_ds = DipserSingleFrameDataset(args.train_csv, args.img_size, train=True,
                                        n_classes=args.n_classes, augment=True)
    val_ds   = DipserSingleFrameDataset(args.val_csv, args.img_size, train=False,
                                        n_classes=args.n_classes, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    model = VisualClassifier(n_classes=args.n_classes, backbone=args.backbone).to(device)
    print("Trainable params:", count_params(model))

    if args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_f1 = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    history = {"args": vars(args), "epochs": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va = validate(model, val_loader, device, args.n_classes)
        dt = time.time() - t0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train: loss={tr['loss']:.4f} acc={tr['acc']:.3f} | "
            f"val: loss={va['loss']:.4f} acc={va['acc']:.3f} f1={va['macro_f1']:.3f} | "
            f"time={dt:.1f}s"
        )

        history["epochs"].append({
            "epoch": epoch,
            "train": tr,
            "val": va,
            "time_sec": dt,
        })

        # guardar último
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": va,
                "args": vars(args),
            },
            os.path.join(args.out_dir, "last.pt"),
        )
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # guardar mejor por F1
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": va,
                    "args": vars(args),
                },
                os.path.join(args.out_dir, "best.pt"),
            )

    print(f"Done. Best val macro-F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
