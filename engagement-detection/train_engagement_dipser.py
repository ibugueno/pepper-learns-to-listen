#!/usr/bin/env python3
# train_engagement_dipser_noaudio.py
# Engagement detection over DIPSER using ResNet18 + GRU/TCN.

import os
import csv
import glob
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


# ----------------------------------------------------
# Utils
# ----------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class DIPSERDataset(Dataset):
    """
    Manifest CSV columns:
        clip_id,frames_dir,labels_path

    - frames_dir: folder with ordered face crops 000001.jpg, ...
    - labels_path: .npy array of shape [N] with values in {0,1,2}
    """

    def __init__(self, manifest_path, num_frames, img_size, train, n_classes, fps=25.0, augment=True):
        self.entries = self._read_manifest(manifest_path)
        self.num_frames = num_frames
        self.train = train
        self.n_classes = n_classes
        self.fps = fps

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

    def _read_manifest(self, path):
        out = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.jpg")))
                if len(frames) == 0:
                    frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.png")))
                if len(frames) == 0:
                    continue
                labels = row["labels_path"]
                if not os.path.isfile(labels):
                    continue
                out.append({"clip_id": row["clip_id"], "frames": frames, "labels": labels})
        return out

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        frames = e["frames"]
        labels = np.load(e["labels"]).astype(np.int64)

        max_start = max(0, len(frames) - self.num_frames)
        start = random.randint(0, max_start) if self.train else (max_start // 2)

        imgs = []
        for i in range(start, start + self.num_frames):
            i = min(i, len(frames) - 1)
            img = Image.open(frames[i]).convert("RGB")
            imgs.append(self.tf(img))
        imgs = torch.stack(imgs, dim=0)  # [T, C, H, W]

        y = labels[start:start + self.num_frames]
        if len(y) < self.num_frames:
            pad = self.num_frames - len(y)
            y = np.pad(y, (0, pad), mode="edge")
        y = torch.tensor(y, dtype=torch.long)  # [T]

        return imgs, y, {"clip_id": e["clip_id"], "start": start}


# ----------------------------------------------------
# Model (ResNet18 + GRU/TCN)
# ----------------------------------------------------
class VisualBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x_btchw):
        B, T = x_btchw.shape[:2]
        x = x_btchw.reshape(B * T, *x_btchw.shape[2:])
        f = self.backbone(x)
        f = self.proj(f)
        return f.view(B, T, -1)


class TemporalHeadGRU(nn.Module):
    def __init__(self, in_dim, hidden, layers, n_classes):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden,
                          num_layers=layers, batch_first=True,
                          bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, z_btd):
        h, _ = self.gru(z_btd)           # [B, T, 2H]
        return self.fc(h)                # [B, T, C]


class TemporalHeadTCN(nn.Module):
    def __init__(self, in_dim, hidden, layers, n_classes):
        super().__init__()
        ch = [in_dim] + [hidden] * layers
        blocks = []
        for i in range(layers):
            dil = 2 ** i
            blocks += [
                nn.Conv1d(ch[i], ch[i+1], kernel_size=3,
                          padding=dil, dilation=dil),
                nn.ReLU(inplace=True),
            ]
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, z_btd):
        x = z_btd.transpose(1, 2)       # [B, D, T]
        y = self.tcn(x).transpose(1, 2) # [B, T, H]
        return self.fc(y)


class EngageModel(nn.Module):
    def __init__(self, embed_dim, temporal, hidden, layers, n_classes):
        super().__init__()
        self.visual = VisualBackbone(out_dim=embed_dim)
        if temporal == "gru":
            self.temporal = TemporalHeadGRU(embed_dim, hidden, layers, n_classes)
        else:
            self.temporal = TemporalHeadTCN(embed_dim, hidden, layers, n_classes)

    def forward(self, x_btchw):
        v = self.visual(x_btchw)   # [B, T, D]
        return self.temporal(v)    # [B, T, C]


# ----------------------------------------------------
# Training
# ----------------------------------------------------
def loss_fn(logits, labels):
    B, T, C = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B*T, C), labels.reshape(B*T)
    )


@torch.no_grad()
def frame_metrics(logits, labels, C):
    pred = logits.argmax(dim=-1).reshape(-1)
    t = labels.reshape(-1)
    acc = (pred == t).float().mean().item()
    # macro-F1
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (t == c)).sum().item()
        fp = ((pred == c) & (t != c)).sum().item()
        fn = ((pred != c) & (t == c)).sum().item()
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2*prec*rec/(prec+rec))
    return acc, float(np.mean(f1s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-classes", type=int, default=3)

    p.add_argument("--temporal", choices=["gru","tcn"], default="gru")
    p.add_argument("--temporal-hidden", type=int, default=256)
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=256)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--opt", choices=["adamw","adam","sgd"], default="adamw")

    p.add_argument("--out-dir", default="./runs/dipser_noaudio")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DIPSERDataset(args.train_manifest, args.num_frames, args.img_size, True, args.n_classes)
    val_ds   = DIPSERDataset(args.val-manifest, args.num_frames, args.img_size, False, args.n_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EngageModel(args.embed_dim, args.temporal, args.temporal-hidden, args.temporal-layers, args.n_classes).to(device)
    print("Trainable params:", count_params(model))

    if args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_f1 = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        for frames, labels, _ in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(frames)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Eval
        model.eval()
        val_loss = 0; acc=0; f1=0; n=0
        with torch.no_grad():
            for frames, labels, _ in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                logits = model(frames)
                val_loss += loss_fn(logits, labels).item()*frames.size(0)
                a, f = frame_metrics(logits, labels, args.n_classes)
                acc += a*frames.size(0)
                f1 += f*frames.size(0)
                n += frames.size(0)
        val_loss/=n; acc/=n; f1/=n

        print(f"[{epoch}] val_loss={val_loss:.4f} acc={acc:.3f} macroF1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.out_dir}/best.pt")

    print("Done. Best macro-F1 =", best_f1)


if __name__ == "__main__":
    main()
