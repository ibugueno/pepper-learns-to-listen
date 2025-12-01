#!/usr/bin/env python3
# train_ava_activespeaker_imgonly.py
# Active Speaker Detection sólo con imagen (face crops) + modelo temporal (GRU/TCN).
# Manifest CSV columnas: clip_id,frames_dir,labels_path

import os
import csv
import glob
import math
import time
import json
import argparse
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision import models as tvm

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Dataset (imagen sólo)
# ----------------------------
class AVAImgOnlyDataset(Dataset):
    """
    Cada fila en el manifest:
      clip_id,frames_dir,labels_path

    Suposiciones:
      - frames_dir contiene las caras recortadas: una imagen por frame (jpg/png).
      - labels_path es un .npy con vector 0/1 por frame (NOT_SPEAKING / SPEAKING),
        alineado con los frames del directorio.
    Entrenamos ventanas temporales de longitud num_frames (T).
    """
    def __init__(
        self,
        manifest_path: str,
        num_frames: int,
        img_size: int,
        train: bool,
        fps: float = 25.0,
        augment: bool = True,
    ):
        self.items = self._read_manifest(manifest_path)
        self.num_frames = num_frames
        self.train = train
        self.fps = fps

        # Transforms de imagen (sobre el crop, no sobre la imagen completa)
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if train and augment:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                normalize
            ])

    def _read_manifest(self, path: str) -> List[Dict]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            req = {"clip_id", "frames_dir", "labels_path"}
            if not req.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest must contain columns: {req}, got {reader.fieldnames}")
            for row in reader:
                frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.jpg")))
                if len(frames) == 0:
                    frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.png")))
                if len(frames) == 0:
                    continue
                if not os.path.isfile(row["labels_path"]):
                    continue
                out.append({
                    "clip_id": row["clip_id"],
                    "frames": frames,
                    "labels_path": row["labels_path"]
                })
        if len(out) == 0:
            raise ValueError("No valid rows found in manifest.")
        return out

    def __len__(self):
        return len(self.items)

    def _load_frames_window(self, frames: List[str], start: int, T: int) -> torch.Tensor:
        imgs = []
        for i in range(start, start + T):
            i_clamped = min(i, len(frames) - 1)
            img = Image.open(frames[i_clamped]).convert("RGB")
            imgs.append(self.tf(img))
        # [T, C, H, W]
        return torch.stack(imgs, dim=0)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = item["frames"]
        labels = np.load(item["labels_path"]).astype(np.float32).reshape(-1)

        # Por seguridad, recorta/ajusta labels al nº de frames
        n = min(len(frames), len(labels))
        frames = frames[:n]
        labels = labels[:n]

        # Escoger ventana
        max_start = max(0, len(frames) - self.num_frames)
        if self.train:
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = max_start // 2  # centro para validación

        # Ventana visual: [T, C, H, W]
        v = self._load_frames_window(frames, start, self.num_frames)

        # Ventana de labels: [T]
        end = start + self.num_frames
        y = labels[start:min(end, len(labels))]
        if len(y) < self.num_frames:
            pad = self.num_frames - len(y)
            y = np.pad(y, (0, pad), mode="edge")
        y = torch.from_numpy(y)  # float tensor [T]

        meta = {
            "clip_id": item["clip_id"],
            "start": start
        }
        return v, y, meta


# ----------------------------
# Modelo: VisualBackbone + Temporal (GRU/TCN)
# ----------------------------
class VisualBackbone(nn.Module):
    """CNN por frame (ResNet18) que devuelve un embedding."""
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        # x: [B*T, C, H, W]
        f = self.backbone(x)          # [B*T, feat]
        f = self.proj(f)              # [B*T, out_dim]
        return f


class TemporalHeadGRU(nn.Module):
    """Bi-GRU temporal + clasificador por frame."""
    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # logits por frame
        )

    def forward(self, x):
        # x: [B, T, D]
        y, _ = self.gru(x)            # [B, T, 2H]
        logits = self.head(y).squeeze(-1)  # [B, T]
        return logits


class TemporalHeadTCN(nn.Module):
    """Temporal Convolutional Network (1D dilatada) + clasificador por frame."""
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ch = [in_dim] + [hidden] * layers
        blocks = []
        for i in range(layers):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation // 2
            blocks += [
                nn.Conv1d(ch[i], ch[i+1], kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ]
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        y = self.tcn(x)               # [B, H, T]
        y = y.transpose(1, 2)         # [B, T, H]
        logits = self.head(y).squeeze(-1)  # [B, T]
        return logits


class AVASpkImgModel(nn.Module):
    """Solo imagen: backbone visual + cabeza temporal."""
    def __init__(self, img_embed_dim=256, temporal="gru", temporal_hidden=256):
        super().__init__()
        self.v_backbone = VisualBackbone(out_dim=img_embed_dim, pretrained=True)
        if temporal == "gru":
            self.temporal = TemporalHeadGRU(in_dim=img_embed_dim, hidden=temporal_hidden)
        elif temporal == "tcn":
            self.temporal = TemporalHeadTCN(in_dim=img_embed_dim, hidden=temporal_hidden)
        else:
            raise ValueError("temporal must be 'gru' or 'tcn'")

    def forward(self, frames_btchw):
        """
        frames_btchw: [B, T, C, H, W]
        """
        B, T = frames_btchw.shape[:2]
        x = frames_btchw.reshape(B*T, *frames_btchw.shape[2:])
        v = self.v_backbone(x)                   # [B*T, D]
        v = v.view(B, T, -1)                     # [B, T, D]
        logits = self.temporal(v)                # [B, T]
        return logits


# ----------------------------
# Train / Eval (igual que antes, sin audio)
# ----------------------------
def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def frame_metrics(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    t = targets.float()

    tp = (preds * t).sum().item()
    fp = (preds * (1 - t)).sum().item()
    fn = ((1 - preds) * t).sum().item()
    tn = ((1 - preds) * (1 - t)).sum().item()

    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def train_one_epoch(model, loader, optimizer, device, scaler=None, max_norm=1.0):
    model.train()
    total_loss = 0.0
    m_acc = m_prec = m_rec = m_f1 = 0.0
    n_samples = 0

    for frames, labels, _ in loader:
        frames = frames.to(device)          # [B, T, C, H, W]
        labels = labels.to(device)          # [B, T]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                logits = model(frames)
                loss = bce_logits_loss(logits, labels)
            scaler.scale(loss).backward()
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = bce_logits_loss(logits, labels)
            loss.backward()
            if max_norm and max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_loss += loss.item() * frames.size(0)
        mets = frame_metrics(logits.detach(), labels.detach())
        m_acc += mets["acc"] * frames.size(0)
        m_prec += mets["prec"] * frames.size(0)
        m_rec += mets["rec"] * frames.size(0)
        m_f1 += mets["f1"] * frames.size(0)
        n_samples += frames.size(0)

    return {
        "loss": total_loss / n_samples,
        "acc": m_acc / n_samples,
        "prec": m_prec / n_samples,
        "rec": m_rec / n_samples,
        "f1": m_f1 / n_samples
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    m_acc = m_prec = m_rec = m_f1 = 0.0
    n_samples = 0

    for frames, labels, _ in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames)
        loss = bce_logits_loss(logits, labels)

        total_loss += loss.item() * frames.size(0)
        mets = frame_metrics(logits, labels)
        m_acc += mets["acc"] * frames.size(0)
        m_prec += mets["prec"] * frames.size(0)
        m_rec += mets["rec"] * frames.size(0)
        m_f1 += mets["f1"] * frames.size(0)
        n_samples += frames.size(0)

    return {
        "loss": total_loss / n_samples,
        "acc": m_acc / n_samples,
        "prec": m_prec / n_samples,
        "rec": m_rec / n_samples,
        "f1": m_f1 / n_samples
    }


def create_loaders(args):
    train_ds = AVAImgOnlyDataset(
        manifest_path=args.train_manifest,
        num_frames=args.num_frames,
        img_size=args.img_size,
        train=True,
        fps=args.fps,
        augment=not args.no_img_aug,
    )
    val_ds = AVAImgOnlyDataset(
        manifest_path=args.val_manifest,
        num_frames=args.num_frames,
        img_size=args.img_size,
        train=False,
        fps=args.fps,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader


def build_optimizer(model, args):
    if args.opt == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")


def save_checkpoint(state: dict, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))


# ----------------------------
# Argparse / Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Image-only Active Speaker Detection (AVA-style)")

    # Data
    p.add_argument("--train-manifest", type=str, required=True)
    p.add_argument("--val-manifest", type=str, required=True)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--num-frames", type=int, default=32, help="Temporal window length T")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-img-aug", action="store_true", help="Disable image augmentations")

    # Model
    p.add_argument("--img-embed-dim", type=int, default=256)
    p.add_argument("--temporal", type=str, default="gru", choices=["gru", "tcn"])
    p.add_argument("--temporal-hidden", type=int, default=256)

    # Train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="./runs/ava_spk_img")
    p.add_argument("--save-every", type=int, default=0)

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_loaders(args)

    model = AVASpkImgModel(
        img_embed_dim=args.img_embed_dim,
        temporal=args.temporal,
        temporal_hidden=args.temporal_hidden
    ).to(device)
    print(f"Trainable params: {count_params(model):,}")

    optimizer = build_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_f1 = 0.0
    history = {"args": vars(args), "epochs": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler, args.max_grad_norm)
        va = validate(model, val_loader, device)
        dt = time.time() - t0

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train: loss={tr['loss']:.4f} acc={tr['acc']:.3f} prec={tr['prec']:.3f} rec={tr['rec']:.3f} f1={tr['f1']:.3f} | "
              f"val:   loss={va['loss']:.4f} acc={va['acc']:.3f} prec={va['prec']:.3f} rec={va['rec']:.3f} f1={va['f1']:.3f} | "
              f"time={dt:.1f}s")

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": va,
                "args": vars(args)
            }, args.out_dir, "best.pt")

        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": va,
                "args": vars(args)
            }, args.out_dir, f"epoch_{epoch:03d}.pt")

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": va,
            "args": vars(args)
        }, args.out_dir, "last.pt")

        history["epochs"].append({
            "epoch": epoch,
            "train": tr,
            "val": va,
            "time_sec": dt
        })
        save_json(os.path.join(args.out_dir, "history.json"), history)

    print(f"Done. Best val F1: {best_f1:.3f}. Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
