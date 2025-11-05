#!/usr/bin/env python3
# train_ava_activespeaker.py
# Audio-Visual Active Speaker Detection with temporal modeling (Bi-GRU or TCN).
# Manifest CSV columns: clip_id,frames_dir,audio_path,labels_path

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

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
except ImportError as e:
    raise SystemExit("This script requires 'torchaudio'. Install via: pip install torchaudio")

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
# Dataset
# ----------------------------
class AVAManifestDataset(Dataset):
    """
    Each row in the manifest:
      clip_id,frames_dir,audio_path,labels_path

    Assumptions:
      - frames_dir contains sequential images for the clip (one face crop per frame).
      - audio_path is a mono .wav
      - labels_path is a .npy int array of length >= number_of_frames (0/1 per frame)
    We train on fixed-length windows of size num_frames (T).
    """
    def __init__(
        self,
        manifest_path: str,
        num_frames: int,
        img_size: int,
        train: bool,
        fps: float = 25.0,
        sample_rate: int = 16000,
        n_mels: int = 64,
        mel_fmin: float = 50.0,
        mel_fmax: Optional[float] = None,
        augment: bool = True,
        audio_gain_jitter: float = 0.0,
    ):
        self.items = self._read_manifest(manifest_path)
        self.num_frames = num_frames
        self.train = train
        self.fps = fps
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.audio_gain_jitter = audio_gain_jitter

        # Image transforms
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

        # Audio transforms
        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=int(sample_rate / fps),  # ~one hop per video frame
            win_length=1024,
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax
        )
        self.a2db = AmplitudeToDB(stype="power")

    def _read_manifest(self, path: str) -> List[Dict]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            req = {"clip_id", "frames_dir", "audio_path", "labels_path"}
            if not req.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest must contain columns: {req}, got {reader.fieldnames}")
            for row in reader:
                frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.jpg")))
                if len(frames) == 0:
                    frames = sorted(glob.glob(os.path.join(row["frames_dir"], "*.png")))
                if len(frames) == 0:
                    continue
                if not os.path.isfile(row["audio_path"]):
                    continue
                if not os.path.isfile(row["labels_path"]):
                    continue
                out.append({
                    "clip_id": row["clip_id"],
                    "frames": frames,
                    "audio": row["audio_path"],
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
            i_clamped = min(i, len(frames)-1)
            img = Image.open(frames[i_clamped]).convert("RGB")
            imgs.append(self.tf(img))
        # [T, C, H, W]
        return torch.stack(imgs, dim=0)

    def _load_audio_mel_window(self, audio_path: str, start: int, T: int) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)  # [1, N] mono expected
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # to mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # optional gain jitter
        if self.train and self.audio_gain_jitter > 0:
            gain = 10 ** (random.uniform(-self.audio_gain_jitter, self.audio_gain_jitter) / 20.0)
            wav = wav * gain

        # Mel spectrogram -> [n_mels, time_steps]
        mel = self.melspec(wav)  # power mel
        mel_db = self.a2db(mel)  # log-mel
        # Align to frames: hop_length chosen ~ one step per frame
        # Now select mel frames [start:start+T]
        # Convert waveform frames to video-frame index via hop; but we crafted hop=sr/fps,
        # so mel_db.shape[-1] ~= num_video_frames for the full clip (if perfectly aligned).
        # We guard with clamping/padding.

        total_mel_steps = mel_db.shape[-1]
        # If total_mel_steps < total_video_frames, we will clamp/pad.
        # For our window: take [start, start+T)
        idx_end = start + T
        mel_slice = mel_db[:, :, start:min(idx_end, total_mel_steps)]  # [1, n_mels, t']
        if mel_slice.shape[-1] < T:
            pad = T - mel_slice.shape[-1]
            mel_slice = torch.nn.functional.pad(mel_slice, (0, pad), mode="replicate")
        # [T, n_mels] after transpose
        mel_per_frame = mel_slice.squeeze(0).transpose(0, 1)  # [t, n_mels]
        return mel_per_frame  # [T, n_mels]

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = item["frames"]
        labels = np.load(item["labels_path"]).astype(np.float32).reshape(-1)

        # Choose a window
        max_start = max(0, len(frames) - self.num_frames)
        if self.train:
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = max_start // 2  # center for val

        # Visual window: [T, C, H, W]
        v = self._load_frames_window(frames, start, self.num_frames)

        # Audio window aligned to frame indices: [T, n_mels]
        a = self._load_audio_mel_window(item["audio"], start, self.num_frames)

        # Labels window: [T]
        labels_full = labels
        end = start + self.num_frames
        y = labels_full[start:min(end, len(labels_full))]
        if len(y) < self.num_frames:
            pad = self.num_frames - len(y)
            y = np.pad(y, (0, pad), mode="edge")
        y = torch.from_numpy(y)  # float tensor [T]

        meta = {
            "clip_id": item["clip_id"],
            "start": start
        }
        return v, a, y, meta


# ----------------------------
# Model
# ----------------------------
class VisualBackbone(nn.Module):
    """Lightweight per-frame CNN (ResNet18) returning embedding vectors."""
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


class AudioBackbone(nn.Module):
    """Tiny 1D conv over per-frame mel vectors (treated as sequence)."""
    def __init__(self, n_mels=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, T, n_mels] -> [B, n_mels, T]
        x = x.transpose(1, 2)
        y = self.net(x)               # [B, out_dim, T]
        y = y.transpose(1, 2)         # [B, T, out_dim]
        return y


class TemporalHeadGRU(nn.Module):
    """Bi-GRU temporal head + per-frame classifier."""
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
            nn.Linear(hidden, 1)  # logits per frame
        )

    def forward(self, x):
        # x: [B, T, D]
        y, _ = self.gru(x)            # [B, T, 2H]
        logits = self.head(y).squeeze(-1)  # [B, T]
        return logits


class TemporalHeadTCN(nn.Module):
    """Temporal Convolutional Network head (1D dilated conv) + per-frame classifier."""
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


class AVASpkModel(nn.Module):
    """Audio-Visual late fusion + temporal head (GRU or TCN)."""
    def __init__(self, img_embed_dim=256, aud_embed_dim=128, temporal="gru", temporal_hidden=256):
        super().__init__()
        self.v_backbone = VisualBackbone(out_dim=img_embed_dim, pretrained=True)
        self.a_backbone = AudioBackbone(out_dim=aud_embed_dim)
        d_in = img_embed_dim + aud_embed_dim
        if temporal == "gru":
            self.temporal = TemporalHeadGRU(in_dim=d_in, hidden=temporal_hidden)
        elif temporal == "tcn":
            self.temporal = TemporalHeadTCN(in_dim=d_in, hidden=temporal_hidden)
        else:
            raise ValueError("temporal must be 'gru' or 'tcn'")

    def forward(self, frames_btchw, audio_bt_mels):
        """
        frames_btchw: [B, T, C, H, W]
        audio_bt_mels: [B, T, n_mels]
        """
        B, T = frames_btchw.shape[:2]
        x = frames_btchw.reshape(B*T, *frames_btchw.shape[2:])
        v = self.v_backbone(x)                   # [B*T, Dv]
        v = v.view(B, T, -1)                     # [B, T, Dv]
        a = self.a_backbone(audio_bt_mels)       # [B, T, Da]
        z = torch.cat([v, a], dim=-1)            # [B, T, D]
        logits = self.temporal(z)                # [B, T]
        return logits


# ----------------------------
# Train / Eval
# ----------------------------
def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B, T], targets: [B, T]
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def frame_metrics(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    # logits/targets: [B, T]
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

    for frames, mels, labels, _ in loader:
        frames = frames.to(device)          # [B, T, C, H, W]
        mels = mels.to(device)              # [B, T, n_mels]
        labels = labels.to(device)          # [B, T]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                logits = model(frames, mels)
                loss = bce_logits_loss(logits, labels)
            scaler.scale(loss).backward()
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames, mels)
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

    for frames, mels, labels, _ in loader:
        frames = frames.to(device)
        mels = mels.to(device)
        labels = labels.to(device)
        logits = model(frames, mels)
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
    train_ds = AVAManifestDataset(
        manifest_path=args.train_manifest,
        num_frames=args.num_frames,
        img_size=args.img_size,
        train=True,
        fps=args.fps,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        augment=not args.no_img_aug,
        audio_gain_jitter=args.audio_gain_jitter
    )
    val_ds = AVAManifestDataset(
        manifest_path=args.val_manifest,
        num_frames=args.num_frames,
        img_size=args.img_size,
        train=False,
        fps=args.fps,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        augment=False,
        audio_gain_jitter=0.0
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
    p = argparse.ArgumentParser(description="Audio-Visual Active Speaker Detection (AVA-style)")

    # Data
    p.add_argument("--train-manifest", type=str, required=True)
    p.add_argument("--val-manifest", type=str, required=True)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--num-frames", type=int, default=32, help="Temporal window length T")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-img-aug", action="store_true", help="Disable image augmentations")

    # Audio
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--mel-fmin", type=float, default=50.0)
    p.add_argument("--mel-fmax", type=float, default=0.0, help="0 -> None")
    p.add_argument("--audio-gain-jitter", type=float, default=0.0, help="±dB gain jitter for audio augmentation")

    # Model
    p.add_argument("--img-embed-dim", type=int, default=256)
    p.add_argument("--aud-embed-dim", type=int, default=128)
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
    p.add_argument("--out-dir", type=str, default="./runs/ava_spk")
    p.add_argument("--save-every", type=int, default=0)

    args = p.parse_args()
    if args.mel_fmax and args.mel_fmax <= 0:
        args.mel_fmax = None
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_loaders(args)

    model = AVASpkModel(
        img_embed_dim=args.img_embed_dim,
        aud_embed_dim=args.aud_embed_dim,
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

        # Save best by F1
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
