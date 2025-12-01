#!/usr/bin/env python3
# 2_train_biwi_vit_rgbd.py
# ViT-based head pose regression (yaw, pitch, roll) for BIWI-like datasets using
# RGB + Depth crops from a CSV manifest.

import os
import csv
import time
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# You need timm: pip install timm
try:
    import timm
except ImportError as e:
    raise SystemExit(
        "This script requires 'timm'. Please install it via: pip install timm"
    )


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def angular_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple MAE in degrees for (yaw, pitch, roll).
    pred/target shape: [B, 3]
    """
    return torch.mean(torch.abs(pred - target), dim=0)  # returns 3-element tensor


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Dataset (RGB + Depth)
# -----------------------------
@dataclass
class Sample:
    rgb_path: str
    depth_path: str
    yaw: float
    pitch: float
    roll: float


class BiwiRGBDCSV(Dataset):
    """
    Expects a CSV with header at least:
      image_path,yaw,pitch,roll

    Idealmente (como genera 1_build_biwi_crops_and_csv.py):
      image_path,yaw,pitch,roll,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,mask_path,subject,frame_id

    - image_path : ruta al recorte RGB (en rgb_crops_yolo).
    - mask_path  : ruta al recorte de depth/mask (en mask_crops_yolo).
      Si no está 'mask_path' en el CSV, se intenta inferir a partir de image_path.

    Parámetro extra:
    - data_root: si no es None, reescribe rutas antiguas como
      /media/ignacio/KINGSTON/biwi-kinect-head/... a
      {data_root}/...
    """
    def __init__(
        self,
        manifest_path: str,
        img_size: int,
        is_train: bool,
        augment: bool = False,
        data_root: Optional[str] = None,
    ):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.samples: List[Sample] = []
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment
        self.data_root = data_root

        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            req_cols = {"image_path", "yaw", "pitch", "roll"}
            if not req_cols.issubset(fieldnames):
                raise ValueError(
                    f"Manifest must contain columns at least: {req_cols}. "
                    f"Found: {reader.fieldnames}"
                )

            has_mask_col = "mask_path" in fieldnames

            def rewrite_path(old_path: str) -> str:
                """
                Reescribe paths que empiezan con el prefijo antiguo a data_root/...
                Si data_root es None o el prefijo no coincide, devuelve el path tal cual.
                """
                if old_path is None:
                    return None
                if self.data_root is None:
                    return old_path

                old_prefix = "/media/ignacio/KINGSTON/biwi-kinect-head"
                if old_path.startswith(old_prefix):
                    # Tomamos la parte después de 'biwi-kinect-head'
                    suffix = old_path.split("biwi-kinect-head", 1)[1]
                    return os.path.join(self.data_root, suffix.lstrip("/"))
                return old_path

            for row in reader:
                csv_rgb_path = row["image_path"]
                csv_depth_path = row["mask_path"] if has_mask_col else None

                # Reescribir paths si corresponde
                rgb_path = rewrite_path(csv_rgb_path)

                if csv_depth_path:
                    depth_path = rewrite_path(csv_depth_path)
                else:
                    # Infer depth/mask path from RGB crop path:
                    #   .../rgb_crops_yolo/XX/frame_00001_rgb.png
                    # -> .../mask_crops_yolo/XX/frame_00001_mask.png
                    inferred = rgb_path
                    inferred = inferred.replace("rgb_crops_yolo", "mask_crops_yolo")
                    inferred = inferred.replace("_rgb.png", "_mask.png")
                    depth_path = inferred

                yaw = float(row["yaw"])
                pitch = float(row["pitch"])
                roll = float(row["roll"])

                if not os.path.isfile(rgb_path):
                    raise FileNotFoundError(f"RGB file not found: {rgb_path}")
                if not os.path.isfile(depth_path):
                    raise FileNotFoundError(
                        f"Depth/mask file not found: {depth_path}"
                    )

                self.samples.append(
                    Sample(
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
                    )
                )

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

        print(
            f"[BiwiRGBDCSV] Loaded {len(self.samples)} samples from {manifest_path} "
            f"(train={self.is_train}, augment={self.augment}, data_root={self.data_root})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, rgb_path: str, depth_path: str):
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")  # mask/depth recortado
        return rgb, depth

    def _apply_transforms(
        self, rgb: Image.Image, depth: Image.Image
    ) -> torch.Tensor:
        """
        Aplica las mismas transformaciones geométricas a RGB y Depth.
        Luego:
          - RGB: to_tensor + (x - 0.5) / 0.5
          - Depth: to_tensor + (x - 0.5) / 0.5
        Devuelve tensor [4, H, W] con canales [R,G,B,Depth].
        """
        # Resize
        rgb = F.resize(
            rgb,
            (self.img_size, self.img_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        depth = F.resize(
            depth,
            (self.img_size, self.img_size),
            interpolation=InterpolationMode.BILINEAR,
        )

        # Geometric augmentation (compartida)
        if self.is_train and self.augment:
            # Flip horizontal
            if random.random() < 0.5:
                rgb = F.hflip(rgb)
                depth = F.hflip(depth)

            # Pequeño RandomAffine compartido (opcional)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                max_trans = int(0.05 * self.img_size)
                tx = random.randint(-max_trans, max_trans)
                ty = random.randint(-max_trans, max_trans)
                scale = random.uniform(0.95, 1.05)
                shear = random.uniform(-5, 5)

                rgb = F.affine(
                    rgb,
                    angle=angle,
                    translate=(tx, ty),
                    scale=scale,
                    shear=(shear, 0.0),
                    interpolation=InterpolationMode.BILINEAR,
                )
                depth = F.affine(
                    depth,
                    angle=angle,
                    translate=(tx, ty),
                    scale=scale,
                    shear=(shear, 0.0),
                    interpolation=InterpolationMode.BILINEAR,
                )

        # To tensor (0..1)
        rgb_t = F.to_tensor(rgb)      # [3,H,W], float32
        depth_t = F.to_tensor(depth)  # [1,H,W], float32

        # Normalización a [-1,1]
        rgb_t = (rgb_t - 0.5) / 0.5
        depth_t = (depth_t - 0.5) / 0.5

        x = torch.cat([rgb_t, depth_t], dim=0)  # [4,H,W]
        return x

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        rgb, depth = self._load_pair(s.rgb_path, s.depth_path)
        x = self._apply_transforms(rgb, depth)

        target = torch.tensor(
            [s.yaw, s.pitch, s.roll],
            dtype=torch.float32,
        )
        return x, target


# -----------------------------
# Model
# -----------------------------
class ViTRegressor(nn.Module):
    """
    Wrap a timm ViT and replace classifier with a 3D regression head (yaw, pitch, roll).
    Supports arbitrary in_chans (e.g. 4 for RGB+Depth).
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        drop: float = 0.0,
        drop_path: float = 0.0,
        regression_hidden: int = 0,
        out_dim: int = 3,
        in_chans: int = 4,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,         # get features
            drop_rate=drop,
            drop_path_rate=drop_path,
            global_pool="avg",     # average pool token features
            in_chans=in_chans,
        )
        feat_dim = self.vit.num_features

        if regression_hidden > 0:
            self.reg_head = nn.Sequential(
                nn.Linear(feat_dim, regression_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(regression_hidden, out_dim),
            )
        else:
            self.reg_head = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        feats = self.vit(x)      # [B, C]
        out = self.reg_head(feats)  # [B, 3]
        return out


# -----------------------------
# Training / Evaluation
# -----------------------------
def create_loaders(
    train_manifest: str,
    val_manifest: str,
    img_size: int,
    batch_size: int,
    workers: int,
    augment: bool,
    data_root: Optional[str],
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BiwiRGBDCSV(
        train_manifest,
        img_size=img_size,
        is_train=True,
        augment=augment,
        data_root=data_root,
    )
    val_ds = BiwiRGBDCSV(
        val_manifest,
        img_size=img_size,
        is_train=False,
        augment=False,
        data_root=data_root,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    if args.opt == "adam":
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    elif args.opt == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    elif args.opt == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")


def build_scheduler(optimizer: optim.Optimizer, args):
    if args.sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sched == "step":
        return StepLR(
            optimizer,
            step_size=max(1, args.step_size),
            gamma=args.gamma,
        )
    elif args.sched == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {args.sched}")


def build_loss(name: str):
    name = name.lower()
    if name in ["l1", "mae"]:
        return nn.L1Loss()
    elif name in ["mse", "l2"]:
        return nn.MSELoss()
    elif name in ["smoothl1", "huber"]:
        return nn.SmoothL1Loss(beta=1.0)  # Huber
    else:
        raise ValueError(f"Unknown loss: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: Optional[float] = None,
):
    model.train()
    running_loss = 0.0
    running_ang_mae = torch.zeros(3, device=device)

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_ang_mae += torch.sum(torch.abs(preds - targets), dim=0)

    n = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_mae = (running_ang_mae / n).detach().cpu().tolist()  # [yaw, pitch, roll]
    return epoch_loss, epoch_mae


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_ang_mae = torch.zeros(3, device=device)

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(imgs)
        loss = criterion(preds, targets)

        running_loss += loss.item() * imgs.size(0)
        running_ang_mae += torch.sum(torch.abs(preds - targets), dim=0)

    n = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_mae = (running_ang_mae / n).detach().cpu().tolist()
    return epoch_loss, epoch_mae


def maybe_freeze_backbone(model: ViTRegressor, freeze: bool):
    if not freeze:
        return
    for p in model.vit.parameters():
        p.requires_grad = False
    # keep regression head trainable


def save_checkpoint(state: dict, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="ViT Head Pose Regression (BIWI RGB+Depth crops)"
    )

    p.add_argument(
        "--gpu", type=int, default=0,
        help="GPU ID to use (0-7). Use -1 for CPU."
    )


    # Data
    p.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="CSV manifest for training (e.g. train_yolo.csv)",
    )
    p.add_argument(
        "--val-manifest",
        type=str,
        required=True,
        help="CSV manifest for validation (e.g. val_yolo.csv)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Root path of 'biwi-kinect-head' to rewrite old absolute paths "
            "inside CSV. Example: /home/ignacio.bugueno/cachefs/cis/input/biwi-kinect-head"
        ),
    )
    p.add_argument("--img-size", type=int, default=224, help="Input size for ViT")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm ViT model name",
    )
    p.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument(
        "--reg-hidden",
        type=int,
        default=0,
        help="Hidden size for reg head (0 means direct Linear)",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ViT backbone and train only head",
    )

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--opt",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--sched",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
    )
    p.add_argument(
        "--step-size",
        type=int,
        default=15,
        help="StepLR step size (if sched=step)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="StepLR gamma (if sched=step)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "l1", "mae", "mse", "l2"],
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--amp", action="store_true", help="Use mixed precision"
    )

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default="./runs/biwi_vit_rgbd",
        help="Output directory for checkpoints and logs",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0=only best+last)",
    )

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # If --gpu = -1 → CPU
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    train_loader, val_loader = create_loaders(
        args.train_manifest,
        args.val_manifest,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.augment,
        args.data_root,
    )

    # 4 canales: RGB (3) + Depth/Mask (1)
    model = ViTRegressor(
        model_name=args.model,
        pretrained=args.pretrained,
        drop=args.drop,
        drop_path=args.drop_path,
        regression_hidden=args.reg_hidden,
        out_dim=3,
        in_chans=4,
    ).to(device)

    maybe_freeze_backbone(model, args.freeze_backbone)

    total_params = count_params(model)
    print(f"Trainable parameters: {total_params:,}")

    criterion = build_loss(args.loss)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    best_val_loss = float("inf")
    history = {
        "args": vars(args),
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_mae(y/p/r)=({train_mae[0]:.2f}/{train_mae[1]:.2f}/{train_mae[2]:.2f}) "
            f"val_mae(y/p/r)=({val_mae[0]:.2f}/{val_mae[1]:.2f}/{val_mae[2]:.2f}) "
            f"time={dt:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "args": vars(args),
                },
                args.out_dir,
                "best.pt",
            )

        # Optional periodic save
        if args.save_every and (args.save_every > 0) and (epoch % args.save_every == 0):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "args": vars(args),
                },
                args.out_dir,
                f"epoch_{epoch:03d}.pt",
            )

        # Always save last
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "args": vars(args),
            },
            args.out_dir,
            "last.pt",
        )

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mae": [float(x) for x in train_mae],
                "val_mae": [float(x) for x in val_mae],
                "time_sec": dt,
            }
        )
        save_json(os.path.join(args.out_dir, "history.json"), history)

    print(
        f"Training finished. Best val loss: {best_val_loss:.4f}. "
        f"Checkpoints in: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
