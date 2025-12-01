#!/usr/bin/env python3
# 3_eval_biwi_vit_rgbd.py
# Evalúa un ViTRegressor entrenado (RGB+Depth) sobre un manifest CSV (e.g. val_yolo.csv)

import os
import csv
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import timm


# -----------------------------
# Dataset (RGB + Depth) - mismo estilo que en el train
# -----------------------------
@dataclass
class Sample:
    rgb_path: str
    depth_path: str
    yaw: float
    pitch: float
    roll: float


class BiwiRGBDCSV(Dataset):
    def __init__(
        self,
        manifest_path: str,
        img_size: int,
        data_root: Optional[str] = None,
    ):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.samples: List[Sample] = []
        self.img_size = img_size
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
                if old_path is None:
                    return None
                if self.data_root is None:
                    return old_path
                old_prefix = "/media/ignacio/KINGSTON/biwi-kinect-head"
                if old_path.startswith(old_prefix):
                    suffix = old_path.split("biwi-kinect-head", 1)[1]
                    return os.path.join(self.data_root, suffix.lstrip("/"))
                return old_path

            for row in reader:
                csv_rgb_path = row["image_path"]
                csv_depth_path = row["mask_path"] if has_mask_col else None

                rgb_path = rewrite_path(csv_rgb_path)

                if csv_depth_path:
                    depth_path = rewrite_path(csv_depth_path)
                else:
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
                    raise FileNotFoundError(f"Depth/mask file not found: {depth_path}")

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
            f"[BiwiRGBDCSV-Eval] Loaded {len(self.samples)} samples from {manifest_path} "
            f"(data_root={self.data_root})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, rgb_path: str, depth_path: str):
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        return rgb, depth

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rgb, depth = self._load_pair(s.rgb_path, s.depth_path)

        # resize
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

        # to tensor + normalización [-1,1]
        rgb_t = F.to_tensor(rgb)
        depth_t = F.to_tensor(depth)
        rgb_t = (rgb_t - 0.5) / 0.5
        depth_t = (depth_t - 0.5) / 0.5

        x = torch.cat([rgb_t, depth_t], dim=0)  # [4,H,W]

        target = torch.tensor(
            [s.yaw, s.pitch, s.roll],
            dtype=torch.float32,
        )
        meta = {
            "rgb_path": s.rgb_path,
            "depth_path": s.depth_path,
        }
        return x, target, meta


# -----------------------------
# Model wrapper
# -----------------------------
class ViTRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        drop: float,
        drop_path: float,
        regression_hidden: int,
        out_dim: int = 3,
        in_chans: int = 4,
        pretrained: bool = False,  # para eval, normalmente False
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop,
            drop_path_rate=drop_path,
            global_pool="avg",
            in_chans=in_chans,
        )
        feat_dim = self.vit.num_features

        if regression_hidden and regression_hidden > 0:
            self.reg_head = nn.Sequential(
                nn.Linear(feat_dim, regression_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(regression_hidden, out_dim),
            )
        else:
            self.reg_head = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        feats = self.vit(x)
        out = self.reg_head(feats)
        return out


# -----------------------------
# Argumentos
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Eval BIWI ViT RGBD model")

    p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest to evaluate (e.g. val_yolo.csv)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root path of 'biwi-kinect-head' to rewrite old absolute paths",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pt (or any .pt checkpoint from training)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to store evaluation results",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0, help="GPU id (use -1 for CPU)")

    return p.parse_args()


# -----------------------------
# Métricas
# -----------------------------
def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    # gt, pred: [N,3] en grados
    err = np.abs(pred - gt)
    mae = err.mean(axis=0)  # [3]
    mae_mean = err.mean()

    rmse = np.sqrt(((pred - gt) ** 2).mean(axis=0))
    rmse_mean = np.sqrt(((pred - gt) ** 2).mean())

    percentiles = np.percentile(err, [50, 90, 95], axis=0)  # [3,3]

    metrics = {
        "mae_yaw": float(mae[0]),
        "mae_pitch": float(mae[1]),
        "mae_roll": float(mae[2]),
        "mae_mean": float(mae_mean),
        "rmse_yaw": float(rmse[0]),
        "rmse_pitch": float(rmse[1]),
        "rmse_roll": float(rmse[2]),
        "rmse_mean": float(rmse_mean),
        "p50_yaw": float(percentiles[0, 0]),
        "p50_pitch": float(percentiles[0, 1]),
        "p50_roll": float(percentiles[0, 2]),
        "p90_yaw": float(percentiles[1, 0]),
        "p90_pitch": float(percentiles[1, 1]),
        "p90_roll": float(percentiles[1, 2]),
        "p95_yaw": float(percentiles[2, 0]),
        "p95_pitch": float(percentiles[2, 1]),
        "p95_roll": float(percentiles[2, 2]),
    }
    return metrics, err


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] Using GPU {args.gpu}")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model", "vit_base_patch16_224")
    drop = ckpt_args.get("drop", 0.0)
    drop_path = ckpt_args.get("drop_path", 0.0)
    reg_hidden = ckpt_args.get("reg_hidden", 0)
    img_size = ckpt_args.get("img_size", 224)

    print("[INFO] Checkpoint loaded from:", args.checkpoint)
    print("[INFO] Model:", model_name)
    print("[INFO] img_size:", img_size)

    model = ViTRegressor(
        model_name=model_name,
        drop=drop,
        drop_path=drop_path,
        regression_hidden=reg_hidden,
        out_dim=3,
        in_chans=4,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # dataset + loader
    ds = BiwiRGBDCSV(
        manifest_path=args.manifest,
        img_size=img_size,
        data_root=args.data_root,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_gt = []
    all_pred = []
    rows_out = []

    with torch.no_grad():
        for x, target, meta in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(x)  # [B,3]

            all_gt.append(target.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

            # guardar fila por muestra
            for i in range(x.size(0)):
                gt_i = target[i].cpu().numpy()
                pr_i = pred[i].cpu().numpy()
                err_i = np.abs(pr_i - gt_i)

                rows_out.append(
                    {
                        "rgb_path": meta["rgb_path"][i],
                        "depth_path": meta["depth_path"][i],
                        "yaw_gt": float(gt_i[0]),
                        "pitch_gt": float(gt_i[1]),
                        "roll_gt": float(gt_i[2]),
                        "yaw_pred": float(pr_i[0]),
                        "pitch_pred": float(pr_i[1]),
                        "roll_pred": float(pr_i[2]),
                        "err_yaw": float(err_i[0]),
                        "err_pitch": float(err_i[1]),
                        "err_roll": float(err_i[2]),
                    }
                )

    all_gt = np.concatenate(all_gt, axis=0)   # [N,3]
    all_pred = np.concatenate(all_pred, axis=0)

    metrics, err = compute_metrics(all_gt, all_pred)

    print("\n===== EVALUATION RESULTS =====")
    print(f"MAE (yaw, pitch, roll): {metrics['mae_yaw']:.3f}, {metrics['mae_pitch']:.3f}, {metrics['mae_roll']:.3f}")
    print(f"MAE mean: {metrics['mae_mean']:.3f} deg")
    print(f"RMSE (yaw, pitch, roll): {metrics['rmse_yaw']:.3f}, {metrics['rmse_pitch']:.3f}, {metrics['rmse_roll']:.3f}")
    print(f"RMSE mean: {metrics['rmse_mean']:.3f} deg")
    print("Percentiles (deg):")
    print(f"  P50 (yaw, pitch, roll): {metrics['p50_yaw']:.3f}, {metrics['p50_pitch']:.3f}, {metrics['p50_roll']:.3f}")
    print(f"  P90 (yaw, pitch, roll): {metrics['p90_yaw']:.3f}, {metrics['p90_pitch']:.3f}, {metrics['p90_roll']:.3f}")
    print(f"  P95 (yaw, pitch, roll): {metrics['p95_yaw']:.3f}, {metrics['p95_pitch']:.3f}, {metrics['p95_roll']:.3f}")

    # save metrics JSON
    metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", metrics_path)

    # save predictions CSV
    pred_csv = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print("Saved predictions to:", pred_csv)


if __name__ == "__main__":
    main()
