#!/usr/bin/env python3
# 3_eval_biwi_vit_rgbd.py
#
# Evaluate ViT-B/16 RGB(D) head pose model on BIWI and
# generate K PDF (vector) reports for the most accurate
# validation samples, showing:
#   - RGB image
#   - Depth image (optional, dummy if not available)
#   - Mask image
#   - GT and predicted head pose drawn
#   - Text with per-axis MAE and mean MAE
#
# Text is in English.

import os
import csv
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def round_recursive(x, ndigits=4):
    if isinstance(x, float):
        return round(x, ndigits)
    if isinstance(x, int):
        return x
    if isinstance(x, list):
        return [round_recursive(v, ndigits) for v in x]
    if isinstance(x, dict):
        return {k: round_recursive(v, ndigits) for k, v in x.items()}
    return x


# -----------------------------
# Dataset (adapted to your CSV)
# -----------------------------
class BiwiHeadPoseRGBDDataset(Dataset):
    """
    BIWI RGB(D) dataset for evaluation.

    CSV expected columns (like your val_yolo.csv):
        image_path, yaw, pitch, roll,
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
        mask_path, subject, frame_id

    We use:
        - image_path  -> RGB
        - mask_path   -> Mask
        - depth: *optional*, we try to infer from image_path by replacing tokens
    """
    def __init__(self, csv_path: str, data_root: str):
        self.data_root = data_root
        self.samples = self._read_csv(csv_path)

    def _read_csv(self, path: str) -> List[Dict]:
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"image_path", "yaw", "pitch", "roll", "mask_path"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV {path} must contain columns {required}, "
                    f"got {reader.fieldnames}"
                )
            for row in reader:
                out.append(row)
        if not out:
            raise ValueError(f"No rows found in {path}")
        return out

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def _infer_depth_path(self, rgb_path: str) -> str:
        """
        Try to infer a plausible depth path from the RGB path.
        You may need to adapt this to your actual directory layout.
        """
        # Example heuristics: replace 'rgb'->'depth' or 'color'->'depth'
        candidates = []
        dirname, fname = os.path.split(rgb_path)

        # 1) Replace 'rgb' with 'depth' in the directory path
        if "rgb" in dirname:
            candidates.append(os.path.join(dirname.replace("rgb", "depth"), fname))
        if "color" in dirname:
            candidates.append(os.path.join(dirname.replace("color", "depth"), fname))

        # 2) Same dir, different subfolder 'depth'
        # e.g. /.../rgb/frame.png -> /.../depth/frame.png
        parent = os.path.dirname(dirname)
        depth_dir = os.path.join(parent, "depth")
        candidates.append(os.path.join(depth_dir, fname))

        for c in candidates:
            if os.path.isfile(c):
                return c

        # If nothing found, return empty string
        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        rgb_path_rel = row["image_path"]
        mask_path_rel = row["mask_path"]

        rgb_path = self._resolve_path(rgb_path_rel)
        mask_path = self._resolve_path(mask_path_rel)

        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb)  # H,W,3

        # Try to get depth
        depth_path = self._infer_depth_path(rgb_path)
        if depth_path and os.path.isfile(depth_path):
            depth = Image.open(depth_path)
            depth_np = np.array(depth)
        else:
            # Dummy depth: zeros with same H,W
            depth_np = np.zeros(rgb_np.shape[:2], dtype=np.float32)

        # Mask
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)

        # For the model, we only need RGB resized to 224x224
        rgb_resized = rgb.resize((224, 224), Image.BILINEAR)
        rgb_t = torch.from_numpy(np.array(rgb_resized)).float() / 255.0  # H,W,3
        rgb_t = rgb_t.permute(2, 0, 1)  # C,H,W

        # Angles
        yaw = float(row["yaw"])
        pitch = float(row["pitch"])
        roll = float(row["roll"])
        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)

        meta = {
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "subject": row.get("subject", ""),
            "frame_id": row.get("frame_id", ""),
        }

        return rgb_t, angles, rgb_np, depth_np, mask_np, meta


# -----------------------------
# Model
# -----------------------------
class ViTHeadPoseModel(nn.Module):
    """
    ViT-B/16 (vit_base_patch16_224) with a 3D regression head (yaw, pitch, roll).
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        self.head = nn.Linear(in_features, 3)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        out = self.head(feat)
        return out


# -----------------------------
# Visualization utilities
# -----------------------------
def draw_head_pose_arrow(ax, img, yaw_deg, pitch_deg, color="lime", label=""):
    """
    Draw a simple 2D arrow on the image to represent head pose direction.
    Qualitative only.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    dx = np.sin(yaw) * np.cos(pitch)
    dy = -np.sin(pitch)

    length = min(h, w) * 0.25
    x2 = cx + dx * length
    y2 = cy + dy * length

    ax.arrow(
        cx, cy,
        x2 - cx, y2 - cy,
        head_width=h * 0.02,
        head_length=h * 0.03,
        length_includes_head=True,
        color=color,
        linewidth=2,
        alpha=0.9,
    )
    if label:
        ax.text(
            cx, cy,
            label,
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none")
        )


def save_sample_pdf(sample: Dict, out_dir: str, rank: int):
    os.makedirs(out_dir, exist_ok=True)
    rgb = sample["rgb_np"]
    depth = sample["depth_np"]
    mask = sample["mask_np"]
    gt = sample["gt_angles"]
    pred = sample["pred_angles"]
    mae = sample["mae_per_axis"]
    mae_mean = sample["mae_mean"]
    meta = sample["meta"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle(f"BIWI Head Pose – Best validation sample #{rank+1}", fontsize=14)

    # RGB + pose arrows
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("RGB image")
    ax.axis("off")
    draw_head_pose_arrow(ax, rgb, gt[0], gt[1], color="lime", label="GT")
    draw_head_pose_arrow(ax, rgb, pred[0], pred[1], color="red", label="Pred")

    # Depth
    ax = axes[0, 1]
    ax.imshow(depth, cmap="viridis")
    ax.set_title("Depth image")
    ax.axis("off")

    # Mask
    ax = axes[0, 2]
    ax.imshow(mask, cmap="gray")
    ax.set_title("Mask image")
    ax.axis("off")

    # Text with metrics
    ax = axes[1, 0]
    ax.axis("off")

    text_lines = [
        "Head pose (degrees)",
        "",
        f"GT yaw / pitch / roll: {gt[0]:.2f} / {gt[1]:.2f} / {gt[2]:.2f}",
        f"Pred yaw / pitch / roll: {pred[0]:.2f} / {pred[1]:.2f} / {pred[2]:.2f}",
        "",
        "Absolute error (degrees)",
        f"yaw:   {mae[0]:.2f}",
        f"pitch: {mae[1]:.2f}",
        f"roll:  {mae[2]:.2f}",
        "",
        f"Mean MAE: {mae_mean:.2f} deg",
        "",
        "Paths:",
        os.path.basename(meta.get("rgb_path", "")),
        os.path.basename(meta.get("mask_path", "")),
    ]
    ax.text(
        0.0, 1.0,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )

    # GT vs Pred bar plot
    ax = axes[1, 1]
    ax.set_title("GT vs Pred angles")
    labels = ["yaw", "pitch", "roll"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, gt, width, label="GT")
    ax.bar(x + width/2, pred, width, label="Pred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Degrees")
    ax.legend()

    # Error per axis
    ax = axes[1, 2]
    ax.set_title("Absolute error per axis")
    ax.bar(labels, mae)
    ax.set_ylabel("Degrees")

    pdf_name = os.path.join(out_dir, f"biwi_best_sample_{rank+1:02d}.pdf")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(pdf_name, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {pdf_name}")


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_and_generate_pdfs(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    val_ds = BiwiHeadPoseRGBDDataset(
        csv_path=args.val_manifest,
        data_root=args.data_root
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = ViTHeadPoseModel(
        model_name=args.model_name,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_samples = []
    mae_sum = np.zeros(3, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for rgb_t, angles_t, rgb_np, depth_np, mask_np, meta in val_loader:
            rgb_t = rgb_t.to(device)
            angles_t = angles_t.to(device)

            preds = model(rgb_t)
            gt_np = angles_t.cpu().numpy()
            pred_np = preds.cpu().numpy()

            abs_err = np.abs(pred_np - gt_np)  # [B,3]
            mae_sum += abs_err.sum(axis=0)
            count += abs_err.shape[0]

            for i in range(abs_err.shape[0]):
                mae_per_axis = abs_err[i]
                mae_mean = float(mae_per_axis.mean())

                all_samples.append({
                    "rgb_np": rgb_np[i].numpy(),
                    "depth_np": depth_np[i].numpy(),
                    "mask_np": mask_np[i].numpy(),
                    "gt_angles": gt_np[i],
                    "pred_angles": pred_np[i],
                    "mae_per_axis": mae_per_axis,
                    "mae_mean": mae_mean,
                    "meta": {k: meta[k][i] for k in meta},
                })

    overall_mae = mae_sum / max(1, count)
    print(f"[INFO] Global MAE (yaw, pitch, roll): {overall_mae}")
    print(f"[INFO] Mean MAE: {overall_mae.mean():.3f} deg")

    all_samples.sort(key=lambda d: d["mae_mean"])

    k = min(args.num_pdfs, len(all_samples))
    print(f"[INFO] Generating {k} PDF reports (best validation samples).")

    for rank in range(k):
        save_sample_pdf(all_samples[rank], args.output_dir, rank)


# -----------------------------
# Argparse
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ViT-B/16 on BIWI and generate PDF reports."
    )
    p.add_argument("--val-manifest", type=str, required=True,
                   help="CSV file for validation split (with image_path, yaw, pitch, roll, mask_path, ...).")
    p.add_argument("--data-root", type=str, required=True,
                   help="Root directory for BIWI data.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model checkpoint.")
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./biwi_eval_pdfs")
    p.add_argument("--num-pdfs", type=int, default=5,
                   help="Number of best validation samples to export as PDF.")
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_and_generate_pdfs(args)


if __name__ == "__main__":
    main()
