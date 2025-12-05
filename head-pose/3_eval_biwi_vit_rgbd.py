#!/usr/bin/env python3
# 3_eval_biwi_vit_rgbd.py
#
# Evaluate ViT-B/16 RGBD head-pose model on BIWI and
# generate K PDF (vector) reports for the most accurate
# validation samples, showing:
#   - RGB image
#   - Depth image
#   - Mask image
#   - GT and predicted head pose drawn on RGB
#   - Text with per-axis MAE and mean MAE (deg)
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
# Dataset (RGB + Depth + extras)
# -----------------------------
class BiwiHeadPoseRGBDDataset(Dataset):
    """
    BIWI RGB(D) dataset for evaluation.

    CSV expected columns (like val_yolo.csv):
        image_path, yaw, pitch, roll,
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
        mask_path, subject, frame_id

    We use:
        - image_path  -> RGB
        - mask_path   -> Mask/Depth
        - depth_path  is inferred from mask or from path heuristics
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
        """
        Resuelve la ruta del CSV a una ruta válida dentro del contenedor.

        Casos:
          - Si es relativa → la unimos a data_root.
          - Si es absoluta y existe → la usamos tal cual.
          - Si es absoluta y NO existe (ruta del host) → intentamos
            re-mapearla para que cuelgue de data_root, usando como ancla
            'biwi-kinect-head'.
        """
        # Caso 1: ruta relativa → data_root + p
        if not os.path.isabs(p):
            return os.path.join(self.data_root, p)

        # Caso 2: ruta absoluta que ya existe en el contenedor
        if os.path.isfile(p):
            return p

        # Caso 3: ruta absoluta del host → recortar desde 'biwi-kinect-head'
        marker = "biwi-kinect-head"
        if marker in p:
            # p = "/media/.../biwi-kinect-head/crops_yolo/..."
            _, after = p.split(marker, 1)  # after = "/crops_yolo/..."
            after = after.lstrip("/\\")
            candidate = os.path.join(self.data_root, after)
            if os.path.isfile(candidate):
                return candidate

        # Fallback: último intento con el basename
        candidate = os.path.join(self.data_root, os.path.basename(p))
        if os.path.isfile(candidate):
            return candidate

        # Si llegamos aquí, algo está mal de verdad
        raise FileNotFoundError(
            f"Cannot resolve path '{p}' under data_root '{self.data_root}'"
        )


    def _infer_depth_path(self, rgb_path: str) -> str:
        """
        Try to infer a plausible depth path from the RGB path.
        Adapted to your biwi-kinect-head crops layout.
        """
        candidates = []
        dirname, fname = os.path.split(rgb_path)

        # Replace 'rgb_crops_yolo' -> 'mask_crops_yolo'
        if "rgb_crops_yolo" in dirname:
            candidates.append(
                os.path.join(
                    dirname.replace("rgb_crops_yolo", "mask_crops_yolo"),
                    fname.replace("_rgb.png", "_mask.png"),
                )
            )

        # Replace generic 'rgb' -> 'depth'
        if "rgb" in dirname:
            candidates.append(os.path.join(dirname.replace("rgb", "depth"), fname))
        if "color" in dirname:
            candidates.append(os.path.join(dirname.replace("color", "depth"), fname))

        # Parent/depth/fname
        parent = os.path.dirname(dirname)
        depth_dir = os.path.join(parent, "depth")
        candidates.append(os.path.join(depth_dir, fname))

        for c in candidates:
            if os.path.isfile(c):
                return c

        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        rgb_path_rel = row["image_path"]
        mask_path_rel = row["mask_path"]

        rgb_path = self._resolve_path(rgb_path_rel)
        mask_path = self._resolve_path(mask_path_rel)

        # RGB (original resolution, for visualization)
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb)  # H,W,3

        # Depth (optional, same shape as rgb_np if available)
        depth_path = self._infer_depth_path(rgb_path)
        if depth_path and os.path.isfile(depth_path):
            depth = Image.open(depth_path)
            depth_np = np.array(depth)
        else:
            depth_np = np.zeros(rgb_np.shape[:2], dtype=np.float32)

        # Mask (grayscale)
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)

        # --- Build 4-channel input [RGBD] for the model (224x224) ---
        rgb_resized = rgb.resize((224, 224), Image.BILINEAR)
        rgb_arr = np.array(rgb_resized).astype(np.float32) / 255.0  # H,W,3

        depth_img = Image.fromarray(depth_np)
        depth_resized = depth_img.resize((224, 224), Image.NEAREST)
        depth_arr = np.array(depth_resized).astype(np.float32)

        # Normalize depth to [0,1] if possible (avoid div by zero)
        if depth_arr.max() > 0:
            depth_arr = depth_arr / depth_arr.max()
        depth_arr = depth_arr[..., None]  # H,W,1

        # Concatenate → H,W,4
        rgbd_arr = np.concatenate([rgb_arr, depth_arr], axis=-1)
        # To tensor C,H,W
        rgbd_t = torch.from_numpy(rgbd_arr).permute(2, 0, 1)  # [4,224,224]

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

        return rgbd_t, angles, rgb_np, depth_np, mask_np, meta


# -----------------------------
# Model (matching training ckpt)
# -----------------------------
class ViTHeadPoseModel(nn.Module):
    """
    ViT-B/16 (vit_base_patch16_224) with a 3D regression head (yaw, pitch, roll).

    The module names are:
        - self.vit      (backbone, in_chans=4, global_pool='avg')
        - self.reg_head (3D regression)

    This matches checkpoints with keys like "vit.*" and "reg_head.*",
    and with "vit.fc_norm.*" instead of "vit.norm.*".
    """
    def __init__(self, model_name: str = "vit_base_patch16_224"):
        super().__init__()
        # num_classes=0 => output features instead of logits
        # in_chans=4 => RGBD
        # global_pool="avg" => uses fc_norm instead of norm
        self.vit = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=4,
            global_pool="avg",
        )
        in_features = getattr(self.vit, "num_features", None)
        if in_features is None:
            in_features = self.vit.head.in_features
        self.reg_head = nn.Linear(in_features, 3)

    def forward(self, x):
        feat = self.vit(x)  # [B, D]
        out = self.reg_head(feat)  # [B, 3]
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
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )


def save_sample_pdf(sample: Dict, out_dir: str, rank: int):
    """
    Produce un PDF con 5 imágenes:
      1) RGB original
      2) Depth
      3) Mask
      4) RGB con pose predicha
      5) RGB con pose GT
    """
    os.makedirs(out_dir, exist_ok=True)

    rgb = sample["rgb_np"]
    depth = sample["depth_np"]
    mask = sample["mask_np"]
    gt = sample["gt_angles"]       # yaw, pitch, roll
    pred = sample["pred_angles"]   # yaw, pitch, roll

    # -------- Precompute images --------
    # Copias para dibujar flechas
    rgb_pred = rgb.copy()
    rgb_gt   = rgb.copy()

    # DIBUJAR flechas (usa tu función existente)
    fig_tmp, ax_tmp = plt.subplots()
    ax_tmp.imshow(rgb_pred)
    draw_head_pose_arrow(ax_tmp, rgb_pred, pred[0], pred[1], color="red")
    plt.close(fig_tmp)

    fig_tmp, ax_tmp = plt.subplots()
    ax_tmp.imshow(rgb_gt)
    draw_head_pose_arrow(ax_tmp, rgb_gt, gt[0], gt[1], color="lime")
    plt.close(fig_tmp)

    # -------- Final layout: 5 imágenes --------
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    panels = [
        (rgb, "RGB"),
        (depth, "Depth"),
        (mask, "Mask"),
        (rgb_pred, "Pred Pose"),
        (rgb_gt, "GT Pose"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    pdf_path = os.path.join(out_dir, f"sample_{rank+1:02d}.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved PDF → {pdf_path}")



# -----------------------------
# Custom collate (clave para evitar el error)
# -----------------------------
def custom_collate_fn(batch):
    """
    batch: list of tuples
      (rgbd_t, angles_t, rgb_np, depth_np, mask_np, meta)

    - Stack only the tensors we need for the model (rgbd_t, angles_t).
    - Keep rgb_np, depth_np, mask_np as lists of numpy arrays.
    - meta as dict of lists.
    """
    rgbd_list, angles_list, rgb_np_list, depth_np_list, mask_np_list, meta_list = zip(*batch)

    rgbd_batch = torch.stack(rgbd_list, dim=0)      # [B,4,224,224]
    angles_batch = torch.stack(angles_list, dim=0)  # [B,3]

    # meta: dict with keys -> list of values
    meta_batch = {}
    for key in meta_list[0].keys():
        meta_batch[key] = [m[key] for m in meta_list]

    return (
        rgbd_batch,
        angles_batch,
        list(rgb_np_list),
        list(depth_np_list),
        list(mask_np_list),
        meta_batch,
    )


# -----------------------------
# Argumentos
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ViT-B/16 RGBD on BIWI and generate PDF reports."
    )
    p.add_argument(
        "--val-manifest",
        type=str,
        required=True,
        help="CSV file for validation split (with image_path, yaw, pitch, roll, mask_path, ...).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory for BIWI data.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./biwi_eval_pdfs")
    p.add_argument(
        "--num-pdfs",
        type=int,
        default=5,
        help="Number of best validation samples to export as PDF.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    return p.parse_args()


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_and_generate_pdfs(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Dataset + loader with custom collate
    val_ds = BiwiHeadPoseRGBDDataset(
        csv_path=args.val_manifest,
        data_root=args.data_root,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Model
    model = ViTHeadPoseModel(model_name=args.model_name).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_samples = []
    mae_sum = np.zeros(3, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for rgbd_t, angles_t, rgb_np_list, depth_np_list, mask_np_list, meta in val_loader:
            rgbd_t = rgbd_t.to(device)      # [B,4,224,224]
            angles_t = angles_t.to(device)  # [B,3]

            preds = model(rgbd_t)
            gt_np = angles_t.cpu().numpy()
            pred_np = preds.cpu().numpy()

            abs_err = np.abs(pred_np - gt_np)  # [B,3]
            mae_sum += abs_err.sum(axis=0)
            count += abs_err.shape[0]

            batch_size = abs_err.shape[0]
            for i in range(batch_size):
                mae_per_axis = abs_err[i]
                mae_mean = float(mae_per_axis.mean())

                all_samples.append(
                    {
                        "rgb_np": rgb_np_list[i],
                        "depth_np": depth_np_list[i],
                        "mask_np": mask_np_list[i],
                        "gt_angles": gt_np[i],
                        "pred_angles": pred_np[i],
                        "mae_per_axis": mae_per_axis,
                        "mae_mean": mae_mean,
                        "meta": {k: meta[k][i] for k in meta},
                    }
                )

    overall_mae = mae_sum / max(1, count)
    print(f"[INFO] Global MAE (yaw, pitch, roll): {overall_mae}")
    print(f"[INFO] Mean MAE: {overall_mae.mean():.3f} deg")

    # Ordenar por MAE medio (menor primero)
    all_samples.sort(key=lambda d: d["mae_mean"])

    k = min(args.num_pdfs, len(all_samples))
    print(f"[INFO] Generating {k} PDF reports (best validation samples).")

    for rank in range(k):
        save_sample_pdf(all_samples[rank], args.output_dir, rank)


def main():
    args = parse_args()
    evaluate_and_generate_pdfs(args)


if __name__ == "__main__":
    main()
