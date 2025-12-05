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

import struct
import numpy as np


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
        - image_path  -> RGB crop (ya cortado por YOLO)
        - mask_path   -> Mask crop (solo para visualizar)
        - depth_path  -> SE CONSTRUYE desde faces_0/<subject>/<frame_id>_depth.bin
                         y se recorta usando el bbox del CSV
    """
    def __init__(self, csv_path: str, data_root: str):
        self.data_root = data_root
        self.samples = self._read_csv(csv_path)

    def _read_csv(self, path: str) -> List[Dict]:
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {
                "image_path", "yaw", "pitch", "roll",
                "mask_path", "bbox_xmin", "bbox_ymin",
                "bbox_xmax", "bbox_ymax", "subject", "frame_id"
            }
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
        """
        if not os.path.isabs(p):
            return os.path.join(self.data_root, p)

        if os.path.isfile(p):
            return p

        marker = "biwi-kinect-head"
        if marker in p:
            _, after = p.split(marker, 1)  # after = "/crops_yolo/..."
            after = after.lstrip("/\\")
            candidate = os.path.join(self.data_root, after)
            if os.path.isfile(candidate):
                return candidate

        candidate = os.path.join(self.data_root, os.path.basename(p))
        if os.path.isfile(candidate):
            return candidate

        raise FileNotFoundError(
            f"Cannot resolve path '{p}' under data_root '{self.data_root}'"
        )

    def _infer_depth_path(self, rgb_path: str) -> str:
        """
        Fallback antiguo: intenta deducir depth desde la ruta del RGB.
        Lo dejamos por si algún sample no tiene subject/frame_id.
        """
        candidates = []
        dirname, fname = os.path.split(rgb_path)

        # Caso rgb_crops_yolo -> depth_crops_yolo
        if "rgb_crops_yolo" in dirname:
            candidates.append(
                os.path.join(
                    dirname.replace("rgb_crops_yolo", "depth_crops_yolo"),
                    fname.replace("_rgb.png", "_depth.png"),
                )
            )

        if "rgb" in dirname:
            candidates.append(os.path.join(dirname.replace("rgb", "depth"), fname))
        if "color" in dirname:
            candidates.append(os.path.join(dirname.replace("color", "depth"), fname))

        parent = os.path.dirname(dirname)
        depth_dir = os.path.join(parent, "depth")
        candidates.append(os.path.join(depth_dir, fname))

        for c in candidates:
            if os.path.isfile(c):
                return c

        return ""

    def _get_depth_path_from_row(self, row: Dict) -> str:
        """
        Construye la ruta al depth .bin original usando subject y frame_id:
            data_root/faces_0/<subject>/frame_<frame_id>_depth.bin
        Si falla, usa la heurística _infer_depth_path como fallback.
        """
        subject = str(row.get("subject", "")).strip()
        frame_id = str(row.get("frame_id", "")).strip()  # p.ej. "00388" o "frame_00388"

        if subject and frame_id:
            # Aseguramos que tenga el prefijo "frame_"
            if not frame_id.startswith("frame_"):
                frame_name = f"frame_{frame_id}"
            else:
                frame_name = frame_id

            cand = os.path.join(
                self.data_root,
                "faces_0",
                subject,
                f"{frame_name}_depth.bin",
            )
            if os.path.isfile(cand):
                return cand

        # Fallback: heurística anterior (por si en algún momento tienes depth_crops_yolo)
        rgb_path = self._resolve_path(row["image_path"])
        return self._infer_depth_path(rgb_path)

    def _load_depth_bin(self, depth_path: str) -> np.ndarray:
        """
        Lee el depth .bin comprimido de BIWI y lo devuelve como imagen 2D (float32).
        Implementa en Python el loadDepthImageCompressed de los autores.
        """
        with open(depth_path, "rb") as f:
            # Leer width y height (2 int de 4 bytes, little-endian)
            header = f.read(8)
            if len(header) != 8:
                raise ValueError(f"Cannot read width/height from {depth_path}")
            im_width, im_height = struct.unpack("ii", header)  # nativo (little-endian)

            num_pixels = im_width * im_height
            depth_flat = np.zeros(num_pixels, dtype=np.int16)

            p = 0
            while p < num_pixels:
                # numempty
                buf = f.read(4)
                if len(buf) != 4:
                    raise ValueError(f"Cannot read numempty from {depth_path} at p={p}")
                (numempty,) = struct.unpack("i", buf)

                # rellenar ceros
                if numempty > 0:
                    end_zero = min(p + numempty, num_pixels)
                    depth_flat[p:end_zero] = 0
                else:
                    end_zero = p

                # numfull
                buf = f.read(4)
                if len(buf) != 4:
                    raise ValueError(f"Cannot read numfull from {depth_path} at p={p}")
                (numfull,) = struct.unpack("i", buf)

                # leer los valores int16 reales
                if numfull > 0:
                    bytes_to_read = numfull * 2
                    buf = f.read(bytes_to_read)
                    if len(buf) != bytes_to_read:
                        raise ValueError(
                            f"Cannot read depth values from {depth_path} at p={p}"
                        )
                    vals = np.frombuffer(buf, dtype=np.int16)
                    start_full = end_zero
                    end_full = min(start_full + numfull, num_pixels)
                    depth_flat[start_full:end_full] = vals[: (end_full - start_full)]

                # avanzar puntero global
                p += numempty + numfull

            # reshape a (H,W) y pasar a float32 (milímetros → opcionalmente metros)
            depth_img = depth_flat.reshape(im_height, im_width).astype(np.float32)

            return depth_img


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        rgb_path  = self._resolve_path(row["image_path"])
        mask_path = self._resolve_path(row["mask_path"])

        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb)
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)

        # --- Depth FULL FRAME desde faces_0 ---
        depth_path_full = self._get_depth_path_from_row(row)

        # DEBUG: ver qué profundidad está encontrando
        if idx < 5:  # o quítalo si quieres ver todo
            print(f"[DEBUG] idx={idx}")
            print(f"  rgb_path      = {rgb_path}")
            print(f"  subject       = {row.get('subject')}")
            print(f"  frame_id      = {row.get('frame_id')}")
            print(f"  depth_path_full = {depth_path_full}")
            print(f"  exists?       = {os.path.isfile(depth_path_full) if depth_path_full else False}")

        depth_full = np.zeros((rgb_np.shape[0], rgb_np.shape[1]), dtype=np.float32)

        if depth_path_full and os.path.isfile(depth_path_full):
            depth_full = self._load_depth_bin(depth_path_full)  # H,W
            H, W = depth_full.shape

            xmin = int(float(row.get("bbox_xmin", 0)))
            ymin = int(float(row.get("bbox_ymin", 0)))
            xmax = int(float(row.get("bbox_xmax", W)))
            ymax = int(float(row.get("bbox_ymax", H)))

            xmin = max(0, min(xmin, W - 1))
            xmax = max(0, min(xmax, W))
            ymin = max(0, min(ymin, H - 1))
            ymax = max(0, min(ymax, H))

            depth_crop = depth_full[ymin:ymax, xmin:xmax]

            # DEBUG: ver si el recorte tiene info
            if idx < 5:
                print(f"  depth_full min/max = {depth_full.min()} / {depth_full.max()}")
                print(f"  depth_crop shape   = {depth_crop.shape}")
                print(f"  depth_crop min/max = {depth_crop.min()} / {depth_crop.max()}")
        else:
            depth_crop = np.zeros(rgb_np.shape[:2], dtype=np.float32)
            if idx < 5:
                print("  [WARN] NO depth file found, depth_crop set to zeros.")

        depth_np = depth_crop


        # --- Build 4-channel input [RGBD] (224x224) ---
        rgb_resized = rgb.resize((224, 224), Image.BILINEAR)
        rgb_arr = np.array(rgb_resized).astype(np.float32) / 255.0  # H,W,3

        depth_img = Image.fromarray(depth_np)
        depth_resized = depth_img.resize((224, 224), Image.NEAREST)
        depth_arr = np.array(depth_resized).astype(np.float32)

        if depth_arr.max() > 0:
            depth_arr = depth_arr / depth_arr.max()
        depth_arr = depth_arr[..., None]  # H,W,1

        rgbd_arr = np.concatenate([rgb_arr, depth_arr], axis=-1)
        rgbd_t = torch.from_numpy(rgbd_arr).permute(2, 0, 1)  # [4,224,224]

        # Angles
        yaw = float(row["yaw"])
        pitch = float(row["pitch"])
        roll = float(row["roll"])
        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)

        meta = {
            "rgb_path":  rgb_path,
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

    # Ajuste simple: izquierda/derecha con yaw, arriba/abajo con pitch
    dx = np.sin(yaw) * np.cos(pitch)
    dy = -np.sin(pitch)

    # Flecha más larga
    length = min(h, w) * 1
    x2 = cx + dx * length
    y2 = cy + dy * length

    ax.arrow(
        cx, cy,
        x2 - cx, y2 - cy,
        head_width=h * 0.04,
        head_length=h * 0.06,
        length_includes_head=True,
        color=color,
        linewidth=3,
        alpha=0.95,
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

    rgb   = sample["rgb_np"]
    depth = sample["depth_np"]
    mask  = sample["mask_np"]
    gt    = sample["gt_angles"]       # [yaw, pitch, roll]
    pred  = sample["pred_angles"]     # [yaw, pitch, roll]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 1) RGB original
    ax = axes[0]
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("RGB", fontsize=20)


   

    # 2) Depth
    ax = axes[1]
    if depth.ndim == 2:
        d = depth.astype(np.float32)

        if np.all(d == 0):
            # Caso típico: no se encontró el archivo de depth y quedó todo en cero
            ax.imshow(d, cmap="gray")
            ax.set_title("Depth (all zeros)", fontsize=10)
        else:
            # Normalización para visualizar mejor
            d_min, d_max = d.min(), d.max()
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = d
            ax.imshow(d_norm, cmap="viridis")
            ax.set_title("Depth", fontsize=20)
    else:
        ax.imshow(depth)
        ax.set_title("Depth", fontsize=20)

    ax.axis("off")




    # 3) Mask
    ax = axes[2]
    ax.imshow(mask, cmap="gray")
    ax.axis("off")
    ax.set_title("Mask", fontsize=20)

    # 4) RGB con pose predicha (flecha roja)
    ax = axes[3]
    ax.imshow(rgb)
    draw_head_pose_arrow(ax, rgb, pred[0], pred[1], color="red")
    ax.axis("off")
    ax.set_title("Pred Pose", fontsize=20)

    # 5) RGB con pose GT (flecha verde)
    ax = axes[4]
    ax.imshow(rgb)
    draw_head_pose_arrow(ax, rgb, gt[0], gt[1], color="lime")
    ax.axis("off")
    ax.set_title("GT Pose", fontsize=20)

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
