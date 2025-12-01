#!/usr/bin/env python3
"""
2_build_biwi_yolo_crops_and_csv.py

Use a YOLO face detector (Ultralytics) on Biwi RGB images to:
  - Detect face bounding boxes on RGB.
  - Crop RGB and corresponding depth-mask images using that bbox.
  - Convert pose.txt -> yaw, pitch, roll (degrees).
  - Save cropped RGB/masks to out_root, and build train/val CSVs.

CSV columns:
    image_path,yaw,pitch,roll,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,mask_path,subject,frame_id

Your existing train_biwi_vit.py will only read:
    image_path,yaw,pitch,roll
and will ignore the extra columns (which you can still use later).
"""

import os
import csv
import math
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import cv2
from ultralytics import YOLO


# -----------------------------------------------------------
# Pose utilities
# -----------------------------------------------------------

def read_pose_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Biwi pose.txt file.

    Format:
        Lines 1-3: 3x3 rotation matrix
        Line 4:    head center (x, y, z)

    Returns:
        R      : np.ndarray, shape (3, 3)
        center : np.ndarray, shape (3,)
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 4:
        raise ValueError(f"pose.txt file {path} does not contain 4 numeric lines.")

    R = np.array([[float(v) for v in lines[i].split()] for i in range(3)], dtype=np.float32)
    center = np.array([float(v) for v in lines[3].split()], dtype=np.float32)
    return R, center


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to Tait-Bryan angles (yaw, pitch, roll)
    using ZYX convention:

        R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Returns:
        angles : np.ndarray (3,) = [yaw, pitch, roll] in radians
    """
    assert R.shape == (3, 3), f"R must be 3x3, got {R.shape}"

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])        # rotation around Z
        pitch = math.atan2(-R[2, 0], sy)          # rotation around Y
        roll = math.atan2(R[2, 1], R[2, 2])       # rotation around X
    else:
        # Gimbal lock: sy ~ 0
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return np.array([yaw, pitch, roll], dtype=np.float32)


# -----------------------------------------------------------
# YOLO face detection utilities
# -----------------------------------------------------------

def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model (Ultralytics) for face detection.

    The model should be trained for faces (e.g., yolov8n-face.pt or similar).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    model = YOLO(model_path)
    return model


def detect_face_bbox_yolo(
    rgb_img: Image.Image,
    model: YOLO,
    conf_thres: float = 0.3,
    img_size: int = 640,
    center_radius: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO on an RGB image and return a single face bbox.

    Selection strategy:
      1) Filter detections by confidence >= conf_thres.
      2) For each detection, compute the normalized distance from its center
         to the image center.
      3) Pick the detection with *minimum* distance to the image center.
      4) If that distance is larger than 'center_radius', return None
         (skip the frame).

    center_radius is expressed in normalized units (0–1) using the image width/height.
    Roughly, 0.25 ~ solo un círculo central de radio 25% de la imagen.
    """
    # PIL -> numpy (RGB)
    img_np = np.array(rgb_img)
    h, w, _ = img_np.shape
    cx_img, cy_img = w / 2.0, h / 2.0

    # YOLO inference
    results = model(
        img_np,
        imgsz=img_size,
        conf=conf_thres,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = results[0].boxes.conf.cpu().numpy()  # (N,)

    # 1) confidence filter
    keep = confs >= conf_thres
    if not np.any(keep):
        return None

    boxes = boxes[keep]
    confs = confs[keep]

    # 2) compute normalized center distance for each box
    best_idx = None
    best_dist2 = None

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0

        # normalizar por ancho/alto para tener algo invariante a resolución
        dx = (cx_box - cx_img) / w
        dy = (cy_box - cy_img) / h
        dist2 = dx * dx + dy * dy  # distancia al centro al cuadrado

        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_idx = i

    if best_idx is None:
        return None

    # 3) check radius constraint
    dist = math.sqrt(best_dist2)
    if dist > center_radius:
        # la mejor cara está demasiado lejos del centro -> descartar frame
        return None

    # 4) return bbox of the best (closest-to-center) detection
    x1, y1, x2, y2 = boxes[best_idx]
    xmin = int(round(x1))
    ymin = int(round(y1))
    xmax = int(round(x2)) - 1  # inclusive
    ymax = int(round(y2)) - 1

    return xmin, ymin, xmax, ymax




def apply_padding_to_bbox(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_w: int,
    img_h: int,
    pad: int,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox by 'pad' pixels in all directions, and clamp to image bounds.

    Args:
        xmin, ymin, xmax, ymax: original bbox (inclusive coords)
        img_w, img_h: image width and height
        pad: number of pixels to pad on each side

    Returns:
        (xmin_p, ymin_p, xmax_p, ymax_p) as ints, still inclusive.
    """
    xmin_p = max(0, xmin - pad)
    ymin_p = max(0, ymin - pad)
    xmax_p = min(img_w - 1, xmax + pad)
    ymax_p = min(img_h - 1, ymax + pad)
    return xmin_p, ymin_p, xmax_p, ymax_p


# -----------------------------------------------------------
# Main processing: build crops + CSV using YOLO bboxes
# -----------------------------------------------------------

def process_biwi_with_yolo(
    faces_dir: Path,
    masks_dir: Path,
    out_root: Path,
    yolo_model: YOLO,
    abs_paths: bool,
    pad: int,
    conf_thres: float,
    yolo_img_size: int,
) -> List[Dict[str, str]]:
    """
    Iterate over subjects and frames, detect face bboxes on RGB with YOLO,
    crop RGB + mask, and collect annotations.

    Returns:
        List of dict rows with keys:
            image_path, yaw, pitch, roll,
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
            mask_path, subject, frame_id
    """
    if not faces_dir.exists():
        raise FileNotFoundError(f"faces_dir not found: {faces_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    rgb_out_root = out_root / "rgb_crops_yolo"
    mask_out_root = out_root / "mask_crops_yolo"

    entries: List[Dict[str, str]] = []

    # Subjects: 01, 02, ..., 24 (or whatever is present)
    for subject_dir in sorted(faces_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name

        subject_mask_dir = masks_dir / subject_id
        if not subject_mask_dir.exists():
            print(f"[WARN] No mask dir for subject {subject_id}, skipping.")
            continue

        # Iterate over RGB frames of this subject
        for rgb_path in sorted(subject_dir.glob("frame_*_rgb.png")):
            frame_id = rgb_path.stem.replace("frame_", "").replace("_rgb", "")

            pose_txt = subject_dir / f"frame_{frame_id}_pose.txt"
            mask_path = subject_mask_dir / f"frame_{frame_id}_depth_mask.png"

            if not pose_txt.exists() or not mask_path.exists():
                # Skip if pose or mask missing
                continue

            # --- Read pose and convert to yaw, pitch, roll (degrees) ---
            R, center = read_pose_txt(pose_txt)
            angles_rad = rotation_matrix_to_euler_zyx(R)
            angles_deg = np.rad2deg(angles_rad)

            # --- Load RGB and detect face bbox with YOLO ---
            rgb_img = Image.open(rgb_path).convert("RGB")
            w, h = rgb_img.size

            bbox = detect_face_bbox_yolo(
                rgb_img,
                yolo_model,
                conf_thres=conf_thres,
                img_size=yolo_img_size,
            )
            if bbox is None:
                print(f"[WARN] No YOLO face detected in {rgb_path}, skipping.")
                continue

            xmin, ymin, xmax, ymax = bbox

            # Apply padding and clamp
            xmin_p, ymin_p, xmax_p, ymax_p = apply_padding_to_bbox(
                xmin, ymin, xmax, ymax, img_w=w, img_h=h, pad=pad
            )

            # Pillow crop uses (left, upper, right, lower) with right/lower exclusive
            crop_box = (xmin_p, ymin_p, xmax_p + 1, ymax_p + 1)

            # Crop RGB
            rgb_crop = rgb_img.crop(crop_box)

            # Crop mask with the same box
            mask_img = Image.open(mask_path).convert("L")
            mask_crop = mask_img.crop(crop_box)

            # --- Build output paths ---
            rgb_out_dir = rgb_out_root / subject_id
            mask_out_dir = mask_out_root / subject_id
            rgb_out_dir.mkdir(parents=True, exist_ok=True)
            mask_out_dir.mkdir(parents=True, exist_ok=True)

            rgb_out_path = rgb_out_dir / f"frame_{frame_id}_rgb.png"
            mask_out_path = mask_out_dir / f"frame_{frame_id}_mask.png"

            rgb_crop.save(rgb_out_path)
            mask_crop.save(mask_out_path)

            # Path to store in CSV
            img_path_str = str(rgb_out_path.resolve()) if abs_paths else str(rgb_out_path)
            mask_path_str = str(mask_out_path.resolve()) if abs_paths else str(mask_out_path)

            entries.append(
                {
                    "image_path": img_path_str,
                    "yaw": float(angles_deg[0]),
                    "pitch": float(angles_deg[1]),
                    "roll": float(angles_deg[2]),
                    "bbox_xmin": int(xmin_p),
                    "bbox_ymin": int(ymin_p),
                    "bbox_xmax": int(xmax_p),
                    "bbox_ymax": int(ymax_p),
                    "mask_path": mask_path_str,
                    "subject": subject_id,
                    "frame_id": frame_id,
                }
            )

    if not entries:
        raise RuntimeError("No valid (RGB + YOLO bbox + mask + pose) samples found.")

    print(f"[INFO] Collected {len(entries)} YOLO-based cropped samples.")
    return entries


def write_csv(path: Path, rows: List[Dict[str, str]]):
    """
    Write a CSV with header including:
        image_path,yaw,pitch,roll,...
    Extra columns are fine for train_biwi_vit.py.
    """
    if not rows:
        raise ValueError("Empty rows list for CSV writing.")

    os.makedirs(path.parent, exist_ok=True)

    base_cols = ["image_path", "yaw", "pitch", "roll"]
    other_cols = [c for c in rows[0].keys() if c not in base_cols]
    fieldnames = base_cols + other_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO-based cropped RGB/mask images and CSV manifests from Biwi."
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        required=True,
        help="Path to archive/faces_0 directory.",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Path to archive/head_pose_masks directory.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root directory for cropped images + CSVs.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Output CSV path for training set.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="Output CSV path for validation set.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (0 < val_split < 1).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=15,
        help="Padding (in pixels) added around the YOLO face bounding box.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--abs-paths",
        action="store_true",
        help="Store absolute image paths in CSV (recommended).",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        required=True,
        help="Path to YOLO face detector .pt file (Ultralytics).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.3,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--yolo-img-size",
        type=int,
        default=640,
        help="Inference size for YOLO.",
    )
    args = parser.parse_args()

    faces_dir = Path(args.faces_dir)
    masks_dir = Path(args.masks_dir)
    out_root = Path(args.out_root)

    # Load YOLO face model
    yolo_model = load_yolo_model(args.yolo_model)

    # Build cropped samples list
    entries = process_biwi_with_yolo(
        faces_dir=faces_dir,
        masks_dir=masks_dir,
        out_root=out_root,
        yolo_model=yolo_model,
        abs_paths=args.abs_paths,
        pad=args.pad,
        conf_thres=args.yolo_conf,
        yolo_img_size=args.yolo_img_size,
    )

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(entries)

    n_total = len(entries)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    train_rows = entries[:n_train]
    val_rows = entries[n_train:]

    print(f"[INFO] Train samples: {n_train} | Val samples: {n_val}")

    # Write CSVs
    write_csv(Path(args.train_csv), train_rows)
    write_csv(Path(args.val_csv), val_rows)


if __name__ == "__main__":
    main()
