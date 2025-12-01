#!/usr/bin/env python3
# prepare_dipser_engagement_single.py
#
# Prepara DIPSER para single-frame engagement:
#   - Recorre group_XX/experiment_YY/subject_ZZ
#   - Usa labeler_01.json (por defecto) y el campo "attention"
#   - NO copia imágenes, sólo genera:
#       out_root/train.csv
#       out_root/val.csv
#
# Columnas CSV: image_path,label,clip_id,datetime

import os
import json
import argparse
import csv
import random
from typing import List, Dict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Prepare DIPSER engagement (single-frame)")
    p.add_argument(
        "--dipser-root",
        type=str,
        required=True,
        help="Ruta a DIPSER/V5/DIPSER (contiene group_01, group_02, ...)",
    )
    p.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Carpeta de salida para los CSV (train/val)",
    )
    p.add_argument(
        "--labeler-json",
        type=str,
        default="labeler_01.json",
        help="Nombre del JSON de anotaciones (p.ej. labeler_01.json)",
    )
    p.add_argument(
        "--min-frames-per-subject",
        type=int,
        default=8,
        help="Descartar sujetos con menos de N frames anotados válidos",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proporción de muestras para validación (resto para train)",
    )
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def datetime_to_filename(dt: str) -> str:
    """
    "10:40:43:047895" -> "10_40_43_047895.png"
    """
    base = dt.replace(":", "_")
    return base + ".png"


def load_attention_map(json_path: str) -> Dict[str, int]:
    """
    Carga el JSON del labeler y devuelve:
        { "10:40:43:047895": attention_int, ... }
    Sólo entradas con campo "attention".
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    att_map = {}
    for entry in data:
        dt = entry.get("datetime", None)
        att = entry.get("attention", None)
        if dt is None or att is None:
            continue
        try:
            att_int = int(att)
        except Exception:
            continue
        att_map[dt] = att_int
    return att_map


def collect_samples(args):
    dipser_root = args.dipser_root
    all_samples = []  # lista de dicts: image_path,label,clip_id,datetime

    for group in sorted(d for d in os.listdir(dipser_root) if d.startswith("group_")):
        group_dir = os.path.join(dipser_root, group)
        if not os.path.isdir(group_dir):
            continue

        for exp in sorted(d for d in os.listdir(group_dir) if d.startswith("experiment_")):
            exp_dir = os.path.join(group_dir, exp)
            if not os.path.isdir(exp_dir):
                continue

            for subj in sorted(d for d in os.listdir(exp_dir) if d.startswith("subject_")):
                subj_dir = os.path.join(exp_dir, subj)
                if not os.path.isdir(subj_dir):
                    continue

                images_dir = os.path.join(subj_dir, "images")
                labels_dir = os.path.join(subj_dir, "labels")
                if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                    continue

                json_path = os.path.join(labels_dir, args.labeler_json)
                if not os.path.isfile(json_path):
                    continue

                att_map = load_attention_map(json_path)
                if not att_map:
                    continue

                subject_samples = []
                clip_id = f"{group}_{exp}_{subj}"

                for dt, att in att_map.items():
                    fname = datetime_to_filename(dt)
                    img_path = os.path.join(images_dir, fname)
                    if os.path.isfile(img_path):
                        # attention {1,2,3,...} -> clases 0,1,2,... (puedes cambiarlo luego)
                        cls = att - 1
                        subject_samples.append({
                            "image_path": os.path.abspath(img_path),
                            "label": cls,
                            "clip_id": clip_id,
                            "datetime": dt
                        })

                if len(subject_samples) < args.min_frames_per_subject:
                    continue

                subject_samples.sort(key=lambda s: s["image_path"])
                all_samples.extend(subject_samples)
                print(f"{clip_id}: {len(subject_samples)} muestras válidas")

    print(f"Total de muestras single-frame: {len(all_samples)}")
    return all_samples


def split_and_write_csv(samples: List[Dict], out_root: str, val_ratio: float):
    ensure_dir(out_root)
    random.shuffle(samples)
    n_total = len(samples)
    n_val = int(round(n_total * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    def write_csv(path, rows):
        fieldnames = ["image_path", "label", "clip_id", "datetime"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    train_csv = os.path.join(out_root, "train.csv")
    val_csv = os.path.join(out_root, "val.csv")
    write_csv(train_csv, train_samples)
    write_csv(val_csv, val_samples)

    print(f"train.csv: {train_csv} ({len(train_samples)} filas)")
    print(f"val.csv:   {val_csv} ({len(val_samples)} filas)")


def main():
    args = parse_args()
    random.seed(42)

    samples = collect_samples(args)
    if not samples:
        print("No se encontraron muestras válidas. Revisa rutas / labeler_json.")
        return

    split_and_write_csv(samples, args.out_root, args.val_ratio)


if __name__ == "__main__":
    main()
