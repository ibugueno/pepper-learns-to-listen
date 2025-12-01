#!/usr/bin/env python3
# prepare_dipser_engagement_no_copy.py
#
# Prepara DIPSER para entrenamiento de engagement SIN copiar imágenes.
# Usa directamente:
#   frames_dir = .../subject_XY/images
#   labels_path = out_root/labels/<group>_<experiment>_<subject>.npy
#
# Manifest columns: clip_id,frames_dir,labels_path

import os
import json
import argparse
import csv
import random
from typing import List, Dict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Prepare DIPSER engagement (no image copy)")
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
        help="Carpeta de salida para labels y manifests",
    )
    p.add_argument(
        "--labeler-json",
        type=str,
        default="labeler_01.json",
        help="Nombre del JSON de anotaciones (p.ej. labeler_01.json)",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=32,
        help="Mínimo de frames etiquetados para mantener un clip",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proporción de clips para validación (resto para train)",
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


def collect_clips(args):
    dipser_root = args.dipser_root
    out_labels_root = os.path.join(args.out_root, "labels")
    ensure_dir(out_labels_root)

    clips = []  # lista de dicts: {clip_id, frames_dir, labels_path}

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

                # Construimos lista (filename, clase) sólo para los frames que existan
                entries = []
                for dt, att in att_map.items():
                    fname = datetime_to_filename(dt)
                    img_path = os.path.join(images_dir, fname)
                    if os.path.isfile(img_path):
                        # attention {1,2,3} -> clases {0,1,2}
                        entries.append((fname, att - 1))

                if len(entries) < args.min_frames:
                    continue

                # Ordenar por nombre (aprox. temporal)
                entries.sort(key=lambda x: x[0])

                clip_id = f"{group}_{exp}_{subj}"
                labels = np.array([cls for _, cls in entries], dtype=np.int64)
                labels_path = os.path.join(out_labels_root, f"{clip_id}.npy")
                np.save(labels_path, labels)

                clips.append(
                    {
                        "clip_id": clip_id,
                        "frames_dir": images_dir,   # 👈 usamos el directorio ORIGINAL
                        "labels_path": labels_path,
                    }
                )

                print(f"OK clip {clip_id}: {len(labels)} frames etiquetados")

    print(f"Total clips válidos: {len(clips)}")
    return clips


def split_and_write_manifests(clips: List[Dict], out_root: str, val_ratio: float):
    random.shuffle(clips)
    n_total = len(clips)
    n_val = int(round(n_total * val_ratio))
    val_clips = clips[:n_val]
    train_clips = clips[n_val:]

    def write_manifest(path, rows):
        fieldnames = ["clip_id", "frames_dir", "labels_path"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    manifests_dir = os.path.join(out_root, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    train_manifest = os.path.join(manifests_dir, "manifest_train.csv")
    val_manifest = os.path.join(manifests_dir, "manifest_val.csv")

    write_manifest(train_manifest, train_clips)
    write_manifest(val_manifest, val_clips)

    print(f"Train manifest: {train_manifest} ({len(train_clips)} clips)")
    print(f"Val   manifest: {val_manifest} ({len(val_clips)} clips)")


def main():
    args = parse_args()
    random.seed(42)

    clips = collect_clips(args)
    if not clips:
        print("No se encontraron clips válidos. Revisa rutas / labeler_json / min_frames.")
        return

    split_and_write_manifests(clips, args.out_root, args.val_ratio)


if __name__ == "__main__":
    main()
