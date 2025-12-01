#!/usr/bin/env python3
# precompute_dipser_bboxes.py
#
# Lee CSVs de DIPSER con columnas:
#   image_path,label,clip_id,datetime
# Detecta la cara más grande en cada imagen y genera nuevos CSV con
# columnas extra: x1,y1,x2,y2.

import os
import csv
import argparse
from tqdm.auto import tqdm
import cv2


def find_face_bbox(img):
    """Devuelve (x1, y1, x2, y2) de la cara más grande, o None si no hay."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(x + w), int(y + h)


def process_csv(in_csv, out_csv):
    """
    Lee in_csv, intenta detectar una cara por imagen, y escribe out_csv
    con columnas extra x1,y1,x2,y2.
    Si no se detecta cara, se SALTA la fila.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(in_csv, "r") as f_in, open(out_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames)

        if "image_path" not in fieldnames:
            raise ValueError(f"Espero columna 'image_path', encontré: {fieldnames}")

        # añadimos columnas bbox si no están
        for c in ["x1", "y1", "x2", "y2"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        rows = list(reader)
        for row in tqdm(rows, desc=f"Procesando {os.path.basename(in_csv)}"):
            img_path = row["image_path"]

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            bbox = find_face_bbox(img)
            if bbox is None:
                # opcional: podrías poner bbox=imagen completa en vez de saltar
                continue

            x1, y1, x2, y2 = bbox
            row["x1"] = x1
            row["y1"] = y1
            row["x2"] = x2
            row["y2"] = y2
            writer.writerow(row)

    print(f"[OK] Escrito con bboxes: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_train = os.path.join(args.out_dir, "train_bbox.csv")
    out_val   = os.path.join(args.out_dir, "val_bbox.csv")

    process_csv(args.train_csv, out_train)
    process_csv(args.val_csv, out_val)


if __name__ == "__main__":
    main()
