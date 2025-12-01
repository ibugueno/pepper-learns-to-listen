#!/usr/bin/env python3
# prepare_ava_activespeaker_imgonly.py
#
# A partir de:
#   - vídeos AVA (p.ej. videos/speech)
#   - anotaciones AVA ActiveSpeaker (*.csv)
# genera:
#   - crops de cara por track (frames por track)
#   - labels.npy por track (0=NOT_SPEAKING, 1=SPEAKING*)
#   - manifests CSV (train/val) para train_ava_activespeaker_imgonly.py
#
# CSV de AVA ActiveSpeaker (sin header), columnas:
#   0: video_id
#   1: timestamp (segundos)
#   2: x1 (normalizado 0-1)
#   3: y1
#   4: x2
#   5: y2
#   6: label_str (NOT_SPEAKING, SPEAKING_AUDIBLE, etc.)
#   7: segment+track, ej: 0f39OWEqJ24_1020_1080:1

import os
import csv
import glob
import argparse
import random
from collections import defaultdict

import numpy as np
import cv2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def find_video_file(videos_root: str, video_id: str):
    """
    Busca el archivo de vídeo correspondiente a video_id en videos_root,
    probando extensiones comunes de AVA (mkv, mp4, webm).
    """
    exts = [".mkv", ".mp4", ".webm"]
    for ext in exts:
        cand = os.path.join(videos_root, video_id + ext)
        if os.path.isfile(cand):
            return cand
    # plan B: cualquier archivo que empiece con video_id.
    pattern = os.path.join(videos_root, video_id + ".*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def parse_activespeaker_csv(csv_path: str):
    """
    Devuelve un dict:
      tracks[track_id] = lista de dicts con:
        {
          "ts": float,
          "x1": float, "y1": float, "x2": float, "y2": float,
          "label": str
        }
    """
    tracks = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 8:
                continue
            video_id = row[0]
            ts = float(row[1])
            x1 = float(row[2])
            y1 = float(row[3])
            x2 = float(row[4])
            y2 = float(row[5])
            label_str = row[6].strip()
            seg_track = row[7].strip()
            # Ejemplo: "0f39OWEqJ24_1020_1080:1" -> track_id="1"
            if ":" in seg_track:
                track_id = seg_track.split(":")[-1]
            else:
                track_id = "0"

            tracks[track_id].append({
                "ts": ts,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": label_str,
                "video_id": video_id,
            })
    # ordena por timestamp cada track
    for t in tracks:
        tracks[t].sort(key=lambda d: d["ts"])
    return tracks


def label_to_int(label_str: str) -> int:
    """
    Mapea las etiquetas de AVA ActiveSpeaker a 0/1.
    Puedes ajustar esto si quieres distinguir más casos.
    """
    s = label_str.upper()
    if "NOT_SPEAKING" in s:
        return 0
    if "SPEAK" in s:  # SPEAKING_AUDIBLE, SPEAKING_UNINTELLIGIBLE, etc.
        return 1
    # fallback: trata cualquier otra cosa como no hablando
    return 0


def crop_track_from_video(video_path: str,
                          track_data,
                          out_frames_dir: str,
                          out_labels_path: str,
                          fps_assumed: float = 25.0,
                          min_box_size: int = 16):
    """
    Dado un vídeo + datos de un track (lista ordenada por ts), genera:
      - frames recortados en out_frames_dir
      - labels.npy (0/1) en out_labels_path

    Aviso: usa cap.set(CAP_PROP_POS_FRAMES, idx) por cada frame (no es lo más rápido,
    pero es simple y suficiente para un prototipo).
    """
    os.makedirs(out_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir vídeo: {video_path}")
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-2:
        video_fps = fps_assumed

    labels = []
    saved = 0

    # obtén tamaño del vídeo
    ret, frame0 = cap.read()
    if not ret:
        print(f"[WARN] No se pudo leer primer frame: {video_path}")
        cap.release()
        return False
    h, w = frame0.shape[:2]
    # volvemos al inicio (por si acaso)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i, entry in enumerate(track_data):
        ts = entry["ts"]
        x1, y1, x2, y2 = entry["x1"], entry["y1"], entry["x2"], entry["y2"]
        label_str = entry["label"]

        # índice de frame aproximado
        frame_idx = int(round(ts * video_fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # no se pudo leer este frame, lo saltamos
            print(f"[WARN] No se pudo leer frame {frame_idx} en {video_path}")
            continue

        # bounding box en píxeles (asumiendo x1,y1,x2,y2 normalizado [0,1])
        x1_px = int(max(0, min(w - 1, x1 * w)))
        y1_px = int(max(0, min(h - 1, y1 * h)))
        x2_px = int(max(0, min(w, x2 * w)))
        y2_px = int(max(0, min(h, y2 * h)))

        if x2_px <= x1_px or y2_px <= y1_px:
            # bbox degenerado
            continue

        crop = frame[y1_px:y2_px, x1_px:x2_px]
        ch, cw = crop.shape[:2]
        if ch < min_box_size or cw < min_box_size:
            # crop demasiado pequeño, lo ignoramos
            continue

        out_name = f"{saved:06d}.jpg"
        out_path = os.path.join(out_frames_dir, out_name)
        cv2.imwrite(out_path, crop)

        y = label_to_int(label_str)
        labels.append(y)
        saved += 1

    cap.release()

    if saved == 0:
        # borra el directorio si no se generó nada
        try:
            os.rmdir(out_frames_dir)
        except OSError:
            pass
        print(f"[INFO] Track sin frames válidos en {video_path}")
        return False

    labels_arr = np.array(labels, dtype=np.float32)
    os.makedirs(os.path.dirname(out_labels_path), exist_ok=True)
    np.save(out_labels_path, labels_arr)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preparar dataset imagen-only para AVA ActiveSpeaker (crops + labels + manifests)."
    )
    parser.add_argument("--videos-root", type=str, required=True,
                        help="Carpeta con los vídeos (p.ej. videos/speech)")
    parser.add_argument("--ann-root", type=str, required=True,
                        help="Carpeta con los .csv de AVA ActiveSpeaker (p.ej. .../ava_activespeaker_train_v1.0)")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Raíz de salida para frames, labels y manifests")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fracción de tracks para train (resto para val)")
    parser.add_argument("--fps-assumed", type=float, default=25.0,
                        help="FPS asumido si el vídeo no reporta fps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    ann_files = sorted(glob.glob(os.path.join(args.ann_root, "*-activespeaker.csv")))
    if not ann_files:
        raise SystemExit(f"No se encontraron CSV en {args.ann_root}")

    frames_root = os.path.join(args.output_root, "frames")
    labels_root = os.path.join(args.output_root, "labels")
    manifests_root = os.path.join(args.output_root, "manifests")
    os.makedirs(frames_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)
    os.makedirs(manifests_root, exist_ok=True)

    samples = []  # para manifest: lista de dicts {clip_id, frames_dir, labels_path}

    for csv_path in ann_files:
        base = os.path.basename(csv_path)
        video_id = base.split("-")[0]
        print(f"=== Procesando video_id={video_id} ({csv_path}) ===")

        video_path = find_video_file(args.videos_root, video_id)
        if video_path is None:
            print(f"[WARN] No se encontró vídeo para {video_id} en {args.videos_root}, se omite.")
            continue

        tracks = parse_activespeaker_csv(csv_path)
        if not tracks:
            print(f"[WARN] Sin tracks en {csv_path}")
            continue

        for track_id, track_data in tracks.items():
            clip_id = f"{video_id}_track{track_id}"
            out_frames_dir = os.path.join(frames_root, clip_id)
            out_labels_path = os.path.join(labels_root, clip_id + ".npy")

            ok = crop_track_from_video(
                video_path=video_path,
                track_data=track_data,
                out_frames_dir=out_frames_dir,
                out_labels_path=out_labels_path,
                fps_assumed=args.fps_assumed,
            )
            if not ok:
                continue

            samples.append({
                "clip_id": clip_id,
                "frames_dir": out_frames_dir,
                "labels_path": out_labels_path
            })

    if not samples:
        raise SystemExit("No se generó ningún sample (revisar paths / anotaciones).")

    # split train/val por track
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(args.train_ratio * n_total)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    def write_manifest(path, rows):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "frames_dir", "labels_path"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    train_manifest = os.path.join(manifests_root, "ava_spk_train.csv")
    val_manifest   = os.path.join(manifests_root, "ava_spk_val.csv")
    write_manifest(train_manifest, train_samples)
    write_manifest(val_manifest,   val_samples)

    print("==========================================")
    print(f"Total tracks: {n_total}")
    print(f"Train tracks: {len(train_samples)} -> {train_manifest}")
    print(f"Val tracks  : {len(val_samples)} -> {val_manifest}")
    print(f"Frames root : {frames_root}")
    print(f"Labels root : {labels_root}")


if __name__ == "__main__":
    main()
