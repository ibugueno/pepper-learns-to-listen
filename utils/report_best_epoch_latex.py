#!/usr/bin/env python3
import json
import argparse
import os
import glob

# -------------------------------
# UTILIDAD: REDONDEAR RECURSIVAMENTE
# -------------------------------
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

# -------------------------------
# DETECCIÓN AUTOMÁTICA DE DATASET
# -------------------------------
def detect_dataset_type(epoch_entry):
    if "val" in epoch_entry and isinstance(epoch_entry["val"], dict):
        return "classification_nested"
    if "val_loss" in epoch_entry:
        return "regression_flat"
    raise ValueError("No se pudo detectar tipo de dataset.")

# -------------------------------
# SELECCIÓN DE MÉTRICA PRINCIPAL
# -------------------------------
def choose_primary_metric(dataset_type, epoch_entry):
    if dataset_type == "classification_nested":
        val = epoch_entry["val"]
        for cand in ["acc", "f1", "macro_f1", "prec", "rec", "loss"]:
            if cand in val:
                mode = "max" if cand != "loss" else "min"
                return f"val.{cand}", mode
        raise ValueError("No hay métricas válidas en val.*")

    if dataset_type == "regression_flat":
        return "val_loss", "min"

    raise ValueError("Tipo de dataset desconocido.")

# -------------------------------
# OBTENER MÉTRICA (NESTED / FLAT)
# -------------------------------
def get_metric(entry, metric_key):
    if "." in metric_key:
        a, b = metric_key.split(".")
        return entry[a][b]
    return entry[metric_key]

# -------------------------------
# EXTRAER SOLO METRICAS val.* (Y REDONDEAR TODAS)
# -------------------------------
def extract_val_metrics(epoch_entry):
    result = {}

    # Clasificación anidada (AVA, DIPSeR)
    if "val" in epoch_entry and isinstance(epoch_entry["val"], dict):
        for k, v in epoch_entry["val"].items():
            result[f"val.{k}"] = round_recursive(v, 4)
        return result

    # Regresión (BIWI)
    for k, v in epoch_entry.items():
        if k.startswith("val"):
            result[k] = round_recursive(v, 4)

    return result

# -------------------------------
# FILTRAR HYPERPARÁMETROS RELEVANTES
# -------------------------------
INTERESTING_ARGS = {
    # comunes
    "img_size",
    "num_frames",
    "fps",
    "batch_size",
    "num_workers",
    "epochs",
    "lr",
    "opt",
    "weight_decay",
    "max_grad_norm",
    "amp",
    "seed",
    # AVA
    "img_embed_dim",
    "temporal",
    "temporal_hidden",
    "no_img_aug",
    # BIWI
    "model",
    "pretrained",
    "augment",
    "drop",
    "drop_path",
    "reg_hidden",
    "freeze_backbone",
    "sched",
    "step_size",
    "gamma",
    "loss",
    # DIPSeR
    "n_classes",
    "backbone",
}

def filter_relevant_args(args_dict):
    filtered = {}
    for k, v in args_dict.items():
        if k in INTERESTING_ARGS:
            filtered[k] = round_recursive(v, 4)
    return filtered

# -------------------------------
# PROCESAR UN ARCHIVO JSON
# -------------------------------
def process_file(path):
    with open(path, "r") as f:
        history = json.load(f)

    epochs = history["epochs"]
    args_cfg = history.get("args", {})

    ds_type = detect_dataset_type(epochs[0])
    metric_key, mode = choose_primary_metric(ds_type, epochs[0])

    best_entry = None
    best_value = None

    for ep in epochs:
        value = get_metric(ep, metric_key)
        if best_value is None:
            best_value = value
            best_entry = ep
        else:
            if mode == "max" and value > best_value:
                best_value = value
                best_entry = ep
            elif mode == "min" and value < best_value:
                best_value = value
                best_entry = ep

    val_metrics = extract_val_metrics(best_entry)
    args_filtered = filter_relevant_args(args_cfg)
    best_value = round_recursive(best_value, 4)

    return {
        "file": path,
        "epoch": best_entry["epoch"],
        "metric_used": metric_key,
        "mode": mode,
        "value": best_value,
        "val_metrics": val_metrics,
        "args": args_filtered,
    }

# -------------------------------
# FORMATEO LATEX
# -------------------------------
def latex_escape(s: str) -> str:
    # escapamos underscores para que no reviente
    return s.replace("_", "\\_")

def format_value_for_latex(v):
    if isinstance(v, list):
        # lista tipo [0.8498, 0.7136, 0.5906]
        return ", ".join(str(x) for x in v)
    return str(v)

def dataset_name_from_path(root, full_path):
    rel = os.path.relpath(full_path, root)
    # por ejemplo: "ava/ava_spk_img_gru32/history.json" → "ava_spk_img_gru32"
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(parts[-1])[0]

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Genera tablas LaTeX de hiperparámetros y métricas de validación para cada history.json"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../output",
        help="Directorio raíz donde buscar history.json"
    )
    args = parser.parse_args()

    pattern = os.path.join(args.root, "**", "history.json")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"% No se encontraron JSON en {args.root}")
        return

    # Ordenamos para que la salida sea estable
    files = sorted(files)

    for f in files:
        try:
            r = process_file(f)
        except Exception as e:
            print(f"% [ERROR] {f}: {e}")
            continue

        ds_name = dataset_name_from_path(args.root, r["file"])
        ds_name_tex = latex_escape(ds_name)

        # ----- Tabla de hiperparámetros -----
        print(f"% ===== HYPERPARAMETERS: {ds_name} =====")
        print("\\begin{table}[ht]")
        print("    \\centering")
        print(f"    \\caption{{Training hyperparameters for {ds_name_tex}.}}")
        print("    \\begin{tabular}{ll}")
        print("        \\hline")
        print("        Hyperparameter & Value \\\\")
        print("        \\hline")

        for k in sorted(r["args"].keys()):
            key_tex = latex_escape(k)
            val_tex = latex_escape(format_value_for_latex(r["args"][k]))
            print(f"        {key_tex} & {val_tex} \\\\")

        print("        \\hline")
        print("    \\end{tabular}")
        print("\\end{table}")
        print()

        # ----- Tabla de métricas de validación -----
        print(f"% ===== VALIDATION METRICS: {ds_name} =====")
        print("\\begin{table}[ht]")
        print("    \\centering")
        print(f"    \\caption{{Validation metrics for {ds_name_tex} (best epoch = {r['epoch']}).}}")
        print("    \\begin{tabular}{ll}")
        print("        \\hline")
        print("        Metric & Value \\\\")
        print("        \\hline")

        # métrica principal primero
        main_key = r["metric_used"]
        main_val = r["val_metrics"].get(main_key, r["value"])
        main_key_tex = latex_escape(main_key)
        main_val_tex = latex_escape(format_value_for_latex(main_val))
        print(f"        {main_key_tex} & {main_val_tex} \\\\")

        # resto de las métricas
        for k in sorted(r["val_metrics"].keys()):
            if k == main_key:
                continue
            key_tex = latex_escape(k)
            val_tex = latex_escape(format_value_for_latex(r["val_metrics"][k]))
            print(f"        {key_tex} & {val_tex} \\\\")

        print("        \\hline")
        print("    \\end{tabular}")
        print("\\end{table}")
        print()
        print("% ------------------------------------------------------------")
        print()

if __name__ == "__main__":
    main()
