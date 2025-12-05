#!/usr/bin/env python3
import json
import argparse
import os
import glob

# -------------------------------
# UTILIDAD: REDONDEAR RECURSIVAMENTE
# -------------------------------
def round_recursive(x, ndigits=4):
    """Redondea:
       - float → float redondeado
       - int   → se deja igual
       - list  → elemento por elemento
       - dict  → cada valor
    """
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
    # AVA / DIPSeR → clasificación anidada
    if "val" in epoch_entry and isinstance(epoch_entry["val"], dict):
        return "classification_nested"
    # BIWI → regresión plana
    if "val_loss" in epoch_entry:
        return "regression_flat"
    raise ValueError("No se pudo detectar tipo de dataset.")


# -------------------------------
# SELECCIÓN DE MÉTRICA PRINCIPAL
# -------------------------------
def choose_primary_metric(dataset_type, epoch_entry):
    if dataset_type == "classification_nested":
        val = epoch_entry["val"]
        # prioridad razonable para tus setups
        for cand in ["acc", "f1", "macro_f1", "prec", "rec", "loss"]:
            if cand in val:
                mode = "max" if cand != "loss" else "min"
                return f"val.{cand}", mode
        raise ValueError("No hay métricas válidas en val.*")

    if dataset_type == "regression_flat":
        # BIWI → minimizar val_loss
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
    args_rounded = round_recursive(args_cfg, 4)

    return {
        "file": path,
        "epoch": best_entry["epoch"],
        "metric_used": metric_key,
        "mode": mode,
        "value": round_recursive(best_value, 4),
        "val_metrics": val_metrics,
        "args": args_rounded,
    }


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reporta la mejor época usando solo métricas de validación redondeadas e hiperparámetros."
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
        print(f"No se encontraron JSON en {args.root}")
        return

    print("\n==================== RESULTADOS ====================\n")

    for f in files:
        try:
            r = process_file(f)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            continue

        print(f"📄 Archivo: {os.path.relpath(r['file'], args.root)}")
        print(f"   → Mejor época: {r['epoch']}")
        print(f"   → Métrica usada: {r['metric_used']} (modo={r['mode']})")
        print(f"   → Valor: {r['value']}")
        print("   → Métricas de validación (val.*):")
        for k, v in r["val_metrics"].items():
            print(f"        {k:20}: {v}")
        print("   → Hiperparámetros (args):")
        for k, v in r["args"].items():
            print(f"        {k:20}: {v}")
        print("\n----------------------------------------------------\n")


if __name__ == "__main__":
    main()
