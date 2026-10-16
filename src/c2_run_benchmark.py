"""Ejecuta TODO el benchmark del Corte 2 en orden, con la misma partición y métricas.

  1. Índice + diagnóstico de fuga          (c2_build_index.py)
  2. Dummy + SVM HSV/LBP (baseline Corte 1)  (c2_baseline.py)
  3. CNNs: desde cero y transfer learning   (c2_train_cnn.py)
  4. Tabla y figuras de análisis            (c2_analyze.py)

Las corridas que ya tienen metrics.json se saltan (útil si se interrumpe).
Use --force para repetirlas.

Uso (desde la raíz del repo, con el entorno .venv activo):
  python src/c2_run_benchmark.py
  python src/c2_run_benchmark.py --quick     # prueba rápida de humo (pocas imágenes, 1 época)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from c2_common import INDEX_CSV, REPORTS

# Ajuste de hiperparámetros acotado y justificado (no búsqueda ciega):
#  - frozen: solo la cabeza aprende -> lr alto (1e-3), pocas épocas.
#  - finetune: toda la red -> lr menor; se prueban 2 valores en el candidato principal.
#  - smallcnn: desde cero -> más épocas y lr 1e-3.
CNN_RUNS = [
    dict(model="smallcnn", mode="scratch", lr=1e-3, epochs=30, patience=7),
    dict(model="mobilenet_v3_large", mode="frozen", lr=1e-3, epochs=10, patience=4),
    dict(model="efficientnet_b0", mode="frozen", lr=1e-3, epochs=10, patience=4),
    dict(model="mobilenet_v3_large", mode="finetune", lr=3e-4, epochs=15, patience=5),
    dict(model="efficientnet_b0", mode="finetune", lr=3e-4, epochs=15, patience=5),
    dict(model="efficientnet_b0", mode="finetune", lr=1e-4, epochs=15, patience=5),
    dict(model="resnet18", mode="finetune", lr=3e-4, epochs=15, patience=5),
]


def sh(args):
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([sys.executable, *map(str, args)])
    if r.returncode != 0:
        raise SystemExit(f"Falló: {' '.join(map(str, args))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="prueba de humo: 120 imágenes, 1 época por modelo")
    ap.add_argument("--force", action="store_true", help="repetir corridas existentes")
    ap.add_argument("--workers", type=int, default=2,
                    help="procesos que cargan y aumentan imágenes en paralelo (más = GPU más ocupada)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="desactivar precisión mixta en todas las CNN")
    a = ap.parse_args()

    t0 = time.time()
    if a.force or not INDEX_CSV.exists():
        sh([HERE / "c2_build_index.py"])
    if a.force or not (REPORTS / "runs" / "svm_hsv_lbp" / "metrics.json").exists():
        sh([HERE / "c2_baseline.py"])

    for r in CNN_RUNS:
        mode = r["mode"]
        name = f"{r['model']}_{mode}_lr{r['lr']:g}" + ("_QUICK" if a.quick else "")
        if not a.force and (REPORTS / "runs" / name / "metrics.json").exists():
            print(f"(ya existe {name}, se salta)")
            continue
        cmd = [HERE / "c2_train_cnn.py", "--model", r["model"], "--lr", r["lr"], "--name", name,
               "--epochs", 1 if a.quick else r["epochs"], "--patience", r["patience"],
               "--workers", a.workers, "--batch", a.batch]
        if mode in ("frozen", "finetune"):
            cmd += ["--mode", mode]
        if a.quick:
            cmd += ["--limit", 120]
        if a.no_pretrained:
            cmd += ["--no-pretrained"]
        if a.no_amp:
            cmd += ["--no-amp"]
        sh(cmd)

    sh([HERE / "c2_analyze.py"] + (["--include-quick"] if a.quick else []))
    print(f"\nBenchmark completo en {(time.time()-t0)/60:.1f} min. Resultados en {REPORTS}")
