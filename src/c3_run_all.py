"""Corte 3 — ejecuta todo en orden (≈15-25 min en una GTX 1650):

  1. c3_final_test.py   evaluación final en test reservado (una sola vez)
  2. c3_gradcam.py      interpretabilidad (Grad-CAM + atención sobre la hoja)
  3. c3_robustness.py   perturbaciones de captura + fuera de dominio sintético
  4. c3_external.py     fotos de campo (PlantDoc)

Requiere los pesos del Corte 2 en models/ (python src/c2_run_benchmark.py).
Uso (desde la raíz del repo, con el entorno activo):
  python src/c3_run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["c3_final_test.py", "c3_gradcam.py", "c3_robustness.py", "c3_external.py"]

if __name__ == "__main__":
    t0 = time.perf_counter()
    for s in STEPS:
        print(f"\n{'=' * 70}\n>> {s}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)])
        if r.returncode != 0:
            sys.exit(f"Falló {s} (código {r.returncode}). Corrija y vuelva a lanzar; los pasos ya hechos se conservan.")
    print(f"\nCorte 3 completo en {(time.perf_counter() - t0) / 60:.1f} min. Resultados en reports/corte3/")
