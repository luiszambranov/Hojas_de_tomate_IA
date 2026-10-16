"""Paso 1 — Construye el índice del Corte 2 y el diagnóstico de fuga.

Salidas (reports/corte2/):
  index_corte2.csv       una fila por imagen, con split, clase, hoja de origen y marcas de fuga
  fuga_resumen.csv       conteo por partición y clase de imágenes aumentadas y contaminadas

Uso (desde la raíz del repo):
  python src/c2_build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from c2_common import INDEX_CSV, REPORTS, SHORT, build_index, dataset_root

if __name__ == "__main__":
    root = dataset_root()
    print(f"Dataset: {root}")
    df = build_index(root)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)

    resumen = (
        df.groupby(["split", "class"])
        .agg(
            imagenes=("rel_path", "size"),
            aumentadas=("is_aug", "sum"),
            hojas_unicas=("source", "nunique"),
            contaminadas=("leak", "sum"),
            duplicado_exacto=("leak_exact", "sum"),
        )
        .reset_index()
    )
    resumen.to_csv(REPORTS / "fuga_resumen.csv", index=False)

    print(f"\nTotal: {len(df)} imágenes")
    print(df.groupby("split").size().to_string())
    print("\nImágenes de val/test cuya hoja de origen (o archivo idéntico) está en train:")
    for s in ["val", "test"]:
        d = df[df.split == s]
        print(f"  {s}: {int(d.leak.sum())} de {len(d)} ({100*d.leak.mean():.1f} %)")
        by = d[d.leak].groupby("class").size()
        for c, n in by.items():
            print(f"      {SHORT[c]:<11} {n}")
    print(f"\nÍndice guardado en {INDEX_CSV}")
    print("Nota: test solo se indexa; no se usa para entrenar ni para decidir.")
