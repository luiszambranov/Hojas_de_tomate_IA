"""
Utilidades comunes del Corte 2 — Proyecto 7 (hojas de tomate).

- Rutas relativas a la raíz del repositorio (funciona en Windows y Linux).
- Índice del dataset con detección de fuga de información:
  el dataset trae copias aumentadas ("..._original_<archivo>_<uuid>.JPG")
  repartidas entre particiones, de modo que una misma hoja puede aparecer
  en train y en val/test. Aquí se identifica la hoja de origen de cada
  imagen para construir una validación "limpia".
- Funciones de métricas compartidas por todos los modelos, para que la
  comparación sea justa (misma partición, mismas métricas).

REGLA: el conjunto de test NO se usa para entrenar ni para decidir nada
en este corte. Solo se indexa (nombres de archivo) para documentar la fuga.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# --------------------------------------------------------------------------
# Rutas y configuración
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]

# Ubicación del dataset. Por defecto, la carpeta tal como viene de Mendeley
# dentro del repo. Se puede cambiar con la variable de entorno TOMATE_DATA.
_DEFAULT_CANDIDATES = [
    REPO / "Processed Tomato Leaf Disease Image Dataset with T" / "Dataset",
    REPO / "data" / "Dataset",
]


def dataset_root() -> Path:
    env = os.environ.get("TOMATE_DATA")
    if env:
        return Path(env)
    for c in _DEFAULT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No se encontró el dataset. Colóquelo en uno de estos lugares o defina "
        "TOMATE_DATA:\n  " + "\n  ".join(str(c) for c in _DEFAULT_CANDIDATES)
    )


REPORTS = REPO / "reports" / "corte2"
MODELS_DIR = REPO / "models"          # pesos entrenados (no se versionan)
INDEX_CSV = REPORTS / "index_corte2.csv"

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]
CLASS_ES = {
    "Tomato___Bacterial_spot": "Mancha bacteriana",
    "Tomato___Early_blight": "Tizón temprano",
    "Tomato___Late_blight": "Tizón tardío",
    "Tomato___Leaf_Mold": "Moho de la hoja",
    "Tomato___Septoria_leaf_spot": "Mancha Septoria",
    "Tomato___healthy": "Sana",
}
SHORT = {
    "Tomato___Bacterial_spot": "Bact.",
    "Tomato___Early_blight": "T.temprano",
    "Tomato___Late_blight": "T.tardío",
    "Tomato___Leaf_Mold": "Moho",
    "Tomato___Septoria_leaf_spot": "Septoria",
    "Tomato___healthy": "Sana",
}
SPLITS = ["train", "val", "test"]
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Índice + fuga
# --------------------------------------------------------------------------
_AUG_RE = re.compile(r"_original_(.+?\.(?:jpe?g|png))_[0-9a-f\-]{36}\.(?:jpe?g|png)$", re.I)


def source_of(filename: str) -> tuple[str, bool]:
    """Devuelve (nombre de la hoja original, es_aumentada)."""
    m = _AUG_RE.search(filename)
    if m:
        return m.group(1), True
    return filename, False


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_index(root: Path | None = None, with_hash: bool = True) -> pd.DataFrame:
    """Recorre root/{split}/{clase}/ y arma el índice con columnas:
    rel_path, split, class, label, filename, is_aug, source, md5,
    leak_source (la hoja de origen también está en train),
    leak_exact (archivo idéntico byte a byte en train),
    leak (cualquiera de las dos) y clean (= no leak).
    """
    root = root or dataset_root()
    rows = []
    for split in SPLITS:
        for label, cls in enumerate(CLASSES):
            d = root / split / cls
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                src, aug = source_of(f.name)
                rows.append(
                    dict(
                        rel_path=f.relative_to(root).as_posix(),
                        split=split,
                        **{"class": cls},
                        label=label,
                        filename=f.name,
                        is_aug=aug,
                        source=f"{cls}/{src}",
                        md5=_md5(f) if with_hash else "",
                    )
                )
    df = pd.DataFrame(rows)
    train = df[df.split == "train"]
    train_src = set(train.source)
    train_md5 = set(train.md5) if with_hash else set()
    df["leak_source"] = (df.split != "train") & df.source.isin(train_src)
    df["leak_exact"] = (df.split != "train") & with_hash & df.md5.isin(train_md5)
    df["leak"] = df.leak_source | df.leak_exact
    df["clean"] = ~df.leak
    return df


def load_index() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(
            f"No existe {INDEX_CSV}. Ejecute primero: python src/c2_build_index.py"
        )
    return pd.read_csv(INDEX_CSV)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metrics_block(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
        "per_class": {
            CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(len(CLASSES))
        },
    }


def evaluate_split(df_split: pd.DataFrame, y_pred) -> dict:
    """Métricas sobre la partición completa y sobre su versión limpia."""
    y_pred = np.asarray(y_pred)
    clean = df_split["clean"].to_numpy(bool)
    return {
        "full": metrics_block(df_split["label"], y_pred),
        "clean": metrics_block(df_split["label"].to_numpy()[clean], y_pred[clean]),
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
