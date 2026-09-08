"""
Funciones reutilizables para el proyecto 7 - Diagnóstico visual de
enfermedades en hojas de tomate (Corte 1).

Contiene utilidades para:
- listar el dataset y su partición train/val/test
- extraer features clásicas (color HSV + textura LBP) para el baseline
- calcular una métrica simple de nitidez (para detectar imágenes borrosas)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

CLASS_LABELS_ES = {
    "Tomato___Bacterial_spot": "Mancha bacteriana",
    "Tomato___Early_blight": "Tizón temprano",
    "Tomato___Late_blight": "Tizón tardío",
    "Tomato___Leaf_Mold": "Moho de la hoja",
    "Tomato___Septoria_leaf_spot": "Mancha por Septoria",
    "Tomato___healthy": "Sana",
}

SPLITS = ["train", "val", "test"]


def list_dataset(root: Path) -> pd.DataFrame:
    """Recorre root/{split}/{clase}/*.jpg y arma un DataFrame con metadatos
    básicos (ruta, split, clase, tamaño en bytes, ancho, alto)."""
    rows = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                rows.append(
                    {
                        "path": str(f),
                        "split": split,
                        "class": cls,
                        "class_es": CLASS_LABELS_ES[cls],
                        "size_bytes": f.stat().st_size,
                    }
                )
    return pd.DataFrame(rows)


def sharpness_score(img_bgr: np.ndarray) -> float:
    """Varianza del laplaciano: proxy estándar de nitidez.
    Valores bajos sugieren imagen borrosa."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_hue_saturation(img_bgr: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean())


def extract_features(path: str, size: int = 128) -> np.ndarray:
    """Extrae un vector de features clásico para el baseline:
    - histograma de color en HSV (color de la lesión / hoja)
    - histograma de Local Binary Pattern (textura de la lesión)

    Ambas son señales que un agrónomo usaría a ojo: color y textura de la
    mancha, sin necesitar aprender representaciones profundas.
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Histograma de color HSV (8 bins por canal)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color_hist = np.concatenate([hist_h, hist_s, hist_v])
    color_hist = color_hist / (color_hist.sum() + 1e-8)

    # Textura: Local Binary Pattern sobre el canal de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([color_hist, lbp_hist]).astype(np.float32)
