"""Paso 2 — Re-evalúa las líneas base del Corte 1 con el protocolo del Corte 2.

Modelos:
  dummy        DummyClassifier (clase mayoritaria) — referencia mínima
  svm_hsv_lbp  histograma HSV + LBP + SVM RBF (mismo baseline del Corte 1)

Se entrena con train y se mide en train y en validación (completa y limpia).
Salidas: reports/corte2/runs/<modelo>/{metrics.json, preds_val.csv}

Uso:
  python src/c2_baseline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from c2_common import CLASSES, REPORTS, SEED, dataset_root, evaluate_split, load_index, metrics_block, save_json
from data_utils import extract_features

FEATS = REPORTS / "features_hsv_lbp.npz"


def get_features(df: pd.DataFrame, root: Path) -> np.ndarray:
    """Extrae (o reutiliza de caché) las features de color+textura del Corte 1."""
    if FEATS.exists():
        d = np.load(FEATS, allow_pickle=True)
        if list(d["rel_path"]) == list(df.rel_path):
            print("Usando features en caché:", FEATS)
            return d["X"]
    print(f"Extrayendo features HSV+LBP de {len(df)} imágenes (1-3 min)...")
    X = []
    for i, rp in enumerate(df.rel_path):
        X.append(extract_features(str(root / rp)))
        if i % 1000 == 0:
            print(f"  {i}/{len(df)}")
    X = np.stack(X).astype(np.float32)
    np.savez_compressed(FEATS, X=X, rel_path=df.rel_path.to_numpy())
    return X


def run(name: str, model, family: str, Xtr, ytr, Xval, dval, n_params: int | None, desc: str):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_val = model.predict(Xval)
    infer_ms = 1000 * (time.perf_counter() - t0) / len(Xval)
    pred_tr = model.predict(Xtr)

    out = REPORTS / "runs" / name
    m = {
        "name": name,
        "family": family,
        "description": desc,
        "params": n_params,
        "train_time_s": round(train_time, 2),
        "infer_ms_per_img": round(infer_ms, 3),
        "infer_note": "solo el clasificador; la extracción de features no está incluida",
        "train": metrics_block(ytr, pred_tr),
        "val": evaluate_split(dval, pred_val),
    }
    save_json(m, out / "metrics.json")
    p = dval[["rel_path", "class", "label", "clean", "is_aug"]].copy()
    p["pred"] = pred_val
    p.to_csv(out / "preds_val.csv", index=False)
    v = m["val"]
    print(
        f"{name:<14} train F1={m['train']['macro_f1']:.4f} | val F1 completo={v['full']['macro_f1']:.4f} "
        f"limpio={v['clean']['macro_f1']:.4f} | fit {train_time:.1f}s"
    )
    return m


if __name__ == "__main__":
    root = dataset_root()
    idx = load_index()
    df = idx[idx.split.isin(["train", "val"])].reset_index(drop=True)  # test NO se toca
    X = get_features(df, root)
    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    Xtr, ytr = X[tr], df.label.to_numpy()[tr]
    Xval, dval = X[va], df[va].reset_index(drop=True)

    run("dummy", DummyClassifier(strategy="most_frequent"), "referencia",
        Xtr, ytr, Xval, dval, 0, "Predice siempre la clase más frecuente de train")

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                              class_weight="balanced", random_state=SEED))
    m = run("svm_hsv_lbp", svm, "clásico", Xtr, ytr, Xval, dval, None,
            "Histograma HSV (48) + LBP uniforme (10) + SVM RBF C=10 (baseline Corte 1)")
    # "parámetros" del SVM: vectores de soporte x dimensión (aprox. del tamaño del modelo)
    sv = svm[-1].support_vectors_.shape
    m["params"] = int(sv[0] * sv[1])
    m["params_note"] = f"{sv[0]} vectores de soporte x {sv[1]} features"
    save_json(m, REPORTS / "runs" / "svm_hsv_lbp" / "metrics.json")
