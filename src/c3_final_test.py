"""Corte 3 · Paso 1 — Evaluación FINAL sobre el test reservado (una sola vez).

Evalúa con el test (completo y LIMPIO, sin las 82 copias de hojas que están en train):
  - Dummy (mayoritaria)          referencia mínima
  - SVM HSV+LBP                  línea base del Corte 1 (se reentrena igual, con train)
  - ResNet-18 fine-tune          modelo provisional del Corte 2 = modelo final
  - MobileNetV3 fine-tune        alternativa liviana (solo se reporta, no se elige con test)

Nada de lo que sale de aquí se usa para cambiar el modelo, el umbral ni los hiperparámetros.
Si los resultados ya existen, el script se niega a repetir (use --force solo si cambió el código
de evaluación, nunca para "probar otra vez").

Salidas en reports/corte3/:
  test_resumen.csv, test_f1_por_clase.csv, test_results.json, preds_test_<modelo>.csv,
  comparacion_cortes.csv, figures/c3_01..c3_02, c3_04
Uso:
  python src/c3_final_test.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from c2_common import CLASSES, REPORTS, SEED, SHORT, dataset_root, evaluate_split, load_index, load_json, save_json
from c3_common import (ALT, BLUE, CONF_THRESHOLD, FIG3, FINAL, GRAY, INK, INK2, NICE3, ORANGE, REPORTS3, device,
                       load_cnn, open_rgb, predict_pils)
from data_utils import extract_features

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.dpi": 110, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def bootstrap_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[s], p[s], average="macro", labels=list(range(len(CLASSES))), zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    b01 = int((a_ok & ~b_ok).sum())   # A acierta, B falla
    b10 = int((~a_ok & b_ok).sum())   # A falla, B acierta
    p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_acierta_b_falla": b01, "a_falla_b_acierta": b10, "p_value": float(p)}


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def selective(probs, y, thr=CONF_THRESHOLD):
    conf, pred = probs.max(1), probs.argmax(1)
    acc_mask = conf >= thr
    return {
        "umbral": thr,
        "cobertura": float(acc_mask.mean()),
        "no_concluyentes": int((~acc_mask).sum()),
        "accuracy_aceptadas": float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else float("nan"),
        "errores_con_confianza_alta": int(((pred != y) & acc_mask).sum()),
        "errores_totales": int((pred != y).sum()),
    }


def svm_predictions(idx, dtest, root):
    """Reentrena el SVM del Corte 1/2 exactamente igual (solo train) y predice test."""
    cache = np.load(REPORTS / "features_hsv_lbp.npz", allow_pickle=True)
    pos = {rp: i for i, rp in enumerate(cache["rel_path"])}
    dtr = idx[idx.split == "train"]
    Xtr = cache["X"][[pos[r] for r in dtr.rel_path]]
    ytr = dtr.label.to_numpy()
    ft = REPORTS3 / "features_test_hsv_lbp.npz"
    if ft.exists() and list(np.load(ft, allow_pickle=True)["rel_path"]) == list(dtest.rel_path):
        Xte = np.load(ft, allow_pickle=True)["X"]
    else:
        print(f"Extrayendo features HSV+LBP de {len(dtest)} imágenes de test...")
        Xte = np.stack([extract_features(str(root / rp)) for rp in dtest.rel_path]).astype(np.float32)
        np.savez_compressed(ft, X=Xte, rel_path=dtest.rel_path.to_numpy())
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                              class_weight="balanced", random_state=SEED))
    svm.fit(Xtr, ytr)
    return svm.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    REPORTS3.mkdir(parents=True, exist_ok=True)
    FIG3.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS3 / "test_results.json"
    if out_json.exists() and not args.force:
        print(f"El test final ya se evaluó ({out_json}). No se repite: el test es de un solo uso.\n"
              "Si solo cambió el código de las figuras, use --force.")
        return

    root = dataset_root()
    idx = load_index()
    dtest = idx[idx.split == "test"].reset_index(drop=True)
    y = dtest.label.to_numpy()
    clean = dtest.clean.to_numpy(bool)
    print(f"Test: {len(dtest)} imágenes | limpias: {clean.sum()} | con fuga: {(~clean).sum()}")

    preds, probs = {}, {}
    majority = int(idx[idx.split == "train"].label.value_counts().idxmax())
    preds["dummy"] = np.full(len(dtest), majority)
    preds["svm_hsv_lbp"] = svm_predictions(idx, dtest, root)

    dev = device()
    pils = [open_rgb(root / rp) for rp in dtest.rel_path]
    for run in [FINAL, ALT]:
        model = load_cnn(run, dev)
        probs[run] = predict_pils(model, pils, dev)
        preds[run] = probs[run].argmax(1)
        del model

    # --- predicciones por imagen
    for run, p in preds.items():
        d = dtest[["rel_path", "class", "label", "clean", "is_aug"]].copy()
        d["pred"] = p
        if run in probs:
            for i in range(len(CLASSES)):
                d[f"prob_{i}"] = probs[run][:, i]
            d["conf"] = probs[run].max(1)
        d.to_csv(REPORTS3 / f"preds_test_{run}.csv", index=False)

    # --- métricas
    res = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "n_test": int(len(dtest)),
           "n_test_limpio": int(clean.sum()), "modelo_final": FINAL, "modelos": {}}
    rows, pc_rows = [], []
    for run, p in preds.items():
        ev = evaluate_split(dtest, p)
        lo, hi = bootstrap_ci(y[clean], p[clean])
        entry = {"test": ev, "ic95_macro_f1_limpio": [lo, hi],
                 "errores_limpio": int((p[clean] != y[clean]).sum())}
        if run in probs:
            entry["ece_limpio"] = ece(probs[run][clean], y[clean])
            entry["selectiva_limpio"] = selective(probs[run][clean], y[clean])
        res["modelos"][run] = entry
        rows.append({
            "run": run, "modelo": NICE3[run],
            "test_macroF1_limpio": round(ev["clean"]["macro_f1"], 4),
            "ic95_inf": round(lo, 4), "ic95_sup": round(hi, 4),
            "test_macroF1_completo": round(ev["full"]["macro_f1"], 4),
            "test_acc_limpio": round(ev["clean"]["accuracy"], 4),
            "errores_limpio": entry["errores_limpio"],
            "ECE": round(entry.get("ece_limpio", float("nan")), 4),
        })
        for c in CLASSES:
            pcm = ev["clean"]["per_class"][c]
            pc_rows.append({"run": run, "modelo": NICE3[run], "clase": c, "precision": round(pcm["precision"], 4),
                            "recall": round(pcm["recall"], 4), "f1": round(pcm["f1"], 4), "soporte": pcm["support"]})
    res["mcnemar_limpio"] = {
        "final_vs_svm": mcnemar(y[clean], preds[FINAL][clean], preds["svm_hsv_lbp"][clean]),
        "final_vs_alt": mcnemar(y[clean], preds[FINAL][clean], preds[ALT][clean]),
    }
    # validación (Corte 2) para la comparación entre cortes
    for run in preds:
        mv = load_json(REPORTS / "runs" / run / "metrics.json")
        res["modelos"][run]["val_macro_f1_limpio_c2"] = mv["val"]["clean"]["macro_f1"]
        res["modelos"][run]["val_macro_f1_completo_c2"] = mv["val"]["full"]["macro_f1"]
    for run in probs:   # selectiva en validación (para ver si el umbral fijado a priori se comporta igual)
        pv = pd.read_csv(REPORTS / "runs" / run / "preds_val.csv")
        pv = pv[pv.clean.astype(bool)]
        res["modelos"][run]["selectiva_val_limpio"] = selective(
            pv[[f"prob_{i}" for i in range(len(CLASSES))]].to_numpy(), pv.label.to_numpy())
    save_json(res, out_json)

    tab = pd.DataFrame(rows)
    tab["val_macroF1_limpio_C2"] = [round(res["modelos"][r]["val_macro_f1_limpio_c2"], 4) for r in tab.run]
    tab["caida_val_a_test"] = (tab.val_macroF1_limpio_C2 - tab.test_macroF1_limpio).round(4)
    tab.to_csv(REPORTS3 / "test_resumen.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(REPORTS3 / "test_f1_por_clase.csv", index=False)

    # Comparación explícita entre cortes
    comp = pd.DataFrame([
        {"corte": "C1", "rol": "línea base", "modelo": NICE3["svm_hsv_lbp"]},
        {"corte": "C2", "rol": "modelo provisional", "modelo": "ResNet-18 FT"},
        {"corte": "C3", "rol": "modelo final (mismo provisional + Grad-CAM, robustez, umbral de abstención)",
         "modelo": NICE3[FINAL]},
    ])
    comp["val_macroF1_limpio"] = [tab.set_index("run").loc[r, "val_macroF1_limpio_C2"] for r in ["svm_hsv_lbp", FINAL, FINAL]]
    comp["test_macroF1_limpio"] = [tab.set_index("run").loc[r, "test_macroF1_limpio"] for r in ["svm_hsv_lbp", FINAL, FINAL]]
    comp.to_csv(REPORTS3 / "comparacion_cortes.csv", index=False)

    print("\n" + tab[["modelo", "val_macroF1_limpio_C2", "test_macroF1_limpio", "ic95_inf", "ic95_sup",
                      "test_macroF1_completo", "errores_limpio", "ECE"]].to_string(index=False))
    print("\nMcNemar final vs SVM:", res["mcnemar_limpio"]["final_vs_svm"])
    print("McNemar final vs alternativa:", res["mcnemar_limpio"]["final_vs_alt"])
    print("Abstención (test limpio):", res["modelos"][FINAL]["selectiva_limpio"])

    figures(tab, dtest, preds, probs, clean, root)


def figures(tab, dtest, preds, probs, clean, root):
    # c3_01: validación (C2) vs test (C3)
    order = ["svm_hsv_lbp", ALT, FINAL]
    d = tab.set_index("run").loc[order]
    yy = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    h = 0.38
    ax.barh(yy + h / 2, d.val_macroF1_limpio_C2, h - 0.04, color=GRAY, label="validación limpia (Corte 2, decide)")
    ax.barh(yy - h / 2, d.test_macroF1_limpio, h - 0.04, color=BLUE, label="test limpio (Corte 3, una sola vez)")
    for k, (v, t, lo, hi) in enumerate(zip(d.val_macroF1_limpio_C2, d.test_macroF1_limpio, d.ic95_inf, d.ic95_sup)):
        ax.errorbar(t, k - h / 2, xerr=[[t - lo], [hi - t]], color=INK, capsize=3, lw=1)
        ax.text(max(hi, t) + 0.002, k - h / 2, f"{t:.3f}", va="center", fontsize=9, color=INK)
        ax.text(v + 0.002, k + h / 2, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(yy, [NICE3[r] for r in order])
    lo = max(0.0, float(np.floor((min(d.test_macroF1_limpio.min(), d.ic95_inf.min()) - 0.02) * 50) / 50))
    ax.set_xlim(lo, 1.015)
    dm = tab.set_index("run").loc["dummy", "test_macroF1_limpio"]
    ax.set_xlabel(f"macro-F1 (eje desde {lo:.2f}; barras = IC 95 % bootstrap)  ·  Dummy en test: {dm:.3f}")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    ax.set_title("Línea base vs. modelo final: validación y test reservado", loc="left", pad=24)
    fig.savefig(FIG3 / "c3_01_val_vs_test.png")
    plt.close(fig)

    # c3_02: matrices de confusión en test limpio (baseline vs final)
    y = dtest.label.to_numpy()[clean]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    for ax, run in zip(axes, ["svm_hsv_lbp", FINAL]):
        cm = confusion_matrix(y, preds[run][clean], labels=list(range(len(CLASSES))))
        ax.imshow(cm, cmap=cmap)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8.5,
                            color="white" if cm[i, j] > cm.max() * 0.5 else INK)
        lab = [SHORT[c] for c in CLASSES]
        ax.set_xticks(range(6), lab, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(6), lab, fontsize=8.5)
        ax.set_xlabel("predicha")
        ax.set_ylabel("real")
        err = int((preds[run][clean] != y).sum())
        ax.set_title(f"{NICE3[run]}: {err} errores / {len(y)}", loc="left", fontsize=10)
    fig.suptitle("Matriz de confusión en test limpio", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_02_confusion_test.png")
    plt.close(fig)

    # c3_04: galería de errores del modelo final en test (todos, completo, máx. 12)
    p = preds[FINAL]
    yall = dtest.label.to_numpy()
    err = np.where(p != yall)[0]
    conf = probs[FINAL].max(1)
    err = err[np.argsort(-conf[err])][:12]
    if len(err):
        n = len(err)
        cols = min(n, 6)
        rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(2.3 * cols, 2.7 * rows_), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), err):
            ax.imshow(open_rgb(root / dtest.rel_path[i]))
            tag = "" if dtest.clean[i] else " [fuga]"
            ax.set_title(f"real: {SHORT[CLASSES[yall[i]]]}\npred: {SHORT[CLASSES[p[i]]]} ({conf[i]:.0%}){tag}",
                         fontsize=8, color=INK)
        fig.suptitle(f"Errores del modelo final en test ({int((p != yall).sum())} en {len(yall)} imágenes)",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG3 / "c3_04_errores_test.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
