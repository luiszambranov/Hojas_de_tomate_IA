"""Paso 4 — Consolida el benchmark y genera tablas y figuras del Corte 2.

Lee reports/corte2/runs/*/ y produce en reports/corte2/:
  benchmark.csv                tabla comparativa (desempeño, brecha train/val, costo)
  f1_por_clase.csv             F1 por clase en validación limpia
  comparacion_errores.csv      imágenes que el baseline falla y el mejor modelo acierta (y viceversa)
  significancia.json           IC 95 % bootstrap del macro-F1 y prueba de McNemar (mejor vs. baseline)
  figures/*.png                figuras para el notebook y la presentación

Todas las decisiones se toman con VALIDACIÓN (limpia). Test no se usa.

Uso:
  python src/c2_analyze.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from c2_common import CLASSES, REPORTS, SHORT, dataset_root, load_index, load_json, save_json

FIG = REPORTS / "figures"

# Paleta (validada, ver skill dataviz): azul = principal, naranja = secundaria, gris = contexto
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9", "#0b0b0b", "#52514e"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.grid": False, "savefig.dpi": 200,
})

NICE = {
    "dummy": "Dummy (mayoritaria)",
    "svm_hsv_lbp": "SVM HSV+LBP (Corte 1)",
    "smallcnn_scratch_lr0.001": "CNN pequeña (desde cero)",
    "mobilenet_v3_large_frozen_lr0.001": "MobileNetV3 congelado",
    "efficientnet_b0_frozen_lr0.001": "EfficientNet-B0 congelado",
    "mobilenet_v3_large_finetune_lr0.0003": "MobileNetV3 fine-tune",
    "efficientnet_b0_finetune_lr0.0003": "EfficientNet-B0 fine-tune (lr 3e-4)",
    "efficientnet_b0_finetune_lr0.0001": "EfficientNet-B0 fine-tune (lr 1e-4)",
    "resnet18_finetune_lr0.0003": "ResNet-18 fine-tune",
}


def nice(name: str) -> str:
    return NICE.get(name.replace("_QUICK", ""), name) + (" [quick]" if name.endswith("_QUICK") else "")


def load_runs(include_quick: bool):
    runs = {}
    for d in sorted((REPORTS / "runs").glob("*")):
        if not (d / "metrics.json").exists():
            continue
        if d.name.endswith("_QUICK") and not include_quick:
            continue
        m = load_json(d / "metrics.json")
        m["preds"] = pd.read_csv(d / "preds_val.csv")
        m["history"] = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
        runs[d.name] = m
    return runs


def bootstrap_f1(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(f1_score(y[s], p[s], average="macro", labels=list(range(len(CLASSES))), zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def table(runs):
    rows = []
    for name, m in runs.items():
        v = m["val"]
        rows.append({
            "run": name,
            "modelo": nice(name),
            "familia": m["family"],
            "params_M": round((m.get("params") or 0) / 1e6, 3),
            "entrenables_M": round((m.get("trainable_params") or m.get("params") or 0) / 1e6, 3),
            "tiempo_entreno_min": round(m["train_time_s"] / 60, 2),
            "ms_por_imagen": m["infer_ms_per_img"],
            "mejor_epoca": m.get("best_epoch"),
            "train_macroF1": round(m["train"]["macro_f1"], 4),
            "val_macroF1_completo": round(v["full"]["macro_f1"], 4),
            "val_macroF1_limpio": round(v["clean"]["macro_f1"], 4),
            "inflado_por_fuga": round(v["full"]["macro_f1"] - v["clean"]["macro_f1"], 4),
            "brecha_train_val": round(m["train"]["macro_f1"] - v["clean"]["macro_f1"], 4),
            "val_acc_limpio": round(v["clean"]["accuracy"], 4),
            "F1_T.temprano": round(v["clean"]["per_class"]["Tomato___Early_blight"]["f1"], 4),
            "F1_T.tardío": round(v["clean"]["per_class"]["Tomato___Late_blight"]["f1"], 4),
            "recall_Sana": round(v["clean"]["per_class"]["Tomato___healthy"]["recall"], 4),
        })
    return pd.DataFrame(rows).sort_values("val_macroF1_limpio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_fuga(idx):
    d = idx[idx.split.isin(["val", "test"])]
    c = d[d.leak].groupby(["class", "split"]).size().unstack(fill_value=0).reindex(CLASSES, fill_value=0)
    for s in ["val", "test"]:
        if s not in c:
            c[s] = 0
    y = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    h = 0.38
    ax.barh(y - h / 2, c["val"], h - 0.04, color=BLUE, label="validación")
    ax.barh(y + h / 2, c["test"], h - 0.04, color=ORANGE, label="test")
    off = max(c.values.max(), 1) * 0.015
    for yy, (a, b) in enumerate(zip(c["val"], c["test"])):
        ax.text(a + off, yy - h / 2, str(a), va="center", fontsize=9, color=INK)
        ax.text(b + off, yy + h / 2, str(b), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [SHORT[k] for k in CLASSES])
    ax.invert_yaxis()
    ax.set_xlabel("imágenes cuya hoja de origen también está en train")
    nv, nt = int(d[d.split == "val"].leak.sum()), int(d[d.split == "test"].leak.sum())
    ax.set_title(f"Fuga de información en el dataset: {nv} imágenes en val y {nt} en test", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "c2_01_fuga_por_clase.png")
    plt.close(fig)


def fig_benchmark(df):
    dummy = df[df.run == "dummy"]
    d = df[df.run != "dummy"].iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.6))
    h = 0.38
    ax.barh(y + h / 2, d.val_macroF1_limpio, h - 0.04, color=BLUE, label="validación limpia (decide)")
    ax.barh(y - h / 2, d.val_macroF1_completo, h - 0.04, color=GRAY, label="validación completa (con fuga)")
    for yy, v in zip(y, d.val_macroF1_limpio):
        ax.text(v + 0.002, yy + h / 2, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, d.modelo)
    lo = float(np.floor((d.val_macroF1_limpio.min() - 0.03) * 50) / 50)
    ax.set_xlim(lo, 1.012)
    nota = f"  ·  Dummy: {dummy.val_macroF1_limpio.iloc[0]:.3f} (fuera de escala)" if len(dummy) else ""
    ax.set_xlabel(f"macro-F1 (eje desde {lo:.2f}){nota}")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    ax.set_title("Benchmark en validación: macro-F1 por modelo", loc="left", pad=24)
    fig.savefig(FIG / "c2_02_benchmark_macroF1.png")
    plt.close(fig)


def fig_costo(df, best):
    d = df[df.run != "dummy"].copy()
    d["x"] = d.ms_por_imagen.clip(lower=1e-3)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xscale("log")
    ax.grid(True, color="#e8e7e3", linewidth=0.6)
    # Etiquetas: los modelos casi empatados arriba se listan escalonados para no solaparse
    d = d.sort_values("val_macroF1_limpio", ascending=False)
    top = d[d.val_macroF1_limpio >= 0.99]
    for _, r in d.iterrows():
        c = BLUE if r.run == best else INK2
        ax.scatter(r.x, r.val_macroF1_limpio, s=70 if r.run == best else 42, color=c,
                   edgecolor="white", linewidth=1.5, zorder=3)
    for k, (_, r) in enumerate(top.iterrows()):
        ty = 0.992 - k * 0.0055
        ax.annotate(f"{r.modelo} ({r.val_macroF1_limpio:.3f}, {r.ms_por_imagen:.1f} ms)",
                    xy=(r.x, r.val_macroF1_limpio), xytext=(0.03, ty), textcoords=("axes fraction", "data"),
                    fontsize=8, color=INK if r.run == best else INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.8))
    for _, r in d[d.val_macroF1_limpio < 0.99].iterrows():
        ax.annotate(r.modelo, (r.x, r.val_macroF1_limpio), xytext=(7, -10), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_xlabel("latencia de inferencia por imagen (ms, escala log)")
    ax.set_ylabel("macro-F1 validación limpia")
    ax.set_title("Desempeño vs. costo de inferencia", loc="left")
    fig.text(0.01, -0.03, "SVM: solo el clasificador (sin extracción de features). CNN: batch 1 en GPU (GTX 1650, float32).",
             fontsize=7.5, color=INK2)
    fig.savefig(FIG / "c2_03_desempeno_vs_costo.png")
    plt.close(fig)


def fig_f1_clase(runs, df):
    order = [r for r in df.run if r != "dummy"]
    M = np.array([[runs[r]["val"]["clean"]["per_class"][c]["f1"] for c in CLASSES] for r in order])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    vmin = max(0.0, min(0.8, M.min()))
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if (v - vmin) / (1 - vmin + 1e-9) > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES])
    ax.set_yticks(range(len(order)), [nice(r) for r in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("F1 por clase (validación limpia)", loc="left")
    fig.savefig(FIG / "c2_04_f1_por_clase.png")
    plt.close(fig)


def _cm(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap
    ax.imshow(cmn, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES], rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES])
    ax.set_xlabel("predicción")
    ax.set_ylabel("real")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10)


def fig_confusion(runs, base, best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, r in zip(axes, [base, best]):
        p = runs[r]["preds"]
        p = p[p.clean]
        f1 = runs[r]["val"]["clean"]["macro_f1"]
        _cm(ax, p.label.to_numpy(), p.pred.to_numpy(), f"{nice(r)}\nmacro-F1 = {f1:.3f}")
    fig.suptitle("Matrices de confusión en validación limpia (conteos; color = proporción por fila)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "c2_05_confusion_baseline_vs_mejor.png")
    plt.close(fig)


def fig_curvas(runs):
    cnn = [r for r, m in runs.items() if m["history"] is not None]
    if not cnn:
        return
    n = len(cnn)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False, sharey=True)
    for ax, r in zip(axes.flat, cnn):
        h = runs[r]["history"]
        ax.plot(h.epoch, h.train_acc_aug, color=ORANGE, lw=2, marker="o", ms=3, label="accuracy train (con aug.)")
        ax.plot(h.epoch, h.val_f1_clean, color=BLUE, lw=2, marker="o", ms=3, label="macro-F1 val limpia")
        be = runs[r]["best_epoch"]
        ax.axvline(be, color=GRAY, lw=1, ls="--")
        ax.set_title(nice(r), fontsize=8.5, loc="left")
        ax.set_xlabel("época", fontsize=8)
        ax.grid(True, color="#eeede9", linewidth=0.6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Curvas de entrenamiento (línea punteada = época elegida por validación)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "c2_06_curvas_entrenamiento.png")
    plt.close(fig)


def gallery(rows, root, path, title, max_n=12):
    rows = rows.head(max_n)
    if rows.empty:
        return
    cols = min(6, len(rows))
    nr = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(max(2.1 * cols, 4.2), 2.85 * nr), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        try:
            ax.imshow(Image.open(root / r.rel_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "imagen no disponible", ha="center", va="center", fontsize=7)
        ax.set_title(r.caption, fontsize=7.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-quick", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    runs = load_runs(a.include_quick)
    if not runs:
        raise SystemExit("No hay corridas en reports/corte2/runs/")

    df = table(runs)
    df.to_csv(REPORTS / "benchmark.csv", index=False)
    pd.DataFrame(
        [{"modelo": nice(r), **{SHORT[c]: round(runs[r]["val"]["clean"]["per_class"][c]["f1"], 4) for c in CLASSES}}
         for r in df.run]
    ).to_csv(REPORTS / "f1_por_clase.csv", index=False)

    base = next((r for r in df.run if r.startswith("svm_hsv_lbp")), None)
    cands = [r for r in df.run if r not in ("dummy", base)]
    best = cands[0] if cands else base   # mayor macro-F1 en val limpia (referencia para las figuras)

    fig_fuga(idx)
    fig_benchmark(df)
    fig_costo(df, best)
    fig_f1_clase(runs, df)
    fig_curvas(runs)

    # ---- Significancia y comparación de errores: mejor CNN vs baseline (val limpia)
    sig = {}
    if base and best and best != base:
        fig_confusion(runs, base, best)
        pb = runs[base]["preds"]
        pm = runs[best]["preds"]
        mg = pb[["rel_path", "class", "label", "clean", "pred"]].rename(columns={"pred": "pred_base"}).merge(
            pm[["rel_path", "pred"]].rename(columns={"pred": "pred_best"}), on="rel_path", how="inner")
        mg = mg[mg.clean].reset_index(drop=True)
        ok_b, ok_m = mg.pred_base == mg.label, mg.pred_best == mg.label
        b, c = int((ok_b & ~ok_m).sum()), int((~ok_b & ok_m).sum())
        y = mg.label.to_numpy()
        sig = {
            "baseline": base, "mejor": best, "n_val_limpia": int(len(mg)),
            "macroF1_baseline": runs[base]["val"]["clean"]["macro_f1"],
            "macroF1_mejor": runs[best]["val"]["clean"]["macro_f1"],
            "IC95_baseline": bootstrap_f1(y, mg.pred_base.to_numpy()),
            "IC95_mejor": bootstrap_f1(y, mg.pred_best.to_numpy()),
            "errores_baseline": int((~ok_b).sum()), "errores_mejor": int((~ok_m).sum()),
            "corregidos_por_mejor": c, "nuevos_errores_del_mejor": b,
            "mcnemar_p": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0,
        }
        save_json(sig, REPORTS / "significancia.json")
        lab = lambda i: SHORT[CLASSES[int(i)]]
        diff = mg[ok_b != ok_m].copy()
        diff["tipo"] = np.where(ok_m[ok_b != ok_m], "corregido_por_mejor", "nuevo_error_del_mejor")
        diff["real"] = diff.label.map(lab)
        diff["pred_baseline"] = diff.pred_base.map(lab)
        diff["pred_mejor"] = diff.pred_best.map(lab)
        diff[["rel_path", "tipo", "real", "pred_baseline", "pred_mejor"]].to_csv(
            REPORTS / "comparacion_errores.csv", index=False)

        try:
            root = dataset_root()
            err = pm[pm.clean & (pm.pred != pm.label)].copy()
            if "prob_0" in err:
                err["conf"] = err[[f"prob_{i}" for i in range(len(CLASSES))]].max(axis=1)
                err = err.sort_values("conf", ascending=False)
                err["caption"] = [f"real: {lab(l)}\npred: {lab(p)} ({c:.0%})" for l, p, c in zip(err.label, err.pred, err.conf)]
            else:
                err["caption"] = [f"real: {lab(l)}\npred: {lab(p)}" for l, p in zip(err.label, err.pred)]
            gallery(err, root, FIG / "c2_07_errores_mejor_modelo.png",
                    f"Errores de {nice(best)} en validación limpia (más confiados primero)")
            corr = diff[diff.tipo == "corregido_por_mejor"].copy()
            # primero la confusión crítica del baseline: tizón temprano <-> tizón tardío
            tiz = {"T.temprano", "T.tardío"}
            corr["_prio"] = ~(corr.real.isin(tiz) & corr.pred_baseline.isin(tiz))
            corr = corr.sort_values(["_prio", "real"])
            corr["caption"] = [f"real: {r}\nSVM: {b} | CNN: {m}" for r, b, m in zip(corr.real, corr.pred_baseline, corr.pred_mejor)]
            gallery(corr, root, FIG / "c2_08_errores_corregidos.png",
                    f"Casos que el SVM falla y {nice(best)} acierta")
            nuevos = diff[diff.tipo == "nuevo_error_del_mejor"].copy()
            nuevos["caption"] = [f"real: {r}\nSVM: {b} | CNN: {m}" for r, b, m in zip(nuevos.real, nuevos.pred_baseline, nuevos.pred_mejor)]
            gallery(nuevos, root, FIG / "c2_09_errores_nuevos.png",
                    f"Casos que el SVM acierta y {nice(best)} falla")
        except FileNotFoundError:
            print("(dataset no disponible: se omiten las galerías de imágenes)")

    pd.set_option("display.width", 200)
    print(df[["modelo", "train_macroF1", "val_macroF1_completo", "val_macroF1_limpio",
              "brecha_train_val", "params_M", "tiempo_entreno_min", "ms_por_imagen"]].to_string(index=False))
    if sig:
        print(f"\nMejor por métrica: {nice(best)}  vs  baseline {nice(base)}")
        print(f"  macro-F1 limpio: {sig['macroF1_mejor']:.4f} IC95 {np.round(sig['IC95_mejor'], 4)}  vs  "
              f"{sig['macroF1_baseline']:.4f} IC95 {np.round(sig['IC95_baseline'], 4)}")
        print(f"  errores: {sig['errores_mejor']} vs {sig['errores_baseline']} | corrige {sig['corregidos_por_mejor']}, "
              f"introduce {sig['nuevos_errores_del_mejor']} | McNemar p = {sig['mcnemar_p']:.4g}")
    print(f"\nTablas y figuras en {REPORTS}")


if __name__ == "__main__":
    main()
