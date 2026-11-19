"""Corte 3 · Paso 2 — Profundización: interpretabilidad visual con Grad-CAM.

Pregunta que responde: ¿el modelo mira la HOJA (lesiones) o el FONDO?

1. Calcula Grad-CAM del modelo final (ResNet-18, capa layer4) para TODO el test.
2. Estima una máscara de la hoja y mide qué fracción del mapa de atención cae
   dentro de la hoja. Se compara con la fracción de área que ocupa la hoja
   (lo que daría un mapa uniforme, sin preferencia).
3. Galerías: aciertos por clase, errores, tizón temprano vs. tardío y verificación de máscaras.

Salidas: reports/corte3/gradcam_test.csv, gradcam_resumen.csv, figures/c3_03, c3_05..c3_07
Uso:
  python src/c3_gradcam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c2_common import CLASSES, SHORT, dataset_root, load_index
from c3_common import (BLUE, FIG3, FINAL, GRAY, INK, INK2, ORANGE, REPORTS3, GradCAM, cam_on_leaf, device,
                       leaf_mask, load_cnn, open_rgb, overlay, target_layer)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.bbox": "tight", "savefig.dpi": 200,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def main():
    FIG3.mkdir(parents=True, exist_ok=True)
    root = dataset_root()
    idx = load_index()
    dtest = idx[idx.split == "test"].reset_index(drop=True)
    dev = device()
    model = load_cnn(FINAL, dev)
    cam = GradCAM(model, target_layer(model, FINAL))

    rows = []
    print(f"Grad-CAM sobre {len(dtest)} imágenes de test...")
    for i, r in dtest.iterrows():
        pil = open_rgb(root / r.rel_path)
        heat, probs = cam(pil)
        m = leaf_mask(pil)
        rows.append({"rel_path": r.rel_path, "class": r["class"], "label": r.label, "clean": r.clean,
                     "pred": int(probs.argmax()), "conf": float(probs.max()),
                     "area_hoja": float(m.mean()), "cam_en_hoja": cam_on_leaf(heat, m)})
        if i % 200 == 0:
            print(f"  {i}/{len(dtest)}")
    g = pd.DataFrame(rows)
    g["acierto"] = g.pred == g.label
    # Si la máscara falla (casi toda la imagen o casi nada), no se usa en el resumen
    g["mascara_valida"] = g.area_hoja.between(0.05, 0.95)
    g["ganancia"] = g.cam_en_hoja / g.area_hoja.clip(lower=1e-3)
    g.to_csv(REPORTS3 / "gradcam_test.csv", index=False)

    v = g[g.mascara_valida & g.clean.astype(bool)]
    res = v.groupby("class").agg(n=("rel_path", "size"), area_hoja_media=("area_hoja", "mean"),
                                 cam_en_hoja_media=("cam_en_hoja", "mean"),
                                 pct_imgs_cam_mayoria_en_hoja=("cam_en_hoja", lambda s: (s > 0.5).mean()))
    res = res.reindex(CLASSES)
    tot = pd.DataFrame([{"n": len(v), "area_hoja_media": v.area_hoja.mean(), "cam_en_hoja_media": v.cam_en_hoja.mean(),
                         "pct_imgs_cam_mayoria_en_hoja": (v.cam_en_hoja > 0.5).mean()}], index=["TOTAL"])
    res = pd.concat([res, tot]).round(4)
    res.to_csv(REPORTS3 / "gradcam_resumen.csv")
    print(res.to_string())
    print(f"Máscaras válidas: {g.mascara_valida.mean():.1%}")

    figs(g, dtest, root, cam)


def show_pair(ax1, ax2, pil, cam, title):
    heat, probs = cam(pil)
    ax1.imshow(pil.resize((224, 224)))
    ax2.imshow(overlay(pil, heat))
    ax1.set_title(title, fontsize=8, loc="left")
    for a in (ax1, ax2):
        a.axis("off")
    return probs


def figs(g, dtest, root, cam):
    rng = np.random.default_rng(0)
    # c3_03 — verificación visual de máscaras
    sample = g.sample(12, random_state=1)
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        pil = open_rgb(root / r.rel_path)
        img = np.asarray(pil.resize((224, 224))).copy()
        m = leaf_mask(pil).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (235, 104, 52), 2)
        ax.imshow(img)
        ax.set_title(f"{SHORT[r['class']]} · hoja {r.area_hoja:.0%}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Verificación de la máscara de hoja (contorno naranja)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_03_mascaras.png")
    plt.close(fig)

    # c3_05 — galería por clase: 2 aciertos por clase (original + Grad-CAM)
    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(8.6, 2.2 * len(CLASSES)))
    for k, c in enumerate(CLASSES):
        ok = g[(g["class"] == c) & g.acierto & g.clean.astype(bool)]
        pick = ok.sample(min(2, len(ok)), random_state=k)
        for j, (_, r) in enumerate(pick.iterrows()):
            show_pair(axes[k, 2 * j], axes[k, 2 * j + 1], open_rgb(root / r.rel_path), cam,
                      f"{SHORT[c]} · {r.conf:.0%} · atención en hoja {r.cam_en_hoja:.0%}")
    fig.suptitle("Grad-CAM del modelo final: aciertos por clase (rojo = más atención)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_05_gradcam_por_clase.png")
    plt.close(fig)

    # c3_06 — errores + casos tizón temprano/tardío
    err = g[~g.acierto].sort_values("conf", ascending=False).head(4)
    pool = g[g["class"].isin(["Tomato___Early_blight", "Tomato___Late_blight"]) & g.acierto]
    tt = pool.sample(min(6 - len(err), len(pool)), random_state=3)
    sel = pd.concat([err.assign(tipo="ERROR"), tt.assign(tipo="acierto")]).head(6)
    if len(sel):
        fig, axes = plt.subplots(len(sel), 2, figsize=(4.6, 2.3 * len(sel)), squeeze=False)
        for k, (_, r) in enumerate(sel.iterrows()):
            show_pair(axes[k, 0], axes[k, 1], open_rgb(root / r.rel_path), cam,
                      f"{r.tipo}: real {SHORT[r['class']]} → pred {SHORT[CLASSES[r.pred]]} ({r.conf:.0%})")
        fig.suptitle("Grad-CAM: errores y casos tizón temprano/tardío", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG3 / "c3_06_gradcam_errores.png")
        plt.close(fig)

    # c3_07 — ¿atención en hoja vs. área de hoja?
    v = g[g.mascara_valida]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.grid(True, color="#e8e7e3", lw=0.6)
    ax.scatter(v.area_hoja, v.cam_en_hoja, s=9, alpha=0.45, color=BLUE, edgecolor="none", label="imagen de test")
    e = v[~v.acierto]
    if len(e):
        ax.scatter(e.area_hoja, e.cam_en_hoja, s=40, color=ORANGE, edgecolor="white", label="error del modelo")
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    ax.text(0.62, 0.55, "atención uniforme\n(sin preferencia)", color=INK2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fracción de la imagen ocupada por la hoja")
    ax.set_ylabel("fracción de la atención Grad-CAM sobre la hoja")
    above = (v.cam_en_hoja > v.area_hoja).mean()
    ax.set_title(f"El modelo atiende a la hoja: {above:.0%} de las imágenes sobre la diagonal", loc="left",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(FIG3 / "c3_07_atencion_hoja_vs_area.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
