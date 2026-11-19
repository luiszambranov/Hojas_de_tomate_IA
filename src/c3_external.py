"""Corte 3 · Paso 4 — Prueba con fotografías de CAMPO (fuera de PlantVillage).

Usa imágenes del split de test de PlantDoc (Singh et al., 2020; CC BY 4.0;
https://github.com/pratikkayal/PlantDoc-Dataset): fotos reales de hojas de tomate
tomadas en campo/internet, con fondos naturales, varias hojas, sombras, etc.

  data/externas/plantdoc_campo/<CLASE>/*.jpg     53 fotos de las 6 clases del proyecto
  data/externas/plantdoc_campo/_fuera_de_clase/  16 fotos de virus del mosaico y de la hoja
                                                 amarilla (clases que el modelo NO conoce)

NO se usan para entrenar ni para decidir nada: solo para medir el salto laboratorio → campo.
Las etiquetas de PlantDoc son de anotadores, no de un diagnóstico de laboratorio.

Salidas: reports/corte3/externas_resultados.csv, externas_preds.csv, figures/c3_10
Uso:
  python src/c3_external.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from c2_common import CLASSES, REPO, SHORT
from c3_common import (ALT, CONF_THRESHOLD, FIG3, FINAL, INK, INK2, NICE3, REPORTS3, GradCAM, device,
                       features_from_pil, fit_svm, load_cnn, open_rgb, overlay, predict_pils, target_layer)

EXT = REPO / "data" / "externas" / "plantdoc_campo"
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 200, "axes.titleweight": "bold"})


def main():
    if not EXT.exists():
        print(f"No existe {EXT}; se omite la prueba de campo.")
        return
    files, labels = [], []
    for d in sorted(EXT.iterdir()):
        if not d.is_dir():
            continue
        lab = CLASSES.index(d.name) if d.name in CLASSES else -1
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append(f)
                labels.append(lab)
    y = np.array(labels)
    pils = [open_rgb(f) for f in files]
    known = y >= 0
    print(f"Fotos de campo: {known.sum()} de clases conocidas + {(~known).sum()} fuera de clase")

    dev = device()
    preds = pd.DataFrame({"archivo": [f.relative_to(EXT).as_posix() for f in files],
                          "clase_real": [CLASSES[k] if k >= 0 else "fuera_de_clase" for k in y]})
    rows = []
    for run in [FINAL, ALT, "svm_hsv_lbp"]:
        if run == "svm_hsv_lbp":
            p = fit_svm().predict(np.stack([features_from_pil(x) for x in pils]))
            conf = np.full(len(p), np.nan)
        else:
            pr = predict_pils(load_cnn(run, dev), pils, dev)
            p, conf = pr.argmax(1), pr.max(1)
        preds[f"pred_{run}"] = [SHORT[CLASSES[k]] for k in p]
        preds[f"conf_{run}"] = conf
        acc = (p[known] == y[known]).mean()
        row = dict(modelo=NICE3[run], n_conocidas=int(known.sum()), accuracy=acc,
                   macro_f1=f1_score(y[known], p[known], average="macro", labels=list(range(6)), zero_division=0))
        if run != "svm_hsv_lbp":
            ok = conf[known] >= CONF_THRESHOLD
            row.update(conf_media_conocidas=conf[known].mean(), pct_no_concluyente_conocidas=(~ok).mean(),
                       accuracy_si_concluyente=(p[known][ok] == y[known][ok]).mean() if ok.any() else np.nan,
                       conf_media_fuera_de_clase=conf[~known].mean() if (~known).any() else np.nan,
                       pct_fuera_de_clase_con_conf_alta=(conf[~known] >= CONF_THRESHOLD).mean()
                       if (~known).any() else np.nan)
        rows.append(row)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(REPORTS3 / "externas_resultados.csv", index=False)
    preds.round(4).to_csv(REPORTS3 / "externas_preds.csv", index=False)
    print(res.to_string(index=False))

    # c3_10 — galería con Grad-CAM (una foto por clase + 2 fuera de clase)
    model = load_cnn(FINAL, dev)
    cam = GradCAM(model, target_layer(model, FINAL))
    pick = []
    for k in list(range(6)) + [-1, -1]:
        cand = [i for i in range(len(files)) if y[i] == k and i not in pick]
        if cand:
            pick.append(cand[0] if k >= 0 else cand[len([j for j in pick if y[j] == -1]) * 5 % len(cand)])
    fig, axes = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 5.2), squeeze=False)
    for j, i in enumerate(pick):
        heat, pr = cam(pils[i])
        axes[0, j].imshow(pils[i].resize((224, 224)))
        axes[1, j].imshow(overlay(pils[i], heat))
        real = SHORT[CLASSES[y[i]]] if y[i] >= 0 else "fuera de clase"
        ok = "✓" if pr.argmax() == y[i] else "✗"
        axes[0, j].set_title(f"real: {real}\npred: {SHORT[CLASSES[pr.argmax()]]} ({pr.max():.0%}) {ok}", fontsize=8,
                             color=INK)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    fig.suptitle("Fotos de campo (PlantDoc): predicción y Grad-CAM del modelo final", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_10_campo_gradcam.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
