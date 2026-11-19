"""Corte 3 · Paso 3 — Pruebas de robustez y sensibilidad.

Sobre el test LIMPIO se aplican perturbaciones que imitan condiciones reales de captura
y se mide cuánto cae el macro-F1 de: modelo final (ResNet-18), alternativa (MobileNetV3)
y línea base (SVM HSV+LBP).

  brillo        sub/sobreexposición                 (factor 0,4 … 1,6)
  contraste     luz plana / dura                    (factor 0,5 … 1,5)
  rotación      hoja girada                         (15° … 180°)
  desenfoque    foto movida o desenfocada           (radio 1 … 4 px)
  ruido         sensor de celular con poca luz      (σ 10 … 50)
  jpeg          compresión de WhatsApp/web          (calidad 50 … 10)
  zoom          foto más cercana (recorte central)  (70 %, 50 %)
  fondo         ¿depende del fondo gris de laboratorio?  (negro, blanco, ruido, follaje sintético)

Además: entradas fuera de dominio sintéticas (ruido, colores planos) para ver si el
modelo "se abstiene" o responde con alta confianza (clasificador de mundo cerrado).

Salidas: reports/corte3/robustez.csv, robustez_ood.csv, figures/c3_08, c3_09
Uso:
  python src/c3_robustness.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score

from c2_common import CLASSES, SHORT, dataset_root, load_index
from c3_common import (ALT, BLUE, CONF_THRESHOLD, FIG3, FINAL, GRAY, INK, INK2, NICE3, ORANGE, REPORTS3, device,
                       features_from_pil, fit_svm, leaf_mask, load_cnn, open_rgb, predict_pils)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.bbox": "tight", "savefig.dpi": 200,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
RNG_SEED = 0


# --------------------------------------------------------------------------
# Perturbaciones (todas deterministas con semilla)
# --------------------------------------------------------------------------
def border_color(pil):
    a = np.asarray(pil)
    b = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in b.mean(0))


def jpeg(pil, q):
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise(pil, sigma, rng):
    a = np.asarray(pil).astype(np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, sigma, a.shape), 0, 255).astype(np.uint8))


def zoom(pil, frac):
    w, h = pil.size
    cw, ch = int(w * frac), int(h * frac)
    l, t = (w - cw) // 2, (h - ch) // 2
    return pil.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)


def foliage(shape, rng):
    """Fondo sintético tipo follaje: manchas verdes/cafés suavizadas."""
    h, w = shape
    base = rng.uniform(0, 1, (h // 8 + 1, w // 8 + 1, 1))
    base = np.asarray(Image.fromarray((base[:, :, 0] * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255
    g = np.stack([40 + 60 * base, 80 + 100 * base, 30 + 40 * (1 - base)], -1)
    return np.clip(g + rng.normal(0, 12, g.shape), 0, 255)


def replace_bg(pil, mode, rng):
    a = np.asarray(pil).astype(np.float32)
    m = leaf_mask(pil, size=a.shape[0])
    if m.shape != a.shape[:2]:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]))) > 127
    if mode == "negro":
        bg = np.zeros_like(a)
    elif mode == "blanco":
        bg = np.full_like(a, 255)
    elif mode == "ruido":
        bg = rng.uniform(0, 255, a.shape)
    else:
        bg = foliage(a.shape[:2], rng)
    out = np.where(m[:, :, None], a, bg)
    return Image.fromarray(out.astype(np.uint8))


def conditions():
    c = [("original", "—", lambda p, r: p)]
    for f in [0.4, 0.6, 0.8, 1.2, 1.4, 1.6]:
        c.append(("brillo", f"{f:g}", lambda p, r, f=f: ImageEnhance.Brightness(p).enhance(f)))
    for f in [0.5, 0.75, 1.25, 1.5]:
        c.append(("contraste", f"{f:g}", lambda p, r, f=f: ImageEnhance.Contrast(p).enhance(f)))
    for d in [15, 45, 90, 180]:
        c.append(("rotación", f"{d}°", lambda p, r, d=d: p.rotate(d, fillcolor=border_color(p))))
    for rad in [1, 2, 4]:
        c.append(("desenfoque", f"r={rad}", lambda p, r, rad=rad: p.filter(ImageFilter.GaussianBlur(rad))))
    for s in [10, 25, 50]:
        c.append(("ruido", f"σ={s}", lambda p, r, s=s: noise(p, s, r)))
    for q in [50, 20, 10]:
        c.append(("jpeg", f"q={q}", lambda p, r, q=q: jpeg(p, q)))
    for z in [0.7, 0.5]:
        c.append(("zoom", f"{int(z * 100)}%", lambda p, r, z=z: zoom(p, z)))
    for mode in ["negro", "blanco", "ruido", "follaje"]:
        c.append(("fondo", mode, lambda p, r, mode=mode: replace_bg(p, mode, r)))
    return c


def main():
    t0 = time.perf_counter()
    FIG3.mkdir(parents=True, exist_ok=True)
    root = dataset_root()
    idx = load_index()
    dt = idx[(idx.split == "test") & idx.clean.astype(bool)].reset_index(drop=True)
    y = dt.label.to_numpy()
    pils = [open_rgb(root / rp) for rp in dt.rel_path]
    print(f"Robustez sobre test limpio: {len(dt)} imágenes")

    dev = device()
    models = {FINAL: load_cnn(FINAL, dev), ALT: load_cnn(ALT, dev)}
    svm = fit_svm()

    rows, examples = [], {}
    for fam, lvl, fn in conditions():
        rng = np.random.default_rng(RNG_SEED)
        imgs = [fn(p, rng) for p in pils]
        examples[(fam, lvl)] = imgs[7]
        for run, m in models.items():
            pr = predict_pils(m, imgs, dev)
            pred, conf = pr.argmax(1), pr.max(1)
            rows.append(dict(familia=fam, nivel=lvl, run=run, modelo=NICE3[run],
                             macro_f1=f1_score(y, pred, average="macro"), accuracy=(pred == y).mean(),
                             conf_media=conf.mean(), pct_no_concluyente=(conf < CONF_THRESHOLD).mean(),
                             **{f"recall_{SHORT[c]}": ((pred == k) & (y == k)).sum() / max((y == k).sum(), 1)
                                for k, c in enumerate(CLASSES)}))
        X = np.stack([features_from_pil(p) for p in imgs])
        pred = svm.predict(X)
        rows.append(dict(familia=fam, nivel=lvl, run="svm_hsv_lbp", modelo=NICE3["svm_hsv_lbp"],
                         macro_f1=f1_score(y, pred, average="macro"), accuracy=(pred == y).mean(),
                         **{f"recall_{SHORT[c]}": ((pred == k) & (y == k)).sum() / max((y == k).sum(), 1)
                            for k, c in enumerate(CLASSES)}))
        r = [x for x in rows if x["familia"] == fam and x["nivel"] == lvl]
        print(f"{fam:<11}{lvl:<8} " + " | ".join(f"{x['modelo'][:12]} {x['macro_f1']:.3f}" for x in r))
    rob = pd.DataFrame(rows)
    base = rob[rob.familia == "original"].set_index("run").macro_f1
    rob["caida_vs_original"] = rob.apply(lambda r: base[r.run] - r.macro_f1, axis=1)
    rob.round(4).to_csv(REPORTS3 / "robustez.csv", index=False)

    # ---- Fuera de dominio sintético
    rng = np.random.default_rng(RNG_SEED)
    ood = {
        "ruido uniforme": [Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)) for _ in range(50)],
        "color plano": [Image.new("RGB", (256, 256), tuple(int(v) for v in rng.integers(0, 256, 3))) for _ in range(50)],
        "verde plano": [Image.new("RGB", (256, 256), (int(g * 0.4), int(g), int(g * 0.3)))
                        for g in rng.integers(80, 220, 50)],
    }
    orows = []
    for name, imgs in ood.items():
        for run, m in models.items():
            pr = predict_pils(m, imgs, dev)
            conf = pr.max(1)
            top = pd.Series(pr.argmax(1)).map(lambda k: SHORT[CLASSES[k]]).value_counts()
            orows.append(dict(entrada=name, modelo=NICE3[run], n=len(imgs), conf_media=conf.mean(),
                              pct_conf_mayor_umbral=(conf >= CONF_THRESHOLD).mean(),
                              clase_mas_predicha=f"{top.index[0]} ({top.iloc[0]}/{len(imgs)})"))
    pd.DataFrame(orows).round(4).to_csv(REPORTS3 / "robustez_ood.csv", index=False)
    print(pd.DataFrame(orows).round(3).to_string(index=False))

    figs(rob, examples, models[FINAL], dev)
    print(f"Listo en {(time.perf_counter() - t0) / 60:.1f} min")


def figs(rob, examples, model, dev):
    fams = ["brillo", "contraste", "rotación", "desenfoque", "ruido", "jpeg", "zoom", "fondo"]
    colors = {FINAL: BLUE, ALT: AQUA_, "svm_hsv_lbp": ORANGE}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharey=True)
    for ax, fam in zip(axes.ravel(), fams):
        for run, col in colors.items():
            o = rob[(rob.familia == "original") & (rob.run == run)]
            d = rob[(rob.familia == fam) & (rob.run == run)]
            lv = list(d.nivel)
            vals = list(d.macro_f1)
            if fam in ("brillo", "contraste"):   # insertar el original (factor 1) en su lugar
                lv_num = [float(v) for v in lv] + [1.0]
                order = np.argsort(lv_num)
                lv = [(["%g" % float(v) for v in d.nivel] + ["1"])[i] for i in order]
                vals = [(vals + [o.macro_f1.iloc[0]])[i] for i in order]
            else:
                lv = ["orig."] + lv
                vals = [o.macro_f1.iloc[0]] + vals
            ax.plot(range(len(lv)), vals, marker="o", ms=4, lw=1.8, color=col, label=NICE3[run])
            ax.set_xticks(range(len(lv)), lv, fontsize=8)
        ax.set_title(fam, loc="left", fontsize=10)
        ax.set_ylim(0, 1.03)
        ax.grid(True, axis="y", color="#e8e7e3", lw=0.6)
    axes[0, 0].set_ylabel("macro-F1 (test limpio)")
    axes[1, 0].set_ylabel("macro-F1 (test limpio)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Robustez: macro-F1 bajo perturbaciones de captura", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_08_robustez.png")
    plt.close(fig)

    # c3_09 — ejemplos visuales de cada perturbación con la predicción del modelo final
    keys = [("original", "—"), ("brillo", "0.4"), ("brillo", "1.6"), ("contraste", "0.5"), ("rotación", "45°"),
            ("desenfoque", "r=4"), ("ruido", "σ=50"), ("jpeg", "q=10"), ("zoom", "50%"), ("fondo", "negro"),
            ("fondo", "ruido"), ("fondo", "follaje")]
    keys = [k for k in keys if k in examples]
    pr = predict_pils(model, [examples[k] for k in keys], dev)
    fig, axes = plt.subplots(2, 6, figsize=(13.5, 5))
    for ax, k, p in zip(axes.ravel(), keys, pr):
        ax.imshow(examples[k])
        ax.set_title(f"{k[0]} {k[1] if k[1] != '—' else ''}\n→ {SHORT[CLASSES[p.argmax()]]} ({p.max():.0%})", fontsize=8.5)
        ax.axis("off")
    fig.suptitle("Ejemplos de perturbación y predicción del modelo final (misma hoja)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG3 / "c3_09_ejemplos_perturbacion.png")
    plt.close(fig)


AQUA_ = "#1baf7a"

if __name__ == "__main__":
    main()
