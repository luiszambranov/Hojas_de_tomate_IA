"""Inferencia sobre imágenes nuevas — Proyecto 7 (hojas de tomate).

Uso desde la terminal:
  python src/predict.py ruta/a/hoja.jpg
  python src/predict.py carpeta_con_fotos/ --gradcam reports/corte3/inferencia
  python src/predict.py hoja.jpg --modelo alternativa       # MobileNetV3 (más liviano)

Uso desde Python / notebook:
  from predict import Predictor
  pr = Predictor()
  r = pr.predict("hoja.jpg")              # dict con clase, confianza, top-3, advertencia
  r, heat = pr.predict("hoja.jpg", gradcam=True)

IMPORTANTE (condiciones de uso): herramienta de apoyo académico. NO es un diagnóstico
agronómico. Fue entrenada con hojas aisladas sobre fondo de laboratorio (PlantVillage) y
solo conoce 6 clases: cualquier otra enfermedad o cultivo se forzará a una de ellas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image

from c2_common import CLASS_ES, CLASSES
from c3_common import ALT, CONF_THRESHOLD, FINAL, GradCAM, device, load_cnn, open_rgb, overlay, target_layer

AVISO = ("Resultado orientativo: modelo entrenado con fotos de laboratorio (PlantVillage); "
         "confirme con un agrónomo antes de tomar decisiones.")


class Predictor:
    def __init__(self, run: str = FINAL, threshold: float = CONF_THRESHOLD):
        self.run = run
        self.threshold = threshold
        self.dev = device()
        self.model = load_cnn(run, self.dev)
        self.cam = GradCAM(self.model, target_layer(self.model, run))

    def predict(self, img, gradcam: bool = False):
        pil = img if isinstance(img, Image.Image) else open_rgb(img)
        pil = pil.convert("RGB")
        heat, probs = self.cam(pil)
        order = np.argsort(-probs)
        k = int(order[0])
        conf = float(probs[k])
        concluyente = conf >= self.threshold
        r = {
            "clase": CLASSES[k],
            "clase_es": CLASS_ES[CLASSES[k]],
            "confianza": conf,
            "concluyente": concluyente,
            "top3": [(CLASS_ES[CLASSES[i]], float(probs[i])) for i in order[:3]],
            "probabilidades": {CLASS_ES[c]: float(p) for c, p in zip(CLASSES, probs)},
            "advertencia": (AVISO if concluyente else
                            f"NO CONCLUYENTE (confianza {conf:.0%} < {self.threshold:.0%}): la imagen puede ser "
                            "de otra enfermedad, otro cultivo o estar mal tomada. " + AVISO),
            "modelo": self.run,
        }
        return (r, heat) if gradcam else r


def _iter_images(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        else:
            yield p


def main():
    ap = argparse.ArgumentParser(description="Clasifica fotos de hojas de tomate (6 clases).")
    ap.add_argument("imagenes", nargs="+", help="archivos o carpetas")
    ap.add_argument("--modelo", choices=["final", "alternativa"], default="final")
    ap.add_argument("--gradcam", metavar="CARPETA", help="guardar el mapa Grad-CAM de cada imagen aquí")
    args = ap.parse_args()
    pr = Predictor(FINAL if args.modelo == "final" else ALT)
    out = Path(args.gradcam) if args.gradcam else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for f in _iter_images(args.imagenes):
        r, heat = pr.predict(f, gradcam=True)
        estado = "OK " if r["concluyente"] else "?? "
        top = " | ".join(f"{n} {p:.1%}" for n, p in r["top3"])
        print(f"{estado}{f.name:<40} → {r['clase_es']:<18} {r['confianza']:6.1%}   (top-3: {top})")
        if not r["concluyente"]:
            print("    " + r["advertencia"].split(":")[0])
        if out:
            pil = open_rgb(f)
            vis = np.concatenate([np.asarray(pil.resize((224, 224))) / 255, overlay(pil, heat)], axis=1)
            Image.fromarray((vis * 255).astype(np.uint8)).save(out / f"{f.stem}_gradcam.png")
    print("\n" + AVISO)


if __name__ == "__main__":
    main()
