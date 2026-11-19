"""Demo interactiva (presentación final): sube una foto de hoja de tomate y muestra
la predicción, el top-3, el mapa Grad-CAM y la advertencia de uso.

Instalar una vez:   pip install gradio
Ejecutar:           python src/demo_app.py      → abre http://127.0.0.1:7860

Si Gradio no está disponible, la misma demo funciona por terminal:
  python src/predict.py data/externas/plantdoc_campo --gradcam reports/corte3/inferencia
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gradio as gr
import numpy as np

from c2_common import REPO, dataset_root
from c3_common import overlay
from predict import Predictor

pr = Predictor()


def run(img):
    if img is None:
        return None, None, ""
    r, heat = pr.predict(img, gradcam=True)
    estado = "✅ Concluyente" if r["concluyente"] else "⚠️ NO concluyente"
    md = (f"### {r['clase_es']} — {r['confianza']:.1%}\n**{estado}** (umbral {pr.threshold:.0%})\n\n"
          f"> {r['advertencia']}")
    return r["probabilidades"], (overlay(img, heat) * 255).astype(np.uint8), md


def examples():
    ex = []
    try:
        root = dataset_root() / "test"
        for d in sorted(root.iterdir()):
            f = sorted(d.iterdir())[3]
            ex.append([str(f)])
    except Exception:
        pass
    camp = REPO / "data" / "externas" / "plantdoc_campo"
    if camp.exists():
        for d in sorted(camp.iterdir())[:4]:
            fs = sorted(d.iterdir())
            if fs:
                ex.append([str(fs[0])])
    return ex


with gr.Blocks(title="Hojas de tomate — Proyecto 7") as demo:
    gr.Markdown("# Diagnóstico visual de hojas de tomate (Proyecto 7)\n"
                "ResNet-18 fine-tune · 6 clases · **herramienta académica, no diagnóstico agronómico**")
    with gr.Row():
        inp = gr.Image(type="pil", label="Foto de la hoja")
        cam = gr.Image(label="Grad-CAM (dónde mira el modelo)")
    with gr.Row():
        lab = gr.Label(num_top_classes=3, label="Probabilidades")
        txt = gr.Markdown()
    gr.Examples(examples(), inputs=inp, label="Ejemplos: test (laboratorio) y campo (PlantDoc)")
    inp.change(run, inp, [lab, cam, txt])

if __name__ == "__main__":
    demo.launch()
