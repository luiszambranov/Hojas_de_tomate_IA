"""Genera notebooks/Corte3_Final_Tomate.ipynb (solo lee resultados de reports/corte3/;
la única celda que calcula es la demo de inferencia). Uso: python src/build_notebook_c3.py"""
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Proyecto 7 — Corte 3: integración, profundización y cierre responsable
**Diagnóstico visual de enfermedades en hojas de tomate** · Profundización III (IA) · 2026-II

Integrantes: Luis Alejandro Zambrano · Jhoan David Santamaría · Juan Norberto Pardo · Edwin Santiago Sandoval

Este cuaderno **no reentrena ni reevalúa**: lee los resultados generados por `python src/c3_run_all.py`.
La única celda que calcula algo es la demo de inferencia (sección 7).

| Sección | Entregable de la guía |
|---|---|
| 1 | Protocolo y regla del test reservado |
| 2 | Evaluación final sobre datos reservados |
| 3 | Comparación con la línea base del Corte 1 y con el provisional del Corte 2 |
| 4 | Profundización I: interpretabilidad visual (Grad-CAM) |
| 5 | Profundización II: robustez y sensibilidad |
| 6 | Prueba fuera del dominio de laboratorio (campo) |
| 7 | Función de inferencia y demo |
| 8 | Conclusiones, límites y condiciones de uso |
""")
code("""
import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display, Markdown
REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(REPO / 'src'))
R3 = REPO / 'reports' / 'corte3'; F3 = R3 / 'figures'
res = json.load(open(R3 / 'test_results.json', encoding='utf-8'))
pd.set_option('display.precision', 4)
print('Test evaluado el', res['fecha'], '| imágenes:', res['n_test'], '| limpias:', res['n_test_limpio'])
""")
md("""
## 1. Protocolo

- **Modelo final = modelo provisional del Corte 2**: ResNet-18 con fine-tuning (lr 3·10⁻⁴), elegido **solo con la
  validación limpia**. En el Corte 3 no se cambió ningún hiperparámetro.
- **Test reservado**: 990 imágenes. **Test limpio**: 908, sin las 82 copias de hojas que también están en train
  (fuga detectada en el Corte 2). Se reportan ambos; las conclusiones se apoyan en el limpio.
- **Un solo uso**: `c3_final_test.py` se niega a reevaluar si ya existen resultados.
- **Umbral de abstención** 0,70, fijado *antes* de ver el test: en validación, el único error tuvo confianza 0,54.
- La profundización (Grad-CAM, robustez, campo) **no modifica el modelo**: mide su comportamiento.
""")
md("## 2. Evaluación final sobre el test reservado")
code("""
t = pd.read_csv(R3 / 'test_resumen.csv')
display(t[['modelo','val_macroF1_limpio_C2','test_macroF1_limpio','ic95_inf','ic95_sup','test_macroF1_completo','errores_limpio','ECE']])
m = res['mcnemar_limpio']
print('McNemar final vs SVM:', m['final_vs_svm'])
print('McNemar final vs MobileNetV3:', m['final_vs_alt'])
print('Abstención (test limpio):', res['modelos'][res['modelo_final']]['selectiva_limpio'])
""")
code("display(Image(F3/'c3_01_val_vs_test.png', width=800)); display(Image(F3/'c3_02_confusion_test.png', width=900))")
code("""
pc = pd.read_csv(R3 / 'test_f1_por_clase.csv')
pc.pivot(index='clase', columns='modelo', values='f1')
""")
code("display(Image(F3/'c3_04_errores_test.png', width=800))")
md("""
**Lectura.**
- El modelo final comete **4 errores en 908 imágenes** (macro-F1 0,995; IC 95 % 0,991–0,999). La caída frente a la
  validación es de solo 0,0035, así que no había sobreajuste a la validación.
- Los errores siguen el patrón visto en los Cortes 1 y 2: 3 de **tizón tardío** (dos tomados como moho, uno como
  tizón temprano) y 1 de Septoria tomado como tizón temprano. Todos tienen confianza ≤ 0,83.
- **Ninguna hoja enferma se predijo como sana**, que en este problema es el error más costoso.
- Frente al SVM del Corte 1 la mejora es significativa (30 contra 4 errores no compartidos, p ≈ 6·10⁻⁶).
- Frente a MobileNetV3 hay **empate** (p = 1,0). Mantenemos ResNet-18 porque la decisión se tomó con validación en
  el Corte 2 y el test no debe usarse para reabrirla. MobileNetV3 queda como opción liviana para celular.
""")
md("## 3. Comparación explícita entre cortes")
code("pd.read_csv(R3 / 'comparacion_cortes.csv')")
md("""
| Corte | Qué cambió | Evidencia |
|---|---|---|
| C1 | Línea base de color y textura (HSV+LBP+SVM) | macro-F1 ≈ 0,97; confunde tizón temprano con tardío |
| C2 | Benchmark de 9 modelos, fuga detectada, validación limpia, transfer learning | ResNet-18 FT 0,999 en validación limpia; corrige 31 de los 32 errores del SVM |
| C3 | Test único, Grad-CAM, robustez, campo, abstención, inferencia | 0,995 en test; robusto a la luz, frágil al desenfoque, al fondo y al campo |

El modelo final **es el mismo** que el provisional. Lo que agrega el Corte 3 es **evidencia** sobre cuándo se
puede confiar en él y cuándo no.
""")
md("## 4. Profundización I — ¿el modelo mira la hoja o el fondo? (Grad-CAM)")
code("pd.read_csv(R3 / 'gradcam_resumen.csv', index_col=0)")
code("display(Image(F3/'c3_05_gradcam_por_clase.png', width=700))")
code("display(Image(F3/'c3_07_atencion_hoja_vs_area.png', width=550)); display(Image(F3/'c3_06_gradcam_errores.png', width=420))")
code("display(Image(F3/'c3_03_mascaras.png', width=900))")
md("""
**Lectura.**
- En promedio, el **60 % de la atención** cae sobre la hoja, que ocupa solo el 40 % de la imagen. En el 79 % de las
  imágenes la atención se concentra en la hoja, y los máximos coinciden con las lesiones (manchas, bordes necróticos).
- Es evidencia de que el modelo **sí mira la hoja**, pero no de que *solo* mire la hoja: el 40 % restante cae en el
  fondo o en el borde. La sección 5 lo pone a prueba cambiando el fondo.
- *Limitación del método:* la máscara (Otsu sobre saturación) es aproximada y a veces recorta partes de la hoja o
  incluye sombras, como se ve arriba. Grad-CAM tiene una resolución de 7×7 en layer4, así que es una señal gruesa.
""")
md("## 5. Profundización II — robustez y sensibilidad")
code("""
rob = pd.read_csv(R3 / 'robustez.csv')
rob.pivot_table(index=['familia','nivel'], columns='modelo', values='macro_f1', sort=False)
""")
code("display(Image(F3/'c3_08_robustez.png', width=1000)); display(Image(F3/'c3_09_ejemplos_perturbacion.png', width=1000))")
code("pd.read_csv(R3 / 'robustez_ood.csv')")
md("""
**Lectura.**
- **Robusto** a brillo (×0,4 a ×1,6), contraste, rotación, JPEG con calidad ≥ 20 y zoom del 70 %: el macro-F1 se
  mantiene en ≥ 0,98. El SVM baseline **colapsa** en esas mismas condiciones porque depende del histograma de color.
  Esa es una ventaja real del transfer learning con augmentación.
- **Frágil** al desenfoque (2 px → 0,60; 4 px → 0,07), al ruido fuerte (σ 25 → 0,53) y **al cambio de fondo**
  (0,51–0,61). Esto último confirma lo que sugería Grad-CAM: el fondo gris de laboratorio forma parte de lo que aprendió.
- **Mundo cerrado:** a una imagen de un solo color verde la llama "tizón tardío" con 98,6 % de confianza. El modelo
  no sabe abstenerse ante entradas absurdas, y el umbral de confianza no lo detecta.
""")
md("## 6. Prueba fuera del laboratorio: fotos de campo (PlantDoc)")
code("pd.read_csv(R3 / 'externas_resultados.csv')")
code("""
x = pd.read_csv(R3 / 'externas_preds.csv')
SH = {'Tomato___Bacterial_spot':'Bact.','Tomato___Early_blight':'T.temprano','Tomato___Late_blight':'T.tardío',
      'Tomato___Leaf_Mold':'Moho','Tomato___Septoria_leaf_spot':'Septoria','Tomato___healthy':'Sana'}
k = x[x.clase_real != 'fuera_de_clase']
pd.crosstab(k.clase_real.map(SH), k['pred_resnet18_finetune_lr0.0003'], rownames=['real'], colnames=['predicha'])
""")
code("display(Image(F3/'c3_10_campo_gradcam.png', width=1000))")
md("""
**Lectura.** Esta es la respuesta a "¿qué cambia entre una imagen controlada y una foto real de campo?":
- La accuracy baja de **99,6 % a 36 %**, pero la confianza media sigue en **93 %**. El modelo se equivoca con seguridad.
- Casi todo lo predice como **tizón temprano o tardío** (48 de 53). **Ninguna de las 8 hojas sanas** se reconoció
  como sana. 15 de las 16 fotos de virus que el modelo no conoce recibieron confianza ≥ 0,70.
- Las causas coinciden con lo medido en la sección 5: fondos naturales, varias hojas, escala distinta, desenfoque e
  iluminación no controlada. 4 de las 53 imágenes de PlantDoc son diapositivas o infografías; sin ellas el
  resultado casi no cambia (36,7 %).
- La línea base SVM también falla (25 %). El problema es el **dominio de los datos**, no la arquitectura.
""")
md("## 7. Función de inferencia y demo")
code("""
from predict import Predictor
from c2_common import dataset_root
from c3_common import overlay
import matplotlib.pyplot as plt
pr = Predictor()   # ResNet-18 final, umbral 0,70
ejemplos = [sorted((dataset_root()/'test'/c).iterdir())[3] for c in ['Tomato___Early_blight','Tomato___Late_blight','Tomato___healthy']]
ejemplos += [REPO/'data/externas/plantdoc_campo/Tomato___healthy/03.jpg', REPO/'data/externas/plantdoc_campo/_fuera_de_clase/mosaico_02.jpg']
fig, axes = plt.subplots(1, len(ejemplos), figsize=(3.2*len(ejemplos), 3.6))
for ax, f in zip(axes, ejemplos):
    r, heat = pr.predict(f, gradcam=True)
    from PIL import Image as PImage
    ax.imshow(overlay(PImage.open(f).convert('RGB'), heat)); ax.axis('off')
    ax.set_title(f"{f.parent.name.replace('Tomato___','')[:14]}\\n→ {r['clase_es']} {r['confianza']:.0%}" + ('' if r['concluyente'] else ' (NO concl.)'), fontsize=9)
plt.tight_layout(); plt.show()
print(pr.predict(ejemplos[0]))
""")
md("""
Para la presentación: `python src/demo_app.py` abre una página web donde se sube una foto y se ven la predicción, el
top-3, el Grad-CAM y la advertencia. Por terminal: `python src/predict.py <imagen o carpeta>`.
""")
md("""
## 8. Conclusiones, límites y condiciones de uso

**Conclusiones.**
1. El problema y el criterio de éxito se mantuvieron desde el Corte 1 (macro-F1 y F1 por clase). La complejidad se
   aumentó solo cuando la evidencia lo justificó: el SVM se confundía entre tizón temprano y tardío, y el transfer
   learning resolvió esos errores.
2. En el dominio PlantVillage la solución es **excelente y estadísticamente mejor que la línea base**: 0,995 contra
   0,967 en test, 4 errores contra 30, y ningún falso "sano".
3. La profundización mostró **por qué no basta esa cifra**: el modelo depende del fondo de laboratorio, es frágil al
   desenfoque y en campo acierta 36 % con confianza alta.

**Condiciones de uso** (ver `MODEL_CARD.md`): una sola hoja, centrada, sobre fondo neutro, enfocada y bien
iluminada. El resultado es orientativo; si sale "no concluyente", tomar otra foto o consultar a un experto.
**Prohibido presentarlo como diagnóstico agronómico sin validación de campo.**

**Trabajo futuro, en orden de impacto esperado:** (1) entrenar con fotos de campo, por ejemplo el train de PlantDoc,
y con reemplazo aleatorio de fondo; (2) agregar una clase "otro / no es hoja de tomate" o un detector de entradas
fuera de dominio; (3) segmentar la hoja antes de clasificar; (4) validar con un agrónomo sobre fotos de cultivos
locales.

### Respuestas a las preguntas de la guía
- **¿Qué enfermedades son más difíciles de separar visualmente?** Tizón temprano y tizón tardío: concentran la
  confusión del SVM en el Corte 1 y 3 de los 4 errores del modelo final en test. En campo, todo tiende a esas dos clases.
- **¿Qué evidencia muestra que el modelo mira la hoja y no solo el fondo?** Grad-CAM pone el 60 % de la atención en
  la hoja (que ocupa el 40 % de la imagen), con máximos sobre las lesiones. La prueba de cambio de fondo matiza esto:
  el fondo influye, y el macro-F1 cae a 0,51–0,61.
- **¿Qué cambia entre una imagen controlada y una foto de campo?** El fondo, la cantidad de hojas, la escala, la luz
  y el enfoque. El resultado pasa de 99,6 % a 36 % de accuracy con la misma confianza, así que el desempeño en
  PlantVillage no se transfiere al cultivo.
""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = REPO / "notebooks" / "Corte3_Final_Tomate.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("Notebook:", out)
