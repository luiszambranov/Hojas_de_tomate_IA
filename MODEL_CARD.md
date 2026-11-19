# Model card (simplificada) — Clasificador de enfermedades en hojas de tomate

> **Advertencia principal.** Este modelo **NO es un diagnóstico agronómico**. Su desempeño de
> 99,5 % se midió con fotos de laboratorio (PlantVillage). Con fotos reales de campo acertó solo
> **36 %**, casi siempre con confianza alta. No debe usarse para decidir tratamientos, aplicar
> agroquímicos ni descartar una enfermedad sin una **validación de campo** y la confirmación de un
> agrónomo o de un laboratorio fitosanitario.

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Arquitectura | ResNet-18 (torchvision), preentrenada en ImageNet, **fine-tuning completo** |
| Entrada | Imagen RGB redimensionada a 224×224, normalización ImageNet |
| Salida | Probabilidades para 6 clases + regla de abstención (confianza < 0,70 → "no concluyente") |
| Entrenamiento | AdamW lr 3·10⁻⁴, wd 10⁻⁴, coseno, batch 32, 15 épocas, early stopping (paciencia 5), semilla 42, float32 |
| Augmentación (solo train) | RandomResizedCrop(0,7–1), volteos H/V, rotación ±20°, ColorJitter(0,2) |
| Selección | Mejor época por **macro-F1 en validación limpia** (Corte 2). El test no intervino. |
| Tamaño / costo | 11,2 M parámetros (~45 MB); 4,2 ms por imagen en una GTX 1650 (batch 1) |
| Pesos | `models/resnet18_finetune_lr0.0003.pt` (no se versiona; se regenera con `src/c2_run_benchmark.py`) |
| Alternativa | MobileNetV3-Large fine-tune (4,2 M parámetros): **empata estadísticamente** en test |
| Equipo | Luis Alejandro Zambrano, Jhoan David Santamaría, Juan Norberto Pardo, Edwin Santiago Sandoval — Profundización III (IA), Uniagustiniana, 2026-II |

## 2. Uso previsto y fuera de alcance

**Previsto:** ejercicio académico de visión por computador, y prototipo de apoyo para **pre-clasificar**
fotos de **una hoja aislada sobre fondo neutro**, siempre con revisión de un experto.

**Fuera de alcance (no usar):**
- Diagnóstico definitivo o recomendación de tratamientos.
- Fotos de campo de la planta completa, varias hojas, fondo de follaje o suelo.
- Otros cultivos y otras enfermedades del tomate. El modelo **no sabe decir "no sé"** fuera de sus
  6 clases: los virus del mosaico y del rizado amarillo los clasificó como tizón con 94 % de confianza media.
- Fotos borrosas o con mucho ruido (ver §5).

## 3. Datos

- **Entrenamiento/validación/test:** Processed Tomato Leaf Disease Dataset (Mendeley, DOI
  10.17632/3zwdw6y4pn.1, derivado de PlantVillage, CC BY 4.0). 4.586 / 981 / 990 imágenes.
  Ver [`DATASET_CARD.md`](DATASET_CARD.md).
- **Fuga detectada:** 72 imágenes de validación y 82 de test son copias aumentadas de hojas que también
  están en train. Las métricas principales se dan sobre las particiones **limpias** (909 / 908).
- **Campo (solo evaluación):** 53 fotos de PlantDoc de las 6 clases + 16 de clases desconocidas.

## 4. Desempeño en test reservado (evaluado una sola vez, Corte 3)

Test limpio: 908 imágenes. IC 95 % por bootstrap (2.000 remuestreos).

| Modelo | Rol | macro-F1 val. limpia (C2) | **macro-F1 test limpio** | IC 95 % | Errores |
|---|---|---:|---:|---|---:|
| Dummy (clase mayoritaria) | referencia | 0,051 | 0,052 | 0,046–0,057 | 742 |
| SVM HSV+LBP | línea base Corte 1 | 0,965 | 0,967 | 0,956–0,978 | 30 |
| **ResNet-18 FT** | **provisional C2 = final C3** | 0,999 | **0,995** | 0,991–0,999 | **4** |
| MobileNetV3 FT | alternativa | 0,998 | 0,995 | 0,990–0,999 | 4 |

- **ResNet-18 vs. SVM:** McNemar p = 6·10⁻⁶. ResNet acierta 30 imágenes que el SVM falla, y el SVM
  acierta 4 que ResNet falla.
- **ResNet-18 vs. MobileNetV3:** p = 1,0 (4 contra 4), así que no hay diferencia. Se mantiene ResNet-18
  por la decisión del Corte 2 (menor latencia en GPU), sin reabrirla con el test.
- **Caída validación → test:** 0,0035. El modelo no estaba sobreajustado a la validación.
- **Test completo (con fuga):** 0,996. La fuga infla menos de 0,001.
- **F1 por clase en test limpio:** Bact. 1,000 · T. temprano 0,993 · T. tardío 0,991 · Moho 0,992 ·
  Septoria 0,997 · Sana 1,000.
- **Errores (4/908):** 3 hojas de **tizón tardío**, dos clasificadas como moho y una como tizón temprano,
  y 1 de Septoria clasificada como tizón temprano. Sus confianzas fueron 0,52, 0,74, 0,77 y 0,83. Ninguna
  hoja enferma se predijo como sana.
- **Calibración:** ECE = 0,005 en test limpio.
- **Regla de abstención (umbral 0,70, fijado antes de ver el test):** 2 imágenes salen como no concluyentes
  (cobertura del 99,8 %). En las aceptadas, la accuracy es 99,7 %, y 3 de los 4 errores superan el umbral.

## 5. Profundización: interpretabilidad y robustez

**Grad-CAM (layer4)** sobre el test, con una máscara aproximada de la hoja (896 imágenes limpias con máscara válida):
- En promedio, el **60 % de la atención** cae sobre la hoja, que ocupa el 40 % de la imagen.
- En el **79 %** de las imágenes, la mayor parte de la atención está sobre la hoja.
- Las zonas de mayor atención coinciden con las lesiones (ver `reports/corte3/figures/c3_05`).
- **Pero el fondo sí importa.** Al reemplazar el fondo gris de laboratorio por negro, blanco, ruido o
  follaje, el macro-F1 cae a **0,51–0,61**. El modelo usa la hoja, pero aprendió también el contexto del
  laboratorio. *(La máscara es aproximada y recorta parte de la hoja en algunas fotos, así que la caída
  puede estar algo exagerada.)*

**Perturbaciones de captura** (macro-F1 en test limpio; el original es 0,995):

| Perturbación | ResNet-18 | MobileNetV3 | SVM baseline |
|---|---:|---:|---:|
| Brillo ×0,4 / ×1,6 | 0,977 / 0,985 | 0,973 / 0,978 | 0,108 / 0,089 |
| Contraste ×0,5 | 0,984 | 0,997 | 0,366 |
| Rotación 45° / 90° | 0,991 / 0,980 | 0,974 / 0,990 | 0,905 / 0,967 |
| JPEG calidad 20 / 10 | 0,986 / 0,923 | 0,968 / 0,807 | 0,570 / 0,159 |
| Zoom 70 % / 50 % | 0,989 / 0,863 | 0,988 / 0,789 | 0,587 / 0,182 |
| Desenfoque r = 1 / 2 / 4 px | 0,968 / **0,603** / **0,074** | 0,985 / 0,414 / 0,144 | 0,437 / 0,051 / 0,051 |
| Ruido σ = 10 / 25 / 50 | 0,984 / **0,530** / **0,222** | 0,970 / 0,546 / 0,289 | 0,815 / 0,452 / 0,105 |
| Fondo negro / follaje | **0,566 / 0,508** | 0,413 / 0,786 | 0,087 / 0,107 |

La CNN es **muy robusta a la iluminación, la rotación y la compresión**, mientras que el SVM baseline colapsa
porque depende del histograma de color. Las **debilidades** son el desenfoque, el ruido fuerte y el cambio de fondo.

**Fuera de dominio sintético:** a imágenes de un solo color verde las clasifica como tizón tardío con 98,6 %
de confianza (50/50). El modelo **no detecta entradas absurdas**.

## 6. Prueba de campo (PlantDoc)

| Modelo | Accuracy (53 fotos de las 6 clases) | macro-F1 | Confianza media |
|---|---:|---:|---:|
| ResNet-18 FT | **0,36** | 0,23 | 0,93 |
| MobileNetV3 FT | 0,30 | 0,19 | 0,90 |
| SVM HSV+LBP | 0,25 | 0,15 | — |

- Si se excluyen 4 imágenes que son diapositivas o infografías y no fotos, ResNet-18 acierta 36,7 % de
  49 fotos. La conclusión no cambia.
- En campo, el modelo **colapsa hacia tizón temprano y tizón tardío**: 48 de 53 predicciones caen ahí.
  **Ninguna de las 8 hojas sanas** se reconoció como sana.
- **32 de los 34 errores tienen confianza ≥ 0,70**, así que el umbral de abstención no protege en campo.
- De las 16 fotos de virus (clases desconocidas), **15 recibieron confianza ≥ 0,70** en alguna de las 6 clases.

**Lectura:** el 99,5 % del test es válido **solo dentro del dominio de PlantVillage**. Entre una foto
controlada y una de campo cambian el fondo, la iluminación, la cantidad de hojas, la escala y el
desenfoque, y esos son justo los factores a los que el modelo resultó sensible.

## 7. Riesgos, sesgos y condiciones de uso

| Riesgo | Consecuencia | Mitigación |
|---|---|---|
| Uso en campo sin validar | Falsos diagnósticos con confianza alta; tratar una planta sana o no tratar una enferma | Advertencia en `predict.py` y en la demo; esta model card; validación de campo antes de cualquier uso |
| Mundo cerrado (6 clases) | Otra enfermedad se reporta como una de las 6 | Documentado; trabajo futuro: clase "otra/no es hoja" o detección de anomalías |
| Dependencia del fondo de laboratorio | Rendimiento bajo con fondos naturales | Trabajo futuro: entrenar con reemplazo de fondo y fotos de campo (p. ej. PlantDoc train) |
| Exceso de confianza | El usuario cree el resultado | Umbral de abstención, que solo ayuda en el dominio de laboratorio; mostrar top-3 y Grad-CAM |
| Fuga en la partición original | Métricas infladas | Validación y test limpios; diferencia medida menor a 0,001 |
| Licencias | — | Datos CC BY 4.0 (Mendeley/PlantVillage y PlantDoc), con atribución en el repo |
| Privacidad | — | Las imágenes no contienen personas ni datos personales |

**Condiciones mínimas de uso:** una hoja, centrada, sobre fondo neutro y uniforme, enfocada y con buena luz.
El resultado siempre es orientativo. Si sale "no concluyente", hay que tomar otra foto o consultar a un experto.

## 8. Reproducción

```powershell
python src/c2_run_benchmark.py --workers 2   # entrena (Corte 2)
python src/c3_run_all.py                     # test final, Grad-CAM, robustez y campo (Corte 3)
python src/predict.py <imagen_o_carpeta>     # inferencia
python src/demo_app.py                       # demo web (requiere gradio)
```
