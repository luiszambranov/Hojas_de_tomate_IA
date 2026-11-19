# Bitácora de uso de IA generativa

Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate. Cortes 1, 2 y 3.

El uso de IA generativa está permitido como apoyo, siempre que sea
transparente, verificado y adaptado por el equipo. Esta bitácora documenta
dónde se usó, con qué propósito y cómo se verificó cada resultado.

## Corte 1

| Herramienta usada | Propósito | Interacción relevante | ¿Cómo se verificó o adaptó la respuesta? |
|---|---|---|---|
| Claude (Anthropic) | particionar el dataset (train/val/test) y confirmar balance de clases | Se pidió construir un script que recorriera las carpetas del dataset y contara imágenes por clase y partición | Se ejecutó el script contra el dataset real (6.557 imágenes) y se revisaron los conteos manualmente; coinciden con la proporción 70/15/15 declarada por el proveedor |
| Claude (Anthropic) | Diseñar y escribir el código de extracción de features clásicas (histograma de color HSV + Local Binary Pattern) para el baseline | Se solicitó una función reutilizable en `src/data_utils.py` | Se probó la función sobre imágenes de muestra y se revisó que las dimensiones y rangos de los histogramas fueran razonables antes de usarla en el entrenamiento |
| Claude (Anthropic) | Entrenar y evaluar el baseline (SVM sobre color+textura) y compararlo contra un DummyClassifier de clase mayoritaria | Se pidió calcular macro-F1, precision/recall por clase y matriz de confusión en el conjunto de validación | Se ejecutó el entrenamiento sobre los datos reales del equipo; las métricas (macro-F1 ≈ 0.97) y la matriz de confusión se inspeccionaron directamente y se contrastaron con lo observado en el EDA visual (confusión esperable entre tizón temprano y tardío) |
| Claude (Anthropic) | Generar las figuras del EDA (balance de clases, muestras por clase, color por clase, nitidez, matriz de confusión, ejemplos de error) | Se pidió un script de visualización con `matplotlib`/`seaborn` | Cada figura se revisó visualmente antes de incluirla en el notebook; se confirmó que las muestras mostradas correspondían efectivamente a la clase indicada |



Todos los resultados numéricos y las figuras de este corte fueron generados
ejecutando el código sobre el dataset real descargado por el equipo. El
apoyo de IA generativa se usó para estructuración del notebook y
generación de partes de código.


## Corte 2

| Herramienta usada | Propósito | Interacción relevante | ¿Cómo se verificó o adaptó la respuesta? |
|---|---|---|---|
| Claude (Anthropic) | Interpretar la guía del Corte 2 y el cuaderno del miniproyecto de la Semana 2 | Se compartieron ambos documentos y se pidió un plan de entrega alineado con la rúbrica | El equipo revisó el plan contra la guía del banco de proyectos antes de ejecutar nada |
| Claude (Anthropic) | Detectar problemas de partición en el dataset | Se pidió revisar la carpeta del dataset; la IA detectó copias aumentadas (`..._original_...`) de la misma hoja repartidas entre train, val y test | Se confirmó con el índice completo (72 imágenes contaminadas en val, 82 en test) y se inspeccionaron nombres de archivo manualmente |
| Claude (Anthropic) | Configurar el entorno (PyTorch + CUDA en Windows, GTX 1650) | Instalación del entorno virtual, errores de política de ejecución, `WinError 1455` y pérdidas NaN con float16 | Cada solución se probó en el PC del equipo; se dejó la detección automática de GTX 16xx en el código |
| Claude (Anthropic) | Escribir el pipeline del benchmark (`src/c2_*.py`) | Se pidió un pipeline reproducible: índice con fuga, baselines, CNN desde cero y transfer learning, early stopping con validación limpia, augmentación solo en train, análisis estadístico | El código se probó con un dataset sintético y luego se ejecutó completo en el PC del equipo (≈2 h); las métricas se leen de los archivos generados |

Todo el entrenamiento se ejecutó en el PC del equipo con el dataset real. La IA no
tuvo acceso al conjunto de test para tomar decisiones (solo a sus nombres de archivo,
para documentar la fuga).


## Corte 3

| Herramienta usada | Propósito | Interacción relevante | ¿Cómo se verificó o adaptó la respuesta? |
|---|---|---|---|
| Claude (Anthropic, Claude Opus 5.5) | Mapear la guía del Corte 3 a tareas concretas y armar el cronograma | Se pidió listar qué exige la guía para el Proyecto 7 y qué faltaba según el avance del Corte 2 | El equipo contrastó la lista con la sección "Corte 3" y los criterios de aceptación de la guía |
| Claude (Anthropic) | Escribir la evaluación final en test (`src/c3_final_test.py`) | Se pidió evaluar **una sola vez** el modelo provisional del Corte 2, más la línea base y la alternativa, con IC 95 %, McNemar, calibración (ECE) y un umbral de abstención fijado antes de ver el test (0,70) | El script se niega a repetir la evaluación si ya existen resultados. Se ejecutó en el PC del equipo; los números de los documentos se copiaron de `reports/corte3/test_resumen.csv` y `test_results.json` |
| Claude (Anthropic) | Profundización: Grad-CAM y métrica de atención sobre la hoja (`src/c3_gradcam.py`) | Grad-CAM implementado con hooks de PyTorch; máscara de hoja por umbral de Otsu sobre saturación | Las máscaras se revisaron visualmente (`c3_03_mascaras.png`); se documentó que es una aproximación que a veces recorta parte de la hoja |
| Claude (Anthropic) | Pruebas de robustez (`src/c3_robustness.py`) | Perturbaciones de brillo, contraste, rotación, desenfoque, ruido, JPEG, zoom y reemplazo de fondo, más entradas fuera de dominio sintéticas | Se revisaron los ejemplos visuales (`c3_09`) para confirmar que cada perturbación hace lo que dice; el SVM se evaluó con la misma función de features del Corte 1 |
| Claude (Anthropic) | Prueba con fotos de campo (`src/c3_external.py`) | La IA propuso usar el split de test de PlantDoc (CC BY 4.0) y descargó 69 fotos de tomate | Al revisar una hoja de contactos con todas las fotos se encontraron 4 que son diapositivas o infografías; se recalculó sin ellas (36,7 % vs. 35,8 %) y se documentó. El equipo revisó la hoja de contactos y confirmó la exclusión |
| Claude (Anthropic) | Inferencia y demo (`src/predict.py`, `src/demo_app.py`) | Función de inferencia con top-3, Grad-CAM y advertencia "no concluyente"; demo en Gradio | Se probó con imágenes de test y de campo en el PC del equipo |
| Claude (Anthropic) | Borradores de `MODEL_CARD.md`, `DATASET_CARD.md`, README y presentación | Se pidió redactar con los resultados reales, incluidos los negativos (campo y fondo) | Cada cifra se verificó contra los CSV de `reports/corte3/`. El equipo revisó las conclusiones y las preparó para defenderlas con sus propias palabras |

**Decisiones humanas del Corte 3:** mantener ResNet-18 como modelo final, aunque MobileNetV3 empata en test, para
no reabrir con el test una decisión del Corte 2; reportar los resultados de campo aunque sean malos; y declarar
que el modelo no es apto para uso en cultivo.
