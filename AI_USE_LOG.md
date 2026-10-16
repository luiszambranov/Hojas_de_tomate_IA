# Bitácora de uso de IA generativa

Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate. Cortes 1 y 2.

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
| Claude (Anthropic) | Redactar la estructura (ficha del dataset, diagnóstico de calidad, límites de representación) siguiendo el formato usado en el curso | Se compartió el notebook de ejemplo de la Semana 1 | El equipo redacto cada sección acorde con los resultados numéricos reales obtenidos (conteos, macro-F1, matriz de confusión)|


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
