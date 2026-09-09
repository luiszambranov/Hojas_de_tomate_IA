# Bitácora de uso de IA generativa

Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate. Corte 1.

El uso de IA generativa está permitido como apoyo, siempre que sea
transparente, verificado y adaptado por el equipo. Esta bitácora documenta
dónde se usó, con qué propósito y cómo se verificó cada resultado.

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
