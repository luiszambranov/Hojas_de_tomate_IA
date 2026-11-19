# Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate

- **Corte 1:** EDA + protocolo de partición + baseline clásico + presentación.
- **Corte 2:** benchmark de 9 modelos (clásico, CNN desde cero, transfer learning), detección de fuga de información, análisis de errores y modelo provisional.
- **Corte 3 (final):** evaluación única en test reservado, profundización (Grad-CAM + robustez + prueba de campo), inferencia sobre imágenes nuevas, demo, [model card](MODEL_CARD.md) y [ficha del dataset](DATASET_CARD.md).

> **Resultado en una línea:** ResNet-18 con fine-tuning alcanza **macro-F1 0,995** en el test limpio
> (frente a 0,967 de la línea base SVM), pero con fotos reales de campo acierta solo **36 %**. Es un buen
> clasificador de *fotos de laboratorio*, **no un diagnóstico agronómico**.

## Integrantes

- Luis Alejandro Zambrano
- Jhoan David Santamaría
- Juan Norberto Pardo
- Edwin Santiago Sandoval

## Dataset

**Processed Tomato Leaf Disease Image Dataset** (Mendeley Data, derivado de
PlantVillage). Seis clases, dataset depurado y organizado en
train/validation/test 70/15/15. Licencia CC BY 4.0.

- Fuente: Mendeley Data — DOI [10.17632/3zwdw6y4pn.1](https://doi.org/10.17632/3zwdw6y4pn.1)

### Cómo obtener los datos

1. Descargue el dataset desde el DOI anterior (o use la copia ya descargada
   por el equipo).
2. Extraiga el contenido en `data/Dataset/` **o** deje la carpeta tal como viene de
   Mendeley (`Processed Tomato Leaf Disease Image Dataset with T/Dataset/`) en la raíz
   del repo; el código del Corte 2 busca ambas ubicaciones (o use la variable de
   entorno `TOMATE_DATA` con otra ruta):

```text
data/
└── Dataset/
    ├── train/
    │   ├── Tomato___Bacterial_spot/
    │   ├── Tomato___Early_blight/
    │   ├── Tomato___Late_blight/
    │   ├── Tomato___Leaf_Mold/
    │   ├── Tomato___Septoria_leaf_spot/
    │   └── Tomato___healthy/
    ├── val/
    │   └── ... (mismas 6 carpetas)
    └── test/
        └── ... (mismas 6 carpetas)
```

3. No se sube el dataset completo a este repositorio por su tamaño (~116 MB
   descomprimido, ~6.557 imágenes). Sí se incluyen en `reports/` los
   resultados ya calculados (`dataset_index.csv`, `dataset_quality.csv`,
   `features.npz`, `baseline_results.npz` y las figuras) para que el notebook
   pueda revisarse sin volver a descargar el dataset, aunque para una
   ejecución 100% desde cero se necesita el dataset en `data/Dataset/`.

## Estructura del repositorio

```text
Hojas_de_tomate_IA/
├── README.md
├── AI_USE_LOG.md                  # bitácora de uso de IA generativa (Cortes 1, 2 y 3)
├── MODEL_CARD.md                  # ficha del modelo: alcance, desempeño, límites y condiciones de uso
├── DATASET_CARD.md                # ficha del dataset: fuente, licencia, fuga y límites
├── requirements.txt
├── data/Dataset/                  # dataset (no versionado; ver instrucciones arriba)
├── data/externas/plantdoc_campo/  # 69 fotos de campo de PlantDoc (CC BY 4.0), solo evaluación
├── models/                        # pesos entrenados del Corte 2 (no versionados)
├── notebooks/
│   ├── Corte1_EDA_Baseline_Tomate.ipynb
│   ├── Corte2_Benchmark_Tomate.ipynb
│   └── Corte3_Final_Tomate.ipynb
├── src/
│   ├── data_utils.py, build_*.py, train_baseline.py, make_figures.py   # Corte 1
│   ├── c2_common.py               # Corte 2: rutas, índice con detección de fuga, métricas
│   ├── c2_build_index.py          # índice + diagnóstico de fuga
│   ├── c2_baseline.py             # Dummy + SVM HSV/LBP re-evaluados
│   ├── c2_train_cnn.py            # entrenamiento de CNN (desde cero / transfer learning)
│   ├── c2_run_benchmark.py        # ejecuta todo el benchmark en orden
│   ├── c2_analyze.py              # tablas, significancia y figuras
│   ├── c3_common.py               # Corte 3: carga de modelos, Grad-CAM, máscara de hoja
│   ├── c3_final_test.py           # evaluación única en test (IC 95 %, McNemar, ECE, abstención)
│   ├── c3_gradcam.py              # interpretabilidad: ¿mira la hoja o el fondo?
│   ├── c3_robustness.py           # perturbaciones de captura + fuera de dominio
│   ├── c3_external.py             # fotos de campo (PlantDoc)
│   ├── c3_run_all.py              # ejecuta todo el Corte 3 en orden
│   ├── predict.py                 # INFERENCIA sobre imágenes nuevas (terminal o Python)
│   └── demo_app.py                # demo web con Gradio
└── reports/
    ├── ...                        # resultados del Corte 1
    ├── corte2/                    # benchmark.csv, significancia.json, runs/, figures/
    └── corte3/                    # test_resumen.csv, robustez.csv, gradcam_*.csv, externas_*.csv, figures/
```

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/Mac: source .venv/bin/activate
python -m pip install --upgrade pip
# PyTorch con CUDA (GPU NVIDIA). Sin GPU, omita --index-url y se instalará la versión CPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Cómo ejecutar

**Corte 1:** abrir `notebooks/Corte1_EDA_Baseline_Tomate.ipynb` y ejecutar en orden.

**Corte 2:** desde la raíz del repo, con el entorno activo:

```powershell
python src/c2_run_benchmark.py --workers 2   # ≈1,5-2 h en una GTX 1650
jupyter notebook notebooks/Corte2_Benchmark_Tomate.ipynb
```

**Corte 3:**

```powershell
python src/c3_run_all.py                      # ≈15-20 min en una GTX 1650 (test único + Grad-CAM + robustez + campo)
jupyter notebook notebooks/Corte3_Final_Tomate.ipynb
```

**Inferencia sobre imágenes nuevas:**

```powershell
python src/predict.py ruta\a\hoja.jpg
python src/predict.py data\externas\plantdoc_campo --gradcam reports\corte3\inferencia
pip install gradio; python src/demo_app.py   # demo web en http://127.0.0.1:7860
```

`predict.py` devuelve la clase, la confianza, el top-3 y el mapa Grad-CAM. Si la confianza es menor que 0,70,
marca el resultado como **no concluyente**. `c3_final_test.py` se niega a repetir la evaluación del test si ya
existe (el test es de un solo uso).

El benchmark guarda cada modelo en `reports/corte2/runs/<modelo>/`; si se interrumpe,
al relanzarlo se salta lo ya hecho. El cuaderno del Corte 2 solo lee esos resultados
(no reentrena). Notas para Windows: las GTX 16xx se entrenan en float32 automáticamente
(float16 da NaN), y con muchos `--workers` puede aparecer `WinError 1455`
(archivo de paginación); use 2.

## Resultados

### Corte 1
- Dataset balanceado (~765/164/165 imágenes por clase en train/val/test).
- Baseline HSV + LBP + SVM RBF: macro-F1 ≈ 0.97 en validación. Principal
  confusión: tizón temprano vs. tardío.

### Corte 2
- **Fuga de información en el dataset original:** 72 imágenes de validación y 82 de
  test son copias aumentadas de hojas que están en train. Todas las decisiones se
  toman con la **validación limpia** (909 imágenes). Test no se usó.
- **Benchmark (macro-F1, validación limpia):** Dummy 0.051 · EfficientNet-B0 congelado
  0.923 · MobileNetV3 congelado 0.963 · SVM 0.965 · CNN pequeña 0.978 · EfficientNet-B0
  fine-tune 0.997/0.998 · MobileNetV3 fine-tune 0.998 · **ResNet-18 fine-tune 0.999**.
- **Modelo provisional: ResNet-18 con fine-tuning.** 1 error en 909 imágenes frente a
  32 del SVM (corrige los 14 errores tizón temprano↔tardío; McNemar p ≈ 9·10⁻¹⁰).
  Empata estadísticamente con MobileNetV3 y EfficientNet-B0 fine-tune; se prefiere por
  menor latencia en GPU (4,2 ms), cero falsos negativos de enfermedad y error único de
  baja confianza.
- **Límites:** imágenes de laboratorio (hoja aislada, fondo uniforme), posible
  correlación por sesión de captura entre particiones y clasificador de mundo cerrado
  (6 clases). Estos límites se midieron en el Corte 3.

### Corte 3 (final)
Detalle completo en [`MODEL_CARD.md`](MODEL_CARD.md).

| Modelo | macro-F1 val. limpia (C2) | **macro-F1 test limpio (C3)** | IC 95 % | Errores / 908 |
|---|---:|---:|---|---:|
| Dummy | 0,051 | 0,052 | 0,046–0,057 | 742 |
| SVM HSV+LBP (línea base C1) | 0,965 | 0,967 | 0,956–0,978 | 30 |
| **ResNet-18 FT (provisional C2 = final)** | 0,999 | **0,995** | 0,991–0,999 | **4** |
| MobileNetV3 FT (alternativa) | 0,998 | 0,995 | 0,990–0,999 | 4 |

- **Test:** se evaluó una sola vez. ResNet-18 supera al SVM (McNemar p = 6·10⁻⁶) y empata con MobileNetV3
  (p = 1,0). Los 4 errores son 3 de tizón tardío y 1 de Septoria; ninguna hoja enferma se predijo como sana.
  ECE = 0,005.
- **Grad-CAM:** el 60 % de la atención cae sobre la hoja, que ocupa el 40 % de la imagen, y en el 79 % de
  las imágenes la atención se concentra en la hoja. Aun así, al cambiar el fondo de laboratorio el macro-F1
  cae a 0,51–0,61: el modelo también usa el contexto.
- **Robustez:** el modelo es estable ante brillo (×0,4–1,6), contraste, rotación y JPEG ≥ 20, mientras que el
  SVM colapsa. Es frágil ante desenfoque (r = 2 px → 0,60) y ruido (σ = 25 → 0,53).
- **Campo (PlantDoc, 53 fotos):** accuracy de 36 % con 93 % de confianza media. Colapsa hacia tizón
  temprano y tardío, y no reconoce ninguna hoja sana. De 16 fotos de virus desconocidos, 15 recibieron
  confianza ≥ 0,70.
- **Conclusión:** la solución es reproducible y está bien validada *dentro de PlantVillage*. No es apta como
  diagnóstico en cultivo sin datos y validación de campo. Ver condiciones de uso en la model card.

## Fuentes

- Processed Tomato Leaf Disease Image Dataset — Mendeley Data, DOI 10.17632/3zwdw6y4pn.1 (derivado de
  PlantVillage: Hughes & Salathé, 2015, arXiv:1511.08060). CC BY 4.0.
- PlantDoc — Singh et al., 2020, *PlantDoc: A Dataset for Visual Plant Disease Detection*, CODS-COMAD.
  https://github.com/pratikkayal/PlantDoc-Dataset. CC BY 4.0.
- Grad-CAM — Selvaraju et al., 2017, ICCV. ResNet — He et al., 2016. MobileNetV3 — Howard et al., 2019.

## Licencia de los datos

Dataset distribuido bajo licencia CC BY 4.0 por Mendeley Data, derivado de
PlantVillage. No contiene datos personales.
