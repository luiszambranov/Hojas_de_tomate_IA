# Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate

- **Corte 1:** EDA + protocolo de partición + baseline clásico + presentación.
- **Corte 2:** benchmark de 9 modelos (clásico, CNN desde cero, transfer learning), detección de fuga de información, análisis de errores y modelo provisional.

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
├── AI_USE_LOG.md                  # bitácora de uso de IA generativa (Cortes 1 y 2)
├── requirements.txt
├── data/Dataset/                  # dataset (no versionado; ver instrucciones arriba)
├── models/                        # pesos entrenados del Corte 2 (no versionados)
├── notebooks/
│   ├── Corte1_EDA_Baseline_Tomate.ipynb
│   └── Corte2_Benchmark_Tomate.ipynb
├── src/
│   ├── data_utils.py, build_*.py, train_baseline.py, make_figures.py   # Corte 1
│   ├── c2_common.py               # Corte 2: rutas, índice con detección de fuga, métricas
│   ├── c2_build_index.py          # índice + diagnóstico de fuga
│   ├── c2_baseline.py             # Dummy + SVM HSV/LBP re-evaluados
│   ├── c2_train_cnn.py            # entrenamiento de CNN (desde cero / transfer learning)
│   ├── c2_run_benchmark.py        # ejecuta todo el benchmark en orden
│   └── c2_analyze.py              # tablas, significancia y figuras
└── reports/
    ├── ...                        # resultados del Corte 1
    └── corte2/                    # benchmark.csv, significancia.json, runs/, figures/
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
  (6 clases). Pendiente para el Corte 3: test, Grad-CAM y prueba con fotos externas.

## Licencia de los datos

Dataset distribuido bajo licencia CC BY 4.0 por Mendeley Data, derivado de
PlantVillage. No contiene datos personales.
