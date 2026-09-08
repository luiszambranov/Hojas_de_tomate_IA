# Proyecto 7 — Diagnóstico visual de enfermedades en hojas de tomate

Corte 1: EDA + protocolo de partición + baseline clásico + presentación.

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
2. Extraiga el contenido de forma que quede así:

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
proyecto7-tomate/
├── README.md
├── AI_USE_LOG.md              # bitácora de uso de IA generativa
├── requirements.txt
├── data/
│   └── Dataset/                # dataset (no versionado; ver instrucciones arriba)
├── notebooks/
│   └── Corte1_EDA_Baseline_Tomate.ipynb
├── src/
│   ├── data_utils.py           # funciones reutilizables (índice, features, calidad)
│   ├── build_index.py          # construye reports/dataset_index.csv
│   ├── build_features.py       # construye reports/dataset_quality.csv y features.npz
│   ├── train_baseline.py       # entrena y evalúa el baseline clásico
│   └── make_figures.py         # genera las figuras de reports/figures/
└── reports/
    ├── dataset_index.csv
    ├── dataset_quality.csv
    ├── features.npz
    ├── baseline_results.npz
    └── figures/
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cómo ejecutar

Con el dataset en `data/Dataset/` (ver arriba), desde la raíz del proyecto:

```bash
jupyter notebook notebooks/Corte1_EDA_Baseline_Tomate.ipynb
```

Ejecute las celdas en orden. El notebook reutiliza los resultados cacheados en
`reports/` si existen (más rápido); si no existen, los recalcula desde las
imágenes.

Alternativamente, los scripts de `src/` pueden ejecutarse por separado en este
orden: `build_index.py` → `build_features.py` → `train_baseline.py` →
`make_figures.py`.

## Resultados del Corte 1 (resumen)

- Dataset balanceado: ~768/164/166 imágenes por clase en train/val/test
  respectivamente, sin clases vacías.
- Baseline clásico (histograma de color HSV + textura LBP + SVM RBF):
  **macro-F1 ≈ 0.97** en validación, muy por encima del baseline ingenuo de
  clase mayoritaria (macro-F1 ≈ 0.05).
- Principal confusión: **tizón temprano vs. tizón tardío** (enfermedades
  visualmente similares), consistente con lo observado en el EDA de color.
- Límite documentado: el dataset es de origen controlado (fondo uniforme,
  hoja aislada); el buen desempeño aquí no garantiza un diagnóstico confiable
  en campo real. Detalle completo en el notebook, sección 8.

## Licencia de los datos

Dataset distribuido bajo licencia CC BY 4.0 por Mendeley Data, derivado de
PlantVillage. No contiene datos personales.
