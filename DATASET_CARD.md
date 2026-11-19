# Ficha del dataset (simplificada) — Proyecto 7

## Identificación

| Campo | Valor |
|---|---|
| Nombre | Processed Tomato Leaf Disease Image Dataset |
| Fuente canónica | Mendeley Data — DOI [10.17632/3zwdw6y4pn.1](https://doi.org/10.17632/3zwdw6y4pn.1) |
| Origen | Derivado de PlantVillage (Hughes & Salathé, 2015): fotos de laboratorio de hojas aisladas |
| Licencia | CC BY 4.0 (uso permitido con atribución) |
| Tamaño | 6.557 imágenes RGB de 256×256 px, ~116 MB descomprimido |
| Tarea | Clasificación multiclase de imágenes (6 clases) |
| Datos personales | Ninguno |

## Clases y partición provista (70/15/15)

| Clase (carpeta) | Nombre en español | train | val | test |
|---|---|---:|---:|---:|
| Tomato___Bacterial_spot | Mancha bacteriana | 768 | 164 | 166 |
| Tomato___Early_blight | Tizón temprano | 762 | 163 | 164 |
| Tomato___Late_blight | Tizón tardío | 763 | 163 | 165 |
| Tomato___Leaf_Mold | Moho de la hoja | 760 | 163 | 164 |
| Tomato___Septoria_leaf_spot | Mancha por Septoria | 768 | 164 | 166 |
| Tomato___healthy | Sana | 765 | 164 | 165 |
| **Total** | | **4.586** | **981** | **990** |

Clases balanceadas (±1 %), por lo que *macro-F1* y *accuracy* casi coinciden; aun así se reporta
macro-F1 y F1 por clase para no ocultar clases difíciles.

## Hallazgos de calidad (Cortes 1 y 2)

- **Fuga de información en la partición del proveedor.** El dataset incluye copias aumentadas
  (`..._original_<archivo>_<uuid>.JPG`) de la misma hoja repartidas entre particiones:
  **72 imágenes de validación y 82 de test** tienen su hoja de origen en train (sobre todo tizón
  temprano y moho). No hay duplicados byte a byte. Todas las decisiones se tomaron con la
  **validación limpia (909)** y la evaluación final se reporta sobre el **test limpio (908)**, además del
  completo. Detección: `src/c2_common.py::build_index`; resumen en `reports/corte2/fuga_resumen.csv`.
- El preprocesamiento del proveedor eliminó imágenes borrosas y duplicadas; el EDA del Corte 1
  (nitidez por varianza del laplaciano) no encontró imágenes problemáticas adicionales.

## Límites de representación (lo que el dataset NO cubre)

- **Condiciones de laboratorio**: una hoja aislada, centrada, con fondo gris uniforme e iluminación
  controlada. No hay plantas completas, varias hojas, suelo, sombras, gotas ni manos.
- **Mundo cerrado**: solo 5 enfermedades + sana. Faltan, entre otras, virus del mosaico, virus del
  rizado amarillo, ácaros, deficiencias nutricionales, daño por insectos y mezclas de enfermedades.
- **Etapa y variedad**: no se documenta variedad de tomate, etapa de la enfermedad, región ni
  fecha. Posible correlación por sesión de captura entre particiones (misma planta/día).
- **Etiquetas**: provienen de PlantVillage (expertos), sin diagnóstico de laboratorio por imagen.

## Datos externos usados solo para evaluación (Corte 3)

- `data/externas/plantdoc_campo/`: 69 fotos de campo del split de test de **PlantDoc**
  (Singh et al., 2020; CC BY 4.0) — 53 de las 6 clases del proyecto y 16 de virus que el modelo no
  conoce. No se usaron para entrenar ni para elegir nada. Ver `data/externas/README.md`.

## Uso recomendado

Docencia e investigación sobre clasificación de imágenes y su validación. No usar el desempeño
sobre este dataset como evidencia de desempeño en cultivo sin una validación de campo.
