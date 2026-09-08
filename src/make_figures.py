"""Genera las figuras usadas en el notebook y en el reporte (Corte 1)."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import CLASS_LABELS_ES, CLASSES

REPORTS = Path("/home/claude/proyecto7_tomate/reports")
FIGDIR = REPORTS / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = sns.color_palette("Set2", n_colors=6)

meta = pd.read_csv(REPORTS / "dataset_quality.csv")
meta["class_es"] = meta["class"].map(CLASS_LABELS_ES)

# 1. Balance por clase y por split -----------------------------------------
counts = meta.groupby(["split", "class_es"]).size().reset_index(name="n")
order = [CLASS_LABELS_ES[c] for c in CLASSES]
splits_order = ["train", "val", "test"]

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(
    data=counts, x="class_es", y="n", hue="split", order=order,
    hue_order=splits_order, palette="Blues_d", ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Número de imágenes")
ax.set_title("Balance de clases por partición (train/val/test)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGDIR / "01_balance_clases.png", dpi=140)
plt.close()

# 2. Grid de muestras por clase ---------------------------------------------
fig, axes = plt.subplots(6, 5, figsize=(12, 14))
rng = np.random.RandomState(7)
for i, cls in enumerate(CLASSES):
    sub = meta[(meta["class"] == cls) & (meta["split"] == "train")]
    sample_paths = sub["path"].sample(5, random_state=7).tolist()
    for j, p in enumerate(sample_paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(img)
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_ylabel(CLASS_LABELS_ES[cls], fontsize=10)
    axes[i, 0].axis("on")
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_ylabel(CLASS_LABELS_ES[cls], fontsize=10)
fig.suptitle("Muestras de entrenamiento por clase", y=1.0)
plt.tight_layout()
plt.savefig(FIGDIR / "02_muestras_por_clase.png", dpi=130)
plt.close()

# 3. Distribución de tono (hue) y saturación por clase -----------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=meta, x="class_es", y="mean_hue", order=order, palette=PALETTE, ax=axes[0])
axes[0].set_title("Tono (hue) medio por clase")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=meta, x="class_es", y="mean_saturation", order=order, palette=PALETTE, ax=axes[1])
axes[1].set_title("Saturación media por clase")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIGDIR / "03_color_por_clase.png", dpi=140)
plt.close()

# 4. Nitidez (detección de borrosas) -----------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(meta["sharpness"], bins=60, ax=ax, color=PALETTE[0])
ax.axvline(meta["sharpness"].quantile(0.02), color="red", linestyle="--",
           label="percentil 2% (posible borrosidad)")
ax.set_title("Distribución de nitidez (varianza del laplaciano)")
ax.set_xlabel("Nitidez (a mayor valor, más nítida)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "04_nitidez.png", dpi=140)
plt.close()

# 5. Matriz de confusión del baseline -----------------------------------------
res = np.load(REPORTS / "baseline_results.npz", allow_pickle=True)
cm, labels = res["cm"], res["labels"]
labels_es = [CLASS_LABELS_ES[l] for l in labels]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_es,
            yticklabels=labels_es, ax=ax, cbar=False)
ax.set_xlabel("Predicción")
ax.set_ylabel("Clase real")
ax.set_title("Matriz de confusión - baseline clásico (validación)")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGDIR / "05_matriz_confusion.png", dpi=140)
plt.close()

# 6. Ejemplos de error --------------------------------------------------------
yval, pred, path_val = res["yval"], res["pred"], res["path_val"]
err_idx = np.where(yval != pred)[0]
rng.shuffle(err_idx)
err_idx = err_idx[:8]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, idx in zip(axes.flat, err_idx):
    img = cv2.cvtColor(cv2.imread(path_val[idx]), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"real: {CLASS_LABELS_ES[yval[idx]]}\npred: {CLASS_LABELS_ES[pred[idx]]}", fontsize=9)
fig.suptitle("Ejemplos de errores del baseline (validación)")
plt.tight_layout()
plt.savefig(FIGDIR / "06_ejemplos_error.png", dpi=130)
plt.close()

print("figuras generadas en", FIGDIR)
print(list(FIGDIR.iterdir()))
