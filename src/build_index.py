"""Construye el índice del dataset (rutas + metadatos) y lo guarda en CSV."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import list_dataset

ROOT = Path("/home/claude/tomate_raw/inner/Dataset")
OUT = Path("/home/claude/proyecto7_tomate/reports/dataset_index.csv")

if __name__ == "__main__":
    df = list_dataset(ROOT)
    df.to_csv(OUT, index=False)
    print(df.shape)
    print(df.groupby(["split", "class"]).size())
