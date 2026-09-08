"""Extrae sharpness (nitidez) y features clásicas (color+textura) para
todo el índice del dataset, y las guarda en disco para uso del notebook."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import extract_features, sharpness_score, mean_hue_saturation

IDX = Path("/home/claude/proyecto7_tomate/reports/dataset_index.csv")
OUT_META = Path("/home/claude/proyecto7_tomate/reports/dataset_quality.csv")
OUT_FEATS = Path("/home/claude/proyecto7_tomate/reports/features.npz")

if __name__ == "__main__":
    df = pd.read_csv(IDX)
    sharpness = np.zeros(len(df))
    hue = np.zeros(len(df))
    sat = np.zeros(len(df))
    feats = np.zeros((len(df), 64), dtype=np.float32)  # 48 color + 16(ish) lbp -> resize later

    feat_list = []
    for i, row in enumerate(df.itertuples()):
        img = cv2.imread(row.path)
        if img is None:
            sharpness[i] = np.nan
            hue[i] = np.nan
            sat[i] = np.nan
            feat_list.append(None)
            continue
        sharpness[i] = sharpness_score(img)
        h, s = mean_hue_saturation(img)
        hue[i] = h
        sat[i] = s
        feat_list.append(extract_features(row.path))
        if i % 1000 == 0:
            print(f"{i}/{len(df)}")

    df["sharpness"] = sharpness
    df["mean_hue"] = hue
    df["mean_saturation"] = sat
    df.to_csv(OUT_META, index=False)

    feat_dim = len(feat_list[0])
    X = np.zeros((len(df), feat_dim), dtype=np.float32)
    for i, f in enumerate(feat_list):
        if f is not None:
            X[i] = f
    np.savez_compressed(OUT_FEATS, X=X, split=df["split"].values, y=df["class"].values, path=df["path"].values)
    print("done", X.shape)
