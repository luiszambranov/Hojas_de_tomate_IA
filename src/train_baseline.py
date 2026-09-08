"""Entrena el baseline clásico (features de color+textura + SVM lineal)
y reporta métricas macro-F1, precision/recall por clase y matriz de confusión.
"""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA = Path("/home/claude/proyecto7_tomate/reports/features.npz")

if __name__ == "__main__":
    d = np.load(DATA, allow_pickle=True)
    X, split, y, path = d["X"], d["split"], d["y"], d["path"]

    Xtr, ytr = X[split == "train"], y[split == "train"]
    Xval, yval = X[split == "val"], y[split == "val"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)

    # Línea base ingenua: clase mayoritaria
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    dummy_pred = dummy.predict(Xval)
    print("=== Dummy (clase mayoritaria) ===")
    print("Macro-F1:", f1_score(yval, dummy_pred, average="macro"))

    # Baseline clásico: SVM (RBF) sobre color+textura
    clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xval_s)

    print("\n=== Baseline clásico (color HSV + LBP + SVM RBF) ===")
    print("Macro-F1:", f1_score(yval, pred, average="macro"))
    print(classification_report(yval, pred, digits=3))

    labels = sorted(set(y))
    cm = confusion_matrix(yval, pred, labels=labels)
    print("Labels order:", labels)
    print(cm)

    np.savez(
        "/home/claude/proyecto7_tomate/reports/baseline_results.npz",
        cm=cm, labels=labels, yval=yval, pred=pred, path_val=path[split == "val"],
    )
