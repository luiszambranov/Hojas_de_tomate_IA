"""
Utilidades del Corte 3 — Proyecto 7 (hojas de tomate).

- Carga del modelo final (ResNet-18 fine-tune del Corte 2) y de la alternativa
  (MobileNetV3 fine-tune) a partir de los pesos en models/.
- Transformación de evaluación idéntica a la del Corte 2 (resize 224 + normalización ImageNet).
- Grad-CAM implementado a mano (hooks de PyTorch), sin dependencias extra.
- Máscara aproximada de la hoja (umbral de Otsu sobre saturación + morfología)
  para medir si el mapa de atención cae sobre la hoja o sobre el fondo.

REGLA DEL CORTE 3: el test se evalúa UNA sola vez con el modelo ya elegido en el
Corte 2. Ninguna decisión (modelo, umbral, hiperparámetro) se toma mirando el test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from c2_common import CLASS_ES, CLASSES, MODELS_DIR, REPO, SHORT  # noqa: F401
from c2_train_cnn import MEAN, STD, build_model

REPORTS3 = REPO / "reports" / "corte3"
FIG3 = REPORTS3 / "figures"

# Modelo final = modelo provisional del Corte 2 (elegido SOLO con validación limpia).
FINAL = "resnet18_finetune_lr0.0003"
ALT = "mobilenet_v3_large_finetune_lr0.0003"   # alternativa liviana (celular)
ARCH = {FINAL: "resnet18", ALT: "mobilenet_v3_large"}
NICE3 = {
    "dummy": "Dummy (mayoritaria)",
    "svm_hsv_lbp": "SVM HSV+LBP (baseline C1)",
    FINAL: "ResNet-18 FT (modelo final)",
    ALT: "MobileNetV3 FT (alternativa)",
}

# Umbral de "resultado no concluyente". Se fija A PRIORI (no se ajusta con test):
# en validación limpia el único error del modelo tuvo confianza 0,54.
CONF_THRESHOLD = 0.70

IMG_SIZE = 224
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Paleta (la misma del Corte 2)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, RED = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9", "#0b0b0b", "#52514e", "#d03b3b"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn(run: str = FINAL, dev: torch.device | None = None):
    dev = dev or device()
    ckpt = MODELS_DIR / f"{run}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No existe {ckpt}. Entrene primero con: python src/c2_run_benchmark.py")
    model, _ = build_model(ARCH[run], len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    return model.to(dev).eval()


def target_layer(model, run: str):
    """Última capa convolucional (donde se calcula Grad-CAM)."""
    if ARCH[run] == "resnet18":
        return model.layer4[-1]
    return model.features[-1]


@torch.no_grad()
def predict_pils(model, pils, dev=None, batch: int = 64) -> np.ndarray:
    dev = dev or device()
    out = []
    for i in range(0, len(pils), batch):
        x = torch.stack([EVAL_TF(p) for p in pils[i:i + batch]]).to(dev)
        out.append(model(x).float().softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def open_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017). Uso:
        cam = GradCAM(model, target_layer(model, run))
        heat, probs = cam(pil)            # heat: HxW en [0,1] (tamaño 224x224)
    """

    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()
        if o.requires_grad:   # hook sobre el tensor (evita problemas con ReLU inplace)
            o.register_hook(self._save_grad)

    def _save_grad(self, g):
        self.grads = g.detach()

    def __call__(self, pil: Image.Image, class_idx: int | None = None):
        dev = next(self.model.parameters()).device
        x = EVAL_TF(pil).unsqueeze(0).to(dev).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)[0].detach().cpu().numpy()
        c = int(probs.argmax()) if class_idx is None else class_idx
        logits[0, c].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs


def overlay(pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32) / 255
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


# --------------------------------------------------------------------------
# Máscara de hoja (aproximada)
# --------------------------------------------------------------------------
def leaf_mask(pil: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Máscara booleana aproximada de la hoja: en PlantVillage el fondo es gris/poco
    saturado, así que Otsu sobre el canal S (HSV) separa hoja y fondo. Se cierra con
    morfología y se conserva el componente conectado más grande con huecos rellenos.
    Es una aproximación: se revisa visualmente en la figura c3_03."""
    rgb = np.asarray(pil.resize((size, size)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8) * 255
    # rellenar huecos (lesiones oscuras dentro de la hoja)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def cam_on_leaf(cam: np.ndarray, mask: np.ndarray) -> float:
    """Fracción de la 'energía' del mapa Grad-CAM que cae dentro de la hoja."""
    return float(cam[mask].sum() / (cam.sum() + 1e-8))


# --------------------------------------------------------------------------
# Baseline SVM (Corte 1) sobre imágenes en memoria
# --------------------------------------------------------------------------
def features_from_pil(pil: Image.Image, size: int = 128) -> np.ndarray:
    """Mismo vector HSV(48)+LBP(10) que data_utils.extract_features, pero a partir de
    una imagen en memoria (para evaluar el SVM con imágenes perturbadas)."""
    from skimage.feature import local_binary_pattern

    img = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = [cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten(),
            cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten(),
            cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()]
    color_hist = np.concatenate(hist)
    color_hist = color_hist / (color_hist.sum() + 1e-8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    if len(lbp_hist) < 10:   # imágenes degeneradas (p. ej. color sólido)
        lbp_hist = np.pad(lbp_hist, (0, 10 - len(lbp_hist)))
    return np.concatenate([color_hist, lbp_hist]).astype(np.float32)


def fit_svm():
    """Reentrena el SVM baseline exactamente como en los Cortes 1-2 (solo con train)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from c2_common import REPORTS, SEED, load_index

    idx = load_index()
    cache = np.load(REPORTS / "features_hsv_lbp.npz", allow_pickle=True)
    pos = {rp: i for i, rp in enumerate(cache["rel_path"])}
    dtr = idx[idx.split == "train"]
    X = cache["X"][[pos[r] for r in dtr.rel_path]]
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                                              probability=False, random_state=SEED))
    return svm.fit(X, dtr.label.to_numpy())
