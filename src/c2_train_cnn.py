"""Paso 3 — Entrena una CNN (desde cero o transfer learning) con el protocolo del Corte 2.

Protocolo (igual para todos los modelos):
  - Entrena con train; selecciona la mejor época por macro-F1 en VALIDACIÓN LIMPIA
    (sin imágenes cuya hoja de origen está en train). Test no se usa.
  - Augmentación SOLO en train (recortes, volteos, rotación, brillo/contraste).
    Validación: solo redimensionar y normalizar.
  - Early stopping (paciencia configurable), semilla fija, AMP en GPU.

Ejemplos:
  python src/c2_train_cnn.py --model smallcnn --epochs 30 --lr 1e-3
  python src/c2_train_cnn.py --model efficientnet_b0 --mode frozen --epochs 10 --lr 1e-3
  python src/c2_train_cnn.py --model efficientnet_b0 --mode finetune --epochs 15 --lr 3e-4

Salidas:
  reports/corte2/runs/<nombre>/{metrics.json, history.csv, preds_val.csv}
  models/<nombre>.pt   (pesos; no se versionan en git)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from c2_common import (CLASSES, MODELS_DIR, REPORTS, SEED, dataset_root, evaluate_split, load_index,
                       metrics_block, save_json, set_seed)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, tf):
        self.paths = [str(root / p) for p in df.rel_path]
        self.labels = df.label.to_numpy()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def make_transforms(size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """CNN pequeña desde cero: 4 bloques conv-BN-ReLU-pool + GAP + dropout + lineal."""

    def __init__(self, n_classes: int = 6):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, n_classes: int, pretrained: bool):
    """Devuelve (modelo, módulo_cabeza). La cabeza es lo único que se entrena en modo 'frozen'."""
    if name == "smallcnn":
        m = SmallCNN(n_classes)
        return m, m.head
    w = "DEFAULT" if pretrained else None
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m, m.classifier
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m, m.classifier
    if name == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, m.fc
    raise ValueError(f"Modelo desconocido: {name}")


def set_frozen_train_mode(model: nn.Module, head: nn.Module):
    """En modo 'frozen' el backbone queda en eval (BatchNorm no actualiza estadísticas)."""
    model.eval()
    head.train()


# --------------------------------------------------------------------------
# Entrenamiento / evaluación
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, amp: bool):
    model.eval()
    probs, losses = [], []
    crit = nn.CrossEntropyLoss(reduction="sum")
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
        out = out.float()
        losses.append(crit(out, y).item())
        n += len(y)
        probs.append(out.softmax(1).cpu())
    return torch.cat(probs).numpy(), sum(losses) / max(n, 1)


@torch.no_grad()
def latency_ms(model, ds, device, n: int = 50) -> float:
    """Latencia por imagen con batch 1 (escenario de uso: una foto a la vez)."""
    model.eval()
    xs = [ds[i][0].unsqueeze(0) for i in range(min(n + 5, len(ds)))]
    for x in xs[:5]:
        model(x.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in xs[5:]:
        model(x.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / max(len(xs) - 5, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["smallcnn", "mobilenet_v3_large", "efficientnet_b0", "resnet18"])
    ap.add_argument("--mode", default="finetune", choices=["frozen", "finetune"],
                    help="frozen: solo se entrena la cabeza; finetune: toda la red (smallcnn siempre entrena todo)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default=None, help="nombre de la corrida (por defecto se arma con modelo/modo/lr)")
    ap.add_argument("--no-amp", action="store_true", help="desactivar precisión mixta (si aparecen pérdidas NaN)")
    ap.add_argument("--no-pretrained", action="store_true", help="no descargar pesos de ImageNet (solo para pruebas)")
    ap.add_argument("--limit", type=int, default=0, help="usar solo N imágenes por partición (solo para pruebas)")
    args = ap.parse_args()

    set_seed(SEED)
    if args.model == "smallcnn":
        args.mode = "scratch"
    name = args.name or f"{args.model}_{args.mode}_lr{args.lr:g}"
    out = REPORTS / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    if amp and "GTX 16" in torch.cuda.get_device_name(0):
        # Las GTX 16xx (Turing sin tensor cores) dan NaN con float16 en varias operaciones: se usa float32.
        print("GTX 16xx detectada: se desactiva la precisión mixta (float16) y se entrena en float32.")
        amp = False
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # elige los kernels más rápidos para tamaño fijo de imagen
    print(f"== {name} | dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
          + f" | precisión: {'mixta (float16)' if amp else 'float32'}")

    root = dataset_root()
    idx = load_index()
    dtr = idx[idx.split == "train"].reset_index(drop=True)
    dval = idx[idx.split == "val"].reset_index(drop=True)          # test NO se carga
    if args.limit:
        dtr = dtr.groupby("label").head(args.limit // len(CLASSES)).reset_index(drop=True)
        dval = dval.groupby("label").head(max(args.limit // len(CLASSES) // 2, 2)).reset_index(drop=True)

    train_tf, eval_tf = make_transforms(args.img_size)
    pin = device.type == "cuda"
    kw = dict(num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    dl_train = DataLoader(LeafDataset(dtr, root, train_tf), batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    dl_val = DataLoader(LeafDataset(dval, root, eval_tf), batch_size=64, shuffle=False, **kw)
    ds_train_eval = LeafDataset(dtr, root, eval_tf)  # train sin augmentación para medir la brecha train/val

    model, head = build_model(args.model, len(CLASSES), pretrained=not args.no_pretrained)
    if args.mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"Parámetros: {n_params/1e6:.2f} M totales, {n_train/1e6:.3f} M entrenables")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    crit = nn.CrossEntropyLoss()

    hist, best, best_ep, bad = [], -1.0, -1, 0
    ckpt = MODELS_DIR / f"{name}.pt"
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        if args.mode == "frozen":
            set_frozen_train_mode(model, head)
        else:
            model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = crit(logits, y)
            if not math.isfinite(loss.item()):
                raise SystemExit("La pérdida dio NaN/inf. Algunas GTX 16xx fallan con float16: "
                                 "repita la corrida agregando --no-amp.")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(y)
        sched.step()

        probs, vloss = predict(model, dl_val, device, amp)
        pred = probs.argmax(1)
        ev = evaluate_split(dval, pred)
        f1c = ev["clean"]["macro_f1"]
        row = dict(epoch=ep, train_loss=tl / tn, train_acc_aug=tc / tn, val_loss=vloss,
                   val_f1_full=ev["full"]["macro_f1"], val_f1_clean=f1c,
                   lr=opt.param_groups[0]["lr"], secs=time.perf_counter() - t0)
        hist.append(row)
        flag = ""
        if f1c > best + 1e-5:
            best, best_ep, bad = f1c, ep, 0
            torch.save(model.state_dict(), ckpt)
            flag = "  *mejor"
        else:
            bad += 1
        print(f"ep {ep:02d} | loss {row['train_loss']:.4f} acc(aug) {row['train_acc_aug']:.3f} | "
              f"val loss {vloss:.4f} F1 completo {row['val_f1_full']:.4f} limpio {f1c:.4f} | {row['secs']:.0f}s{flag}")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= args.patience:
            print(f"Early stopping: {args.patience} épocas sin mejorar.")
            break
    train_time = time.perf_counter() - t_start

    # ---- Evaluación final con la mejor época
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    probs_val, _ = predict(model, dl_val, device, amp)
    dl_tr_eval = DataLoader(ds_train_eval, batch_size=64, shuffle=False, **kw)
    probs_tr, _ = predict(model, dl_tr_eval, device, amp)
    lat = latency_ms(model, LeafDataset(dval, root, eval_tf), device)

    m = {
        "name": name,
        "family": "cnn_scratch" if args.model == "smallcnn" else f"transfer_{args.mode}",
        "description": f"{args.model} ({args.mode}), lr={args.lr:g}, img={args.img_size}",
        "config": vars(args),
        "params": int(n_params),
        "trainable_params": int(n_train),
        "size_mb": round(n_params * 4 / 1e6, 2),
        "best_epoch": best_ep,
        "epochs_run": len(hist),
        "train_time_s": round(train_time, 1),
        "infer_ms_per_img": round(lat, 3),
        "infer_note": f"batch 1 en {device.type}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "train": metrics_block(dtr.label, probs_tr.argmax(1)),
        "val": evaluate_split(dval, probs_val.argmax(1)),
    }
    save_json(m, out / "metrics.json")
    p = dval[["rel_path", "class", "label", "clean", "is_aug"]].copy()
    p["pred"] = probs_val.argmax(1)
    for i in range(len(CLASSES)):
        p[f"prob_{i}"] = probs_val[:, i]
    p.to_csv(out / "preds_val.csv", index=False)
    print(f"\nMejor época {best_ep} | train F1 {m['train']['macro_f1']:.4f} | "
          f"val F1 completo {m['val']['full']['macro_f1']:.4f} limpio {m['val']['clean']['macro_f1']:.4f} | "
          f"{train_time/60:.1f} min | {lat:.1f} ms/img")


if __name__ == "__main__":
    main()
