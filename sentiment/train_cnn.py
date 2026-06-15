"""Train the FER2013 CNN (optional deep-learning baseline).

1. Install extras:   .venv/bin/python -m pip install -r requirements-ml.txt
2. Get FER2013:      download `fer2013.csv` (Kaggle "FER-2013") to ./data/
3. Train:            python -m sentiment.train_cnn --epochs 30

Saves weights to models/emotion_cnn.pt. Use this to quote a CNN-vs-SVM accuracy
comparison in your report — same task, two very different architectures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sentiment.cnn_fer import WEIGHTS_PATH, EmotionCNN

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "fer2013.csv"


def _load_split(usage: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    df = df[df["Usage"].isin(usage)]
    X = np.stack([np.fromstring(p, sep=" ", dtype=np.float32) for p in df["pixels"]])
    X = (X / 255.0).reshape(-1, 1, 48, 48)
    y = df["emotion"].to_numpy(dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}. Download FER2013 'fer2013.csv' into ./data.")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    Xtr, ytr = _load_split({"Training"})
    Xva, yva = _load_split({"PublicTest", "PrivateTest"})
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256)

    model = EmotionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        sched.step()
        acc = _accuracy(model, val_loader, device)
        print(f"epoch {epoch:3d}/{args.epochs}  val_acc {acc:.3f}")
        if acc > best:
            best = acc
            WEIGHTS_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"\nBest val accuracy: {best:.3f}  -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
