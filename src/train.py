"""Train and evaluate one architecture under the common protocol.

    python -m src.train --model cnn
    python -m src.train --model cnn --permute-features   # feature-order ablation

Writes results/<run>/metrics.json, history.csv, test predictions, and a checkpoint.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader, TensorDataset

from src import config as C
from src.data import load_meta, load_split
from src.models import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, n_classes):
    # Inverse-sqrt frequency: up-weights rare attacks without letting a class
    # with ~1k samples dominate the loss.
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), C.EVAL_BATCH_SIZE):
        out.append(torch.softmax(model(torch.from_numpy(X[i:i + C.EVAL_BATCH_SIZE])), dim=1))
    return torch.cat(out).numpy()


def compute_metrics(y_true, y_pred):
    names = C.CLASS_NAMES
    labels = list(range(len(names)))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)[1],
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # Binary view: benign vs. any attack (the detection question an IDS answers).
    bt, bp = (y_true != C.BENIGN).astype(int), (y_pred != C.BENIGN).astype(int)
    p, r, f, _ = precision_recall_fscore_support(bt, bp, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(bt, bp, labels=[0, 1]).ravel()
    m["binary"] = {
        "accuracy": accuracy_score(bt, bp),
        "benign": {"precision": p[0], "recall": r[0], "f1": f[0]},
        "attack": {"precision": p[1], "recall": r[1], "f1": f[1]},
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return m


def run(model_name, permute=False, seed=C.SEED, max_epochs=C.MAX_EPOCHS, tag=None):
    set_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    meta = load_meta()
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    w_va = np.where(y_va == C.BENIGN, meta["val_benign_weight"], 1.0)
    X_te, y_te = load_split("test")

    if permute:
        # Fixed random column order shared by all splits: destroys whatever
        # locality/ordering the dataset's column layout carries.
        perm = np.random.default_rng(seed + 1000).permutation(X_tr.shape[1])
        X_tr, X_va, X_te = X_tr[:, perm], X_va[:, perm], X_te[:, perm]

    run_name = tag or (model_name + ("_permuted" if permute else ""))
    out_dir = os.path.join(C.RESULTS, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(C.CHECKPOINTS, exist_ok=True)

    n_classes = len(C.CLASS_NAMES)
    model = build_model(model_name, X_tr.shape[1], n_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run_name}] params={n_params:,} train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=C.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_tr, n_classes),
                                    label_smoothing=C.LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    ckpt = os.path.join(C.CHECKPOINTS, f"{run_name}.pt")
    history, best_f1, bad, train_time = [], -1.0, 0, 0.0
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        train_time += time.time() - t0

        va_pred = predict(model, X_va).argmax(1)
        va_f1 = f1_score(y_va, va_pred, average="macro", sample_weight=w_va)
        va_acc = accuracy_score(y_va, va_pred, sample_weight=w_va)
        sched.step(va_f1)
        history.append({"epoch": epoch, "train_loss": total / len(X_tr), "val_macro_f1": va_f1,
                        "val_accuracy": va_acc, "lr": opt.param_groups[0]["lr"],
                        "epoch_seconds": time.time() - t0})
        print(f"[{run_name}] epoch {epoch:2d} loss={total / len(X_tr):.4f} "
              f"val_acc={va_acc:.4f} val_macroF1={va_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if va_f1 > best_f1 + 1e-4:
            best_f1, bad = va_f1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print(f"[{run_name}] early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt))
    t0 = time.time()
    proba = predict(model, X_te)
    infer_s = time.time() - t0
    y_pred = proba.argmax(1)

    metrics = compute_metrics(y_te, y_pred)
    metrics.update({
        "model": model_name, "run": run_name, "permuted_features": permute, "seed": seed,
        "n_params": n_params, "epochs_trained": len(history),
        "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
        "best_val_macro_f1": best_f1, "train_seconds": train_time,
        "inference_us_per_flow": 1e6 * infer_s / len(X_te),
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"), y_true=y_te, y_pred=y_pred,
                        proba=proba.astype(np.float16))
    print(f"[{run_name}] TEST acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"attack_recall={metrics['binary']['attack']['recall']:.4f} "
          f"FPR={metrics['binary']['false_positive_rate']:.5f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "cnn", "lstm", "transformer"])
    ap.add_argument("--permute-features", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--tag")
    a = ap.parse_args()
    run(a.model, a.permute_features, a.seed, a.epochs, a.tag)
