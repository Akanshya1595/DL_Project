"""
Shared evaluation module - used unchanged by ALL four models.
Owner: Krish Uppal.

Reports: accuracy, macro precision/recall/F1, per-class F1, confusion matrix,
parameter count, training time, per-image inference latency, and the
four-stream (disposal-level) regrouped accuracy / macro-F1.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score

from .config import CLASS_TO_STREAM, STREAMS


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), (loss_sum / max(n, 1))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency_ms(model, img_size, device, n=100, warmup=10):
    """Mean single-image (batch=1) forward time in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def metrics_from_preds(y_true, y_pred, class_names):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "per_class_f1": {c: float(v) for c, v in zip(class_names, per_class)},
    }


def four_stream_metrics(y_true, y_pred, class_names):
    streams = list(STREAMS.keys()) + ["Other"]
    s = {c: streams.index(CLASS_TO_STREAM.get(c, "Other")) for c in class_names}
    yt = np.array([s[class_names[i]] for i in y_true])
    yp = np.array([s[class_names[i]] for i in y_pred])
    return {"stream_accuracy": float(accuracy_score(yt, yp)),
            "stream_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def save_confusion_matrix(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_and_save(model, loader, device, class_names, out_dir: Path, tag: str,
                      img_size: int, train_time_s: float, extra: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, _ = predict(model, loader, device)
    res = {"model": tag, **metrics_from_preds(y_true, y_pred, class_names),
           **four_stream_metrics(y_true, y_pred, class_names)}
    total, trainable = count_params(model)
    res.update({"params_total": total, "params_trainable_final_stage": trainable,
                "train_time_s": round(train_time_s, 1),
                "latency_ms_per_image": round(measure_latency_ms(model, img_size, device), 3),
                "test_size": int(len(y_true))})
    if extra:
        res.update(extra)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.savetxt(out_dir / f"{tag}_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    save_confusion_matrix(cm, class_names, out_dir / f"{tag}_confusion_matrix.png",
                          f"{tag} - test confusion matrix")
    (out_dir / f"{tag}_results.json").write_text(json.dumps(res, indent=2))
    return res
