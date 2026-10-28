"""
Shared evaluation module - used unchanged by ALL four models.
Owner: Krish Uppal.

Reports: accuracy, macro precision/recall/F1, weighted F1, per-class
precision/recall/F1/support, confusion matrix, parameter count, training time,
per-image inference latency (mean + median) with the hardware it was measured
on, and the four-stream (disposal-level) regrouped accuracy / macro-F1 plus its
own 4x4 confusion matrix.

NOTE: macro recall is mathematically identical to balanced accuracy, so it is
not reported twice.
"""
import json
import platform
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


def device_info(device) -> dict:
    """Records WHERE a measurement was taken.

    Each member trains on their own machine, so train_time_s and
    latency_ms_per_image are only comparable across models when they were
    produced on the same hardware. Storing the device makes a mismatch visible
    in the results JSON instead of silently corrupting the comparison table.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        name = "Apple Silicon (MPS)"
    else:
        name = platform.processor() or platform.machine() or "cpu"
    return {"device_type": device.type, "device_name": name,
            "torch_version": torch.__version__}


@torch.no_grad()
def measure_latency_ms(model, img_size, device, n=200, warmup=20):
    """Single-image (batch=1) forward time in ms. Returns (mean, median).

    The median is reported alongside the mean because one GC pause or thermal
    hiccup in n runs is enough to skew a mean on a shared Colab/Kaggle VM.
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.median(times))


def metrics_from_preds(y_true, y_pred, class_names):
    """All 12-class metrics.

    `labels` is passed explicitly everywhere so the macro average is always
    taken over all classes, never only over those sklearn happens to find in
    y_true/y_pred. Without it, a model that never predicts a rare class could
    be macro-averaged over a smaller set than another model - which would make
    the headline metric of the comparison silently inconsistent between rows.
    """
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f1, pc_n = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "weighted_f1": float(wf1),
        # per-class precision AND recall, not just F1: an F1 of 0.4 can mean the
        # class is rarely predicted (low recall) or predicted indiscriminately
        # (low precision), and those have opposite explanations.
        "per_class_f1": {c: float(v) for c, v in zip(class_names, pc_f1)},
        "per_class_precision": {c: float(v) for c, v in zip(class_names, pc_p)},
        "per_class_recall": {c: float(v) for c, v in zip(class_names, pc_r)},
        "per_class_support": {c: int(v) for c, v in zip(class_names, pc_n)},
    }


def four_stream_metrics(y_true, y_pred, class_names):
    """Disposal-level metrics: confusing brown-glass with green-glass is harmless,
    confusing battery with paper is not. Returns (metrics, 4x4 cm, stream names).
    """
    unmapped = [c for c in class_names if c not in CLASS_TO_STREAM]
    if unmapped:
        # Fail loudly. Previously these fell into a silent "Other" bucket, so
        # editing STREAMS in config.py and missing a class would quietly
        # corrupt the stream metrics with no error anywhere.
        raise KeyError(
            f"classes not mapped to a disposal stream in config.STREAMS: {unmapped}. "
            f"Add them to STREAMS before evaluating.")
    streams = list(STREAMS.keys())
    idx = {c: streams.index(CLASS_TO_STREAM[c]) for c in class_names}
    yt = np.array([idx[class_names[i]] for i in y_true])
    yp = np.array([idx[class_names[i]] for i in y_pred])
    cm = confusion_matrix(yt, yp, labels=range(len(streams)))
    metrics = {"stream_accuracy": float(accuracy_score(yt, yp)),
               "stream_macro_f1": float(f1_score(yt, yp, labels=range(len(streams)),
                                                 average="macro", zero_division=0))}
    return metrics, cm, streams


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
    stream_metrics, stream_cm, stream_names = four_stream_metrics(y_true, y_pred, class_names)
    res = {"model": tag, **metrics_from_preds(y_true, y_pred, class_names), **stream_metrics}
    total, trainable = count_params(model)
    lat_mean, lat_median = measure_latency_ms(model, img_size, device)
    res.update({"params_total": total, "params_trainable_final_stage": trainable,
                "train_time_s": round(train_time_s, 1),
                "latency_ms_per_image": round(lat_mean, 3),
                "latency_ms_median": round(lat_median, 3),
                "test_size": int(len(y_true)), **device_info(device)})
    if extra:
        res.update(extra)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.savetxt(out_dir / f"{tag}_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    save_confusion_matrix(cm, class_names, out_dir / f"{tag}_confusion_matrix.png",
                          f"{tag} - test confusion matrix")

    # 4x4 disposal-stream matrix: the figure that shows which errors actually matter.
    np.savetxt(out_dir / f"{tag}_stream_confusion_matrix.csv", stream_cm, fmt="%d",
               delimiter=",", header=",".join(stream_names), comments="")
    save_confusion_matrix(stream_cm, stream_names,
                          out_dir / f"{tag}_stream_confusion_matrix.png",
                          f"{tag} - test confusion matrix (4 disposal streams)")

    (out_dir / f"{tag}_results.json").write_text(json.dumps(res, indent=2))
    return res
