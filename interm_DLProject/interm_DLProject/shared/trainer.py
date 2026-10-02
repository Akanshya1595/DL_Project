"""
Common training loop used by all four model folders (identical protocol for every model).
Each model folder supplies only: IMG_SIZE and build() -> (model, stages).
The checkpoint with the best VALIDATION macro-F1 is restored before the single TEST evaluation.
"""
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import RESULTS_DIR, SEED
from .dataset import build_loaders
from .evaluate import evaluate_and_save, predict


def add_common_args(ap, epochs, batch_size):
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--batch_size", type=int, default=batch_size)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_aug", action="store_true", help="disable augmentation (ablation)")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (debug only)")
    ap.add_argument("--class_weights", action="store_true", help="inverse-frequency weighted loss")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data_root", type=str, default=None, help="folder with the class sub-folders")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _freeze_bn_if_frozen(model):
    """BatchNorm layers with frozen weights must not update their running statistics."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def run(name, build_fn, img_size, a):
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    tag = name + ("_noaug" if a.no_aug else "")

    loaders, classes, train_counts = build_loaders(img_size, a.batch_size, not a.no_aug,
                                                   a.workers, a.data_root)
    model, stages = build_fn(len(classes), pretrained=not a.no_pretrained)
    model.to(device)

    weight = None
    if a.class_weights:
        c = torch.tensor(train_counts, dtype=torch.float)
        weight = (c.sum() / (len(c) * c)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history, best_f1, best_state = [], -1.0, None
    t_start = time.perf_counter()
    for st in stages:
        n_ep = max(1, round(a.epochs * st["epoch_frac"]))
        for p in model.parameters():
            p.requires_grad = False
        for p in st["params"]:
            p.requires_grad = True
        opt = torch.optim.AdamW(st["params"], lr=st["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

        for ep in range(1, n_ep + 1):
            model.train()
            _freeze_bn_if_frozen(model)
            run_loss, seen = 0.0, 0
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_cuda):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item() * y.size(0)
                seen += y.size(0)
            sched.step()

            yt, yp, vloss = predict(model, loaders["val"], device, criterion)
            vf1 = f1_score(yt, yp, average="macro", zero_division=0)
            vacc = float((yt == yp).mean())
            history.append(dict(stage=st["name"], epoch=ep, train_loss=run_loss / seen,
                                val_loss=vloss, val_acc=vacc, val_macro_f1=vf1))
            print(f"[{tag}|{st['name']}] ep {ep}/{n_ep}  train_loss {run_loss/seen:.4f}  "
                  f"val_loss {vloss:.4f}  val_acc {vacc:.4f}  val_F1 {vf1:.4f}")
            if vf1 > best_f1:
                best_f1, best_state = vf1, copy.deepcopy(model.state_dict())
    if use_cuda:
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start

    model.load_state_dict(best_state)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "checkpoints").mkdir(exist_ok=True)
    torch.save(best_state, RESULTS_DIR / "checkpoints" / f"{tag}.pt")   # git-ignored
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{tag}_history.csv", index=False)

    res = evaluate_and_save(model, loaders["test"], device, classes, RESULTS_DIR, tag, img_size,
                            train_time, extra={"best_val_macro_f1": best_f1,
                                               "epochs_requested": a.epochs,
                                               "augmentation": not a.no_aug,
                                               "pretrained": not a.no_pretrained, "seed": a.seed})
    print("\nTEST RESULTS")
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "stream_accuracy",
              "params_total", "train_time_s", "latency_ms_per_image"):
        print(f"  {k}: {res[k]}")
    return res
