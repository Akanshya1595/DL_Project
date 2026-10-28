# Changes: repo setup and shared evaluation

Owner: Krish Uppal (230953478) — repo setup, `shared/evaluate.py`
Branch: `repo-hardening-and-leakage-fix`

**No model architecture and no part of the training protocol changed.** Learning
rates, epochs, stages, augmentation, seed and the 70/15/15 ratios are all
untouched, so these changes do not affect comparability between the four models.

---

## 1. `.gitignore` (new file)

The README states that weights in `results/checkpoints/` are git-ignored, but no
`.gitignore` existed. A single `git add .` after training would have committed
them — ResNet-50 is ~100 MB per checkpoint and ViT-B/16 ~330 MB.

- ignores `*.pt` / `*.pth` / `*.ckpt` and `**/results/checkpoints/`
- ignores the image dataset under `**/data/`, but **whitelists `data/splits.csv`**
  so the frozen split is still committed, as the README requires
- ignores `data/data_root.txt` (machine-specific path, differs per member)
- keeps `results/` itself tracked — the metrics JSON, history CSVs, confusion
  matrices and comparison table are deliverables
- ignores `kaggle.json` and `.kaggle/` so credentials cannot be committed
- usual Python / venv / notebook / editor noise

## 2. `shared/evaluate.py` — hardened

This module scores **all four models**, so a defect here invalidates every row of
the comparison rather than one. Five gaps closed:

| | Change | Why |
|---|---|---|
| 1 | Record `device_type`, `device_name`, `torch_version` | Members train on their own machines, so `train_time_s` and `latency_ms_per_image` are only comparable when produced on the same hardware. This makes a mismatch visible instead of silently corrupting the efficiency columns. |
| 2 | Add per-class **precision** and **recall** (+ `per_class_support`) | A per-class F1 of 0.4 can mean the class is rarely predicted (low recall) or predicted indiscriminately (low precision). Those need opposite explanations in the error analysis. |
| 3 | Raise on classes missing from `config.STREAMS` | They previously fell into a silent `"Other"` bucket, so editing `STREAMS` and omitting a class would quietly corrupt the stream metrics with no error anywhere. |
| 4 | Save the 4×4 disposal-stream confusion matrix (CSV + PNG) | Stream accuracy was computed but the matrix discarded. It is the figure that shows which confusions are harmless (glass↔glass) and which are not (battery→recyclable). |
| 5 | Add `weighted_f1` next to `macro_f1` | The gap between them quantifies the rare-class penalty on this imbalanced dataset. |

Also: `labels=` is now passed explicitly to every sklearn averaging call, so the
macro average is always taken over all 12 classes rather than only those present
in `y_true`/`y_pred`; and a median latency is reported next to the mean, because
one GC pause on a shared Colab/Kaggle VM can skew a mean.

Not added: balanced accuracy — it is mathematically identical to macro recall,
which is already reported.

**Timing note.** This had to land *before* training starts. `evaluate_and_save`
writes `<model>_results.json` at the end of each run, so adding a metric
afterwards would require re-evaluating all four checkpoints — and those are
git-ignored and live on four separate machines.

**New files per model:** `<tag>_stream_confusion_matrix.csv` and `.png`.
`scripts/compare.py` is unaffected; it selects a fixed column list.

## 3. `shared/data_prep.py` — remove duplicates before the split

> This file is owned by Akanshya Chakraborty. The change is in its own commit so
> it can be reviewed or reverted independently.

`scan()` already computed an md5 per file and `eda()` already reported
`duplicate_groups_spanning_splits`, but `make_split()` ran first and nothing
removed them. The leakage was therefore **measured and then left in place**: the
same bytes can land in train and in test, so part of the test score is
memorisation and every model's number is inflated by an unknown amount. That
undermines the project's central claim that differences are attributable to
architecture.

`dedupe()` now drops duplicates (keeping one copy per md5) *before* splitting.
Before rather than after, so the 70/15/15 ratios and per-class stratification
stay exact. `eda()` gains `duplicates_removed_before_split`, and the two existing
leakage counters now read 0 as a verification rather than as a warning.

`--keep_duplicates` reproduces the previous behaviour, so the change is explicit
and reversible rather than silent.

Also fixes the module docstring, which told users to run `python -m src.data_prep`
— there is no `src` package, it is `shared`.

---

## Verification

Run against the synthetic dataset from `scripts/make_dummy_data.py` (660 unique
images) with 10 byte-identical duplicates deliberately planted across classes:

| run | removed | `exact_duplicate_files` | `duplicate_groups_spanning_splits` |
|---|---|---|---|
| default | 10 | 0 | **0** |
| `--keep_duplicates` | 0 | 10 | **2** ← leak reproduced |

Two duplicate groups straddled train/test from only 10 planted copies. The real
count on the ~15,150 actual images is unknown until this runs on the real data;
`results/eda/eda_summary.json` will report it.

Also trained MLP for 2 epochs end-to-end on the dummy data: both confusion
matrices written, all new JSON fields present, `per_class_support` sums exactly to
`test_size`, and the unmapped-class guard raises as intended. This was the first
end-to-end run of the pipeline — it had not previously been executed.

## What each member needs to do

- **Akanshya** — review the `dedupe()` change, then generate and commit
  `data/splits.csv`. Everyone else is blocked until that lands.
- **Arushi, Varun** — nothing to change. Pull before you train so your results
  JSON includes the new fields.
- **Everyone** — commit your own folder from your own GitHub account. The
  Individual Contribution Log is generated from git history by
  `scripts/export_commit_log.sh`, and history currently shows the whole project
  arriving as one upload from one account.
- **After all four models are trained** — one person re-measures latency for all
  four checkpoints on a single machine, so the efficiency columns are comparable.
  `device_name` in each results JSON shows whether this is still outstanding.
