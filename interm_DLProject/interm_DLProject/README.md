# Comparative Study of Deep Learning Architectures for Automated Waste Classification

**Course:** ICT-4442 Deep Learning Project, School of Computer Engineering, Manipal Institute of Technology, MAHE

| Member | Reg. No. | Model | Folder | Extra responsibility |
|--------|----------|-------|--------|----------------------|
| Akanshya Chakraborty | 230953042 | Model 1 - MLP baseline | `model1_mlp/` | Dataset acquisition, EDA, frozen split (`shared/data_prep.py`) |
| Arushi Pandey | 230911146 | Model 2 - CNN from scratch (+ augmentation ablation) | `model2_cnn/` | - |
| Krish Uppal | 230953478 | Model 3 - ResNet-50 transfer learning | `model3_resnet50/` | Repo setup, shared evaluation module (`shared/evaluate.py`) |
| Varun | 230953538 | Model 4 - Vision Transformer ViT-B/16 (Swin-T fallback) | `model4_vit/` | Consolidated comparison table (`scripts/compare.py`) |

## Project summary
Waste classification is treated as single-label image classification (12 classes) and used as a controlled testbed to
compare four architecture families. Every model uses the **same dataset, the same frozen stratified 70/15/15 split
(fixed seed), the same preprocessing/augmentation and the same evaluation module**, so differences in performance are
attributable to architecture alone.

**Dataset:** Kaggle *Garbage Classification* (`mostafaabla/garbage-classification`), ~15,150 images, 12 classes: battery,
biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.
**Optional secondary dataset:** TrashNet (~2,527 images, 6 classes) used only as an unseen cross-dataset test.

## Models
| # | Model | Family | Input | Details |
|---|-------|--------|-------|---------|
| 1 | MLP | Fully-connected baseline | 64x64, flattened | 3 dense layers (1024-512-256), ReLU, BatchNorm, Dropout |
| 2 | VGG-style CNN | Convolutional, no pre-training | 128x128 | 5 conv blocks (2 convs each), BatchNorm, MaxPool, global average pooling |
| 3 | ResNet-50 | Convolutional, ImageNet pre-trained | 224x224 | Stage 1: backbone frozen, head trained. Stage 2: layer3/layer4 + head at lower LR |
| 4 | ViT-B/16 | Attention-based, ImageNet pre-trained | 224x224 | Stage 1: head only. Stage 2: full fine-tune at small LR. Swin-T fallback via `--variant swin` |

**Common protocol:** identical split and seed (42); augmentation = horizontal flip, rotation +/-15 deg, colour jitter
(training set only); AdamW + cosine schedule; best-validation-macro-F1 checkpoint is restored; the test set is evaluated once.
**Metrics reported:** accuracy, macro precision / recall / F1, per-class F1, confusion matrix, parameter count, training
time, per-image inference latency, and accuracy / macro-F1 after regrouping into four disposal streams.

## Repository structure
```
shared/                    code common to all models
  config.py                seed, paths, four-stream mapping
  data_prep.py             download/scan, integrity + duplicate check, EDA, frozen split
  dataset.py               preprocessing, augmentation, data loaders
  trainer.py               training loop shared by all models
  evaluate.py              shared evaluation module
  registry.py              model tag -> builder
  trashnet_eval.py         optional TrashNet cross-dataset check
model1_mlp/    model.py  train.py
model2_cnn/    model.py  train.py
model3_resnet50/ model.py  train.py
model4_vit/    model.py  train.py      (ViT-B/16 and Swin-T fallback)
scripts/                   compare.py, export_commit_log.sh, make_dummy_data.py
data/                      splits.csv (frozen split, committed)
results/                   metrics JSON, training history, confusion matrices, EDA
```

## Setup
```bash
pip install -r requirements.txt     # a GPU (local, Colab or Kaggle) is strongly recommended
```
For automatic download, configure Kaggle credentials for `kagglehub`, or download the dataset manually.

## Usage (run everything from the repository root)
```bash
# 1. Dataset + frozen split (ONE member runs this, then commits data/splits.csv)
python -m shared.data_prep --download
#    or: python -m shared.data_prep --data_root /path/to/garbage_classification

# 2. Train (each member runs their own model)
python -m model1_mlp.train       --epochs 30
python -m model2_cnn.train       --epochs 40
python -m model2_cnn.train       --epochs 40 --no_aug        # augmentation ablation
python -m model3_resnet50.train  --epochs 15
python -m model4_vit.train       --epochs 10 --batch_size 32
python -m model4_vit.train       --epochs 10 --variant swin  # fallback

# 3. Consolidate into one comparison table
python scripts/compare.py                                    # -> results/comparison_table.{csv,md}

# 4. Optional: cross-dataset check on TrashNet
python -m shared.trashnet_eval --model resnet50 --trashnet_root /path/to/trashnet

# 5. Contribution log
bash scripts/export_commit_log.sh                            # -> commit_history.txt
```
Useful flags for every model: `--data_root PATH` (dataset location on your machine), `--class_weights`, `--seed N`.

Once the repository is cloned, teammates only need `data/splits.csv` (from Git) and a local copy of the images;
the split stores image paths relative to the dataset folder.

## Outputs (`results/`)
`<model>_results.json` (all metrics), `<model>_history.csv` (learning curves), `<model>_confusion_matrix.{csv,png}`,
`eda/` (class counts, image-size stats, duplicate check), `comparison_table.{csv,md}`. Model weights are saved to
`results/checkpoints/` and are git-ignored.

## Smoke test without the real dataset
```bash
python scripts/make_dummy_data.py /tmp/dummy
python -m shared.data_prep --data_root /tmp/dummy --min_classes 12 --force
python -m model1_mlp.train --epochs 2 --data_root /tmp/dummy
```
Delete `data/splits.csv` afterwards and regenerate it from the real data.

## Notes
- The four-stream grouping in `shared/config.py` (recyclable dry / glass / organic / hazardous & non-recyclable) is an
  assumption; adjust it if the team defines the streams differently.
- Do not regenerate `data/splits.csv` once models have been trained on it.
