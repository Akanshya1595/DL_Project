# Automated Waste Classification Using Deep Learning

### A Comparative Study of MLP, CNN, ResNet-50 and Vision Transformer

Classifying waste correctly is an important step toward automated segregation and recycling. This project compares four deep learning architectures on **15,515 waste images across 12 classes**, evaluating predictive performance alongside model size, training time and inference latency.

**Best reported result:** ResNet-50 achieved **98.02% test accuracy** and **0.978 macro-F1**. Regrouping its predictions into four disposal streams increased accuracy to **99.6%**.

Developed by **Team 06** for the **Deep Learning course (ICT-4442)** at **Manipal Institute of Technology, Manipal Academy of Higher Education**.

## Students

| Student name | Registration number |
| --- | --- |
| Krish Uppal | 230953478 |
| Akanshya Chakraborty | 230953042 |
| Arushi Pandey | 230911146 |
| Varun Bramhankar | 230953538 |

See [Team Contributions](#team-contributions) for each student's responsibilities.

[Project Repository](https://github.com/Akanshya1595/DL_Project) · [Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

## Overview

The study compares four approaches to image classification:

- **Multilayer Perceptron (MLP):** a baseline trained on flattened image pixels.
- **CNN from scratch:** a custom convolutional network that learns spatial features without pretrained weights.
- **ResNet-50:** an ImageNet-pretrained residual network adapted through transfer learning.
- **ViT-B/16:** an ImageNet-pretrained Vision Transformer fine-tuned for waste classification.

All models use the same frozen, stratified dataset split and shared evaluation protocol. Input resolutions and training schedules vary by model, as documented below. **Macro-F1 is the headline metric** because it gives equal importance to each class despite the dataset's imbalance.

## Project Highlights

- Compare fully connected, convolutional and attention-based architectures on a common 12-class benchmark.
- Handle class imbalance with class-weighted cross-entropy.
- Measure accuracy, macro precision, recall and F1, per-class F1, confusion matrices and computational cost.
- Analyze errors at both the individual-class and disposal-stream levels.
- Evaluate the effect of augmentation on the scratch CNN.
- Audit train/test near-duplicates and examine confidence-based prediction rejection.

The project's contribution is an empirical comparison and practical error analysis; it does not introduce a new neural network architecture.

## Dataset

The project uses the **Garbage Classification** dataset by Mostafa Abla on Kaggle. It contains RGB photographs of individual waste items. All files were checked for readability; no unreadable images were removed.

| Class | Images | Assigned disposal stream |
| --- | ---: | --- |
| battery | 945 | Hazardous |
| biological | 985 | Organic |
| brown-glass | 607 | Recyclable |
| cardboard | 891 | Recyclable |
| clothes | 5,325 | Residual/textile |
| green-glass | 629 | Recyclable |
| metal | 769 | Recyclable |
| paper | 1,050 | Recyclable |
| plastic | 865 | Recyclable |
| shoes | 1,977 | Residual/textile |
| trash | 697 | Residual/textile |
| white-glass | 775 | Recyclable |
| **Total** | **15,515** | |

The four-stream mapping is defined for this study and is not a universal waste-disposal standard.

### Data Split

A single stratified split was generated with **random seed 42** and saved as a CSV for reuse across all models.

| Partition | Proportion | Images |
| --- | ---: | ---: |
| Training | 70% | 10,860 |
| Validation | 15% | 2,327 |
| Test | 15% | 2,328 |

Class labels follow alphabetical order, from `battery` (0) to `white-glass` (11). Checkpoints are selected using validation macro-F1, with final performance measured on the held-out test set.

## Preprocessing and Augmentation

Images are initially resized to **224 × 224** without cropping, then adapted to each model:

| Model | Input | Pixel preprocessing |
| --- | --- | --- |
| MLP | 64 × 64, flattened | Scale to [0, 1] |
| Scratch CNN | 128 × 128 | Scale to [0, 1] |
| ResNet-50 | 224 × 224 | ImageNet normalization |
| ViT-B/16 | 224 × 224 | ImageNet normalization |

Training-only GPU augmentation includes random horizontal flips, rotations up to ±15°, brightness/contrast/saturation factors from 0.8 to 1.2, and hue shifts up to ±0.03. The small hue range helps preserve color differences between glass classes. Validation and test images receive only deterministic resizing and scaling or normalization.

Class-weighted cross-entropy uses `weight[c] = N / (12 × n_c)`, where `N` is the number of training images and `n_c` is the training count for class `c`.

## Model Architectures

### MLP

The baseline maps **12,288 → 1,024 → 256 → 12** features through dense layers. Both hidden layers use ReLU, batch normalization and dropout of 0.4. Flattening discards explicit spatial structure.

### CNN from Scratch

Five VGG-style blocks use channel widths **32, 64, 128, 256 and 256**. Each block contains two 3 × 3 convolutions with batch normalization and ReLU, followed by 2 × 2 max pooling. Global average pooling, dropout of 0.4 and a linear classifier produce 12 outputs.

### ResNet-50

An ImageNet-pretrained ResNet-50 receives a replacement head with dropout of 0.2 and a 12-class linear layer. Training proceeds in two stages:

1. Train only the new head for **3 epochs**, with the backbone and its batch-normalization statistics frozen.
2. Fine-tune `layer3`, `layer4` and the head for **7 epochs**.

### ViT-B/16

The ImageNet-1k-pretrained model divides each image into **196 patches of 16 × 16 pixels**, adds a class token and processes the sequence through **12 Transformer encoder layers**, with hidden size **768** and **12 attention heads**. A new 12-class head is added, and all weights are fine-tuned for **5 epochs**, with gradient clipping at 1.0.

## Training Configuration

All models use **AdamW**, a linear learning-rate warm-up over the first **5% of steps**, cosine decay, GPU mixed precision and checkpoint selection by validation macro-F1.

| Model | Peak learning rate | Weight decay | Epochs | Batch size |
| --- | --- | ---: | ---: | ---: |
| MLP | 1e-3 | 5e-4 | 40 | 128 |
| Scratch CNN | 2e-3 | 5e-4 | 30 | 64 |
| ResNet-50 | Stage 1: head 1e-3; stage 2: backbone 1e-4, head 5e-4 | 1e-4 | 3 + 7 | 64 |
| ViT-B/16 | Body 3e-5; head 3e-4 | 0.05 | 5 | 32 |

**Reported environment:** Google Colab, NVIDIA Tesla T4, PyTorch `2.11.0+cu130`, and torchvision pretrained models.

## Results

Results below are from a **single run per model, using seed 42**.

| Model | Accuracy (%) | Macro precision | Macro recall | Macro-F1 | Parameters (M) | Training (min) | Latency (ms/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 67.31 | 0.588 | 0.632 | 0.604 | 12.9 | 3.1 | 0.67 |
| Scratch CNN | 87.37 | 0.843 | 0.863 | 0.851 | 2.4 | 5.9 | 1.87 |
| **ResNet-50** | **98.02** | **0.977** | **0.978** | **0.978** | **23.5** | **4.3** | **8.27** |
| ViT-B/16 | 97.72 | 0.971 | 0.974 | 0.972 | 85.8 | 11.2 | 12.39 |

Training time includes per-epoch validation. Latency is the mean forward-pass time over 100 test images at batch size 1, after 20 warm-up passes on a Tesla T4; it is not end-to-end application latency.

ResNet-50 achieved the highest observed accuracy and macro-F1 while using fewer parameters and lower inference latency than ViT-B/16. However, their accuracy difference was **not statistically significant** in this run (**McNemar p = 0.36**).

### Disposal-Stream Performance

The existing 12-class predictions were regrouped into four streams; no separate four-class model was trained for these results.

| Model | Four-stream accuracy (%) | Four-stream macro-F1 |
| --- | ---: | ---: |
| MLP | 78.8 | 0.700 |
| Scratch CNN | 92.3 | 0.887 |
| ResNet-50 | **99.6** | **0.992** |
| ViT-B/16 | 99.3 | 0.986 |

### Error and Confidence Analysis

- ResNet-50 made **46 errors out of 2,328 test images**.
- **37 of those 46 errors (80.4%)** remained within the same assigned disposal stream.
- The most frequent confusion was **clothes → shoes**, occurring in 13 images.
- Accepting only predictions with confidence **≥ 0.9** retained **96.7%** of test images at **99.1% accuracy on the retained subset**. This suggests a possible rejection mechanism for further validation.

### Augmentation Ablation

| Scratch CNN setting | Test accuracy (%) | Test macro-F1 |
| --- | ---: | ---: |
| With augmentation | 87.37 | 0.851 |
| Without augmentation | 88.02 | 0.859 |

Removing augmentation improved macro-F1 by **0.008** under this fixed training schedule. This experiment does not establish that augmentation is generally harmful.

### Near-Duplicate Audit

A 64-bit difference-hash check with Hamming distance ≤ 4 identified **205 test images (8.8%)** with a near-duplicate in the training set. On the remaining **2,123 test images**, accuracy was **66.8%** for MLP, **86.9%** for the scratch CNN, **98.0%** for ResNet-50 and **97.7%** for ViT-B/16.

## Getting Started

Clone the project repository:

```bash
git clone https://github.com/Akanshya1595/DL_Project.git
cd DL_Project
```

To reproduce the reported experiment:

1. Download and extract the linked Kaggle Garbage Classification dataset.
2. Use the project's training code in a GPU-enabled environment with PyTorch and torchvision. The experiments were run on Google Colab.
3. Configure the dataset location in the relevant notebook or script.
4. Reuse the saved stratified split and alphabetical class mapping for all four models.
5. Apply the preprocessing and training settings documented above, selecting each checkpoint by validation macro-F1.
6. Evaluate the selected checkpoints with the shared evaluation module on the same held-out test set.

> This README is based on the final project report. Exact notebook/script filenames, dependency installation commands, checkpoint locations and command-line arguments are not specified in the report. Consult the repository code for these details before running an experiment.

## Limitations and Future Work

The comparison uses one seed and a fixed training recipe per model, without a hyperparameter search. Differences therefore reflect both architecture and training choices. The dataset contains individual waste items, and performance on cluttered conveyor-belt images, occlusion or different lighting has not been established. A planned cross-dataset evaluation on TrashNet was not carried out.

Future directions include:

- Multiple-seed experiments with uncertainty estimates and model-specific tuning.
- Cross-dataset and real-world evaluation.
- Comparisons with Swin Transformer, ConvNeXt and MobileViT.
- Quantization or knowledge distillation for deployment on limited hardware.
- Validation of confidence-based rejection and hierarchical stream-then-class classification.
- Disposal-stream mappings developed with waste-management partners.

## Team Contributions

Contributions are based on the individual contribution statement in the final project report.

| Student | Model / task owned | Contributions |
| --- | --- | --- |
| **Krish Uppal** | Model 3: ResNet-50; shared evaluation module | Created the GitHub repository with one branch per member. Wrote the shared evaluation module covering accuracy, macro precision, recall and F1, per-class F1, confusion matrices, parameter counts, training time and latency. Implemented, trained and tuned ResNet-50. Wrote the dataset and models sections of the report. |
| **Akanshya Chakraborty** | Model 4: ViT-B/16; comparison table | Built the ViT-B/16 data pipeline and fine-tuning script, and trained the model. Prepared the comparison table for all four models. Wrote the literature review entries for Transformer papers and the risk and plan section. |
| **Arushi Pandey** | Model 2: CNN from scratch; augmentation ablation | Implemented and trained the scratch CNN. Ran the experiments with and without augmentation. Wrote the literature review entries for convolutional-network papers. |
| **Varun Bramhankar** | Model 1: MLP; dataset and split | Downloaded the dataset and performed exploratory data analysis. Generated and froze the stratified training, validation and test split. Implemented and trained the MLP. Wrote the preprocessing section of the report. |

## Acknowledgements

We thank the faculty of the School of Computer Engineering, Manipal Institute of Technology, for their guidance. Experiments used free Google Colab GPU resources and the Kaggle Garbage Classification dataset.

## Documentation Source

*Comparative Study of Deep Learning Architectures for Automated Waste Classification*, Team 06, Deep Learning Project (ICT-4442), November 2026. All experimental configurations and numerical results in this README are taken from the final project report.
