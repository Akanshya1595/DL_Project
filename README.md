# Team 06: Comparative Study of Deep Learning Architectures for Automated Waste Classification
Deep Learning Project (ICT-4442), School of Computer Engineering, MIT Manipal.

Dataset: Kaggle Garbage Classification (12 classes). All four members use the same stratified 70/15/15 split (seed 42), preprocessing, augmentation, class-weighted loss and evaluation code.

| Member | Reg. No. | Folder | Notebook | Model / task |
|---|---|---|---|---|
| Varun Bramhankar | 230953538 | `Varun_Bramhankar/` | `01_..._dataset_EDA_split_MLP.ipynb` | Dataset, EDA, frozen split, near-duplicate check, Model 1 (MLP) |
| Arushi Pandey | 230911146 | `Arushi_Pandey/` | `02_..._CNN_scratch_and_augmentation_ablation.ipynb` | Model 2 (CNN from scratch), augmentation ablation |
| Krish Uppal | 230953478 | `Krish_Uppal/` | `03_..._ResNet50_transfer_learning.ipynb` | Model 3 (ResNet-50), shared evaluation code |
| Akanshya Chakraborty | 230953042 | `Akanshya_Chakraborty/` | `04_..._ViT_B16_and_comparison_table.ipynb` | Model 4 (ViT-B/16), comparison table |

`final_report/` holds the combined notebook that trains all four models in one run and builds the final report.

Each member folder contains the notebook and a `results/` folder with: `<model>_results.json` (all metrics, confusion matrix, per-class F1, training history), `<model>_test_predictions.npz`, `figures/`, and `README_<member>.md` (summary table).

## How to run (each member, in parallel)
1. Open your notebook in Google Colab (File > Upload notebook, or open it from GitHub).
2. Runtime > Change runtime type > T4 GPU, then Runtime > Run all.
3. If the Kaggle download fails, upload `kaggle.json` when asked.
4. A `<Your_Folder>_results.zip` downloads at the end.
5. The line `Split fingerprint: ...` must be identical in all four notebooks.

## Git workflow (everyone, using your own GitHub account so the history shows individual work)
```bash
git clone https://github.com/Akanshya1595/DL_Project.git
cd DL_Project
git config user.name  "Your Name"
git config user.email "your-github-email@example.com"
git checkout -b final/<your-first-name>

# commit 1: your notebook
cp /path/to/<your notebook>.ipynb <Your_Folder>/        # folder name from the table above
git add <Your_Folder>
git commit -m "Add <Model> notebook (<what it does>)"
git push -u origin final/<your-first-name>

# ... run the notebook in Colab, download <Your_Folder>_results.zip ...

# commit 2: your results (unzip INSIDE the repo root so <Your_Folder>/results/ appears)
unzip /path/to/<Your_Folder>_results.zip -d .
git add <Your_Folder>/results
git commit -m "Add <Model> results, figures and metrics"
git push

# commit 3: short note on what you observed (edit <Your_Folder>/results/README_<Your_Folder>.md, add 3-5 lines of your own analysis)
git add -A && git commit -m "Add analysis notes for <Model>" && git push
```
Then open a Pull Request into `main` on GitHub and merge it with **"Create a merge commit"** (not squash), so every commit stays visible in the history. Commit at the real time you do each step; do not rewrite or backdate history.

### Who pushes what, in order
1. **Varun**: `Varun_Bramhankar/` with the notebook and `results/` (includes `split_seed42.csv`, `class_to_index.json`, `dataset_summary.json`, EDA figures, MLP results). Merge to `main` first or in parallel.
2. **Arushi**: `Arushi_Pandey/` with the notebook and `results/` (CNN, CNN without augmentation, `augmentation_ablation.png`).
3. **Krish**: `Krish_Uppal/` with the notebook and `results/` (ResNet-50).
4. **Akanshya**: `Akanshya_Chakraborty/` with the notebook and `results/` (ViT-B/16). After the other three have merged to `main`, run the last cell of her notebook (it reads everyone's `results/*_results.json`), then commit the new `results/comparison/` folder (`comparison_table.csv/.md`, per-class heatmap, trade-off plot, curves). Also add `final_report/` (the combined notebook and `Final_Report_Group6.docx`).
Do not commit the `.zip` files (they are in `.gitignore`).
