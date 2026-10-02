"""Consolidate results/*_results.json into ONE comparison table (owner: Varun).  python scripts/compare.py"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
rows = [json.loads(p.read_text()) for p in sorted(R.glob("*_results.json"))]
if not rows:
    raise SystemExit("No *_results.json found - train at least one model first.")
cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "stream_accuracy",
        "params_total", "train_time_s", "latency_ms_per_image"]
df = pd.DataFrame(rows)[cols].round(4)
df.to_csv(R / "comparison_table.csv", index=False)
try:
    (R / "comparison_table.md").write_text(df.to_markdown(index=False))
except ImportError:                       # tabulate not installed
    (R / "comparison_table.md").write_text(df.to_string(index=False))
print(df.to_string(index=False))
