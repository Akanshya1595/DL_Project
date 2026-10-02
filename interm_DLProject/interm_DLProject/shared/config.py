"""Project-wide constants shared by all four models."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SPLIT_CSV = DATA_DIR / "splits.csv"

SEED = 42
KAGGLE_ID = "mostafaabla/garbage-classification"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Four-stream regrouping used in the error analysis (ASSUMPTION - confirm with team).
STREAMS = {
    "Recyclable dry": ["paper", "cardboard", "plastic", "metal"],
    "Glass": ["brown-glass", "green-glass", "white-glass"],
    "Organic": ["biological"],
    "Hazardous / non-recyclable": ["battery", "trash", "clothes", "shoes"],
}
CLASS_TO_STREAM = {c: s for s, cs in STREAMS.items() for c in cs}
