"""Shared paths, constants and loaders for Task 2 (player segment classification)."""
from pathlib import Path

import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TASK_DIR.parent
SUB_DIR = TASK_DIR / "submissions"
FIG_DIR = TASK_DIR / "reports" / "figures"


def _find(name: str, *dirs: Path) -> Path:
    """First existing location of a data file (the Kaggle CSVs are not in git)."""
    for d in dirs:
        if (d / name).exists():
            return d / name
    return dirs[0] / name  # default location, used in the error message if missing


# Put the Kaggle files in task2/ (like task1/), or in task2/data/.
TRAIN_PATH = _find("train.csv", TASK_DIR, TASK_DIR / "data")
TEST_PATH = _find("test.csv", TASK_DIR, TASK_DIR / "data")
SAMPLE_SUB_PATH = _find("sample_submissionMLgroup.csv", ROOT_DIR, TASK_DIR)

ID_COL = "player_id"
TARGET = "segment"
CLASS_NAMES = {0: "Casual", 1: "Grinder", 2: "Social", 3: "Whale"}

# Datacard states these are pure noise with no relation to the target.
NOISE_COLS = ["random_metric_1", "random_metric_2", "random_metric_3"]

CAT_COLS = [
    "region", "platform", "device_type", "payment_method", "language",
    "account_status", "player_type_tag", "engagement_level", "loyalty_tier",
    "skill_tier",
]

SEED = 42


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """All numeric feature columns (excludes id, target, categoricals)."""
    skip = {ID_COL, TARGET, *CAT_COLS}
    return [c for c in df.columns if c not in skip]
