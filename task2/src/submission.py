"""Write Task 2 predictions in the team submission format.

Two outputs:
  * task2_<name>.csv        -> id, task2   (for the team's final merge)
  * full_<name>.csv         -> all 5 columns, other tasks filled with FIXED
                               placeholders so Task-2-only uploads are
                               comparable with each other on the leaderboard.

Row i of test.csv corresponds to row i of sample_submission (test ids are
P..., submission ids are ANS...). Confirmed by leaderboard uploads.
"""
import numpy as np
import pandas as pd

from data import SAMPLE_SUB_PATH, SUB_DIR

# Keep these constant across every Task-2-only upload.
PLACEHOLDERS = {"task1": 1, "task3": 0.0, "task4": 0, "task5": 1}


def save_submission(pred: np.ndarray, name: str) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    pred = np.asarray(pred).astype(int)
    if len(pred) != len(sample):
        raise ValueError(f"Expected {len(sample)} predictions, got {len(pred)}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": sample["id"], "task2": pred}).to_csv(
        SUB_DIR / f"task2_{name}.csv", index=False)

    full = sample[["id"]].copy()
    for col in ["task1", "task2", "task3", "task4", "task5"]:
        full[col] = pred if col == "task2" else PLACEHOLDERS[col]
    full.to_csv(SUB_DIR / f"full_{name}.csv", index=False)
    return full
