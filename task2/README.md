# Task 2: Player Segment Classification

Classify each player as **Casual (0)**, **Grinder (1)**, **Social (2)** or **Whale (3)**. The metric is **macro F1**.

**Result:** smoothed Naive Bayes, blended with a spline logistic regression and per-class weights.
5-fold CV macro F1 **0.7574**; public leaderboard ≈ 0.755 for Task 2.

## How to run
1. Install the packages: `pip install -r requirements.txt` (Python 3.11).
2. Put the Kaggle files `train.csv` and `test.csv` in this `task2/` folder, the same way as `task1/`. A `task2/data/` subfolder also works.
   To write submission files, also put `sample_submissionMLgroup.csv` in the repo root or in `task2/`.
3. Run the notebooks in order from `notebooks/`. `03_model` needs tuned settings, so run `python src/tune_models.py` once first. It takes about 40 minutes and writes `artifacts/tuned_params.json`.

The data, tuned settings, saved predictions and submission files are not in git. They're recreated by running the steps above.

## Notebooks
| Notebook | What it does | Key result |
|---|---|---|
| `01_eda.ipynb` | Class balance, missing values, data quality, per-segment feature differences, train vs test drift | Segments form a 2×2 grid (engagement × spending). Train and test match (adversarial AUC 0.50). |
| `02_features.ipynb` | Tests feature ideas one at a time; finds why models stop improving | Features are independent within each segment, so Naive Bayes fits best |
| `03_model.ipynb` | Tunes and compares 6 models; blending and class weights scored with nested CV | Averaged histogram Naive Bayes, 0.7562 |
| `04_optimizing.ipynb` | Smoothed densities, SMOTE, balanced weights, stacking, blends | Best 0.7574; SMOTE and balanced weights hurt |

## Code (`src/`)
| File | Contents |
|---|---|
| `data.py` | Paths, column lists, loaders |
| `features.py` | `FeatureBuilder`: drops noise columns, ordinal-encodes tiers, builds composite engagement/spending scores |
| `models.py` | `HistNB`, `AvgHistNB`, `SmoothNB`: Naive Bayes with histogram / smoothed densities, written from scratch |
| `model_zoo.py` | Definitions of the six tuned models |
| `tune_models.py` | Optuna tuning on separate folds |
| `evaluate.py` | Leak-free cross-validation (features fitted inside each fold) |
| `postprocess.py` | Blending, per-class weights, stacking, nested scoring |
| `submission.py` | Writes the Task 2 prediction files |

## Model comparison (5-fold macro F1)
| Model | Score |
|---|---|
| SmoothNB + Spline LR blend, class weights | **0.7574** |
| Smoothed Naive Bayes (`SmoothNB`) | 0.7570 |
| Averaged histogram Naive Bayes | 0.7562 |
| Spline logistic regression | 0.7548 |
| XGBoost | 0.7544 |
| LightGBM | 0.7537 |
| CatBoost | 0.7536 |
| Random Forest | 0.7452 |

## Leakage safeguards
- `player_id` is never a feature.
- Feature statistics are fitted on each fold's training part only.
- Hyperparameters are tuned on different folds (seed 2026) from the ones used for scoring (seed 42).
- Blend weights, stacking and class weights are scored with nested CV.
- The test set is never used for fitting.

## Why the score stops near 0.757
- The learning curve flattens: doubling the training data adds about +0.0016.
- Features are independent within each segment, which makes Naive Bayes close to the best possible model. A LightGBM check could not tell real rows from within-class-shuffled rows (AUC ≈ 0.50).
- About 2.7% of labels look randomly flipped: when the model is 99% confident it is right only about 97% of the time.
- Every model family tried lands at 0.745–0.757.

## `archive/`
The original Task 2 release leaked the label through `player_id`: IDs were issued in blocks by segment, so the ID alone gave 0.968 macro F1.
`archive/03_model_id_old_data.ipynb` keeps that analysis with its saved outputs. The organisers released corrected data on 13 Nov, so this notebook can't be re-run, and the ID is not used anywhere in the current pipeline.
