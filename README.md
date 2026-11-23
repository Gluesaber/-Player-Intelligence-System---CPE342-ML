# Player Intelligence System - CPE342 ML

Repository for CPE342 Machine Learning (1/2026) — KMUTT.

## Repository Structure
- `task1/` — Anti-Cheat Pre-Filter (Binary Classification, Metric: Macro F2)
  - `task1.ipynb` — Full EDA, Feature Engineering, 5-Fold CV & V4.5 Leak-Free Champion Ensemble (`OOF Macro F2 = 0.78179`, `Nested 5-Fold = 0.78096`)
- `task2/` — Player Segment Classification (Multi-class, Metric: Macro F1)
  - `notebooks/01_eda` → `02_features` → `03_model` → `04_optimizing`, with shared code in `src/` (see `task2/README.md`). Final: smoothed Naive Bayes + spline LR blend (`5-Fold Macro F1 = 0.7574`)
- `task3/` — Monthly Spending Prediction
- `task4/` — Game Title Detection
- `task5/` — Account Security Monitoring
