# Player Intelligence System - CPE342 ML

Repository for CPE342 Machine Learning (1/2026) — KMUTT.

## Repository Structure
- `task1/` — Anti-Cheat Pre-Filter (Binary Classification, Metric: Macro F2)
  - `task1.ipynb` — Full EDA, Feature Engineering, 5-Fold CV & V4.5 Leak-Free Champion Ensemble (`OOF Macro F2 = 0.78179`, `Nested 5-Fold = 0.78096`)
- `task2/` — Player Segment Classification (Multi-class, Metric: Macro F1)
  - `task2.ipynb` — EDA, player_id leak audit, Feature Engineering, from-scratch Naive Bayes vs 6 tuned models, SMOTE/stacking/blend tests, ceiling analysis & final prediction (`5-Fold Macro F1 = 0.7574`)
- `task3/` — Monthly Spending Prediction (Regression, Metric: NMAE)
  - `task3.ipynb` — EDA (48.1% zero-inflated spend, leak/identity audit), 3-pass algebraic & rule-based imputation (`historical_spending = total_transactions × avg_transaction_value`, `vip_status ≥ 3 ↔ is_premium_member`) + OOF segment classifier (92.1% acc), seasonal/monetary feature engineering, two-stage L1 hurdle & multiplicative log-ratio GBDT ensemble with segment-wise exact zero-inflated quantile optimization (`5-Fold OOF MAE = 2,605.09 THB`, `NMAE = 0.2482` / `0.00868`, Whale `F2 = 0.9123`)
- `task4/` — Game Title Detection (Image Classification, Metric: Macro F1)
  - `task4.ipynb` — EDA of image corruptions (rotation, color shift, noise) → GPU augmentation set from train statistics, near-duplicate-free group-split validation (DINOv2 embeddings + KMeans), 7 pretrained backbones (EfficientNetV2 / ConvNeXt / DINOv2 / DINOv3) and the best 4-model ensemble with TTA (`group-val Macro F1 = 0.9865`)
- `task5/` — Account Security Monitoring (Unsupervised Anomaly Detection, Metric: Macro F3)
  - `task5.ipynb` — EDA (missing/noise by period), temporal robust-z for sudden late-period changes (type A) + histogram-gap detection on account-level behavior incl. a bot group (type B), cross-checked with PCA / HBOS / Mahalanobis / One-Class SVM / Autoencoder, flags top 8% (`type A + B = 1,811 accounts, 7.0%`)
