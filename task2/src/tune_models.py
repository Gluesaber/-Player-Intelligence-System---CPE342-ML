"""Optuna tuning for the features-only Task 2 models.

Leak-free by construction:
  * player_id is never a feature
  * FeatureBuilder is fitted on the training part of each fold only
  * tuning uses its own 3 folds (seed 2026), different from the 5 folds
    (seed 42) that 04_model_no_leak uses for the reported scores

Objective: macro F1, the competition metric. Boosting rounds are chosen by
early stopping and saved as the mean best round across folds.

Results are saved after each model, so an interrupted run keeps its progress.

Usage:  python tune_models.py [seconds_per_model] [model names...]
"""
import json
import sys
import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score

from data import ID_COL, TARGET, TASK_DIR, load_train
from evaluate import get_folds
from features import FeatureBuilder
from model_zoo import MODEL_VIEW, VIEWS, build

OUT = TASK_DIR / "artifacts" / "tuned_params.json"
TUNE_SEED = 2026
LR, MAX_ROUNDS, PATIENCE = 0.05, 2000, 50

train = load_train()
y = train[TARGET].to_numpy()
folds = get_folds(y, n_splits=3, seed=TUNE_SEED)


def prepare(view):
    data = []
    for tr, va in folds:
        fb = FeatureBuilder(**VIEWS[view]).fit(train.iloc[tr])
        Xtr, Xva = fb.transform(train.iloc[tr]), fb.transform(train.iloc[va])
        assert ID_COL not in Xtr.columns
        data.append((Xtr, y[tr], Xva, y[va]))
    return data


def space(name, t):
    if name == "AvgHistNB":
        return {"bins": t.suggest_categorical("bins", ["narrow", "default", "wide"]),
                "alpha": t.suggest_float("alpha", 0.1, 30, log=True)}
    if name == "SplineLR":
        return {"C": t.suggest_float("C", 0.02, 5, log=True), "n_knots": t.suggest_int("n_knots", 4, 9)}
    if name == "LightGBM":
        return {"learning_rate": LR, "n_estimators": MAX_ROUNDS,
                "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
                "feature_fraction": t.suggest_float("feature_fraction", 0.2, 0.9),
                "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l2": t.suggest_float("lambda_l2", 1e-3, 50, log=True)}
    if name == "XGBoost":
        return {"learning_rate": LR, "n_estimators": MAX_ROUNDS, "early_stopping_rounds": PATIENCE,
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 100, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 0.9),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True)}
    if name == "CatBoost":
        return {"learning_rate": LR, "iterations": MAX_ROUNDS,
                "depth": t.suggest_int("depth", 2, 8),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 50, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0)}
    if name == "RandomForest":
        return {"max_depth": t.suggest_int("max_depth", 8, 24),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 60, log=True),
                "max_features": t.suggest_float("max_features", 0.05, 0.5),
                "max_samples": t.suggest_float("max_samples", 0.3, 1.0),
                "class_weight": t.suggest_categorical("class_weight", [None, "balanced_subsample"])}
    raise ValueError(name)


def fit_fold(name, model, Xtr, ytr, Xva, yva):
    """Fit one fold; return validation macro F1 and the best boosting round (or None)."""
    if name == "LightGBM":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        best = model.best_iteration_
    elif name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best = model.best_iteration + 1
    elif name == "CatBoost":
        model.fit(Xtr, ytr, eval_set=cb.Pool(Xva, yva), early_stopping_rounds=PATIENCE)
        best = model.get_best_iteration() + 1
    else:
        model.fit(Xtr, ytr)
        best = None
    return f1_score(yva, np.asarray(model.predict(Xva)).ravel(), average="macro"), best


def tune(name, seconds):
    data = prepare(MODEL_VIEW[name])

    def objective(t):
        params = space(name, t)
        scores, rounds = [], []
        for Xtr, ytr, Xva, yva in data:
            s, r = fit_fold(name, build(name, params), Xtr, ytr, Xva, yva)
            scores.append(s); rounds.append(r)
        t.set_user_attr("params", params)
        if rounds[0] is not None:
            t.set_user_attr("rounds", int(np.mean(rounds)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, timeout=seconds, callbacks=[
        lambda s, t: print(f"  {name} trial {t.number:3d}  F1 {t.value:.4f}  best {s.best_value:.4f}", flush=True)])

    best = dict(study.best_trial.user_attrs["params"])
    rounds = study.best_trial.user_attrs.get("rounds")
    best.pop("early_stopping_rounds", None)
    if rounds is not None:
        best["iterations" if name == "CatBoost" else "n_estimators"] = rounds
    return {"params": best, "tune_f1": study.best_value, "n_trials": len(study.trials)}


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    names = sys.argv[2:] or list(MODEL_VIEW)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        t0 = time.time()
        results[name] = tune(name, seconds)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"{name}: F1 {results[name]['tune_f1']:.4f}  trials {results[name]['n_trials']}  "
              f"{time.time() - t0:.0f}s  {results[name]['params']}", flush=True)
