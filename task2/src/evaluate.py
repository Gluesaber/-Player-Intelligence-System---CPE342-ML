"""Shared cross-validation helper so every experiment is scored the same way."""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from data import ID_COL, SEED
from features import FeatureBuilder

N_SPLITS = 5


def get_folds(y, n_splits=N_SPLITS, seed=SEED):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def cv_fit_in_fold(model, train, y, view, folds, verbose=True):
    """Like cv_model, but fits the FeatureBuilder on each fold's training part
    only, so no validation-row statistics reach the features.
    Returns (oof_proba, macro_f1, per-fold f1 list)."""
    y = np.asarray(y)
    oof = np.zeros((len(y), len(np.unique(y))))
    fold_f1 = []
    t0 = time.time()
    for tr, va in folds:
        fb = FeatureBuilder(**view).fit(train.iloc[tr])
        Xtr, Xva = fb.transform(train.iloc[tr]), fb.transform(train.iloc[va])
        if ID_COL in Xtr.columns:
            raise ValueError("player_id must not be a feature")
        oof[va] = clone(model).fit(Xtr, y[tr]).predict_proba(Xva)
        fold_f1.append(f1_score(y[va], oof[va].argmax(1), average="macro"))
    score = f1_score(y, oof.argmax(1), average="macro")
    if verbose:
        print(f"macro F1 {score:.4f}  (folds {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f})"
              f"  [{Xtr.shape[1]} feats, {time.time() - t0:.0f}s]")
    return oof, score, fold_f1


def cv_model(model, X, y, folds=None, verbose=True):
    """Return (oof_proba [n, n_classes], macro_f1, per-fold f1 list)."""
    folds = folds or get_folds(y)
    y = np.asarray(y)
    oof = np.zeros((len(y), len(np.unique(y))))
    fold_f1 = []
    t0 = time.time()
    for tr, va in folds:
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])
        fold_f1.append(f1_score(y[va], oof[va].argmax(1), average="macro"))
    score = f1_score(y, oof.argmax(1), average="macro")
    if verbose:
        print(f"macro F1 {score:.4f}  (folds {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f})"
              f"  [{X.shape[1]} feats, {time.time() - t0:.0f}s]")
    return oof, score, fold_f1
