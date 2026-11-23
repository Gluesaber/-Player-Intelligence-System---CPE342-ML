"""Blending OOF probabilities and per-class weight tuning for macro F1.

Both steps are *fitted* on OOF predictions, so their CV score would be
optimistic if measured on the same rows. `nested_score` gives an honest
estimate: for each fold, fit blend + class weights on the other folds' OOF
and score on the held-out fold.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss


# ---- blending -------------------------------------------------------------
def blend(probs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Geometric (log-space) blend; suits well-calibrated probability models."""
    z = sum(wi * np.log(np.clip(p, 1e-12, 1)) for wi, p in zip(w, probs))
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_blend_weights(probs, y) -> np.ndarray:
    """Non-negative weights minimising log-loss (weights need not sum to 1:
    the total also acts as a temperature)."""
    k = len(probs)
    res = minimize(lambda w: log_loss(y, blend(probs, np.abs(w))), np.full(k, 1 / k),
                   method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
    return np.abs(res.x)


# ---- class weights ---------------------------------------------------------
def apply_class_weights(p, cw):
    return (p * cw).argmax(axis=1)


def fit_class_weights(p, y, grid=np.exp(np.linspace(-0.7, 0.7, 29)), n_rounds=3):
    """Coordinate ascent on per-class multipliers to maximise macro F1."""
    cw = np.ones(p.shape[1])
    best = f1_score(y, apply_class_weights(p, cw), average="macro")
    for _ in range(n_rounds):
        for k in range(1, p.shape[1]):          # class 0 fixed as reference
            for g in grid:
                trial = cw.copy(); trial[k] = g
                s = f1_score(y, apply_class_weights(p, trial), average="macro")
                if s > best + 1e-6:
                    best, cw = s, trial
    return cw, best


# ---- stacking --------------------------------------------------------------
def stack_features(probs):
    """Level-2 inputs: log-probabilities of every level-1 model, side by side."""
    return np.hstack([np.log(np.clip(p, 1e-12, 1)) for p in probs])


def fit_stacker(probs, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=3000).fit(stack_features(probs), y)


def nested_stack_score(probs, y, folds, use_cw=True, C=1.0):
    """Meta-model trained on 4 folds of OOF predictions, scored on the 5th."""
    y = np.asarray(y)
    pred = np.zeros(len(y), dtype=int)
    feats = stack_features(probs)
    for tr, va in folds:
        meta = fit_stacker([p[tr] for p in probs], y[tr], C)
        cw = fit_class_weights(meta.predict_proba(feats[tr]), y[tr])[0] if use_cw else np.ones(probs[0].shape[1])
        pred[va] = apply_class_weights(meta.predict_proba(feats[va]), cw)
    return f1_score(y, pred, average="macro")


# ---- honest evaluation -----------------------------------------------------
def nested_score(probs, y, folds, use_blend=True, use_cw=True):
    """Fit blend/class weights on 4 folds of OOF, score on the 5th."""
    y = np.asarray(y)
    pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        ptr = [p[tr] for p in probs]
        w = fit_blend_weights(ptr, y[tr]) if (use_blend and len(probs) > 1) else np.ones(len(probs)) / len(probs)
        cw = fit_class_weights(blend(ptr, w), y[tr])[0] if use_cw else np.ones(probs[0].shape[1])
        pred[va] = apply_class_weights(blend([p[va] for p in probs], w), cw)
    return f1_score(y, pred, average="macro")
