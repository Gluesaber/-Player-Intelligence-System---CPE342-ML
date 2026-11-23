"""ID-block classifier for Task 2.

`player_id` numbers were issued in contiguous blocks per segment
(Casual -> Grinder -> Social -> Whale), so the ID number alone predicts the
segment for ~97% of players. The remaining ~3% are random label flips whose
features still follow their ID block (see 03_model, leakage section).

NOTE: this exploits an artefact of how the dataset was generated (target
leakage via identifier). It would not work for real new players. It is kept
separate from the features-only models so both can be reported.

The block boundaries are *learned* from the training labels (not hard-coded):
for each boundary between consecutive classes we choose the cut point that
maximises training accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


def id_number(ids) -> np.ndarray:
    return pd.Series(ids).str[1:].astype(int).to_numpy()


class IDBlockClassifier(BaseEstimator, ClassifierMixin):
    """Expects X with a `player_id` column; predicts the class of the ID block."""

    def __init__(self, id_col="player_id", smoothing=0.97):
        self.id_col = id_col
        self.smoothing = smoothing   # probability mass given to the block's class

    def fit(self, X, y):
        n = id_number(X[self.id_col])
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        order = np.argsort(n)
        n, y = n[order], y[order]
        self.bounds_ = []
        for k in range(len(self.classes_) - 1):
            # cut between class k (left) and k+1 (right): maximise
            # (#k left of cut) + (#k+1 right of cut), searched over sorted IDs
            left = np.cumsum(y == self.classes_[k])
            right = np.cumsum((y == self.classes_[k + 1])[::-1])[::-1]
            score = left[:-1] + right[1:]
            if self.bounds_:  # only search to the right of the previous cut
                score[n[:-1] < self.bounds_[-1]] = -1
            i = int(np.argmax(score))
            self.bounds_.append((n[i] + n[i + 1]) / 2)
        self.bounds_ = np.array(self.bounds_)
        return self

    def predict(self, X):
        return self.classes_[np.searchsorted(self.bounds_, id_number(X[self.id_col]), side="right")]

    def predict_proba(self, X):
        K = len(self.classes_)
        idx = np.searchsorted(self.bounds_, id_number(X[self.id_col]), side="right")
        p = np.full((len(idx), K), (1 - self.smoothing) / (K - 1))
        p[np.arange(len(idx)), idx] = self.smoothing
        return p
