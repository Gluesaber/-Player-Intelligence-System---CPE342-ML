"""Custom models for Task 2.

HistNB — histogram Naive Bayes.
EDA showed features are (almost exactly) independent *within* each segment
(max within-class |corr| ~ 0.02). Under that condition the Bayes-optimal
classifier is

    log P(class | x) = log P(class) + sum_j log P(x_j | class) + const

so we only need a good 1-D density per feature per class. We estimate it
non-parametrically with quantile histograms (no Gaussian assumption) plus
Laplace smoothing. A missing value simply contributes nothing — the
statistically correct treatment, with no imputation needed.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, ClassifierMixin


class HistNB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=32, alpha=1.0, discrete_max_unique=12, weight=1.0):
        self.n_bins = n_bins
        self.alpha = alpha                      # Laplace smoothing per bin
        self.discrete_max_unique = discrete_max_unique
        self.weight = weight                    # temperature on summed log-likelihood

    # -- helpers ----------------------------------------------------------
    def _bin(self, j, col: pd.Series) -> np.ndarray:
        """Map values to bin index; -1 for missing / unseen category."""
        kind, ref = self.spec_[j]
        if kind == "cat":
            mapped = pd.Series(col.astype(object).map(ref), index=col.index)
            return mapped.fillna(-1).to_numpy(dtype=int)
        v = col.to_numpy(dtype=float)
        idx = np.searchsorted(ref, v, side="right")   # 0..len(ref)
        idx[np.isnan(v)] = -1
        return idx

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yi = np.searchsorted(self.classes_, y)
        self.log_prior_ = np.log(np.bincount(yi, minlength=K) / len(y))
        self.columns_ = list(X.columns)
        self.spec_, self.tables_ = [], []
        for j, c in enumerate(self.columns_):
            col = X[c]
            numeric = pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype)
            if not numeric or col.nunique() <= self.discrete_max_unique:
                cats = pd.Series(col.dropna().unique())
                self.spec_.append(("cat", {v: i for i, v in enumerate(cats)}))
                n = len(cats)
            else:
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                edges = np.unique(np.nanquantile(col.to_numpy(dtype=float), qs))
                self.spec_.append(("num", edges))
                n = len(edges) + 1
            b = self._bin(j, col)
            ok = b >= 0
            counts = np.zeros((K, n))
            np.add.at(counts, (yi[ok], b[ok]), 1)
            self.tables_.append(self._log_table(self.spec_[j][0], counts))
        return self

    def _log_table(self, kind, counts):
        """Per-class log-probabilities of each bin (Laplace smoothed)."""
        counts = counts + self.alpha
        return np.log(counts / counts.sum(axis=1, keepdims=True))

    def log_likelihoods(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature class log-likelihoods, shape (n, n_features, K); 0 where missing."""
        out = np.zeros((len(X), len(self.columns_), len(self.classes_)))
        for j, c in enumerate(self.columns_):
            b = self._bin(j, X[c])
            ok = b >= 0
            out[ok, j, :] = self.tables_[j][:, b[ok]].T
        return out

    def decision_function(self, X):
        return self.log_prior_ + self.weight * self.log_likelihoods(X).sum(axis=1)

    def predict_proba(self, X):
        z = self.decision_function(X)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class SmoothNB(HistNB):
    """Naive Bayes with smoothed (kernel-style) per-class densities.

    Numeric features are cut into many fine quantile bins (default 200),
    then each class's bin counts are smoothed with a Gaussian kernel of
    width `sigma` bins. This is a binned kernel density estimate: it keeps
    the detail of fine bins without their noise, instead of choosing
    between a few coarse bins (HistNB) or averaging several binnings
    (AvgHistNB). Discrete features are handled exactly as in HistNB.
    """

    def __init__(self, n_bins=200, sigma=6.0, alpha=1.0, discrete_max_unique=12, weight=1.0):
        super().__init__(n_bins=n_bins, alpha=alpha, discrete_max_unique=discrete_max_unique, weight=weight)
        self.sigma = sigma

    def _log_table(self, kind, counts):
        if kind == "num" and self.sigma > 0:
            counts = gaussian_filter1d(counts, self.sigma, axis=1, mode="reflect")
        return super()._log_table(kind, counts)


class AvgHistNB(BaseEstimator, ClassifierMixin):
    """Average the log-posteriors of HistNB fitted with several bin counts.

    One histogram is sensitive to where its bin edges fall; averaging over
    several binnings is a cheap way to get a smoother density estimate
    (like an averaged shifted histogram) -> lower variance.
    """

    def __init__(self, bin_list=(10, 12, 14, 16, 20, 24, 28, 32), alpha=1.0):
        self.bin_list = bin_list
        self.alpha = alpha

    def fit(self, X, y):
        self.models_ = [HistNB(n_bins=b, alpha=self.alpha).fit(X, y) for b in self.bin_list]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        z = np.mean([m.decision_function(X) for m in self.models_], axis=0)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
