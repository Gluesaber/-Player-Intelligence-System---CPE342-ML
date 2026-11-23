"""Features-only model definitions, shared by tune_models.py and 04_model_no_leak.

Each model has a feature "view" (FeatureBuilder settings). None of the views
contain player_id: FeatureBuilder.transform drops it.
"""
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from catboost.utils import get_gpu_device_count
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, SplineTransformer

from data import SEED
from features import ORDINAL_MAPS
from models import AvgHistNB

VIEWS = {
    "nb": dict(ordinal=False, composites=False, ratios=False),  # raw values + text categories
    "lr": dict(composites=False, ratios=False),                 # raw values + ordinals
    "gb": dict(ratios=False),                                   # raw + ordinals + composite scores
}
MODEL_VIEW = {"AvgHistNB": "nb", "SplineLR": "lr", "LightGBM": "gb",
              "XGBoost": "gb", "CatBoost": "gb", "RandomForest": "gb"}

CAT_DEVICE = "GPU" if get_gpu_device_count() > 0 else "CPU"
BIN_LISTS = {
    "narrow": (12, 14, 16, 18, 20),
    "default": (10, 12, 14, 16, 20, 24, 28, 32),
    "wide": (8, 10, 12, 16, 20, 24, 32, 40, 48),
}


def _numeric_columns(X):
    return [c for c in X.columns if c not in ORDINAL_MAPS]


def _ordinal_columns(X):
    return [c for c in X.columns if c in ORDINAL_MAPS]


def build(name: str, params: dict):
    """Return an unfitted model. `params` holds tuned values only."""
    p = dict(params)
    if name == "AvgHistNB":
        return AvgHistNB(bin_list=BIN_LISTS[p["bins"]], alpha=p["alpha"])
    if name == "SplineLR":
        numeric = make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(subsample=50000, random_state=0),
            SplineTransformer(n_knots=p["n_knots"], degree=3))
        ordinal = make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                                OneHotEncoder(handle_unknown="ignore"))
        return make_pipeline(make_column_transformer((numeric, _numeric_columns), (ordinal, _ordinal_columns)),
                             LogisticRegression(C=p["C"], max_iter=2000))
    if name == "LightGBM":
        return lgb.LGBMClassifier(**p, bagging_freq=1, verbose=-1, random_state=SEED)
    if name == "XGBoost":
        return xgb.XGBClassifier(**p, objective="multi:softprob", tree_method="hist", random_state=SEED)
    if name == "CatBoost":
        return cb.CatBoostClassifier(**p, loss_function="MultiClass", task_type=CAT_DEVICE,
                                     bootstrap_type="Bernoulli", verbose=0, random_seed=SEED,
                                     allow_writing_files=False)
    if name == "RandomForest":
        return RandomForestClassifier(**p, n_estimators=300, n_jobs=-1, random_state=SEED)
    raise ValueError(name)
