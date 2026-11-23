"""Feature engineering for Task 2.

Key idea from EDA: segments form a 2x2 grid
    engagement (low/high)  x  spending (low/high)
    Casual = low/low, Grinder = high/low, Social = low/high, Whale = high/high.
Each raw column is only a weak, noisy signal for one axis, and columns are
nearly uncorrelated -> averaging many of them into one composite score per
axis should give a much stronger signal (and is robust to ~8% missing cells).

All fitted statistics are learned from TRAIN only (`FeatureBuilder.fit`) and
then applied to any frame, so the test set is never used for fitting.
"""
import numpy as np
import pandas as pd

from data import CAT_COLS, ID_COL, NOISE_COLS, TARGET

# --- column groups (from EDA section 5) ---------------------------------
# sign = +1 if higher value -> more engaged / more spending, -1 if the reverse
ENGAGEMENT_COLS = {
    "play_frequency": 1, "avg_session_duration": 1, "total_playtime_hours": 1,
    "login_streak": 1, "days_since_last_login": -1, "peak_concurrent_hours": 1,
    "friend_count": 1, "friend_invites_sent": 1, "gifts_sent_received": 1,
    "chat_activity_score": 1, "tournament_entries": 1,
    "speed_of_progression": 1, "competitive_rank": 1,
}
SPENDING_COLS = {
    "total_spending_thb": 1, "avg_monthly_spending": 1,
    "spending_frequency": 1, "rare_items_count": 1, "vip_tier": 1,
}

# Categoricals with clear signal -> ordinal; the rest showed none (EDA section 6)
ORDINAL_MAPS = {
    "loyalty_tier": {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3, "Diamond": 4},
    "engagement_level": {"Low": 0, "Mid": 1, "High": 2, "Veteran": 3},
}
USELESS_CATS = [c for c in CAT_COLS if c not in ORDINAL_MAPS]


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


class FeatureBuilder:
    """Configurable feature pipeline; flags let us ablate each idea with CV."""

    def __init__(self, drop_noise=True, drop_useless_cats=True, ordinal=True,
                 composites=True, ratios=True, keep_raw=True):
        self.drop_noise = drop_noise
        self.drop_useless_cats = drop_useless_cats
        self.ordinal = ordinal
        self.composites = composites
        self.ratios = ratios
        self.keep_raw = keep_raw

    # ---- fit: store sorted train values for percentile-rank transform ----
    def fit(self, train: pd.DataFrame):
        cols = list(ENGAGEMENT_COLS) + list(SPENDING_COLS)
        self.sorted_ = {c: np.sort(train[c].dropna().to_numpy()) for c in cols}
        return self

    def _pct_rank(self, s: pd.Series) -> pd.Series:
        ref = self.sorted_[s.name]
        # mid-rank handles ties in discrete columns (e.g. vip_tier)
        lo = np.searchsorted(ref, s.to_numpy(), side="left")
        hi = np.searchsorted(ref, s.to_numpy(), side="right")
        r = (lo + hi) / 2 / len(ref)
        return pd.Series(np.where(s.isna(), np.nan, r), index=s.index)

    def _composite(self, df, group: dict, name: str) -> pd.DataFrame:
        ranks = pd.DataFrame({c: self._pct_rank(df[c]) if sign > 0 else 1 - self._pct_rank(df[c])
                              for c, sign in group.items()})
        return pd.DataFrame({
            f"{name}_score": ranks.mean(axis=1),            # NaN-aware mean
            f"{name}_score_std": ranks.std(axis=1),         # consistency across signals
            f"{name}_n_obs": ranks.notna().sum(axis=1),     # how many signals we saw
        })

    # ---- transform -------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[c for c in (ID_COL, TARGET, "seg_name") if c in df.columns])
        new = []

        if self.composites:
            eng = self._composite(df, ENGAGEMENT_COLS, "eng")
            spd = self._composite(df, SPENDING_COLS, "spend")
            inter = pd.DataFrame({
                "eng_x_spend": eng["eng_score"] * spd["spend_score"],
                "eng_minus_spend": eng["eng_score"] - spd["spend_score"],
            })
            new += [eng, spd, inter]

        if self.ratios:
            new.append(pd.DataFrame({
                "spend_per_hour": _safe_div(X["total_spending_thb"], X["total_playtime_hours"]),
                "spend_per_txn": _safe_div(X["avg_monthly_spending"], X["spending_frequency"]),
                "spend_tenure_months": _safe_div(X["total_spending_thb"], X["avg_monthly_spending"]),
                "rare_per_spend": _safe_div(X["rare_items_count"], X["total_spending_thb"]) * 1000,
                "hours_per_day_played": _safe_div(X["total_playtime_hours"], X["account_age_days"]),
                "friends_per_hour": _safe_div(X["friend_count"], X["total_playtime_hours"]) * 100,
                "gifts_per_friend": _safe_div(X["gifts_sent_received"], X["friend_count"]),
                "session_x_freq": X["avg_session_duration"] * X["play_frequency"],
            }))

        if self.ordinal:
            for c, m in ORDINAL_MAPS.items():
                X[c] = X[c].map(m).astype(float)

        drop = []
        if self.drop_noise:
            drop += NOISE_COLS
        if self.drop_useless_cats:
            drop += USELESS_CATS
        if not self.keep_raw:  # keep only engineered + ordinals
            drop += [c for c in list(ENGAGEMENT_COLS) + list(SPENDING_COLS) if c in X]
        X = X.drop(columns=[c for c in drop if c in X.columns])

        X = pd.concat([X, *new], axis=1)
        for c in X.columns:  # remaining text columns -> pandas category for LightGBM
            if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
                X[c] = X[c].astype("category")
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)
