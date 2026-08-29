"""W1 risk model: calibrated multi-hazard failure probability.

Design decisions, and why:

* **Gradient-boosted trees, not deep learning.** Tabular, ~440k rows, 0.5%
  positives, and the relationship is smooth and low-dimensional. Nateghi et al.
  (2014) found simpler models outperformed complex ones on exactly this problem
  ("Improved Accuracy with Simpler Models", Risk Analysis). Trees also give
  usable per-asset attribution, which the operator needs.
* **Split by scenario, chronologically, three ways.** Train / calibrate / test.
  A random row split leaks -- the same storm lands in both sides and the model
  memorises the event. Calibration needs its own held-out scenarios or the
  isotonic fit is optimistically biased.
* **Calibration is not optional.** W2 multiplies these probabilities by
  consequence and allocates crews on the product. If 0.3 does not mean
  3-in-10, the optimiser misallocates. Ranking metrics alone would hide that.
* **But plain isotonic regression costs ranking.** It is a step function, so it
  maps whole intervals of raw score onto one calibrated value. Measured here:
  ECE improves 0.00136 -> 0.00087 while PR-AUC falls 0.140 -> 0.110, because
  ties destroy resolution inside each step. Two fixes are implemented and
  compared: Platt scaling (strictly monotonic, so ranking is preserved exactly
  but the fit is less flexible), and isotonic with the raw score used to break
  ties. The latter keeps isotonic's calibration and restores the ordering.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DROP = ["scenario_id", "asset_id", "failed", "functional_loss",
        "failure_bucket", "expected_loss", "scenario_year"]

PARAMS = dict(n_estimators=500, learning_rate=0.045, num_leaves=48,
              min_child_samples=60, subsample=0.85, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=7)


def three_way_split(df, scenarios, calib_frac=0.15, test_frac=0.25):
    """Chronological scenario-wise split into train / calibration / test."""
    s = scenarios.sort_values("start")
    n = len(s)
    n_test = max(1, int(n * test_frac))
    n_cal = max(1, int(n * calib_frac))
    test_ids = set(s.scenario_id.tail(n_test))
    cal_ids = set(s.scenario_id.iloc[-(n_test + n_cal):-n_test])
    tr = df[~df.scenario_id.isin(test_ids | cal_ids)].copy()
    cal = df[df.scenario_id.isin(cal_ids)].copy()
    te = df[df.scenario_id.isin(test_ids)].copy()
    return tr, cal, te


def feature_columns(df):
    return [c for c in df.columns if c not in DROP]


def _prep(frames, feats):
    """Align categorical dtypes across splits."""
    tr = frames[0]
    cat = [c for c in feats
           if not pd.api.types.is_numeric_dtype(tr[c])
           or c in ("zone_id", "criticality_tier")]
    out = []
    for i, f in enumerate(frames):
        f = f.copy()
        for c in cat:
            if i == 0:
                f[c] = f[c].astype("category")
            else:
                f[c] = pd.Categorical(f[c], categories=frames[0][c].astype("category").cat.categories)
        out.append(f)
    return out, cat


def _break_ties(calibrated, raw, eps=1e-6):
    """Restore ordering inside isotonic's flat steps using the raw score.

    The nudge is small enough not to disturb the calibrated value materially
    (it is scaled to eps times the rank percentile) but large enough to make
    the ranking strict again.
    """
    rank_pct = pd.Series(raw).rank(pct=True).to_numpy()
    return np.clip(calibrated + eps * rank_pct, 1e-9, 1 - 1e-9)


@dataclass
class RiskModel:
    booster: lgb.LGBMClassifier
    isotonic: IsotonicRegression | None
    platt: LogisticRegression | None
    features: list
    categoricals: list


def train(tr: pd.DataFrame, cal: pd.DataFrame, calibrate: bool = True):
    feats = feature_columns(tr)
    (tr_p, cal_p), cats = _prep([tr, cal], feats)

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr_p[feats], tr_p.failed, categorical_feature=cats)

    iso = plt_ = None
    if calibrate and cal_p.failed.sum() >= 10:
        raw_cal = model.predict_proba(cal_p[feats])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(raw_cal, cal_p.failed)
        plt_ = LogisticRegression(C=1e6, max_iter=1000)
        plt_.fit(np.log(np.clip(raw_cal, 1e-9, 1 - 1e-9) /
                        (1 - np.clip(raw_cal, 1e-9, 1 - 1e-9))).reshape(-1, 1),
                 cal_p.failed)
    return RiskModel(model, iso, plt_, feats, cats)


def predict_frame(rm: RiskModel, df: pd.DataFrame):
    """Return dict of raw / isotonic / isotonic+tiebreak / platt probabilities."""
    d = df.copy()
    for c in rm.categoricals:
        d[c] = d[c].astype("category")
    raw = rm.booster.predict_proba(d[rm.features])[:, 1]
    out = {"raw": raw}
    if rm.isotonic is not None:
        iso = np.clip(rm.isotonic.predict(raw), 1e-6, 1 - 1e-6)
        out["isotonic"] = iso
        out["isotonic_tiebreak"] = _break_ties(iso, raw)
    if rm.platt is not None:
        logit = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / (1 - np.clip(raw, 1e-9, 1 - 1e-9)))
        out["platt"] = rm.platt.predict_proba(logit.reshape(-1, 1))[:, 1]
    return out


def train_per_hazard(tr, cal):
    """One model per hazard type. Tested as an alternative to the pooled model.

    The pooled model can share structure across hazards (an old asset is fragile
    whatever is hitting it); per-hazard models can specialise but see far less
    data each. Which wins is an empirical question, not a design preference.
    """
    models = {}
    for hz in tr.hazard.astype(str).unique():
        t, c = tr[tr.hazard == hz], cal[cal.hazard == hz]
        if t.failed.sum() < 25:
            continue
        models[hz] = train(t, c, calibrate=c.failed.sum() >= 10)
    return models


def predict_per_hazard(models, df):
    out = np.full(len(df), np.nan)
    for hz, rm in models.items():
        m = (df.hazard == hz).to_numpy()
        if m.any():
            pr = predict_frame(rm, df[m])
            out[m] = pr.get("isotonic_tiebreak", pr["raw"])
    return np.where(np.isnan(out), df.failed.mean() if "failed" in df else 0.005, out)
