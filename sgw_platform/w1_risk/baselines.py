"""Baselines the risk model must beat.

A metric without a baseline is not a result. These are the four comparators:
what the client already has, what an experienced planner does by hand, and the
published engineering approach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Lognormal wind fragility parameters: (median capacity m/s, dispersion).
# HAZUS-style form P(fail | v) = Phi( ln(v/theta) / beta ). Values are
# representative rather than sourced from a specific published curve, and the
# limitation is stated: published curves are single-hazard and cover only the
# wind-exposed classes.
FRAGILITY = {
    "transmission_span":       (52.0, 0.42),
    "distribution_feeder":     (38.0, 0.45),
    "recloser":                (45.0, 0.40),
    "transmission_substation": (58.0, 0.35),
    "distribution_substation": (54.0, 0.38),
}


def prevalence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The null model."""
    return np.full(len(test), train.failed.mean())


def condition_score(test: pd.DataFrame) -> np.ndarray:
    """What the CMMS already gives them: rank by how bad the asset looks."""
    return (100 - test.condition_score.fillna(50)) / 100 * 0.02


def planner_heuristic(test: pd.DataFrame) -> np.ndarray:
    """What an experienced resilience planner does in their head:
    old assets in the path of high wind."""
    old = test.wear_ratio.fillna(0.6) > 0.8
    windy = test.wind_gust_ms_max > 25
    return np.where(old & windy, 0.05, 0.001)


def wind_fragility(test: pd.DataFrame) -> np.ndarray:
    """Published-style parametric fragility curve. No learning involved.

    This is the engineering standard-of-practice comparator. It is strong on
    wind-driven classes during storms and blind everywhere else, which is
    precisely the gap a multi-hazard learned model is meant to close.
    """
    v = test.wind_gust_ms_max.to_numpy()
    out = np.full(len(test), 1e-4)
    cls = test.asset_class.astype(str).to_numpy()
    for name, (theta, beta) in FRAGILITY.items():
        m = cls == name
        if m.any():
            out[m] = norm.cdf(np.log(np.maximum(v[m], 1e-3) / theta) / beta)
    return np.clip(out, 1e-6, 1 - 1e-6)


def all_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "Prevalence (null)": prevalence(train, test),
        "Condition score (current practice)": condition_score(test),
        "Planner heuristic (old + windy)": planner_heuristic(test),
        "Wind fragility curve (published form)": wind_fragility(test),
    }
