"""Evaluation for W1. Ranking, calibration, and the two operational metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def core_metrics(y, p):
    return {"PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p)}


def precision_at_k(y, p, k):
    idx = np.argsort(-p)[:k]
    return float(np.asarray(y)[idx].mean())


def loss_capture_at_k(df, p, k, loss_col="consequence_score"):
    """Of the consequence that actually materialised, what share sits in the
    top-k ranked assets? This is the metric an operator cares about: they have
    capacity for k interventions, not for a probability distribution."""
    realised = (df.failed.to_numpy() * df[loss_col].to_numpy())
    total = realised.sum()
    if total <= 0:
        return float("nan")
    idx = np.argsort(-p)[:k]
    return float(realised[idx].sum() / total)


def comparison_table(df, preds: dict, ks=(50, 200)):
    rows = []
    y = df.failed.to_numpy()
    for name, p in preds.items():
        p = np.asarray(p, dtype=float)
        r = {"model": name, **core_metrics(y, p)}
        for k in ks:
            r[f"precision@{k}"] = precision_at_k(y, p, k)
            r[f"loss capture@{k}"] = loss_capture_at_k(df, p, k)
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def reliability(y, p, bins=10):
    """Calibration curve on quantile bins (equal-width bins are useless when
    99.5% of the mass sits near zero)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    g = d.groupby("bin").agg(predicted=("p", "mean"),
                             observed=("y", "mean"), n=("y", "size"))
    return g.reset_index()


def expected_calibration_error(y, p, bins=10):
    r = reliability(y, p, bins)
    w = r.n / r.n.sum()
    return float((w * (r.predicted - r.observed).abs()).sum())


def per_hazard(df, p, min_pos=3):
    rows = []
    d = df.assign(_p=np.asarray(p))
    for h, g in d.groupby("hazard", observed=True):
        if g.failed.sum() >= min_pos:
            rows.append({"hazard": h, "n": len(g), "positives": int(g.failed.sum()),
                         "PR-AUC": average_precision_score(g.failed, g._p),
                         "ROC-AUC": roc_auc_score(g.failed, g._p)})
    return pd.DataFrame(rows).round(4)
