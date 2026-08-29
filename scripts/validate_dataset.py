"""Sanity check on the generated dataset.

Answers one question: does the synthetic estate carry recoverable signal
*without* being trivially learnable? A generator that produces a dataset a
model nails at 0.99 has encoded the answer in the features and proves nothing.

Run:  python scripts/validate_dataset.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sgw_platform.datagen.features import scenario_split

warnings.filterwarnings("ignore")

DROP = ["scenario_id", "asset_id", "failed", "functional_loss",
        "failure_bucket", "expected_loss", "scenario_year"]


def main(data_dir: str = "data"):
    df = pd.read_parquet(f"{data_dir}/training_table.parquet")
    sc = pd.read_csv(f"{data_dir}/scenarios.csv", parse_dates=["start"])
    tr, te = scenario_split(df, sc, test_frac=0.25)
    print(f"train {len(tr):,} rows ({tr.failed.mean()*100:.2f}% positive), "
          f"{tr.scenario_id.nunique()} scenarios")
    print(f"test  {len(te):,} rows ({te.failed.mean()*100:.2f}% positive), "
          f"{te.scenario_id.nunique()} scenarios  [chronological holdout]")

    feats = [c for c in tr.columns if c not in DROP]
    cat = [c for c in feats
           if not pd.api.types.is_numeric_dtype(tr[c])
           or c in ("zone_id", "criticality_tier")]
    for c in cat:
        tr[c] = tr[c].astype("category")
        te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)

    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=48, min_child_samples=60,
                               verbose=-1, random_state=7)
    model.fit(tr[feats], tr.failed, categorical_feature=cat)
    p = model.predict_proba(te[feats])[:, 1]

    # Baselines the client already has. A model must beat these to be worth
    # deploying, and quoting a metric without them is meaningless.
    baselines = {
        "prevalence (null)": np.full(len(te), tr.failed.mean()),
        "heuristic: old + windy": (((te.wear_ratio.fillna(0.6) > 0.8)
                                    & (te.wind_gust_ms_max > 25)).astype(float)
                                   * 0.05 + 0.001),
        "condition score only": (100 - te.condition_score.fillna(50)) / 100 * 0.02,
        "LightGBM": p,
    }
    print(f"\n{'model':<26}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>10}")
    for name, pred in baselines.items():
        print(f"{name:<26}{average_precision_score(te.failed, pred):>9.4f}"
              f"{roc_auc_score(te.failed, pred):>9.4f}"
              f"{brier_score_loss(te.failed, pred):>10.5f}")

    print("\nper-hazard PR-AUC (test):")
    t2 = te.assign(p=p)
    for h, g in t2.groupby("hazard", observed=True):
        if g.failed.sum() >= 3:
            print(f"  {h:<14} n={len(g):>6} pos={int(g.failed.sum()):>4} "
                  f"PR-AUC={average_precision_score(g.failed, g.p):.4f}")

    print("\ncalibration by decile of predicted risk (test):")
    t2["decile"] = pd.qcut(t2.p, 10, labels=False, duplicates="drop")
    cal = t2.groupby("decile").agg(predicted=("p", "mean"),
                                   observed=("failed", "mean"),
                                   n=("failed", "size"))
    print(cal.round(5).to_string())
    print("\nNote: the model is uncalibrated -- middle deciles under-predict and "
          "the top decile over-predicts.\nCalibrating it is a real, "
          "demonstrable improvement, not a formality.")

    print("\ntop 12 features by gain:")
    print(pd.Series(model.feature_importances_, index=feats).nlargest(12).to_string())


if __name__ == "__main__":
    main()
