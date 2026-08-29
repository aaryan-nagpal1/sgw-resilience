"""W1 pipeline: clean -> train -> calibrate -> evaluate -> figures.

    python scripts/run_w1.py
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sgw_platform.datagen.features import build_training_table
from sgw_platform.w1_risk import baselines, evaluate as ev, model as M
from sgw_platform.w1_risk.cleaning import (clean_assets, clean_work_orders,
                                           score_cleaning)

DATA = Path("data"); OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)


def main():
    print("=" * 74); print("W1  MULTI-HAZARD ASSET RISK"); print("=" * 74)

    # ---------------------------------------------------------------- clean
    gis = pd.read_csv(DATA / "gis_asset_extract.csv")
    truth = pd.read_csv(DATA / "_assets_truth.csv")
    wo = pd.read_csv(DATA / "cmms_work_orders.csv")
    clean, report = clean_assets(gis)
    _, wo_stats = clean_work_orders(wo, clean)
    scores = score_cleaning(clean, truth)

    print("\n--- 1. data cleaning -------------------------------------------")
    print(report.to_frame()[["step", "records_detected", "action"]].to_string(index=False))
    print(f"\norphan work orders: {wo_stats['orphan_work_orders']} "
          f"({wo_stats['orphan_share']*100:.1f}%), of which "
          f"{wo_stats['lost_failure_labels']} are corrective -- "
          f"failure labels that cannot be used")
    print("\ncleaning accuracy, scored against held-back truth:")
    print(scores.to_string(index=False))
    report.to_frame().to_csv(OUT / "w1_cleaning_report.csv", index=False)
    scores.to_csv(OUT / "w1_cleaning_scores.csv", index=False)

    # ------------------------------------------------------------- features
    exposure = pd.read_parquet(DATA / "hazard_exposure.parquet")
    failures = pd.read_csv(DATA / "failure_events.csv")
    cons = pd.read_csv(DATA / "asset_consequence.csv")
    scen = pd.read_csv(DATA / "scenarios.csv", parse_dates=["start"])

    # Rebuild the training table from the *cleaned* assets, and add the
    # cleaning provenance flags as features -- the model should know when a
    # value was imputed rather than measured.
    c = clean[~clean.is_duplicate_record].copy()
    c["install_year"] = c.install_year_clean
    c["material"] = c.material_clean
    c["diameter_mm"] = c.diameter_mm_clean
    tt = build_training_table(c, exposure, failures, cons, scen)
    flags = c[["asset_id", "install_year_imputed", "position_low_confidence",
               "condition_stale"]]
    tt = tt.merge(flags, on="asset_id", how="left")

    tr, cal, te = M.three_way_split(tt, scen)
    print(f"\n--- 2. split (chronological, by scenario) ----------------------")
    print(f"train {len(tr):>7,} rows / {tr.scenario_id.nunique():>3} scenarios "
          f"({tr.failed.sum():>4} failures)")
    print(f"calib {len(cal):>7,} rows / {cal.scenario_id.nunique():>3} scenarios "
          f"({cal.failed.sum():>4} failures)")
    print(f"test  {len(te):>7,} rows / {te.scenario_id.nunique():>3} scenarios "
          f"({te.failed.sum():>4} failures)")

    # ---------------------------------------------------------------- train
    rm = M.train(tr, cal)
    pr = M.predict_frame(rm, te)
    # Platt is the production calibrator: it preserves ranking exactly (it is
    # strictly monotonic) and gave the best ECE and Brier of the three tested.
    raw, calib = pr["raw"], pr["platt"]

    per_hz = M.train_per_hazard(tr, cal)
    p_hz = M.predict_per_hazard(per_hz, te)

    preds = baselines.all_baselines(tr, te)
    preds["LightGBM (uncalibrated)"] = raw
    preds["LightGBM + Platt"] = pr["platt"]
    preds["LightGBM + isotonic"] = pr["isotonic"]
    preds["LightGBM + isotonic, ties broken"] = pr["isotonic_tiebreak"]
    preds["Per-hazard models"] = p_hz

    print("\n--- 3. model vs baselines (test set) ---------------------------")
    table = ev.comparison_table(te, preds)
    print(table.to_string(index=False))
    table.to_csv(OUT / "w1_comparison.csv", index=False)

    print("\n--- 4. calibration ---------------------------------------------")
    for nm, p in [("uncalibrated", raw), ("Platt", pr["platt"]),
                  ("isotonic", pr["isotonic"]),
                  ("isotonic + tiebreak", pr["isotonic_tiebreak"])]:
        print(f"  {nm:<14} ECE={ev.expected_calibration_error(te.failed, p):.5f}  "
              f"Brier={ev.core_metrics(te.failed, p)['Brier']:.5f}")
    rel_raw = ev.reliability(te.failed, raw); rel_cal = ev.reliability(te.failed, calib)
    rel_raw.to_csv(OUT / "w1_reliability_raw.csv", index=False)
    rel_cal.to_csv(OUT / "w1_reliability_calibrated.csv", index=False)

    print("\n--- 5. per-hazard (calibrated) ---------------------------------")
    ph = ev.per_hazard(te, calib); print(ph.to_string(index=False))
    ph.to_csv(OUT / "w1_per_hazard.csv", index=False)

    # ------------------------------------------------------------ risk output
    te_out = te[["scenario_id", "asset_id", "asset_class", "hazard", "zone_id",
                 "consequence_score", "customers_affected",
                 "cascade_population_water", "critical_facilities_affected",
                 "failed"]].copy()
    te_out["p_failure"] = calib
    te_out["risk"] = te_out.p_failure * te_out.consequence_score
    te_out.to_parquet(OUT / "w1_risk_scores.parquet", index=False)
    print(f"\nwrote {OUT/'w1_risk_scores.parquet'} ({len(te_out):,} rows)")

    # top risks in the single worst test scenario -- the demo view
    # ------------------------------------------------------------- figures
    from sgw_platform.w1_risk import figures as F
    FIG = Path("figures")
    ece_raw = ev.expected_calibration_error(te.failed, raw)
    ece_cal = ev.expected_calibration_error(te.failed, calib)
    F.reliability_plot(rel_raw, ev.reliability(te.failed, calib),
                       FIG / "w1_calibration.png", ece_raw, ece_cal)
    F.baseline_bars(table, FIG / "w1_baselines.png")
    F.per_hazard_bars(ph, FIG / "w1_per_hazard.png")
    F.importance_plot(rm.features, rm.booster.feature_importances_,
                      FIG / "w1_importance.png")
    cons_full = pd.read_csv(DATA / "asset_consequence.csv").merge(
        c[["asset_id", "asset_class"]], on="asset_id")
    F.cascade_plot(cons_full, FIG / "cascade_consequence.png")
    print(f"\nfigures written to {FIG}/")

    worst = te_out.groupby("scenario_id").failed.sum().idxmax()
    top = (te_out[te_out.scenario_id == worst]
           .nlargest(8, "risk")[["asset_id", "asset_class", "p_failure",
                                 "consequence_score", "cascade_population_water", "risk"]])
    print(f"\n--- 6. top risks in worst test scenario ({worst}) --------------")
    print(top.round(4).to_string(index=False))
    return rm, te, calib


if __name__ == "__main__":
    main()
