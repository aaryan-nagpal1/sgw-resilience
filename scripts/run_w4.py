"""W4 pipeline: SCADA condition monitoring."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
from sgw_platform.w4_anomaly import detect

DATA, ART = Path("data"), Path("artifacts")


def main():
    print("=" * 74); print("W4  CONDITION MONITORING (blue-sky day)"); print("=" * 74)
    tel = pd.read_parquet(DATA / "scada_telemetry.parquet")
    labels = pd.read_csv(DATA / "scada_labels.csv", parse_dates=["degradation_onset"])
    print(f"\n{tel.asset_id.nunique()} instrumented pumps, {len(tel):,} readings, "
          f"{tel.timestamp.min():%Y-%m-%d} to {tel.timestamp.max():%Y-%m-%d}")

    detail, summary = detect.run(tel)
    res = detect.evaluate(summary, labels)
    print("\n--- detection ---------------------------------------------------")
    for k, v in res.items():
        print(f"  {k:<28} {v}")
    print("\n--- data quality flags raised -----------------------------------")
    print(summary.data_quality.value_counts().to_string())

    alerts = summary[summary.alerted].merge(labels, on="asset_id", how="left")
    if len(alerts):
        print("\n--- alerting assets ---------------------------------------------")
        print(alerts[["asset_id", "first_alert", "min_z", "final_efficiency_ratio",
                      "is_degrading"]].head(10).to_string(index=False))
    detail.to_parquet(ART / "w4_detail.parquet", index=False)
    summary.to_csv(ART / "w4_summary.csv", index=False)
    pd.Series(res).to_csv(ART / "w4_metrics.csv")
    print(f"\nwrote artifacts/w4_summary.csv")
    return summary, res


if __name__ == "__main__":
    main()
