"""W4: condition monitoring on instrumented pumping assets.

This is the blue-sky-day workflow. It exists because a platform used only
during named storms is used four days a year, and a tool nobody opens on a
Tuesday will not have its data maintained or its users fluent when it matters.

Method is deliberately simple and explainable: seasonal-trend decomposition of
the efficiency signal, then a robust z-score on the residual. Utilities already
have threshold alarms; the complaint is that thresholds fire once the pump has
already failed. Detecting the *drift* is the thing that buys lead time.

Two failure modes are handled explicitly because both occur in real historians:
sensor dropout (gaps) and stuck sensors (variance collapses to zero, which
fools any detector keyed on deviation).

CAVEAT ON THE RESULTS. Recall and precision both come out at 1.0 on the
synthetic fleet. That is a property of the generator, not evidence the method
is that good: injected bearing wear is a clean ~35% fall in efficiency against
~5% measurement noise, i.e. a 7-sigma signal. Real degradation is noisier,
partially masked by duty changes, and sometimes not present in the channels
that happen to be instrumented. Treat these numbers as an upper bound and the
method as demonstrated, not validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESAMPLE = "6h"
MAD_TO_SIGMA = 1.4826
ALERT_Z = 3.5
PERSISTENCE = 3          # consecutive windows above threshold before alerting


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = MAD_TO_SIGMA * mad
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(x)
    return (x - med) / scale


def pump_efficiency(df: pd.DataFrame) -> pd.Series:
    """Flow delivered per unit of electrical current drawn.

    Chosen over raw vibration because it is robust to the pump simply being
    asked to do more work: demand rises in a heatwave and so do flow and
    current together, leaving the ratio stable. A detector on raw current
    would fire on every hot day.
    """
    cur = df.motor_current_a.replace(0, np.nan)
    return df.flow_m3_h / cur


def detect_stuck_sensor(s: pd.Series, window=32, tol=1e-9) -> pd.Series:
    """Flag stretches where a sensor stops changing. Zero variance reads as
    'perfectly healthy' to a deviation-based detector, so it must be caught
    separately or it silently suppresses alerts."""
    return s.rolling(window).std().fillna(1.0) < tol


def analyse_asset(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").set_index("timestamp")
    stuck = detect_stuck_sensor(g.discharge_pressure_bar)
    eff = pump_efficiency(g)

    r = eff.resample(RESAMPLE).median()
    coverage = eff.notna().resample(RESAMPLE).mean()
    stuck_w = stuck.resample(RESAMPLE).mean()

    # Compare against the asset's own early-life behaviour, NOT against a
    # rolling window of itself. An earlier version detrended with a centred
    # rolling median and detected nothing -- it was subtracting out the slow
    # drift that IS the degradation. Bearing wear is a trend, so the reference
    # has to be a fixed healthy period, not a moving one.
    #
    # Per-asset baselines also matter: pumps differ, and a fleet-wide threshold
    # would flag the inherently inefficient ones forever.
    n_base = max(len(r) // 4, 8)
    baseline = r.iloc[:n_base].dropna()
    base_med = baseline.median()
    base_mad = np.nanmedian(np.abs(baseline - base_med))
    scale = MAD_TO_SIGMA * base_mad if base_mad > 1e-9 else np.nan
    ratio = r / base_med
    z = ((r - base_med) / scale) if np.isfinite(scale) else pd.Series(0.0, index=r.index)

    out = pd.DataFrame({
        "efficiency": r.round(4), "efficiency_ratio": ratio.round(4),
        "z": z.round(3), "coverage": coverage.round(3),
        "stuck_fraction": stuck_w.round(3),
    })
    # Alert on sustained degradation only (z below -ALERT_Z for PERSISTENCE
    # windows). One bad reading is noise; three in a row is a trend.
    bad = (out.z < -ALERT_Z) & (out.coverage > 0.5) & (out.stuck_fraction < 0.5)
    out["alert"] = bad.rolling(PERSISTENCE).sum().fillna(0) >= PERSISTENCE
    out["data_quality_flag"] = np.where(
        out.coverage <= 0.5, "sparse",
        np.where(out.stuck_fraction >= 0.5, "stuck_sensor", "ok"))
    return out.reset_index()


def run(telemetry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-window detail, per-asset summary)."""
    details, summary = [], []
    for aid, g in telemetry.groupby("asset_id"):
        d = analyse_asset(g)
        d.insert(0, "asset_id", aid)
        details.append(d)
        first = d.loc[d.alert, "timestamp"]
        summary.append({
            "asset_id": aid,
            "alerted": bool(d.alert.any()),
            "first_alert": first.iloc[0] if len(first) else pd.NaT,
            "min_z": float(d.z.min()),
            "final_efficiency_ratio": float(d.efficiency_ratio.dropna().iloc[-1])
            if d.efficiency_ratio.notna().any() else np.nan,
            # Report the worst flag seen, not the most common one: a sensor
            # that was stuck for a tenth of the window is still a sensor
            # whose alerts cannot be trusted, and a modal "ok" hides that.
            "data_quality": ("stuck_sensor" if (d.data_quality_flag == "stuck_sensor").any()
                             else "sparse" if (d.data_quality_flag == "sparse").any()
                             else "ok"),
            "windows_degraded_or_suspect": int((d.data_quality_flag != "ok").sum()),
        })
    return pd.concat(details, ignore_index=True), pd.DataFrame(summary)


def evaluate(summary: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Score against the known degradation onsets, including lead time --
    the metric that actually determines whether the alert was useful."""
    m = summary.merge(labels, on="asset_id", how="left")
    tp = int((m.alerted & m.is_degrading).sum())
    fp = int((m.alerted & ~m.is_degrading).sum())
    fn = int((~m.alerted & m.is_degrading).sum())
    tn = int((~m.alerted & ~m.is_degrading).sum())
    hit = m[m.alerted & m.is_degrading].copy()
    lead = ((pd.to_datetime(hit.first_alert) - pd.to_datetime(hit.degradation_onset))
            .dt.total_seconds() / 86400.0)
    return {
        "instrumented_assets": len(m),
        "degrading": int(m.is_degrading.sum()),
        "detected": tp, "missed": fn, "false_alarms": fp, "correct_quiet": tn,
        "recall": round(tp / max(tp + fn, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "median_days_after_onset": round(float(lead.median()), 1) if len(lead) else None,
    }
