"""SCADA telemetry for instrumented pump stations.

Feeds the blue-sky-day anomaly detection workflow (W4). Most pumps run
normally; a minority degrade slowly over weeks before failing. The degradation
is gradual and partly masked by seasonality, which is what makes this an
anomaly-detection problem rather than a threshold alarm -- utilities already
have threshold alarms, and the complaint is that they fire too late.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def build_telemetry(rng, assets: pd.DataFrame):
    pumps = assets[assets.asset_class == "pump_station"]
    n_inst = max(1, int(len(pumps) * C.TELEMETRY_ASSET_FRACTION))
    inst = pumps.sample(n_inst, random_state=int(rng.integers(1e9)))

    end = pd.Timestamp(C.HISTORY_END)
    idx = pd.date_range(end - pd.Timedelta(days=C.TELEMETRY_DAYS), end,
                        freq=f"{C.TELEMETRY_FREQ_MIN}min")
    n = len(idx)
    hod = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear.to_numpy()

    # Diurnal demand: morning and evening peaks. Seasonal: summer irrigation.
    diurnal = (1.0 + 0.32 * np.exp(-((hod - 7.5) ** 2) / 4.5)
               + 0.40 * np.exp(-((hod - 19.0) ** 2) / 7.0)
               - 0.22 * np.exp(-((hod - 3.5) ** 2) / 6.0))
    seasonal = 1.0 + 0.20 * np.sin((doy - 100) / 365.0 * 2 * np.pi)

    n_degrade = max(1, int(n_inst * C.DEGRADING_FRACTION / C.TELEMETRY_ASSET_FRACTION))
    degrading = set(inst.asset_id.sample(
        min(n_degrade, n_inst), random_state=int(rng.integers(1e9))))

    frames, labels = [], []
    for r in inst.itertuples(index=False):
        rated = float(r.pump_rated_kw) if not np.isnan(r.pump_rated_kw) else 150.0
        base_flow = rated * 0.42
        flow = base_flow * diurnal * seasonal * rng.normal(1.0, 0.035, n)
        pressure = 4.2 - 0.0016 * (flow - base_flow) + rng.normal(0, 0.07, n)
        vibration = rng.gamma(4.0, 0.28, n)
        current = rated * 0.9 * diurnal * seasonal / 400.0 * rng.normal(1.0, 0.03, n)

        onset = np.nan
        if r.asset_id in degrading:
            # Bearing wear: vibration and current climb, efficiency (flow per
            # unit current) falls. Onset partway through the window.
            k = int(rng.uniform(0.35, 0.75) * n)
            onset = idx[k]
            ramp = np.clip((np.arange(n) - k) / max(n - k, 1), 0, 1) ** 1.7
            vibration *= 1.0 + 2.4 * ramp
            current *= 1.0 + 0.28 * ramp
            flow *= 1.0 - 0.17 * ramp
            pressure -= 0.55 * ramp

        frames.append(pd.DataFrame({
            "asset_id": r.asset_id, "timestamp": idx,
            "flow_m3_h": flow.round(2),
            "discharge_pressure_bar": pressure.round(3),
            "vibration_mm_s": vibration.round(3),
            "motor_current_a": current.round(2),
        }))
        labels.append({"asset_id": r.asset_id,
                       "is_degrading": r.asset_id in degrading,
                       "degradation_onset": onset})

    tel = pd.concat(frames, ignore_index=True)
    # Sensor dropout: real historians have gaps. ~1.2% of readings missing.
    drop = rng.random(len(tel)) < 0.012
    for col in ["flow_m3_h", "discharge_pressure_bar", "vibration_mm_s", "motor_current_a"]:
        tel.loc[drop, col] = np.nan
    # A few stuck sensors -- constant value for a stretch. Classic failure mode
    # that naive anomaly detectors miss because variance goes to zero.
    stuck_assets = inst.asset_id.sample(max(1, n_inst // 12),
                                        random_state=int(rng.integers(1e9)))
    for aid in stuck_assets:
        m = tel.asset_id == aid
        sub = tel.index[m][:int(m.sum() * 0.1)]
        tel.loc[sub, "discharge_pressure_bar"] = tel.loc[sub[0], "discharge_pressure_bar"]

    return tel, pd.DataFrame(labels)
