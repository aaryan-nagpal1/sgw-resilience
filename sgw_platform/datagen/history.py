"""Generative failure model and three years of synthetic operating history.

This module is the ground truth a downstream risk model has to recover. It is
deliberately built so that recovery is possible but not trivial:

  * several hazard pathways act on different asset classes, with interactions
    (age x inundation, vegetation x wind, load factor x temperature);
  * one driver is *latent* -- soil corrosivity is only partially published to
    the modelling dataset, so no model can reach the Bayes rate;
  * failures are Bernoulli draws, not thresholds, so identical assets under
    identical hazards do not share a fate;
  * functional loss propagates over the power->water dependency edges rather
    than appearing in any feature column, so a purely tabular model cannot
    see the cascade. That is the argument for keeping the graph in the system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import hazards as H
from .geography import Territory

WIND_CLASSES = {"transmission_span", "distribution_feeder", "recloser"}
FLOOD_CLASSES = {"transmission_substation", "distribution_substation",
                 "water_treatment_plant", "pump_station", "raw_water_intake",
                 "recloser"}
THERMAL_CLASSES = {"transmission_substation", "distribution_substation"}
FIRE_CLASSES = {"transmission_span", "distribution_feeder"}
GROUND_CLASSES = {"water_main"}
DUTY_CLASSES = {"pump_station"}

BASE_LOGIT = -11.4      # blue-sky hazard rate per asset per 6h bucket


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def failure_probability(ex: pd.DataFrame, a: pd.DataFrame) -> np.ndarray:
    """P(physical failure) for each (asset, bucket) row of `ex`.

    `a` must be aligned row-for-row with `ex` (asset attributes broadcast).
    """
    cls = a.asset_class.to_numpy()
    age = 2026 - a.install_year.to_numpy()
    life = a.design_life_years.to_numpy()
    wear = np.clip(age / life, 0, 1.8)
    frag = a["_material_fragility"].to_numpy()
    cond = a.condition_score.to_numpy() / 100.0
    veg = a.vegetation_density.to_numpy()
    soil = a["_soil_corrosivity_true"].to_numpy()
    lf = np.nan_to_num(a.summer_load_factor.to_numpy(), nan=0.0)
    dia = np.nan_to_num(a.diameter_mm.to_numpy(), nan=200.0)

    gust = ex.wind_gust_ms.to_numpy()
    hrs20 = ex.hours_wind_above_20.to_numpy()
    inund = ex.inundation_depth_m.to_numpy()
    temp = ex.temp_c.to_numpy()
    fwi = ex.fire_weather_index.to_numpy()
    sat = ex.soil_saturation.to_numpy()

    z = np.full(len(ex), BASE_LOGIT)
    # Condition and wear raise the floor everywhere: a decrepit asset fails on
    # an ordinary Tuesday too.
    z += 1.9 * wear * frag ** 0.5 + 1.4 * (1.0 - cond)

    # --- wind pathway: gust^2, amplified by vegetation and weak materials ----
    m = np.isin(cls, list(WIND_CLASSES))
    z += m * (0.0042 * gust ** 2 * frag * (0.55 + 1.15 * veg)
              + 0.011 * hrs20                      # fatigue from duration
              + 0.9 * wear * np.clip(gust / 30, 0, 2))   # age x wind interaction

    # --- flood pathway: switchgear inundation is close to a step function ----
    m = np.isin(cls, list(FLOOD_CLASSES))
    z += m * (3.1 * np.clip(inund, 0, 3.0) ** 0.75
              + 1.5 * wear * np.clip(inund, 0, 2))       # age x flood

    # --- thermal pathway: the heatwave mechanism -----------------------------
    # Transformers derate in high ambient. Stress is driven by how far the
    # asset is loaded past its de-rated capability, not by temperature alone.
    m = np.isin(cls, list(THERMAL_CLASSES))
    # Ambient above ~32 C erodes usable capacity; the asset is then loaded past
    # what it can actually carry. Failure is driven by that overload, not by
    # temperature alone -- a lightly loaded transformer survives a heatwave.
    derate = np.clip((temp - 32.0) / C.THERMAL_HALVING_DELTA_C, 0, None)
    effective_capacity = np.clip(1.0 - 0.09 * derate, 0.55, 1.0)
    overload = np.clip(lf / effective_capacity - 0.88, 0, None)
    z += m * (9.0 * overload + 1.2 * derate * wear + 4.5 * overload * derate)

    # --- fire pathway --------------------------------------------------------
    m = np.isin(cls, list(FIRE_CLASSES))
    z += m * (5.2 * np.clip(fwi - 0.62, 0, None) * (0.15 + 1.7 * veg) * frag)

    # --- ground movement / pressure transient pathway for mains -------------
    # soil corrosivity is the latent term: strongly predictive, only partly
    # observable. This caps achievable model performance, as in reality.
    m = np.isin(cls, list(GROUND_CLASSES))
    z += m * (1.9 * soil * wear
              + 1.0 * sat * (0.4 + soil)
              + 0.9 * frag * wear
              - 0.0016 * dia)                            # big mains break less

    # --- pump duty under heat: the water side of the feedback loop ----------
    m = np.isin(cls, list(DUTY_CLASSES))
    demand_mult = 1.0 + 0.42 * np.clip((temp - 28.0) / 12.0, 0, 1.4)
    z += m * (1.7 * np.clip(demand_mult - 1.0, 0, None) * (0.5 + wear))

    return _sig(z)


def water_demand_multiplier(temp_c: np.ndarray) -> np.ndarray:
    """System water demand rises sharply with temperature. Drives pumping load,
    which in turn drives the electrical peak that degrades the transformers
    feeding the pumps."""
    return 1.0 + 0.42 * np.clip((temp_c - 28.0) / 12.0, 0, 1.4)


def simulate_scenario(rng, sc: H.Scenario, assets: pd.DataFrame, terr: Territory,
                      dep_parent: dict, backup: dict, runtime: dict):
    """Run one scenario. Returns (exposure, failure_events)."""
    ex = H.asset_exposure(sc, assets, terr)
    a = assets.set_index("asset_id").loc[ex.asset_id].reset_index()
    p = failure_probability(ex, a)
    hit = rng.random(len(p)) < p

    ev = ex.loc[hit, ["scenario_id", "hazard", "asset_id", "bucket", "timestamp"]].copy()
    ev["failure_type"] = "physical"
    # An asset fails once per scenario; keep the first occurrence.
    ev = ev.sort_values("bucket").drop_duplicates("asset_id", keep="first")

    # ---- functional loss over the power dependency edges -------------------
    # A water site whose feeding substation is down loses function once its
    # backup generation (if any) is exhausted.
    down_sub = dict(zip(ev.asset_id, ev.bucket))
    func = []
    for site, parent in dep_parent.items():
        b = down_sub.get(parent)
        if b is None or site in down_sub:
            continue
        delay_buckets = int(runtime.get(site, 0) // H.BUCKET_HOURS) if backup.get(site) else 0
        b_loss = b + delay_buckets
        if b_loss < sc.n_buckets:
            func.append({
                "scenario_id": sc.scenario_id, "hazard": sc.hazard,
                "asset_id": site, "bucket": b_loss,
                "timestamp": sc.start + pd.Timedelta(hours=b_loss * H.BUCKET_HOURS),
                "failure_type": "functional_power_loss",
            })
    if func:
        ev = pd.concat([ev, pd.DataFrame(func)], ignore_index=True)

    # ---- restoration duration ---------------------------------------------
    if len(ev):
        cls = assets.set_index("asset_id").loc[ev.asset_id, "asset_class"].to_numpy()
        base = np.where(np.isin(cls, ["transmission_substation", "water_treatment_plant"]), 34,
                np.where(np.isin(cls, ["distribution_substation", "pump_station"]), 19,
                np.where(cls == "water_main", 14, 7)))
        # Restoration is slower when many assets are down at once -- crews are
        # the binding constraint, and this is what makes pre-positioning pay.
        congestion = 1.0 + 0.9 * np.clip(len(ev) / 220.0, 0, 3.0)
        ev["restore_hours"] = np.round(
            base * congestion * rng.lognormal(0, 0.42, len(ev)), 1)
        ev["is_functional"] = ev.failure_type != "physical"
    return ex, ev


def build_history(rng, assets: pd.DataFrame, terr: Territory, G):
    """Three years of scenarios, exposures and failures."""
    dep_parent = {v: u for u, v, d in G.edges(data=True)
                  if d.get("edge_type") == "power_dependency"}
    backup = dict(zip(assets.asset_id, assets.has_backup_generation))
    runtime = dict(zip(assets.asset_id, assets.backup_runtime_hours))

    end = pd.Timestamp(C.HISTORY_END)
    start = end - pd.DateOffset(years=C.HISTORY_YEARS)

    scenarios, all_ev, all_ex, i = [], [], [], 0
    for year in range(C.HISTORY_YEARS):
        y0 = start + pd.DateOffset(years=year)
        for hz, n in C.EVENTS_PER_YEAR.items():
            for _ in range(n):
                i += 1
                # Seasonality: hurricanes Jun-Nov, heat Jun-Sep, fire Mar-Oct.
                season = {"hurricane": (150, 330), "heatwave": (150, 270),
                          "wildfire": (60, 300), "inland_flood": (1, 365),
                          "baseline": (1, 365)}[hz]
                doy = rng.integers(*season)
                t0 = (y0 + pd.Timedelta(days=int(doy))).normalize()
                sc = H.make_scenario(rng, hz, t0, i)
                ex, ev = simulate_scenario(rng, sc, assets, terr,
                                           dep_parent, backup, runtime)
                scenarios.append({"scenario_id": sc.scenario_id, "hazard": hz,
                                  "start": sc.start, "duration_h": sc.duration_h,
                                  **{f"param_{k}": v for k, v in sc.params.items()}})
                all_ex.append(ex)
                if len(ev):
                    all_ev.append(ev)

    return (pd.DataFrame(scenarios),
            pd.concat(all_ex, ignore_index=True),
            pd.concat(all_ev, ignore_index=True))


def build_work_orders(rng, assets, failures):
    """CMMS extract: corrective orders from failures, plus routine maintenance."""
    corr = failures[["asset_id", "timestamp", "scenario_id", "failure_type"]].copy()
    corr["work_order_type"] = "corrective"
    corr["raised_at"] = corr.timestamp
    corr["cost_usd"] = np.round(rng.lognormal(8.4, 0.8, len(corr)), 2)

    n_routine = int(len(assets) * 1.6)
    idx = rng.choice(len(assets), n_routine)
    end = pd.Timestamp(C.HISTORY_END)
    rt = pd.DataFrame({
        "asset_id": assets.asset_id.to_numpy()[idx],
        "timestamp": [end - pd.Timedelta(days=int(d))
                      for d in rng.integers(1, 365 * C.HISTORY_YEARS, n_routine)],
        "scenario_id": None,
        "failure_type": None,
        "work_order_type": rng.choice(["routine", "inspection", "vegetation"],
                                      n_routine, p=[0.45, 0.35, 0.20]),
        "cost_usd": np.round(rng.lognormal(6.6, 0.7, n_routine), 2),
    })
    rt["raised_at"] = rt.timestamp
    wo = pd.concat([corr, rt], ignore_index=True).sort_values("raised_at")
    wo["work_order_id"] = [f"WO-{i:07d}" for i in range(1, len(wo) + 1)]
    return wo.reset_index(drop=True)
