"""Aggregate per-bucket exposure into the modelling grain.

The 6-hour exposure series is the right grain for the operational timeline the
dispatcher scrubs through, but the wrong grain for training: consecutive
buckets for one asset in one storm are heavily autocorrelated, and treating
them as independent rows inflates the effective sample size by ~10x and will
produce a model that looks excellent in cross-validation and fails in service.

The training grain is therefore one row per (scenario, asset): "given the
forecast for this event, does this asset fail at any point during it?"
Splits must be **by scenario**, never by row, for the same reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGGS = {
    "wind_gust_ms": ["max", "mean"],
    "rainfall_trailing_mm": ["max"],
    "surge_m": ["max"],
    "inundation_depth_m": ["max"],
    "temp_c": ["max", "min"],
    "humidity": ["min"],
    "fire_weather_index": ["max"],
    "soil_saturation": ["max"],
    "hours_above_35c": ["max"],
    "hours_wind_above_20": ["max"],
}

STATIC_COLS = [
    "asset_id", "asset_class", "network", "install_year", "design_life_years",
    "material", "condition_score", "criticality_tier", "elevation_m",
    "site_elevation_m", "dist_to_coast_km", "vegetation_density",
    "drainage_quality", "drought_susceptibility", "zone_id",
    "flood_defence_m", "has_backup_generation", "backup_runtime_hours",
    "diameter_mm", "rating_mva", "summer_load_factor", "pump_rated_kw",
    "last_inspected_year", "soil_corrosivity_surveyed",
]

CONSEQUENCE_COLS = [
    "asset_id", "downstream_assets", "customers_affected",
    "population_water_affected", "critical_facilities_affected",
    "cascade_population_water", "consequence_score",
]


def build_training_table(gis: pd.DataFrame, exposure: pd.DataFrame,
                         failures: pd.DataFrame, consequence: pd.DataFrame,
                         scenarios: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, asset) with aggregated exposure and the label."""
    agg = (exposure.groupby(["scenario_id", "asset_id"])
           .agg(AGGS))
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()

    # Derived features that only make sense post-aggregation.
    agg["diurnal_recovery_c"] = agg.temp_c_max - agg.temp_c_min
    agg["wind_gust_range"] = agg.wind_gust_ms_max - agg.wind_gust_ms_mean

    static = gis[[c for c in STATIC_COLS if c in gis.columns]].drop_duplicates("asset_id")
    df = agg.merge(static, on="asset_id", how="left")
    df = df.merge(consequence[CONSEQUENCE_COLS], on="asset_id", how="left")
    df = df.merge(scenarios[["scenario_id", "hazard", "start"]], on="scenario_id", how="left")

    # Age is derived at scenario time, not "now" -- using today's age for a
    # storm three years ago leaks the future into the features.
    df["scenario_year"] = pd.to_datetime(df["start"]).dt.year
    df["age_years"] = df.scenario_year - df.install_year
    df["wear_ratio"] = df.age_years / df.design_life_years
    df["years_since_inspection"] = df.scenario_year - df.last_inspected_year

    # ---- labels -----------------------------------------------------------
    phys = failures[failures.failure_type == "physical"]
    df["failed"] = df.set_index(["scenario_id", "asset_id"]).index.isin(
        phys.set_index(["scenario_id", "asset_id"]).index).astype(int)
    # Kept separate on purpose: functional loss is a *consequence of the
    # network*, not a property of the asset. A model trained to predict it from
    # asset features would be learning a graph relationship through a keyhole.
    func = failures[failures.failure_type == "functional_power_loss"]
    df["functional_loss"] = df.set_index(["scenario_id", "asset_id"]).index.isin(
        func.set_index(["scenario_id", "asset_id"]).index).astype(int)

    first_b = (failures.groupby(["scenario_id", "asset_id"])["bucket"].min()
               .rename("failure_bucket").reset_index())
    df = df.merge(first_b, on=["scenario_id", "asset_id"], how="left")

    df["expected_loss"] = df.failed * df.consequence_score
    return df.drop(columns=["start"])


def scenario_split(df: pd.DataFrame, scenarios: pd.DataFrame, test_frac=0.25, seed=7):
    """Split by scenario and by time -- the last events are the test set.

    Random row splits leak: the same storm appears in train and test, and the
    model memorises the event rather than the relationship. Splitting by
    scenario chronologically is the only honest evaluation here.
    """
    s = scenarios.sort_values("start")
    n_test = max(1, int(len(s) * test_frac))
    test_ids = set(s.scenario_id.tail(n_test))
    return df[~df.scenario_id.isin(test_ids)].copy(), df[df.scenario_id.isin(test_ids)].copy()
