"""Asset portfolio generation for SGW: electric and water estates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geography import Territory

# Placement rules per class: (coastal_bias, city_weighted, is_linear)
PLACEMENT = {
    "transmission_substation": (0.15, False, False),
    "distribution_substation": (0.10, True, False),
    "transmission_span":       (0.10, False, True),
    "distribution_feeder":     (0.05, True, True),
    "recloser":                (0.05, True, False),
    "water_treatment_plant":   (0.70, False, False),   # sited low, near source
    "pump_station":            (0.25, True, False),
    "storage_tank":            (0.10, True, False),
    "water_main":              (0.10, True, True),
    "raw_water_intake":        (0.85, False, False),   # on the water
}

# (mean_install_year, sd, design_life_years)
VINTAGE = {
    "transmission_substation": (1979, 17, 60),
    "distribution_substation": (1987, 18, 50),
    "transmission_span":       (1976, 19, 70),
    "distribution_feeder":     (1991, 17, 45),
    "recloser":                (2008, 9, 30),
    "water_treatment_plant":   (1981, 16, 60),
    "pump_station":            (1994, 16, 40),
    "storage_tank":            (1985, 18, 65),
    "water_main":              (1972, 22, 80),
    "raw_water_intake":        (1977, 17, 60),
}


def _pick_material(rng, cls, n):
    spec = C.MATERIALS.get(cls, C.DEFAULT_MATERIAL)
    names = [s[0] for s in spec]
    probs = np.array([s[1] for s in spec], dtype=float)
    probs /= probs.sum()
    frag = {s[0]: s[2] for s in spec}
    chosen = rng.choice(names, size=n, p=probs)
    return chosen, np.array([frag[m] for m in chosen])


def build_assets(rng: np.random.Generator, terr: Territory) -> pd.DataFrame:
    rows = []
    for cls, n in C.ASSET_COUNTS.items():
        coastal_bias, city_weighted, is_linear = PLACEMENT[cls]
        if city_weighted:
            lat, lon = terr.sample_near_cities(n)
        else:
            lat, lon = terr.sample_points(n, coastal_bias)

        # Linear assets get an end point; exposure is evaluated along the run.
        if is_linear:
            length_km = rng.gamma(2.0, 1.1, n) + 0.25
            bearing = rng.uniform(0, 2 * np.pi, n)
            lat2 = lat + (length_km / 111.0) * np.cos(bearing)
            lon2 = lon + (length_km / (111.0 * np.cos(np.radians(lat)))) * np.sin(bearing)
        else:
            length_km = np.zeros(n)
            lat2, lon2 = lat.copy(), lon.copy()

        mu_year, sd_year, design_life = VINTAGE[cls]
        install_year = np.clip(rng.normal(mu_year, sd_year, n).round(), 1928, 2025)
        age = 2026 - install_year
        material, mat_frag = _pick_material(rng, cls, n)

        # Condition score 0 (failed) .. 100 (as new). Driven by age against
        # design life plus material and unobserved local factors, then blurred:
        # inspection is subjective and infrequent.
        wear = np.clip(age / design_life, 0, 1.6)
        condition = np.clip(100 - 62 * wear * mat_frag ** 0.6
                            + rng.normal(0, 11, n), 2, 100)

        soil = terr.soil_corrosivity(lat, lon)
        veg = terr.vegetation_density(lat, lon)
        elev = terr.elevation_m(lat, lon)
        dcoast = terr.dist_to_coast_km(lat, lon)
        drain = terr.drainage_quality(lat, lon)
        drought = terr.drought_susceptibility(lat, lon)
        dens = terr.customer_density(lat, lon)

        crit = rng.choice([1, 2, 3], size=n, p=C.CRITICALITY_WEIGHTS)

        rows.append(pd.DataFrame({
            "asset_class": cls,
            "lat": lat, "lon": lon, "lat_end": lat2, "lon_end": lon2,
            "length_km": length_km.round(3),
            "install_year": install_year.astype(int),
            "design_life_years": design_life,
            "material": material,
            "_material_fragility": mat_frag,      # generative truth, dropped later
            "condition_score": condition.round(1),
            "criticality_tier": crit,
            "elevation_m": elev.round(2),
            "dist_to_coast_km": dcoast.round(2),
            "vegetation_density": veg.round(3),
            "drainage_quality": drain.round(3),
            "drought_susceptibility": drought.round(3),
            "_soil_corrosivity_true": soil.round(3),   # latent
            "_customer_density": dens.round(4),
            "zone_id": terr.zone_of(lat, lon),
        }))

    df = pd.concat(rows, ignore_index=True)

    # ---- identity ---------------------------------------------------------
    prefix = {
        "transmission_substation": "TSS", "distribution_substation": "DSS",
        "transmission_span": "TLS", "distribution_feeder": "DFR",
        "recloser": "RCL", "water_treatment_plant": "WTP",
        "pump_station": "PMP", "storage_tank": "TNK",
        "water_main": "WMN", "raw_water_intake": "INT",
    }
    df["asset_id"] = [f"{prefix[c]}-{i:05d}" for i, c in enumerate(df.asset_class, 1)]

    # ---- network role attributes -----------------------------------------
    df["network"] = np.where(df.asset_class.isin(C.ELECTRIC_CLASSES), "electric", "water")

    # Site elevation above *local* grade -- the variable that actually decides
    # whether a site floods. Regional elevation gates storm surge; a substation
    # 40 km inland still floods if it sits in a hollow. Water assets are sited
    # low by necessity (gravity, proximity to source), which is precisely why
    # they are the flood-exposed population.
    low_sited = df.asset_class.isin(
        ["water_treatment_plant", "pump_station", "raw_water_intake"])
    df["site_elevation_m"] = np.where(
        low_sited, np.round(rng.gamma(1.6, 0.55, len(df)), 2),
        np.round(rng.gamma(2.6, 0.95, len(df)), 2))
    # Coastal sites of any class sit lower.
    df["site_elevation_m"] *= np.clip(0.45 + df.dist_to_coast_km / 150.0, 0.45, 1.0)
    df["site_elevation_m"] = df.site_elevation_m.round(2)

    # Flood defence: some sites have been hardened already. Coastal sites are
    # more likely to have been done, because they flooded before.
    p_defended = np.clip(0.10 + 0.45 * np.exp(-df.dist_to_coast_km / 40.0), 0, 0.7)
    df["flood_defence_m"] = np.where(
        rng.random(len(df)) < p_defended,
        np.round(rng.uniform(0.5, 2.5, len(df)), 2), 0.0)

    # Backup generation matters enormously for water assets during grid loss.
    is_water_site = df.asset_class.isin(["pump_station", "water_treatment_plant"])
    p_backup = np.where(df.criticality_tier == 1, 0.85,
                        np.where(df.criticality_tier == 2, 0.45, 0.16))
    df["has_backup_generation"] = (is_water_site & (rng.random(len(df)) < p_backup))
    df["backup_runtime_hours"] = np.where(
        df.has_backup_generation, rng.choice([8, 12, 24, 48, 72], len(df),
                                             p=[.28, .27, .27, .12, .06]), 0)

    # Pipe diameter (mm) for water mains; transformer rating (MVA) for substations.
    df["diameter_mm"] = np.where(
        df.asset_class == "water_main",
        rng.choice([100, 150, 200, 250, 300, 400, 600, 900], len(df),
                   p=[.16, .21, .19, .14, .12, .09, .06, .03]), np.nan)
    df["rating_mva"] = np.where(
        df.asset_class == "transmission_substation", rng.uniform(120, 600, len(df)).round(1),
        np.where(df.asset_class == "distribution_substation",
                 rng.uniform(8, 75, len(df)).round(1), np.nan))

    # Summer peak load factor -- how close a substation runs to its rating on a
    # hot day. This is the variable that turns a heatwave into a failure.
    df["summer_load_factor"] = np.where(
        df.asset_class.isin(["transmission_substation", "distribution_substation"]),
        np.clip(rng.beta(6, 3, len(df)) * 1.05, 0.25, 1.12).round(3), np.nan)

    # Pump duty
    df["pump_rated_kw"] = np.where(
        df.asset_class == "pump_station", rng.gamma(3.0, 60, len(df)).round(1), np.nan)

    df["last_inspected_year"] = np.clip(
        2026 - rng.gamma(2.0, 1.6, len(df)).round(), 2005, 2026).astype(int)

    return df


def allocate_customers(rng, df: pd.DataFrame) -> pd.DataFrame:
    """Distribute electric customers and water-served population across the
    assets that directly serve load."""
    # Electric customers sit behind distribution feeders.
    fe = df.asset_class == "distribution_feeder"
    w = df.loc[fe, "_customer_density"].to_numpy() * rng.uniform(0.5, 1.5, fe.sum())
    df.loc[fe, "customers_served"] = (w / w.sum() * C.TOTAL_CUSTOMERS).round()

    # Water-served population sits behind pump stations.
    pm = df.asset_class == "pump_station"
    w = df.loc[pm, "_customer_density"].to_numpy() * rng.uniform(0.5, 1.5, pm.sum())
    df.loc[pm, "population_served_water"] = (w / w.sum() * C.TOTAL_POPULATION).round()

    df["customers_served"] = df.get("customers_served", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["population_served_water"] = df.get(
        "population_served_water", pd.Series(0.0, index=df.index)).fillna(0.0)

    # Critical facilities (hospitals, dialysis, care homes, emergency services)
    n_crit = 420
    load_bearing = df.index[fe | pm]
    weights = df.loc[load_bearing, "_customer_density"].to_numpy()
    weights = weights / weights.sum()
    picks = rng.choice(load_bearing, size=n_crit, replace=True, p=weights)
    counts = pd.Series(picks).value_counts()
    df["critical_facilities"] = 0
    df.loc[counts.index, "critical_facilities"] = counts.to_numpy()
    return df
