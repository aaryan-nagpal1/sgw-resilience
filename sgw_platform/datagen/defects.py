"""Deliberate data-quality defects.

The PRD asks for data quality considerations. A pristine synthetic dataset
would let a candidate skip that section entirely, so the published extracts
carry the defects a real GIS/CMMS integration actually presents. Every defect
here is one that shows up in genuine utility data.

The *clean* frame is retained separately so the resolution logic can be
scored rather than merely asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

MATERIAL_VARIANTS = {
    "cast_iron": ["Cast Iron", "CI", "cast-iron", "CAST IRON", "C.I."],
    "ductile_iron": ["Ductile Iron", "DI", "ductile-iron", "D.I."],
    "asbestos_cement": ["Asbestos Cement", "AC", "A/C", "asbestos-cement"],
    "pvc": ["PVC", "P.V.C.", "pvc", "Poly Vinyl Chloride"],
    "wood_pole": ["Wood", "WOOD POLE", "timber", "Wood Pole"],
}


def apply_defects(rng, assets: pd.DataFrame):
    """Return (gis_extract, cmms_extract) -- two imperfect views of one estate."""
    d = C.DEFECTS
    n = len(assets)
    gis = assets.copy()

    # 1. Missing install years -- pre-digitisation records were never captured.
    m = rng.random(n) < d["missing_install_year"]
    gis.loc[m, "install_year"] = np.nan

    # 2. Missing material.
    m = rng.random(n) < d["missing_material"]
    gis.loc[m, "material"] = None

    # 3. Coordinate drift -- 50-400 m, from historical digitisation of paper maps.
    m = rng.random(n) < d["coordinate_drift"]
    off = rng.normal(0, 0.0022, (int(m.sum()), 2))
    gis.loc[m, "lat"] = gis.loc[m, "lat"].to_numpy() + off[:, 0]
    gis.loc[m, "lon"] = gis.loc[m, "lon"].to_numpy() + off[:, 1]
    gis["position_source"] = np.where(m, "digitised_paper", "gps_survey")

    # 4. Stale condition scores.
    m = rng.random(n) < d["stale_condition"]
    gis.loc[m, "last_inspected_year"] = rng.integers(2005, 2019, int(m.sum()))

    # 5. Mixed units on pipe diameter -- some records in inches.
    wm = gis.asset_class == "water_main"
    m = wm & (rng.random(n) < d["diameter_unit_mix"])
    gis.loc[m, "diameter_mm"] = (gis.loc[m, "diameter_mm"] / 25.4).round(1)
    gis["diameter_unit"] = np.where(m, "in", np.where(wm, "mm", None))

    # 6. Duplicate asset records -- the same physical asset surveyed twice.
    n_dup = int(n * d["duplicate_assets"])
    dups = gis.sample(n_dup, random_state=int(rng.integers(1e9))).copy()
    dups["asset_id"] = [f"{a}-DUP" for a in dups.asset_id]
    dups["lat"] += rng.normal(0, 0.0004, n_dup)
    dups["lon"] += rng.normal(0, 0.0004, n_dup)
    gis = pd.concat([gis, dups], ignore_index=True)

    # Drop the generative-truth columns: a modeller must not see them.
    gis = gis.drop(columns=[c for c in gis.columns if c.startswith("_")])

    # Soil corrosivity is published only where a survey was actually done
    # (~40% of the network). This is the latent variable that caps model
    # performance, and its partial availability is itself realistic.
    surveyed = rng.random(len(gis)) < 0.40
    gis["soil_corrosivity_surveyed"] = np.where(
        surveyed, np.round(assets["_soil_corrosivity_true"].reindex(
            gis.index, fill_value=np.nan), 2), np.nan)
    return gis


def corrupt_work_orders(rng, work_orders: pd.DataFrame, assets: pd.DataFrame):
    """CMMS extract with its own identity problems."""
    wo = work_orders.copy()
    d = C.DEFECTS

    # Orphan work orders: reference asset ids that are not in the GIS at all,
    # usually because the asset was replaced and re-numbered.
    n_orph = int(len(wo) * d["orphan_work_orders"])
    orph = rng.choice(wo.index, n_orph, replace=False)
    wo.loc[orph, "asset_id"] = [f"LEGACY-{i:06d}" for i in rng.integers(1, 99999, n_orph)]

    # Free-text material recorded by the technician, inconsistently.
    mat = assets.set_index("asset_id")["material"].to_dict()
    raw = []
    for aid in wo.asset_id:
        base = mat.get(aid)
        if base is None:
            raw.append(None)
        elif base in MATERIAL_VARIANTS and rng.random() < d["material_spelling_noise"]:
            raw.append(rng.choice(MATERIAL_VARIANTS[base]))
        else:
            raw.append(base)
    wo["material_as_recorded"] = raw

    # Timestamps in local time with no timezone, and a handful entered as the
    # date the crew filed the paperwork rather than the date of the fault.
    late = rng.random(len(wo)) < 0.06
    wo.loc[late, "raised_at"] = wo.loc[late, "raised_at"] + pd.Timedelta(days=3)
    wo["timestamp_quality"] = np.where(late, "reported_late", "as_occurred")
    return wo
