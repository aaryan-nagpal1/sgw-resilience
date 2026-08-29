"""Resolve the data-quality defects in the GIS and CMMS extracts.

Deliberately does NOT trust the metadata columns that would make this trivial.
A real integration gets a `diameter_unit` field that is wrong often enough that
you have to check it against the values, and duplicate records do not arrive
helpfully suffixed. Each fix is therefore inferred and then scored against the
held-back truth table by `score_cleaning`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Canonical material names -> the free-text variants seen in the wild.
MATERIAL_MAP = {
    "cast_iron": ["cast iron", "ci", "c.i.", "cast-iron", "castiron"],
    "ductile_iron": ["ductile iron", "di", "d.i.", "ductile-iron"],
    "asbestos_cement": ["asbestos cement", "ac", "a/c", "asbestos-cement"],
    "pvc": ["pvc", "p.v.c.", "poly vinyl chloride", "polyvinyl chloride"],
    "wood_pole": ["wood", "wood pole", "timber", "woodpole"],
    "steel": ["steel"], "concrete": ["concrete"],
    "steel_lattice": ["steel lattice", "lattice"],
    "steel_monopole": ["steel monopole", "monopole"],
    "composite_pole": ["composite pole", "composite"],
    "concrete_pole": ["concrete pole"],
    "underground": ["underground", "ug"],
    "standard": ["standard"],
}
_LOOKUP = {v: k for k, vs in MATERIAL_MAP.items() for v in vs}

# Water mains are manufactured in a known set of diameters. Anything far below
# the smallest plausible mm value is an imperial record, whatever the unit
# column claims.
MIN_PLAUSIBLE_MM = 50.0


@dataclass
class CleaningReport:
    steps: list = field(default_factory=list)

    def add(self, step, detected, action, note=""):
        self.steps.append({"step": step, "records_detected": detected,
                           "action": action, "note": note})

    def to_frame(self):
        return pd.DataFrame(self.steps)


def _norm_material(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"[^a-z0-9./ _-]", "", str(v).strip().lower())
    s = re.sub(r"[\s_-]+", " ", s).strip()
    if s in _LOOKUP:
        return _LOOKUP[s]
    return s.replace(" ", "_")


def clean_assets(gis: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    df = gis.copy()
    rep = CleaningReport()

    # --- 1. material normalisation -----------------------------------------
    before = df.material.astype("object").nunique(dropna=True)
    df["material_clean"] = df.material.map(_norm_material)
    rep.add("material normalisation", int(before),
            f"{before} raw values -> {df.material_clean.nunique()} canonical",
            "free-text CMMS variants collapsed")

    # --- 2. diameter units --------------------------------------------------
    # Infer from the values, then report how often the stated unit disagreed.
    wm = df.asset_class == "water_main"
    looks_imperial = wm & (df.diameter_mm < MIN_PLAUSIBLE_MM)
    df["diameter_mm_clean"] = df.diameter_mm.where(~looks_imperial,
                                                   df.diameter_mm * 25.4)
    stated = df.get("diameter_unit")
    disagree = int((looks_imperial & (stated != "in")).sum()) if stated is not None else 0
    rep.add("diameter unit reconciliation", int(looks_imperial.sum()),
            "converted inches -> mm by value range",
            f"{disagree} records where the stated unit column disagreed")

    # --- 3. duplicate GIS records -------------------------------------------
    # Same class, same install year, within ~150 m. No reliance on the id.
    df["_k"] = (df.asset_class.astype(str) + "|"
                + df.install_year.fillna(-1).astype(int).astype(str) + "|"
                + (df.lat * 200).round().astype(int).astype(str) + "|"
                + (df.lon * 200).round().astype(int).astype(str))
    dup_mask = df.duplicated("_k", keep="first")
    df["is_duplicate_record"] = dup_mask
    rep.add("duplicate asset records", int(dup_mask.sum()),
            "flagged; first record retained as canonical",
            "matched on class + vintage + 150 m proximity")

    # --- 4. missing install year -------------------------------------------
    # Condition score is a function of wear, and wear is a function of age, so
    # a per-class regression on condition recovers vintage far better than the
    # class median. The median is kept as the fallback where a class has too
    # few observations to fit.
    miss = df.install_year.isna()
    est = pd.Series(np.nan, index=df.index)
    for cls, g in df.groupby("asset_class"):
        known = g[g.install_year.notna() & g.condition_score.notna()]
        if len(known) >= 30:
            b, a = np.polyfit(known.condition_score, known.install_year, 1)
            est.loc[g.index] = a + b * g.condition_score
    med = df.groupby("asset_class").install_year.transform("median")
    df["install_year_clean"] = (df.install_year
                                .fillna(est.round())
                                .fillna(med).round())
    df["install_year_imputed"] = miss
    rep.add("missing install year", int(miss.sum()),
            "imputed by per-class regression on condition score",
            "median fallback; imputation flag carried into the model")

    # --- 5. positional confidence ------------------------------------------
    drift = df.get("position_source", pd.Series("gps_survey", index=df.index)) \
              == "digitised_paper"
    df["position_low_confidence"] = drift
    rep.add("positional uncertainty", int(drift.sum()),
            "flagged; hazard sampled over a wider radius for these",
            "digitised from paper maps, 50-400 m error")

    # --- 6. stale inspection ------------------------------------------------
    stale = (2026 - df.last_inspected_year) > 5
    df["condition_stale"] = stale
    rep.add("stale condition score", int(stale.sum()),
            "flagged; condition down-weighted where stale",
            "last inspected more than 5 years ago")

    return df.drop(columns=["_k"]), rep


def clean_work_orders(wo: pd.DataFrame, assets: pd.DataFrame):
    """Identify orphan work orders. These are lost failure history, not noise --
    every orphan is a training label that cannot be used."""
    known = set(assets.asset_id)
    wo = wo.copy()
    wo["is_orphan"] = ~wo.asset_id.isin(known)
    corrective_orphans = int((wo.is_orphan & (wo.work_order_type == "corrective")).sum())
    return wo, {
        "orphan_work_orders": int(wo.is_orphan.sum()),
        "orphan_share": round(float(wo.is_orphan.mean()), 4),
        "lost_failure_labels": corrective_orphans,
    }


def score_cleaning(cleaned: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Score the inferred fixes against the held-back truth table.

    This is the point of holding truth back: data-quality handling becomes a
    measured claim rather than a described intention.
    """
    t = truth.set_index("asset_id")
    c = cleaned[~cleaned.is_duplicate_record].set_index("asset_id")
    common = c.index.intersection(t.index)
    c, t = c.loc[common], t.loc[common]
    rows = []

    ok = (c.material_clean == t.material)
    known = c.material_clean.notna()
    rows.append({"check": "material recovered", "n": int(known.sum()),
                 "accuracy": round(float(ok[known].mean()), 4)})

    # Relative tolerance: the imperial records were stored to 1 d.p., so an
    # exact round-trip is impossible. The question is whether the unit was
    # identified, not whether the rounding was undone.
    wm = t.asset_class == "water_main"
    dm = ((c.diameter_mm_clean[wm] - t.diameter_mm[wm]).abs()
          / t.diameter_mm[wm]) < 0.01
    rows.append({"check": "diameter unit recovered (1% tol)", "n": int(wm.sum()),
                 "accuracy": round(float(dm.mean()), 4)})

    imp = c.install_year_imputed
    err = (c.install_year_clean[imp] - t.install_year[imp]).abs()
    rows.append({"check": "install year imputed (within 10 yrs)", "n": int(imp.sum()),
                 "accuracy": round(float((err <= 10).mean()), 4)})

    # Duplicate detection measured against the known duplicate ids.
    true_dups = set(a for a in truth.asset_id if str(a).endswith("-DUP"))
    flagged = set(cleaned.loc[cleaned.is_duplicate_record, "asset_id"])
    all_dups = set(a for a in cleaned.asset_id if str(a).endswith("-DUP"))
    if all_dups:
        rec = len(flagged & all_dups) / len(all_dups)
        prec = len(flagged & all_dups) / max(len(flagged), 1)
        rows.append({"check": "duplicate detection (recall)", "n": len(all_dups),
                     "accuracy": round(rec, 4)})
        rows.append({"check": "duplicate detection (precision)", "n": len(flagged),
                     "accuracy": round(prec, 4)})
    return pd.DataFrame(rows)
