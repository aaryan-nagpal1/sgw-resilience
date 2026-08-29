"""Entry point: build the complete synthetic SGW dataset.

    python -m sgw_platform.datagen.generate --out data

Everything is seeded from config.SEED, so the dataset is reproducible. Rerun
with --seed to produce an independent estate for out-of-sample testing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .assets import build_assets, allocate_customers
from .crews import build_crews
from .defects import apply_defects, corrupt_work_orders
from .geography import Territory
from .history import build_history, build_work_orders
from .network import build_network, compute_consequence
from .features import build_training_table
from .telemetry import build_telemetry


def generate(out_dir: str = "data", seed: int = C.SEED, quiet: bool = False):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def log(msg):
        if not quiet:
            print(f"  [{time.time() - t0:6.1f}s] {msg}")

    log("building territory")
    terr = Territory(rng)

    log("building asset estate")
    assets = allocate_customers(rng, build_assets(rng, terr))

    log("building network topology and dependency edges")
    edges, G = build_network(rng, assets)
    consequence = compute_consequence(assets, G)

    log(f"simulating {C.HISTORY_YEARS} years of operating history")
    scenarios, exposure, failures = build_history(rng, assets, terr, G)

    log("building work orders")
    work_orders = build_work_orders(rng, assets, failures)

    log("building SCADA telemetry")
    telemetry, telemetry_labels = build_telemetry(rng, assets)

    log("building crews and depots")
    depots, crews = build_crews(rng, terr, assets)

    log("injecting data quality defects")
    gis_extract = apply_defects(rng, assets)
    cmms_extract = corrupt_work_orders(rng, work_orders, assets)

    log("building training table")
    training = build_training_table(gis_extract, exposure, failures,
                                    consequence, scenarios)

    # ---- write -----------------------------------------------------------
    # Two views are published deliberately:
    #   *_extract  -- what the integration actually receives (defects included)
    #   assets_truth -- the clean estate, held back for scoring the cleaning
    log("writing files")
    tables = {
        "gis_asset_extract": gis_extract,
        "cmms_work_orders": cmms_extract,
        "network_edges": edges,
        "asset_consequence": consequence,
        "scenarios": scenarios,
        "hazard_exposure": exposure,
        "failure_events": failures,
        "training_table": training,
        "scada_telemetry": telemetry,
        "scada_labels": telemetry_labels,
        "crews": crews,
        "depots": depots,
        "_assets_truth": assets,
    }
    # Large tables go to parquet (10-20x smaller and typed); small ones stay
    # as CSV so a reviewer can open them without tooling.
    BIG = {"hazard_exposure", "scada_telemetry", "training_table"}
    for name, df in tables.items():
        if name in BIG:
            df.to_parquet(out / f"{name}.parquet", index=False)
        else:
            df.to_csv(out / f"{name}.csv", index=False)

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "seed": seed,
        "config": {
            "history_years": C.HISTORY_YEARS,
            "history_end": C.HISTORY_END,
            "bucket_hours": 6,
            "grid_res_deg": C.GRID_RES_DEG,
            "total_customers": C.TOTAL_CUSTOMERS,
            "total_population": C.TOTAL_POPULATION,
        },
        "tables": {n: {"rows": int(len(d)), "cols": int(d.shape[1])}
                   for n, d in tables.items()},
        "summary": {
            "assets": int(len(assets)),
            "asset_classes": assets.asset_class.value_counts().to_dict(),
            "network_edges": int(len(edges)),
            "power_dependency_edges": int((edges.edge_type == "power_dependency").sum()),
            "scenarios": int(len(scenarios)),
            "scenarios_by_hazard": scenarios.hazard.value_counts().to_dict(),
            "failure_events": int(len(failures)),
            "failures_by_hazard": failures.hazard.value_counts().to_dict(),
            "functional_cascade_failures": int((failures.failure_type
                                                == "functional_power_loss").sum()),
            "exposure_rows": int(len(exposure)),
            "overall_failure_rate_per_asset_bucket": round(
                float(len(failures) / max(len(exposure), 1)), 6),
            "training_rows": int(len(training)),
            "training_positive_rate": round(float(training.failed.mean()), 5),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"done -> {out.resolve()}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic SGW dataset.")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    m = generate(a.out, a.seed, a.quiet)
    print(json.dumps(m["summary"], indent=2))


if __name__ == "__main__":
    main()
