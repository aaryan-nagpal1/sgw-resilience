"""Field resources: depots and crews. Inputs to the dispatch optimiser (W2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geography import Territory


def build_crews(rng, terr: Territory, assets: pd.DataFrame):
    lat, lon = terr.sample_near_cities(C.N_DEPOTS)
    depots = pd.DataFrame({
        "depot_id": [f"DEP-{i:03d}" for i in range(1, C.N_DEPOTS + 1)],
        "lat": lat.round(4), "lon": lon.round(4),
        "zone_id": terr.zone_of(lat, lon),
        "staging_capacity": rng.integers(6, 22, C.N_DEPOTS),
    })

    rows = []
    for ctype, (count, classes, cost) in C.CREW_TYPES.items():
        d = rng.choice(depots.depot_id, count)
        rows.append(pd.DataFrame({
            "crew_type": ctype,
            "home_depot": d,
            "cost_per_shift_usd": cost,
            "shift_hours": 12,
            # Crews vary in productivity; storm managers know which are quick.
            "productivity": rng.normal(1.0, 0.13, count).clip(0.6, 1.5).round(3),
            "can_work": ";".join(sorted(classes)),
        }))
    crews = pd.concat(rows, ignore_index=True)
    crews["crew_id"] = [f"CRW-{i:04d}" for i in range(1, len(crews) + 1)]

    # Mutual aid: crews available from neighbouring utilities at a premium,
    # but only with 48h notice. A real and frequently binding constraint.
    aid = pd.DataFrame({
        "crew_id": [f"AID-{i:04d}" for i in range(1, 41)],
        "crew_type": rng.choice(["line", "water", "vegetation"], 40, p=[.6, .25, .15]),
        "home_depot": "MUTUAL_AID",
        "cost_per_shift_usd": rng.choice([7800, 8600, 9400], 40),
        "shift_hours": 12,
        "productivity": rng.normal(0.88, 0.14, 40).clip(0.5, 1.3).round(3),
        "can_work": "",
        "lead_time_hours": 48,
    })
    crews["lead_time_hours"] = 0
    return depots, pd.concat([crews, aid], ignore_index=True)
