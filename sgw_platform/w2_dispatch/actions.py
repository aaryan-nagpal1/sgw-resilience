"""Pre-emptive action catalogue and mitigation assumptions.

Every effect size here is an ASSUMPTION with no published source we could find
for this asset mix. They are therefore (a) collected in one place, (b) exposed
to the sensitivity sweep in `sensitivity.py`, and (c) reported as a range in
the deliverables rather than as a point estimate. The claim we make is "the
optimiser wins across the plausible range", not "it wins at the number we chose".
"""
from __future__ import annotations

from dataclasses import dataclass

WIND_FIRE_CLASSES = {"transmission_span", "distribution_feeder"}
WATER_SITES = {"pump_station", "water_treatment_plant"}


@dataclass(frozen=True)
class Action:
    name: str
    crew_type: str
    crew_shifts: int
    cost_usd: int
    risk_multiplier: float      # applied to P(failure)
    severs_cascade: bool = False


CATALOGUE = {
    # Inspect and spot-repair: the general-purpose intervention.
    "inspect_repair": Action("inspect_repair", "line", 1, 4_200, 0.55),
    "inspect_repair_water": Action("inspect_repair_water", "water", 1, 3_600, 0.55),
    # Vegetation clearance: only meaningful on overhead wind/fire-exposed plant.
    "vegetation_clear": Action("vegetation_clear", "vegetation", 1, 2_800, 0.60),
    # Temporary generation: reduces no failure probability at all. It removes a
    # *dependency*. Only a model that carries the power->water graph can value
    # this action, which makes it the clearest demonstration of the coupling.
    "temp_generation": Action("temp_generation", "water", 1, 11_500, 1.00,
                              severs_cascade=True),
}


def available_actions(asset_class: str) -> list[str]:
    acts = []
    if asset_class in WATER_SITES:
        acts += ["inspect_repair_water", "temp_generation"]
    elif asset_class.startswith(("water_", "storage", "raw_")):
        acts += ["inspect_repair_water"]
    else:
        acts += ["inspect_repair"]
    if asset_class in WIND_FIRE_CLASSES:
        acts += ["vegetation_clear"]
    return acts
