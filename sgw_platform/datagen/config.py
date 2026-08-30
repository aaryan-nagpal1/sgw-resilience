"""Configuration for the synthetic SGW estate.

Every number here is an *assumption*. They are collected in one place so the
assumptions register in the PRD can cite them directly, and so a reviewer can
see exactly what was invented rather than hunting through the code.
"""
from __future__ import annotations

SEED = 20260831

# ---------------------------------------------------------------- geography
# A fictional coastal/inland service territory in the US Southeast. The
# bounding box is deliberately plausible (Gulf-adjacent latitudes) so that
# hurricane, heatwave, inland flood and wildfire are all credible hazards,
# but the territory itself does not correspond to any real utility.
LAT_MIN, LAT_MAX = 29.60, 33.00
LON_MIN, LON_MAX = -86.00, -81.50

N_ZONES = 12          # operating zones (dispatch + reporting boundaries)
N_CITIES = 9          # population centres driving customer density

GRID_RES_DEG = 0.10   # hazard grid cell size (~11 km) -- matches the
                      # resolution of a regional NWP forecast product

# ------------------------------------------------------------------ estate
# Counts scaled for a utility serving ~8M residents (~3.2M electric accounts).
# Linear assets are already segmented: a "span" or "main" row is one
# maintainable segment, not a whole circuit.
ASSET_COUNTS = {
    # electric
    "transmission_substation": 60,
    "distribution_substation": 350,
    "transmission_span": 900,
    "distribution_feeder": 700,
    "recloser": 200,
    # water
    "water_treatment_plant": 15,
    "pump_station": 300,
    "storage_tank": 150,
    "water_main": 800,
    "raw_water_intake": 40,
}

ELECTRIC_CLASSES = {
    "transmission_substation", "distribution_substation",
    "transmission_span", "distribution_feeder", "recloser",
}
WATER_CLASSES = {
    "water_treatment_plant", "pump_station", "storage_tank",
    "water_main", "raw_water_intake",
}
LINEAR_CLASSES = {"transmission_span", "distribution_feeder", "water_main"}

TOTAL_CUSTOMERS = 3_200_000
TOTAL_POPULATION = 8_000_000

# Residents behind one electricity account. DERIVED, not chosen: it is simply
# the ratio implied by the service territory above. An earlier version
# hard-coded 2.4, which contradicted this config (8.0M/3.2M = 2.5) and had no
# basis. Deriving it means the figure cannot drift from the estate it describes.
#
# For corroboration rather than justification: US average household size was
# ~2.5 persons in the 2024 ACS 1-year estimates. Note that utility customer
# accounts include commercial and industrial connections, so residents-per-
# account and household size are not the same quantity in general -- for SGW
# the territory ratio is the correct derivation.
RESIDENTS_PER_ACCOUNT = TOTAL_POPULATION / TOTAL_CUSTOMERS   # = 2.5

# Materials by class, with (name, weight, relative_fragility).
# relative_fragility multiplies the baseline hazard response: cast iron mains
# and wood poles are the classic problem populations.
MATERIALS = {
    "water_main": [
        ("cast_iron", 0.30, 1.85), ("ductile_iron", 0.32, 1.00),
        ("pvc", 0.22, 0.70), ("asbestos_cement", 0.11, 1.55),
        ("steel", 0.05, 1.10),
    ],
    "transmission_span": [
        ("steel_lattice", 0.62, 0.80), ("steel_monopole", 0.28, 0.70),
        ("concrete", 0.10, 0.95),
    ],
    "distribution_feeder": [
        ("wood_pole", 0.71, 1.40), ("composite_pole", 0.12, 0.85),
        ("concrete_pole", 0.09, 0.90), ("underground", 0.08, 0.25),
    ],
}
DEFAULT_MATERIAL = [("standard", 1.0, 1.0)]

# Criticality tiers. Tier 1 assets serve designated critical load
# (hospitals, dialysis centres, emergency services, care homes).
CRITICALITY_WEIGHTS = [0.06, 0.19, 0.75]   # tier 1, 2, 3

# How many people-equivalent one critical facility (hospital, dialysis centre,
# care home) counts for in the consequence score. THIS ONE IS ARBITRARY and we
# say so: it encodes a value judgement about how much more a hospital matters
# than a household. No research supports a specific figure. It is exposed as a
# tunable so the client sets it, and the risk ranking's sensitivity to it is
# reported rather than hidden.
CRITICAL_FACILITY_WEIGHT = 500.0

# ------------------------------------------------------------------ hazards
HAZARDS = ("hurricane", "heatwave", "inland_flood", "wildfire", "baseline")

# Wind speeds in m/s. 33 m/s is the Cat-1 threshold.
HURRICANE_MAX_WIND_RANGE = (28.0, 62.0)
# Heatwave daily maxima in degrees C.
HEATWAVE_PEAK_RANGE = (36.0, 44.0)
# Transformer thermal ageing: IEEE loading guidance puts the halving of
# insulation life at roughly a 6-8 C sustained rise above rating.
THERMAL_HALVING_DELTA_C = 7.0

# ------------------------------------------------------------------ history
HISTORY_YEARS = 3
HISTORY_END = "2026-08-01"
# Event mix per year of synthetic history.
EVENTS_PER_YEAR = {
    "hurricane": 2, "heatwave": 5, "inland_flood": 6,
    "wildfire": 3, "baseline": 26,
}

# ----------------------------------------------------------------- telemetry
TELEMETRY_ASSET_FRACTION = 0.22   # share of pump stations with SCADA history
TELEMETRY_DAYS = 180
TELEMETRY_FREQ_MIN = 15
DEGRADING_FRACTION = 0.08         # of instrumented pumps, share that decay

# --------------------------------------------------------------------- crews
N_DEPOTS = 15
CREW_TYPES = {          # type -> (count, assets it can work, cost per shift USD)
    "line": (58, ELECTRIC_CLASSES, 4200),
    "water": (34, WATER_CLASSES, 3600),
    "vegetation": (22, {"distribution_feeder", "transmission_span"}, 2800),
    "assessment": (18, ELECTRIC_CLASSES | WATER_CLASSES, 1900),
}

# ------------------------------------------------------- data quality defects
# These are injected deliberately. The PRD asks for "data quality
# considerations"; showing the resolution logic is worth more than shipping
# an implausibly clean dataset.
DEFECTS = {
    "missing_install_year": 0.08,
    "missing_material": 0.05,
    "coordinate_drift": 0.03,       # assets with GIS positions off by 50-400 m
    "stale_condition": 0.17,        # last inspected > 5 years ago
    "material_spelling_noise": 0.30,  # free-text variants in the CMMS extract
    "orphan_work_orders": 0.04,     # work orders referencing unknown asset ids
    "duplicate_assets": 0.015,      # same physical asset, two GIS records
    "diameter_unit_mix": 0.25,      # water mains recorded in inches not mm
}
