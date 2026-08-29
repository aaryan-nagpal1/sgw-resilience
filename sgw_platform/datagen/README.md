# SGW synthetic data generator

Generates a complete, reproducible synthetic estate for **Southeastern Grid & Water**:
a dual power/water utility serving ~8M residents across a fictional US Southeast
territory exposed to hurricanes, inland flooding, heatwaves and wildfires.

```bash
python -m sgw_platform.datagen.generate --out data
python scripts/validate_dataset.py
```

Runs in ~15 s and produces ~34 MB. Everything is seeded (`config.SEED`); pass
`--seed` for an independent estate to test out-of-sample generalisation.

## Why a generator rather than a fixed dataset

The case permits mocked data. A generator earns more than a static CSV because
it makes the **assumptions executable**: every number a reviewer might question
lives in `config.py` and can be changed to see the consequence. It also lets the
same pipeline produce the training history *and* the live forecast scenario the
demo scrubs through, which is how the real system would work.

## Design principles

**1. Multi-hazard by construction.** The brief names four hazards; the research
literature's open problem is *compound* events. Four hazard types share one
pipeline (`hazards.py`) so a heatwave coinciding with drought is representable.
A per-hazard codepath would have made that impossible.

**2. Spatially coherent environment.** `geography.py` builds continuous fields —
coastline, elevation, population, vegetation, soil, drainage — and assets sample
them. Drawing each asset's environment independently would produce a dataset in
which no spatial model could ever help, quietly flattering any model trained on it.

**3. The power→water dependency is a first-class edge.** SGW owns both networks,
so every pump station and treatment plant is an electrical load on SGW's own
distribution grid (`network.py`, `edge_type == "power_dependency"`). A substation
failure darkens homes *and* silences pumps downstream, and a different population
loses water pressure hours later. In the generated history, a severe heatwave
produces **more pump stations lost to grid failure than to their own equipment**.

**4. Recoverable, but not trivially so.** `history.py` is the generative truth:
- several hazard pathways (wind, flood, thermal, fire, ground movement, pump duty)
  acting on different asset classes, with genuine interactions;
- **a latent driver** — soil corrosivity is published for only ~40% of the network,
  so no model can reach the Bayes rate, exactly as in reality;
- failures are Bernoulli draws, so identical assets under identical hazards do
  not share a fate;
- **functional loss propagates over graph edges, not feature columns**, so a
  purely tabular model structurally cannot see the cascade. That is the argument
  for keeping the network in the system rather than flattening it into features.

**5. Realistic data-quality defects, injected on purpose** (`defects.py`). The PRD
asks for data-quality considerations; a pristine dataset lets that section be
skipped. The published extracts carry missing install years, missing materials,
coordinate drift from digitised paper maps, stale inspection dates, mixed pipe
diameter units (in/mm), duplicate GIS records, orphan work orders referencing
retired asset IDs, free-text material spellings (`Cast Iron` / `CI` / `C.I.`),
and late-entered timestamps. `_assets_truth.csv` is held back so cleaning logic
can be **scored**, not merely asserted.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Every invented number, in one place, so the assumptions register can cite it |
| `geography.py` | Coastline, elevation, population, vegetation, soil, drainage fields |
| `assets.py` | ~3,500 assets across 10 classes with vintage, material, condition, criticality |
| `network.py` | Power and water topology, **power→water dependency edges**, consequence propagation |
| `hazards.py` | Four hazard scenario types → gridded fields → **per-asset exposure** |
| `history.py` | Generative failure model; 3 years of scenarios, failures and work orders |
| `telemetry.py` | SCADA for instrumented pumps, with slow degradation, dropout and stuck sensors |
| `crews.py` | Depots, crews by type, mutual aid with 48h lead time |
| `defects.py` | Deliberate data-quality corruption of the published extracts |
| `features.py` | Aggregation to the modelling grain + **scenario-wise chronological split** |
| `generate.py` | Orchestration and manifest |

## The four-step risk decomposition

The code follows the standard infrastructure risk decomposition rather than
collapsing everything into one "risk score":

| Step | Where | Example |
|---|---|---|
| **Hazard** — physical forcing on the grid | `hazards.hazard_fields` | 58 m/s gust, 2.1 m surge |
| **Exposure** — forcing sampled at the asset | `hazards.asset_exposure` | this substation, this bucket |
| **Vulnerability** — P(fail \| exposure) | `history.failure_probability` | 1987 vintage, 0.7 m inundation → 0.34 |
| **Consequence** — what is lost if it fails | `network.compute_consequence` | 18,400 customers + 40,000 lose water |

Linear assets (spans, feeders, mains) are sampled at both endpoints and the
midpoint and the **maximum** is taken — a line fails at its worst point, not its
average one. Sampling a linear asset at its centroid is a common and
consequential modelling error.

## Modelling grain and splitting

`features.build_training_table` aggregates the 6-hour exposure series to **one row
per (scenario, asset)**: *given the forecast for this event, does this asset fail
at any point during it?*

The per-bucket series is the right grain for the operational timeline, but the
wrong grain for training — consecutive buckets for one asset in one storm are
heavily autocorrelated, and treating them as independent inflates the effective
sample size roughly tenfold, producing a model that cross-validates beautifully
and fails in service.

`features.scenario_split` splits **by scenario, chronologically**. A random row
split leaks: the same storm appears in train and test and the model memorises
the event rather than the relationship.

## Validation results

`scripts/validate_dataset.py` on the default seed, chronological holdout
(95 train scenarios / 31 test):

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Prevalence (null) | 0.0041 | 0.500 | 0.00405 |
| Heuristic: old asset + high wind | 0.0062 | 0.514 | 0.00405 |
| Condition score only (what the CMMS already gives) | 0.0204 | 0.822 | 0.00405 |
| **LightGBM** | **0.1241** | **0.881** | **0.00390** |

**~30× lift over prevalence and ~6× over the condition-score ranking the client
already has.** Positive rate is 0.4–0.6%, so PR-AUC and Brier are reported rather
than accuracy — predicting "nothing fails" scores 99.6% accuracy and is useless.

Per-hazard PR-AUC: hurricane 0.19, inland flood 0.20, heatwave 0.17, wildfire
0.05, baseline 0.015. Baseline being hardest is correct — random equipment
failure on an ordinary day *is* genuinely unpredictable, and a generator that
made it learnable would be lying.

The model is **deliberately left uncalibrated** at this stage: middle deciles
under-predict and the top decile over-predicts. Calibration is therefore a real,
measurable improvement to demonstrate rather than a box to tick.

## Known limitations

- **Inundation is a simplification**: surge plus un-drained rainfall minus site
  elevation and defences. Production would need a coupled hydraulic/hydrologic model.
- **Hurricane winds use a Holland-style radial profile** with exponential overland
  decay — no asymmetry from forward motion, no terrain roughness.
- **No vegetation growth dynamics**; vegetation density is static, whereas real
  vegetation risk depends on time since last trim cycle.
- **Restoration duration** uses a congestion multiplier rather than simulating the
  crew queue. The dispatch optimiser (W2) is where that gets modelled properly.
- **Consequence weights are policy choices**, not physics — a critical facility
  counts as 500 people-equivalent. That belongs in the assumptions register and
  should be set with the client, not by the modeller.
