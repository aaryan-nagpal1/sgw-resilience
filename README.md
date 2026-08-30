# DSS V1 — Resilience Operations Decision Support

Prototype built for the AECOM AI Solution Engineer case.

Southeastern Grid & Water (SGW) runs the electricity network and the water network for
about 8 million residents across a region exposed to hurricanes, flooding, heatwaves and
wildfires. The data needed to prepare for those events already exists, but it is spread
across asset records, maintenance systems, weather feeds and field operations.

DSS V1 brings that information together to answer three questions during an event: what is
likely to fail, how many people will be affected, and what SGW should do about it today. It
is a decision support layer. It sits above SGW's existing systems, recommends actions, and
records what a person decided.

---

## Repository map

```
sgw_platform/
  datagen/          Synthetic SGW estate: geography, assets, network,
                    hazards, three years of operating history
  w1_risk/          Data cleaning, baselines, risk model, calibration,
                    evaluation, figures
  w2_dispatch/      Action catalogue, CP-SAT response planner,
                    baseline policies, Monte Carlo evaluation
  w3_assist/        Typed tool layer, AI assistant, situation reports
  w4_anomaly/       Pump condition monitoring

app/main.py         Prototype interface (5 tabs, solver runs live)
scripts/            One runner per stage, plus sensitivity and validation
data/               Generated dataset (large tables are regenerated, not committed)
artifacts/          Model outputs, metrics and plan files
figures/            Charts written by the evaluation scripts
```

---

## Running it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Each stage writes what the next one reads, so the order matters:

```bash
./.venv/bin/python -m sgw_platform.datagen.generate --out data   # ~15 s
./.venv/bin/python scripts/run_w1.py                             # ~60 s
./.venv/bin/python scripts/run_w2.py                             # ~30 s
./.venv/bin/python scripts/run_w4.py                             # ~40 s
./.venv/bin/streamlit run app/main.py                            # the demo
```

Two optional checks: `scripts/run_w2_sensitivity.py` sweeps the response-planning
assumptions, and `scripts/validate_dataset.py` confirms the generated data carries
recoverable signal.

`ANTHROPIC_API_KEY` is optional. Without it the assistant uses a deterministic router over
the same tools and returns the same numbers, so nothing in the demo depends on it.

---

## What each part does

| Part | Purpose |
|---|---|
| **Data generation** | Builds a plausible SGW estate: 3,515 assets, four hazard types, 126 scenarios and three years of failures. Realistic data problems are added deliberately, and a clean copy is held back so the cleaning can be scored |
| **W1 — Risk scoring** | Cleans the extracts, estimates the probability of failure for every asset every six hours over 72 hours, and multiplies it by the number of people affected |
| **W2 — Response planning** | Chooses pre-emptive actions and crew staging within the available budget and crews, then tests the plan against sampled failure outcomes |
| **W3 — Assistant and reporting** | Exposes the results as typed functions. The assistant selects which function to call and explains the answer, but does not calculate anything |
| **W4 — Condition monitoring** | Detects pumps losing efficiency during normal operations, so the system is used outside emergencies |

### Completeness of each part

Not every part is built to the same standard. The table below states which is which,
rather than leaving it to be discovered.

| Level | Meaning | Applies to |
|---|---|---|
| Production-shaped | Real evaluation, baselines and error analysis | W1, W2 |
| Working | Runs and can be inspected, but lightly evaluated | Interface, W4 |
| Thin by design | Demonstrates the pattern, no evaluation harness | W3 assistant and reports |
| Scoped, not built | Specified in the PRD and deliberately not coded | Damage assessment from imagery, multi-year capital planning |

---

## Key design choices

**The network model carries power-to-water dependencies.** Pumping stations and treatment
plants are electrical loads on SGW's own grid, so a substation failure also removes water
supply once backup generation runs out. Consequence propagates across that connection.
Three distribution substations reach the estate's six highest-impact assets for this
reason alone.

**External risk models are treated as inputs, not competitors.** Storm outage prediction,
water main failure prediction and vegetation risk can already be purchased. Rebuilding
them adds cost without addressing what happens after a risk signal is produced, so the
architecture allows those feeds to replace or sit alongside the internal model.

**The language model never produces a number.** W3 defines typed functions; the assistant
chooses which to call and how to word the reply. Removing the model changes the wording and
nothing else. In an emergency tool, a confident wrong number reaching a decision-maker is
the failure that matters most.

---

## Results

| Measure | Result |
|---|---|
| Impact captured by acting on the 200 highest-ranked assets | **27.8%**, compared with 2.1% using condition-score ranking |
| Calibration | Platt scaling halves expected calibration error (0.00131 to 0.00071) with ranking preserved exactly |
| Response plan quality | **14.0% lower** median realised impact than the strongest baseline, better in 98.4% of 500 simulated storms, at 7% lower cost |
| Sensitivity | The advantage holds between +10.2% and +19.1% across the full range of tested assumptions |
| Data cleaning, scored against the held-back truth | Material 100%, pipe units 100%, duplicate records 88.5% recall, installation year 72.5% |
| Condition monitoring | 24 of 24 degrading pumps detected with no false alarms |
| Planning speed | 3,515 assets replanned in 0.11 seconds, proven optimal |

Every figure above is written to `artifacts/` as CSV by the scripts, so each number can be
checked against the run that produced it.

---

## Assumptions

Every invented value is collected in `sgw_platform/datagen/config.py` and
`sgw_platform/w2_dispatch/actions.py`, so an assumption can be changed and its effect
re-run rather than argued about. The ones that shape the result most:

1. **Outage history can be linked to individual assets.** This is the largest delivery risk
   in a real engagement. In the synthetic extract, 4% of work orders reference retired
   asset IDs and are unusable as training labels.
2. **Electrical connections to pumping facilities are recorded somewhere.** The whole
   cross-network model depends on this. If the mapping does not exist it must be estimated
   from location data and validated with operations staff.
3. **Consequence weighting is a policy choice.** Electricity customers are weighted at 2.5
   residents per account, which is derived from the service territory, but the 500
   people-equivalent per critical facility is a value judgement that belongs to the client.
4. **Mitigation effect sizes are estimates.** No published source was found for this asset
   mix, which is why the sensitivity sweep exists. The claim is that the conclusion holds
   across the plausible range, not that one value is correct.
5. **"Real time" means five minutes, not sub-second**, so the pipeline runs in short timed
   batches rather than continuously.

---

## Limitations

**The data is simulated, so the model partly learns the generator.** Signal was checked to
be recoverable but not trivially so: one driver of pipe deterioration is published for only
40% of assets, so no model can be perfect. Performance on real data would differ.

**Flood depth is simplified** to surge plus un-drained rainfall minus site elevation and
defences. Production would need a coupled hydraulic model. Hurricane winds use a radial
profile with no asymmetry from forward motion.

**There is no hydraulic simulation.** Water consequence propagates over the network graph.
An incorrect hydraulic model would be worse than an honest graph-based one.

**Condition monitoring results are an upper bound.** Perfect recall and precision reflect a
clean injected signal, roughly a 35% efficiency loss against 5% measurement noise, rather
than the quality of the method.

**Mutual aid crews are excluded** from response planning, because their 48-hour lead time
is longer than the 72-hour decision window.

### Corrections made during development

Both are recorded because either would otherwise have produced a flattering but false
result.

The response-planning baselines originally committed every available crew to pre-emptive
work and kept none back for restoration, which no storm manager would do. That produced a
100% win rate for the optimiser which came from the baseline's resource split rather than
better asset selection. With a realistic 60% restoration reserve, the advantage is 14.0%.

Condition monitoring originally detected nothing, because it removed the trend using a
centred rolling median and therefore subtracted out the slow decline it was looking for. It
now compares each pump against a fixed healthy baseline period.
