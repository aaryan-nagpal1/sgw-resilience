"""Monte Carlo evaluation of dispatch plans.

Scoring plans on *expected* consequence is circular: it evaluates the optimiser
using the same numbers the optimiser maximised, so it always wins. Instead we
sample actual failure realisations from the calibrated probabilities, apply each
plan to that realisation, and measure realised loss.

The output is a distribution, not a point estimate -- including the fraction of
storms in which the optimiser *loses*, which is the honest part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .actions import CATALOGUE
from .optimise import restoration_curve


def apply_plan(risk: pd.DataFrame, plan) -> pd.DataFrame:
    """Return risk with p_failure and consequence adjusted for the plan."""
    r = risk.copy()
    r["p_after"] = r.p_failure
    r["consequence_after"] = r.consequence_score
    if len(plan.actions):
        act_by_asset = dict(zip(plan.actions.asset_id, plan.actions.action))
        for aid, aname in act_by_asset.items():
            act = CATALOGUE[aname]
            m = r.asset_id == aid
            if act.severs_cascade:
                r.loc[m, "consequence_after"] = (r.loc[m, "consequence_score"]
                                                 - r.loc[m, "cascade_population_water"])
            else:
                r.loc[m, "p_after"] = r.loc[m, "p_failure"] * act.risk_multiplier
    return r


def realised_loss(r: pd.DataFrame, plan, rng, base_hours=18.0):
    """One sampled storm: draw failures, weight by restoration time in the zone."""
    failed = rng.random(len(r)) < r.p_after.to_numpy()
    stage = dict(zip(plan.staging.zone_id, plan.staging.crews_staged))
    k = r.zone_id.map(stage).fillna(0).to_numpy()
    hours = base_hours / (1.0 + 0.55 * k)
    return float((failed * r.consequence_after.to_numpy() * hours).sum())


def monte_carlo(risk, plans: dict, n=500, seed=11):
    """Replay every plan against the SAME sampled storms (common random
    numbers), so differences are due to the plans and not to sampling noise."""
    prepared = {name: apply_plan(risk, p) for name, p in plans.items()}
    out = {name: np.empty(n) for name in plans}
    for i in range(n):
        seeds = np.random.default_rng(seed + i)
        state = seeds.bit_generator.state
        for name, r in prepared.items():
            rng = np.random.default_rng(seed + i)   # identical draw per plan
            out[name][i] = realised_loss(r, plans[name], rng)
    return pd.DataFrame(out)


def summarise(losses: pd.DataFrame, reference: str, plans: dict | None = None):
    ref = losses[reference]
    rows = []
    for name in losses.columns:
        v = losses[name]
        rows.append({
            "plan": name,
            "median realised loss": v.median(),
            "mean": v.mean(),
            "p90 (bad storms)": v.quantile(0.90),
            f"median reduction vs {reference}": 1 - v.median() / ref.median(),
            f"win rate vs {reference}": float((v < ref).mean()),
            "spend_usd": plans[name].spend if plans else np.nan,
        })
    return pd.DataFrame(rows).sort_values("median realised loss")
