"""Baseline dispatch policies, held to the same budget and crew limits.

Comparing an optimiser against a deliberately hobbled alternative proves
nothing. Two fairness rules apply here:

1. Every policy faces the same budget and the same crew pool.
2. Every policy **reserves crews for restoration**. An earlier version let the
   greedy policies spend the entire line/water pool on pre-emptive work and
   stage nobody, which no storm manager would ever do; it handed the optimiser
   a 100% win rate that came from the baseline's resource split, not from
   better asset selection. The baselines now hold back `RESERVE_FRACTION` of
   the response pool, which is the standard heuristic, and the optimiser has to
   beat that on the merits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .actions import CATALOGUE, available_actions
from .optimise import Plan


# Share of the line/water pool a storm manager holds back for restoration
# rather than committing to pre-emptive work. An assumption; swept in
# sensitivity.py.
RESERVE_FRACTION = 0.60


def _greedy(risk, crews, budget_usd, order_col, max_stage_per_zone=12,
            ascending=False, rng=None, label="greedy",
            reserve_fraction=RESERVE_FRACTION):
    supply = crews[crews.home_depot != "MUTUAL_AID"].crew_type.value_counts().to_dict()
    response_pool = supply.get("line", 0) + supply.get("water", 0)
    action_budget_crews = int(round(response_pool * (1 - reserve_fraction)))
    used = {k: 0 for k in supply}
    spend, rows = 0.0, []
    ordered = (risk.sample(frac=1, random_state=0) if order_col == "_random"
               else risk.sort_values(order_col, ascending=ascending))
    for r in ordered.itertuples(index=False):
        best = None
        for a in available_actions(r.asset_class):
            act = CATALOGUE[a]
            if act.crew_type in ("line", "water"):
                spent = used.get("line", 0) + used.get("water", 0)
                if spent + act.crew_shifts > action_budget_crews:
                    continue
            elif used.get(act.crew_type, 0) + act.crew_shifts > supply.get(act.crew_type, 0):
                continue
            if spend + act.cost_usd > budget_usd:
                continue
            base = r.p_failure * r.consequence_score
            after = (r.p_failure * (r.consequence_score - r.cascade_population_water)
                     if act.severs_cascade
                     else r.p_failure * act.risk_multiplier * r.consequence_score)
            gain = max(base - after, 0.0)
            if best is None or gain > best[1]:
                best = (a, gain, act)
        if best is None:
            continue
        a, gain, act = best
        used[act.crew_type] = used.get(act.crew_type, 0) + act.crew_shifts
        spend += act.cost_usd
        rows.append({"asset_id": r.asset_id, "action": a, "cost": act.cost_usd,
                     "crew_type": act.crew_type, "zone_id": r.zone_id, "avoided": gain})

    # Staging: proportional to each zone's share of expected consequence.
    remaining = max(response_pool - sum(used.get(t, 0) for t in ("line", "water")), 0)
    zr = risk.groupby("zone_id").apply(
        lambda g: float((g.p_failure * g.consequence_score).sum()), include_groups=False)
    share = (zr / zr.sum()) if zr.sum() > 0 else zr * 0
    stage = pd.DataFrame({"zone_id": zr.index,
                          "crews_staged": np.minimum(
                              (share * remaining).round().astype(int),
                              max_stage_per_zone)})
    acts = pd.DataFrame(rows, columns=["asset_id", "action", "cost", "crew_type",
                                       "zone_id", "avoided"])
    return Plan(acts, stage, float("nan"), spend,
                acts.crew_type.value_counts().to_dict() if len(acts) else {},
                label, 0.0)


def rank_by_risk(risk, crews, budget_usd, **kw):
    """The obvious policy: act on the highest risk = P x consequence first."""
    r = risk.assign(_r=risk.p_failure * risk.consequence_score)
    return _greedy(r, crews, budget_usd, "_r", label="greedy rank-by-risk", **kw)


def rank_by_customers(risk, crews, budget_usd, **kw):
    """Closest to current practice: protect the biggest customer counts."""
    return _greedy(risk, crews, budget_usd, "customers_affected",
                   label="rank-by-customers", **kw)


def rank_by_probability(risk, crews, budget_usd, **kw):
    """What you get if you use a risk model but ignore consequence."""
    return _greedy(risk, crews, budget_usd, "p_failure",
                   label="rank-by-probability", **kw)


def status_quo(risk, crews, budget_usd, max_stage_per_zone=12, **kw):
    """No pre-emptive action; crews stay at home depots, spread evenly."""
    supply = crews[crews.home_depot != "MUTUAL_AID"].crew_type.value_counts().to_dict()
    total = supply.get("line", 0) + supply.get("water", 0)
    zones = sorted(risk.zone_id.unique())
    per = min(total // max(len(zones), 1), max_stage_per_zone)
    stage = pd.DataFrame({"zone_id": zones, "crews_staged": per})
    empty = pd.DataFrame(columns=["asset_id", "action", "cost", "crew_type",
                                  "zone_id", "avoided"])
    return Plan(empty, stage, float("nan"), 0.0, {}, "status quo (no action)", 0.0)


def random_policy(risk, crews, budget_usd, seed=0, **kw):
    """Sanity floor. If the optimiser cannot beat random, nothing else matters."""
    return _greedy(risk, crews, budget_usd, "_random", label="random", **kw)
