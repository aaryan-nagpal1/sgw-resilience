"""W2: constrained pre-emptive action selection and crew staging.

The decision is not "which assets are riskiest" -- W1 already answers that.
It is: **given a fixed number of crews and a fixed budget, is this crew better
spent hardening an asset now, or held in reserve to restore faster afterwards?**
That trade-off is the reason a solver is warranted rather than a sorted list.

Formulation (CP-SAT):

  minimise   sum_a  P_a(after action) * consequence_a          [residual risk]
           + sum_z  restoration_loss[z][k_z]                   [response burden]

  subject to at most one action per asset
             crew-shifts consumed by actions + crews staged <= crews available
             total action cost <= budget

The restoration term is piecewise-linear in the number of crews staged per zone,
pre-computed and selected by a one-hot integer variable. That keeps an
inherently non-linear term exact and the model solvable in under a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from .actions import CATALOGUE, available_actions

SCALE = 100  # objective is integerised; consequence is in people-equivalents


@dataclass
class Plan:
    actions: pd.DataFrame          # asset_id, action, cost, crew_type
    staging: pd.DataFrame          # zone_id, crews_staged
    objective: float
    spend: float
    crews_used: dict
    status: str
    solve_seconds: float


def restoration_curve(zone_risk: float, max_crews: int, base_hours: float = 18.0):
    """Expected restoration burden in a zone as a function of crews staged.

    Diminishing returns: the first crew helps a lot, the tenth much less.
    Burden is expected-consequence x expected restoration hours.
    """
    k = np.arange(max_crews + 1)
    hours = base_hours / (1.0 + 0.55 * k)
    return zone_risk * hours


def build_and_solve(risk: pd.DataFrame, crews: pd.DataFrame, *, budget_usd: float,
                    max_stage_per_zone: int = 12, time_limit_s: float = 20.0,
                    locked: set | None = None, forbidden: set | None = None) -> Plan:
    """risk: one row per asset with p_failure, consequence_score,
    cascade_population_water, asset_class, zone_id."""
    m = cp_model.CpModel()
    risk = risk.reset_index(drop=True)
    zones = sorted(risk.zone_id.unique())

    # Crew supply by type (mutual aid excluded: 48 h lead time, out of scope
    # for a 72 h decision window -- stated as a limitation).
    supply = crews[crews.home_depot != "MUTUAL_AID"].crew_type.value_counts().to_dict()

    # ---- decision variables: at most one action per asset ------------------
    x, meta = {}, []
    for i, r in risk.iterrows():
        for a in available_actions(r.asset_class):
            act = CATALOGUE[a]
            v = m.NewBoolVar(f"x_{i}_{a}")
            x[(i, a)] = v
            # Avoided expected consequence if this action is taken.
            base = r.p_failure * r.consequence_score
            if act.severs_cascade:
                after = r.p_failure * (r.consequence_score - r.cascade_population_water)
            else:
                after = r.p_failure * act.risk_multiplier * r.consequence_score
            meta.append({"i": i, "action": a, "var": v,
                         "avoided": max(base - after, 0.0),
                         "cost": act.cost_usd, "crew_type": act.crew_type,
                         "shifts": act.crew_shifts, "asset_id": r.asset_id,
                         "zone_id": r.zone_id})
    md = pd.DataFrame(meta)
    for i, g in md.groupby("i"):
        m.AddAtMostOne(list(g["var"]))

    # Operator overrides: locked actions must be taken, forbidden ones cannot.
    for aid in (locked or set()):
        g = md[md.asset_id == aid]
        if len(g):
            m.AddExactlyOne(list(g["var"]))
    for aid in (forbidden or set()):
        for v in md[md.asset_id == aid]["var"]:
            m.Add(v == 0)

    # ---- staging: one-hot over crew counts per zone -------------------------
    y, stage_terms, staged_expr = {}, [], []
    zone_risk = risk.groupby("zone_id").apply(
        lambda g: float((g.p_failure * g.consequence_score).sum()), include_groups=False)
    for z in zones:
        curve = restoration_curve(float(zone_risk.get(z, 0.0)), max_stage_per_zone)
        vs = [m.NewBoolVar(f"y_{z}_{k}") for k in range(max_stage_per_zone + 1)]
        m.AddExactlyOne(vs)
        for k, v in enumerate(vs):
            y[(z, k)] = v
            stage_terms.append((v, curve[k]))
        staged_expr.append(sum(k * vs[k] for k in range(max_stage_per_zone + 1)))

    # ---- constraints --------------------------------------------------------
    m.Add(sum(int(r.cost) * r["var"] for _, r in md.iterrows()) <= int(budget_usd))

    # Line and water crews are ONE shared pool: a crew spent hardening an asset
    # now is a crew not available to restore afterwards. That shared constraint
    # is the whole reason this is an optimisation rather than a sorted list.
    response_pool = supply.get("line", 0) + supply.get("water", 0)
    shared = md[md.crew_type.isin(["line", "water"])]
    m.Add(sum(int(r.shifts) * r["var"] for _, r in shared.iterrows())
          + sum(staged_expr) <= int(response_pool))

    # Vegetation crews cannot restore plant, so they are a separate pool.
    for ctype in ("vegetation", "assessment"):
        used = [int(r.shifts) * r["var"] for _, r in md[md.crew_type == ctype].iterrows()]
        if used:
            m.Add(sum(used) <= int(supply.get(ctype, 0)))

    # ---- objective: minimise residual risk + restoration burden -------------
    total_base = float((risk.p_failure * risk.consequence_score).sum())
    obj = (int(total_base * SCALE)
           - sum(int(r.avoided * SCALE) * r["var"] for _, r in md.iterrows())
           + sum(int(c * SCALE / 18.0) * v for v, c in stage_terms))
    m.Minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(m)

    chosen = md[[solver.Value(v) == 1 for v in md["var"]]].copy()
    stage = pd.DataFrame([
        {"zone_id": z,
         "crews_staged": next(k for k in range(max_stage_per_zone + 1)
                              if solver.Value(y[(z, k)]) == 1)}
        for z in zones])
    return Plan(
        actions=chosen[["asset_id", "action", "cost", "crew_type", "zone_id", "avoided"]],
        staging=stage,
        objective=solver.ObjectiveValue() / SCALE,
        spend=float(chosen.cost.sum()) if len(chosen) else 0.0,
        crews_used=chosen.crew_type.value_counts().to_dict() if len(chosen) else {},
        status=solver.StatusName(status),
        solve_seconds=solver.WallTime(),
    )
