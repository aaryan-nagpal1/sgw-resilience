"""Situation report generation.

The report is assembled from typed data, not written by a model from scratch.
Every number is computed here and passed in; the language model is only allowed
to phrase the narrative paragraph, and the demo works with no model at all.

That split is deliberate. During an emergency the failure mode that matters is
a fluent, confident, wrong number reaching a decision-maker. Keeping the
arithmetic out of the model removes that class of error entirely.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field


class ZoneLine(BaseModel):
    zone_id: int
    assets_at_risk: int
    crews_staged: int
    planned_actions: int
    expected_consequence: int


class SitRep(BaseModel):
    """Structured situation report. Rendered to text; never free-form."""
    scenario_id: str
    hazard: str
    generated_at: datetime
    headline: str
    assets_assessed: int
    assets_above_1pct: int
    total_expected_consequence: int
    critical_facilities_exposed: int
    water_population_at_cascade_risk: int
    planned_actions: int
    planned_spend_usd: int
    crews_staged: int
    top_risks: list[dict] = Field(default_factory=list)
    zones: list[ZoneLine] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    narrative: str | None = None


def build(risk: pd.DataFrame, actions: pd.DataFrame, staging: pd.DataFrame,
          scenario_id: str) -> SitRep:
    d = risk[risk.scenario_id == scenario_id]
    casc = d[d.cascade_population_water > 0]
    top = d.nlargest(5, "risk")

    zones = []
    for z, g in d.groupby("zone_id"):
        az = actions[actions.zone_id == z] if "zone_id" in actions else actions.iloc[:0]
        sz = staging[staging.zone_id == z]
        zones.append(ZoneLine(
            zone_id=int(z), assets_at_risk=int((g.p_failure > 0.01).sum()),
            crews_staged=int(sz.crews_staged.iloc[0]) if len(sz) else 0,
            planned_actions=len(az),
            expected_consequence=int((g.p_failure * g.consequence_score).sum())))

    exp = int((d.p_failure * d.consequence_score).sum())
    n_hi = int((d.p_failure > 0.01).sum())
    return SitRep(
        scenario_id=scenario_id, hazard=str(d.hazard.iloc[0]),
        generated_at=datetime.now(),
        headline=(f"{n_hi} assets above 1% failure probability; "
                  f"{len(actions)} pre-emptive actions planned; "
                  f"{int(staging.crews_staged.sum())} crews staged"),
        assets_assessed=len(d), assets_above_1pct=n_hi,
        total_expected_consequence=exp,
        critical_facilities_exposed=int(
            d.loc[d.p_failure > 0.01, "critical_facilities_affected"].sum()),
        water_population_at_cascade_risk=int(
            (casc.p_failure * casc.cascade_population_water).sum()),
        planned_actions=len(actions),
        planned_spend_usd=int(actions.cost.sum()) if len(actions) else 0,
        crews_staged=int(staging.crews_staged.sum()),
        top_risks=[{"asset_id": r.asset_id, "asset_class": r.asset_class,
                    "p_failure": round(float(r.p_failure), 3),
                    "consequence": int(r.consequence_score)}
                   for r in top.itertuples()],
        zones=sorted(zones, key=lambda z: -z.expected_consequence)[:6],
        caveats=[
            "Failure probabilities are model estimates, calibrated on 18 held-out "
            "scenarios; they are not guarantees.",
            "Consequence weighting counts one critical facility as 500 "
            "people-equivalent. That is a policy choice, not a measurement.",
            "Mutual aid crews are excluded: 48-hour lead time exceeds this "
            "decision window.",
        ])


def render_markdown(s: SitRep) -> str:
    """Deterministic rendering. No model involved."""
    L = [f"# Situation Report — {s.scenario_id}",
         f"**Hazard:** {s.hazard}  |  **Generated:** {s.generated_at:%Y-%m-%d %H:%M}",
         "", f"**{s.headline}**", ""]
    if s.narrative:
        L += [s.narrative, ""]
    L += ["## Position",
          f"- Assets assessed: {s.assets_assessed:,}",
          f"- Above 1% failure probability: {s.assets_above_1pct}",
          f"- Expected consequence: {s.total_expected_consequence:,} people-equivalent",
          f"- Critical facilities exposed: {s.critical_facilities_exposed}",
          f"- Water population at risk **via grid failure**: "
          f"{s.water_population_at_cascade_risk:,}",
          "", "## Response",
          f"- Pre-emptive actions planned: {s.planned_actions} "
          f"(${s.planned_spend_usd:,})",
          f"- Crews staged: {s.crews_staged}", "", "## Highest risk assets", "",
          "| Asset | Class | P(fail) | Consequence |", "|---|---|---|---|"]
    for t in s.top_risks:
        L.append(f"| {t['asset_id']} | {t['asset_class'].replace('_',' ')} | "
                 f"{t['p_failure']:.3f} | {t['consequence']:,} |")
    L += ["", "## By zone", "",
          "| Zone | Assets at risk | Actions | Crews | Expected consequence |",
          "|---|---|---|---|---|"]
    for z in s.zones:
        L.append(f"| {z.zone_id} | {z.assets_at_risk} | {z.planned_actions} | "
                 f"{z.crews_staged} | {z.expected_consequence:,} |")
    L += ["", "## Caveats", ""] + [f"- {c}" for c in s.caveats]
    return "\n".join(L)
