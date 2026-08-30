"""Typed tools over the platform's own outputs.

These are the functions the assistant is allowed to call. Two deliberate
choices:

1. **Tools, not retrieval.** The assistant does not search a document pile and
   paraphrase. It calls a function that returns numbers computed by W1/W2, so
   every figure in an answer is traceable to a specific call with specific
   arguments. Retrieval-augmented answers over operational data are how you get
   a confident wrong number during an emergency.
2. **Pydantic-validated arguments and returns.** The same schemas serialise to
   the JSON Schema the model is given, so there is one definition rather than a
   prompt and a validator that drift apart.

In production these become FastAPI endpoints and are exposed to other AECOM
agents over MCP. They are plain functions here so the demo runs in one process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

ART, DATA = Path("artifacts"), Path("data")
_CACHE: dict = {}


def _load(name, loader):
    if name not in _CACHE:
        _CACHE[name] = loader()
    return _CACHE[name]


def risk_table() -> pd.DataFrame:
    return _load("risk", lambda: pd.read_parquet(ART / "w1_risk_scores.parquet"))


def plan_actions() -> pd.DataFrame:
    return _load("plan", lambda: pd.read_csv(ART / "w2_plan_actions.csv"))


def plan_staging() -> pd.DataFrame:
    return _load("stage", lambda: pd.read_csv(ART / "w2_plan_staging.csv"))


def anomaly_summary() -> pd.DataFrame:
    return _load("anom", lambda: pd.read_csv(ART / "w4_summary.csv"))


# ----------------------------------------------------------------- schemas
class AssetsAtRiskArgs(BaseModel):
    """Rank assets by risk (probability x consequence) for a scenario."""
    scenario_id: str = Field(description="Scenario identifier, e.g. HURR-20251231-086")
    top_n: int = Field(10, ge=1, le=100, description="How many assets to return")
    zone_id: int | None = Field(None, description="Restrict to one operating zone")
    network: Literal["electric", "water", "any"] = "any"


class ExplainAssetArgs(BaseModel):
    """Explain why one asset is ranked where it is."""
    asset_id: str
    scenario_id: str


class ZoneSummaryArgs(BaseModel):
    """Summarise risk and planned response for one zone."""
    scenario_id: str
    zone_id: int


class CascadeArgs(BaseModel):
    """Find assets whose failure would cut water supply through the power grid."""
    scenario_id: str
    top_n: int = Field(10, ge=1, le=50)


# ------------------------------------------------------------------ tools
def get_assets_at_risk(args: AssetsAtRiskArgs) -> dict:
    d = risk_table()
    d = d[d.scenario_id == args.scenario_id]
    if args.zone_id is not None:
        d = d[d.zone_id == args.zone_id]
    if args.network != "any":
        water = {"pump_station", "water_treatment_plant", "storage_tank",
                 "water_main", "raw_water_intake"}
        d = d[d.asset_class.isin(water)] if args.network == "water" \
            else d[~d.asset_class.isin(water)]
    d = d.nlargest(args.top_n, "risk")
    return {"scenario_id": args.scenario_id, "count": len(d),
            "assets": [{"asset_id": r.asset_id, "asset_class": r.asset_class,
                        "zone_id": int(r.zone_id),
                        "p_failure": round(float(r.p_failure), 4),
                        "consequence": int(r.consequence_score),
                        "risk": int(r.risk)} for r in d.itertuples()]}


def explain_asset_risk(args: ExplainAssetArgs) -> dict:
    d = risk_table()
    row = d[(d.asset_id == args.asset_id) & (d.scenario_id == args.scenario_id)]
    if row.empty:
        return {"error": f"{args.asset_id} not found in {args.scenario_id}"}
    r = row.iloc[0]
    plan = plan_actions()
    act = plan[plan.asset_id == args.asset_id]
    rank = int((d[d.scenario_id == args.scenario_id].risk > r.risk).sum()) + 1
    return {
        "asset_id": r.asset_id, "asset_class": r.asset_class,
        "rank_in_scenario": rank,
        "p_failure": round(float(r.p_failure), 4),
        "consequence_people_equivalent": int(r.consequence_score),
        "customers_affected": int(r.customers_affected),
        "water_population_lost_via_grid": int(r.cascade_population_water),
        "critical_facilities_affected": int(r.critical_facilities_affected),
        "planned_action": (act.action.iloc[0] if len(act) else None),
        "action_cost_usd": (int(act.cost.iloc[0]) if len(act) else None),
    }


def get_zone_summary(args: ZoneSummaryArgs) -> dict:
    d = risk_table()
    d = d[(d.scenario_id == args.scenario_id) & (d.zone_id == args.zone_id)]
    plan, stage = plan_actions(), plan_staging()
    pz = plan[plan.zone_id == args.zone_id]
    sz = stage[stage.zone_id == args.zone_id]
    return {
        "zone_id": args.zone_id, "assets": len(d),
        "total_expected_consequence": int((d.p_failure * d.consequence_score).sum()),
        "assets_above_1pct_risk": int((d.p_failure > 0.01).sum()),
        "critical_facilities": int(d.critical_facilities_affected.sum()),
        "planned_actions": len(pz),
        "planned_spend_usd": int(pz.cost.sum()) if len(pz) else 0,
        "crews_staged": int(sz.crews_staged.iloc[0]) if len(sz) else 0,
    }


def get_cascade_risks(args: CascadeArgs) -> dict:
    """Assets whose failure takes out water supply via the power dependency.
    This is the view no siloed electric or water tool can produce."""
    d = risk_table()
    d = d[(d.scenario_id == args.scenario_id) & (d.cascade_population_water > 0)]
    d = d.assign(cascade_risk=d.p_failure * d.cascade_population_water)
    d = d.nlargest(args.top_n, "cascade_risk")
    plan = set(plan_actions().asset_id)
    return {"count": len(d), "assets": [
        {"asset_id": r.asset_id, "asset_class": r.asset_class,
         "p_failure": round(float(r.p_failure), 4),
         "water_population_at_risk": int(r.cascade_population_water),
         "expected_water_population_lost": int(r.cascade_risk),
         "protected_by_plan": r.asset_id in plan} for r in d.itertuples()]}


def get_maintenance_alerts(args: BaseModel | None = None) -> dict:
    """Blue-sky-day condition monitoring alerts from W4."""
    s = anomaly_summary()
    a = s[s.alerted]
    return {"instrumented": len(s), "alerting": len(a), "alerts": [
        {"asset_id": r.asset_id, "first_alert": str(r.first_alert),
         "efficiency_vs_baseline": round(float(r.final_efficiency_ratio), 3),
         "data_quality": r.data_quality} for r in a.head(10).itertuples()]}


REGISTRY = {
    "get_assets_at_risk": (get_assets_at_risk, AssetsAtRiskArgs),
    "explain_asset_risk": (explain_asset_risk, ExplainAssetArgs),
    "get_zone_summary": (get_zone_summary, ZoneSummaryArgs),
    "get_cascade_risks": (get_cascade_risks, CascadeArgs),
    "get_maintenance_alerts": (get_maintenance_alerts, None),
}


def tool_specs() -> list[dict]:
    """Anthropic tool-use schemas, generated from the pydantic models so the
    prompt and the validator cannot drift apart."""
    specs = []
    for name, (fn, model) in REGISTRY.items():
        schema = (model.model_json_schema() if model
                  else {"type": "object", "properties": {}})
        schema.pop("title", None)
        desc = (fn.__doc__ or "").strip() or (model.__doc__ or "").strip() if model \
            else (fn.__doc__ or "").strip()
        schema.pop("description", None)
        specs.append({"name": name, "description": desc.split("\n\n")[0],
                      "input_schema": schema})
    return specs


def call(name: str, arguments: dict):
    fn, model = REGISTRY[name]
    return fn(model(**arguments)) if model else fn()
