"""Operations copilot: natural language over the platform's own tools.

Architecture in one line: **the model chooses which function to call and how to
phrase the answer; it never computes a number.**

Two execution paths, and the demo works on either:

* `llm`      -- Claude with tool use, when ANTHROPIC_API_KEY is set.
* `fallback` -- a deterministic intent router over the same tool registry.

The fallback exists because a demo that dies without network access is not a
demo, but it also makes a design point that is easy to lose: if the answers are
still correct with the model removed, the model was never load-bearing for
correctness. That is the property you want in an emergency-operations tool.
"""
from __future__ import annotations

import json
import os
import re

from . import tools as T

MODEL = "claude-sonnet-5"
SYSTEM = """You are an operations assistant for Southeastern Grid & Water, a
combined electricity and water utility. You support emergency dispatchers and
duty managers during severe weather.

Rules:
- Answer only from the tools. Never estimate, infer or recall a number.
- If a tool cannot answer the question, say so plainly.
- Probabilities are model estimates, not guarantees. Do not overstate them.
- Be brief. Dispatchers are busy. Lead with the answer, then the evidence.
- Asset ids look like DSS-00168, PMP-02465, WTP-02219.
"""


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------- fallback
_INTENTS = [
    (r"cascad|water.*(grid|power)|(grid|power).*water|knock.*water",
     "get_cascade_risks"),
    (r"maintenance|anomal|degrad|pump.*health|blue.?sky", "get_maintenance_alerts"),
    (r"zone\s*(\d+)", "get_zone_summary"),
    (r"\b([A-Z]{3}-\d{5})\b", "explain_asset_risk"),
    (r"risk|worst|top|worry|priorit", "get_assets_at_risk"),
]


def _route(question: str, scenario_id: str):
    q = question.strip()
    for pattern, tool in _INTENTS:
        m = re.search(pattern, q, re.I)
        if not m:
            continue
        if tool == "get_zone_summary":
            return tool, {"scenario_id": scenario_id, "zone_id": int(m.group(1))}
        if tool == "explain_asset_risk":
            return tool, {"scenario_id": scenario_id, "asset_id": m.group(1).upper()}
        if tool == "get_maintenance_alerts":
            return tool, {}
        args = {"scenario_id": scenario_id, "top_n": 5}
        if tool == "get_assets_at_risk":
            if re.search(r"\bwater\b", q, re.I):
                args["network"] = "water"
            elif re.search(r"electric|grid|power", q, re.I):
                args["network"] = "electric"
        return tool, args
    return "get_assets_at_risk", {"scenario_id": scenario_id, "top_n": 5}


def _phrase(tool: str, result: dict) -> str:
    """Deterministic phrasing. Plain, and honest about being a template."""
    if tool == "get_assets_at_risk":
        lines = [f"Top {result['count']} by risk (probability x consequence):"]
        for a in result["assets"]:
            lines.append(f"  {a['asset_id']}  {a['asset_class'].replace('_',' ')}  "
                         f"zone {a['zone_id']}  P={a['p_failure']:.3f}  "
                         f"consequence {a['consequence']:,}")
        return "\n".join(lines)
    if tool == "get_cascade_risks":
        lines = ["Assets whose failure would cut water supply through the grid:"]
        for a in result["assets"]:
            flag = "covered by plan" if a["protected_by_plan"] else "NOT covered"
            lines.append(f"  {a['asset_id']}  P={a['p_failure']:.3f}  "
                         f"{a['water_population_at_risk']:,} people on water  ({flag})")
        return "\n".join(lines)
    if tool == "explain_asset_risk":
        if "error" in result:
            return result["error"]
        r = result
        s = (f"{r['asset_id']} ({r['asset_class'].replace('_',' ')}) ranks "
             f"#{r['rank_in_scenario']} this scenario.\n"
             f"  P(failure) {r['p_failure']:.3f}; consequence "
             f"{r['consequence_people_equivalent']:,} people-equivalent\n"
             f"  {r['customers_affected']:,} customers, "
             f"{r['critical_facilities_affected']} critical facilities")
        if r["water_population_lost_via_grid"]:
            s += (f"\n  {r['water_population_lost_via_grid']:,} people would lose "
                  f"water supply through the power dependency")
        s += (f"\n  Planned action: {r['planned_action']} "
              f"(${r['action_cost_usd']:,})" if r["planned_action"]
              else "\n  No action planned within budget")
        return s
    if tool == "get_zone_summary":
        r = result
        return (f"Zone {r['zone_id']}: {r['assets']} assets, "
                f"{r['assets_above_1pct_risk']} above 1% failure probability.\n"
                f"  Expected consequence {r['total_expected_consequence']:,}; "
                f"{r['critical_facilities']} critical facilities\n"
                f"  Plan: {r['planned_actions']} actions (${r['planned_spend_usd']:,}), "
                f"{r['crews_staged']} crews staged")
    if tool == "get_maintenance_alerts":
        lines = [f"{result['alerting']} of {result['instrumented']} instrumented "
                 f"pumps are showing sustained efficiency loss:"]
        for a in result["alerts"][:6]:
            q = "" if a["data_quality"] == "ok" else f"  [{a['data_quality']}]"
            lines.append(f"  {a['asset_id']}  running at "
                         f"{a['efficiency_vs_baseline']:.0%} of baseline  "
                         f"since {a['first_alert'][:10]}{q}")
        return "\n".join(lines)
    return json.dumps(result, indent=1)[:1200]


def ask(question: str, scenario_id: str, force_fallback: bool = False) -> dict:
    """Return {'answer', 'tool_calls', 'mode'}."""
    if force_fallback or not available():
        tool, args = _route(question, scenario_id)
        result = T.call(tool, args)
        return {"answer": _phrase(tool, result),
                "tool_calls": [{"tool": tool, "arguments": args, "result": result}],
                "mode": "deterministic (no API key set)"}

    import anthropic
    client = anthropic.Anthropic()
    msgs = [{"role": "user",
             "content": f"Current scenario is {scenario_id}.\n\n{question}"}]
    calls = []
    for _ in range(5):
        resp = client.messages.create(model=MODEL, max_tokens=900, system=SYSTEM,
                                      tools=T.tool_specs(), messages=msgs)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "tool_calls": calls, "mode": f"llm ({MODEL})"}
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                out = T.call(b.name, b.input)
            except Exception as e:                      # surfaced, not swallowed
                out = {"error": str(e)}
            calls.append({"tool": b.name, "arguments": b.input, "result": out})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out, default=str)})
        msgs.append({"role": "user", "content": results})
    return {"answer": "Stopped after 5 tool-use rounds without a final answer.",
            "tool_calls": calls, "mode": f"llm ({MODEL})"}
