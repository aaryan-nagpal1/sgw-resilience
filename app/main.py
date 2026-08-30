"""SGW Resilience Operations Layer -- prototype UI.

Scope note, stated here and in the README: this is a demonstration of the
decision workflow, not a production console. It is intentionally plain. The
engineering effort in this project went into the risk model, the optimiser and
their evaluation; the UI exists to make those inspectable by a human.

Run:  streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pydeck as pdk
import streamlit as st

from sgw_platform.w2_dispatch import baselines as B, optimise as O
from sgw_platform.w3_assist import copilot as C, sitrep as SR

DATA, ART = Path("data"), Path("artifacts")
NAVY = "#10345C"

st.set_page_config(page_title="SGW Resilience Operations",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load():
    risk = pd.read_parquet(ART / "w1_risk_scores.parquet")
    gis = pd.read_csv(DATA / "gis_asset_extract.csv",
                      usecols=["asset_id", "lat", "lon"]).drop_duplicates("asset_id")
    risk = risk.merge(gis, on="asset_id", how="left")
    return (risk,
            pd.read_csv(DATA / "crews.csv"),
            pd.read_csv(ART / "w2_plan_actions.csv"),
            pd.read_csv(ART / "w2_plan_staging.csv"),
            pd.read_csv(ART / "w4_summary.csv"),
            pd.read_parquet(ART / "w2_monte_carlo.parquet"),
            pd.read_parquet(DATA / "hazard_exposure.parquet",
                            columns=["scenario_id", "bucket", "timestamp",
                                     "wind_gust_ms", "temp_c", "inundation_depth_m"]))


risk, crews, plan_actions, plan_staging, anomalies, mc, exposure = load()


@st.cache_data(show_spinner=False)
def solve_cached(scenario_id, budget, forbidden):
    sub = risk[risk.scenario_id == scenario_id]
    return O.build_and_solve(sub, crews, budget_usd=budget, forbidden=set(forbidden))


@st.cache_data(show_spinner=False)
def greedy_cached(scenario_id, budget):
    sub = risk[risk.scenario_id == scenario_id]
    return B.rank_by_risk(sub, crews, budget)

st.sidebar.markdown(f"### SGW Resilience Operations")
st.sidebar.caption("Prototype — decision workflow demonstration")
scenarios = (risk.groupby(["scenario_id", "hazard"]).failed.sum()
             .reset_index().sort_values("failed", ascending=False))
labels = {f"{r.scenario_id}  ({r.hazard})": r.scenario_id for r in scenarios.itertuples()}
choice = st.sidebar.selectbox("Scenario", list(labels), index=0)
sid = labels[choice]
r = risk[risk.scenario_id == sid].copy()

st.sidebar.metric("Assets assessed", f"{len(r):,}")
st.sidebar.metric("Above 1% failure risk", int((r.p_failure > 0.01).sum()))
st.sidebar.metric("Expected consequence",
                  f"{int((r.p_failure * r.consequence_score).sum()):,}")
st.sidebar.caption("Consequence is in people-equivalent: customers x 2.5, "
                   "plus water population, plus 500 per critical facility.")

tabs = st.tabs(["Situation", "Response plan", "Copilot", "Situation report",
                "Condition monitoring"])

# ------------------------------------------------------------------ SITUATION
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Asset risk")
        m = r.dropna(subset=["lat", "lon"]).copy()
        m["radius"] = (m.risk / m.risk.max() * 5500 + 400).clip(400, 6000)
        hi = m.risk.quantile(0.98)
        m["color"] = m.risk.apply(
            lambda v: [217, 140, 31, 200] if v >= hi else [31, 111, 178, 110])
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=float(m.lat.mean()),
                                             longitude=float(m.lon.mean()),
                                             zoom=6.2, pitch=0),
            layers=[pdk.Layer("ScatterplotLayer", m, get_position=["lon", "lat"],
                              get_radius="radius", get_fill_color="color",
                              pickable=True, opacity=0.7)],
            tooltip={"text": "{asset_id}\n{asset_class}\nrisk {risk}"}))
        st.caption("Amber = top 2% by risk. Risk = calibrated P(failure) x consequence.")
    with c2:
        st.subheader("Hazard through the event")
        e = exposure[exposure.scenario_id == sid]
        if len(e):
            agg = e.groupby("timestamp").agg(
                peak_gust=("wind_gust_ms", "max"), peak_temp=("temp_c", "max"),
                peak_inundation=("inundation_depth_m", "max"))
            st.line_chart(agg[["peak_gust", "peak_temp"]], height=180)
            st.caption("Peak gust (m/s) and temperature (C) across the estate, "
                       "6-hourly. Exposure is sampled per asset, not per region.")
        st.subheader("Highest risk")
        st.dataframe(
            r.nlargest(10, "risk")[["asset_id", "asset_class", "p_failure",
                                    "consequence_score", "cascade_population_water"]]
            .rename(columns={"consequence_score": "consequence",
                             "cascade_population_water": "water via grid"})
            .style.format({"p_failure": "{:.3f}", "consequence": "{:,.0f}",
                           "water via grid": "{:,.0f}"}),
            use_container_width=True, hide_index=True)

    casc = r[r.cascade_population_water > 0].assign(
        cr=lambda d: d.p_failure * d.cascade_population_water).nlargest(6, "cr")
    if len(casc):
        st.subheader("Cross-network cascade")
        st.caption("Grid assets whose failure would interrupt water supply. "
                   "No electric-only or water-only tool produces this view.")
        st.dataframe(casc[["asset_id", "asset_class", "p_failure",
                           "cascade_population_water", "cr"]]
                     .rename(columns={"cascade_population_water": "people on water",
                                      "cr": "expected people losing water"})
                     .style.format({"p_failure": "{:.3f}",
                                    "people on water": "{:,.0f}",
                                    "expected people losing water": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------- PLAN
with tabs[1]:
    st.subheader("Recommended pre-emptive plan")
    budget = st.slider("Budget (USD)", 100_000, 1_500_000, 750_000, 50_000)
    veto = st.multiselect("Dispatcher veto — actions to forbid",
                          options=sorted(plan_actions.asset_id.tolist()),
                          help="Forbid an action. The plan re-solves immediately and "
                               "the cost of the override is shown.")
    # Solve on every change rather than behind a button. An earlier version
    # required clicking "Re-solve" after choosing a veto; if you forgot, the
    # override cost silently read +0 because the displayed plan was still the
    # un-vetoed one. At ~0.1 s a solve, gating it behind a button bought
    # nothing and created a trap.
    with st.spinner("Solving..."):
        p = solve_cached(sid, budget, tuple(sorted(veto)))
        unvetoed = solve_cached(sid, budget, ()) if veto else p
        base = greedy_cached(sid, budget)

    k = st.columns(5 if veto else 4)
    k[0].metric("Actions", len(p.actions))
    k[1].metric("Spend", f"${p.spend:,.0f}", f"budget ${budget:,.0f}")
    k[2].metric("Crews staged", int(p.staging.crews_staged.sum()))
    k[3].metric("Solve time", f"{p.solve_seconds:.2f}s", p.status)
    if veto:
        delta = p.objective - unvetoed.objective
        pct = delta / max(unvetoed.objective, 1) * 100
        k[4].metric("Cost of override", f"+{delta:,.0f}",
                    f"+{pct:.2f}% expected harm", delta_color="inverse")

    if veto:
        st.warning(f"{len(veto)} action(s) vetoed by the dispatcher. The plan below is "
                   f"the best available under that constraint. The override is permitted; "
                   f"'Cost of override' is the extra expected harm it causes, measured in "
                   f"people-equivalent.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Actions**")
        st.dataframe(p.actions.sort_values("avoided", ascending=False)
                     .rename(columns={"avoided": "expected consequence avoided"})
                     .style.format({"cost": "${:,.0f}",
                                    "expected consequence avoided": "{:,.0f}"}),
                     use_container_width=True, hide_index=True, height=300)
    with c2:
        st.markdown("**Crew staging by zone**")
        st.bar_chart(p.staging.set_index("zone_id"), height=300)

    st.markdown("**Realised loss across 500 simulated storms**")
    st.caption("Plans are replayed against sampled failure outcomes, not scored on "
               "the optimiser's own expected values. Lower is better.")
    st.dataframe((mc.describe().T[["50%", "mean", "max"]] / 1e6).round(2)
                 .rename(columns={"50%": "median (M)", "mean": "mean (M)",
                                  "max": "worst (M)"}),
                 use_container_width=True)

# ------------------------------------------------------------------ COPILOT
with tabs[2]:
    st.subheader("Operations copilot")
    st.caption(f"Mode: **{'Claude tool-use' if C.available() else 'deterministic router'}**. "
               "The assistant selects which function to call; it never computes a "
               "number. Answers are identical with the model removed — that is the "
               "point, not a limitation.")
    ex = st.columns(4)
    presets = ["What should I be worried about?",
               "Which substations would knock out water supply?",
               "How does zone 7 look?", "Any pumps degrading?"]
    for col, q in zip(ex, presets):
        if col.button(q, use_container_width=True):
            st.session_state.q = q
    q = st.text_input("Ask", value=st.session_state.get("q", presets[0]))
    if q:
        out = C.ask(q, sid)
        st.code(out["answer"], language=None)
        with st.expander(f"Tool calls ({len(out['tool_calls'])}) — every number traceable"):
            for tc in out["tool_calls"]:
                st.markdown(f"`{tc['tool']}({tc['arguments']})`")
                st.json(tc["result"], expanded=False)

# ---------------------------------------------------------------- SITUATION REPORT
with tabs[3]:
    st.subheader("Leadership situation report")
    st.caption("Assembled from typed data. The numbers are computed, not generated; "
               "a language model may phrase the narrative but cannot alter a figure.")
    s = SR.build(risk, plan_actions, plan_staging, sid)
    md = SR.render_markdown(s)
    st.download_button("Download report (.md)", md,
                       file_name=f"sitrep_{sid}.md", type="primary")
    st.markdown(md)

# ------------------------------------------------------- CONDITION MONITORING
with tabs[4]:
    st.subheader("Condition monitoring — blue-sky day")
    st.caption("The reason the platform gets opened on an ordinary Tuesday. "
               "Sustained efficiency loss on instrumented pumping assets.")
    a = anomalies[anomalies.alerted]
    k = st.columns(3)
    k[0].metric("Instrumented pumps", len(anomalies))
    k[1].metric("Alerting", len(a))
    k[2].metric("Flagged sensor quality issues",
                int((anomalies.data_quality != "ok").sum()))
    st.dataframe(a[["asset_id", "first_alert", "final_efficiency_ratio",
                    "min_z", "data_quality"]]
                 .rename(columns={"final_efficiency_ratio": "efficiency vs baseline"})
                 .style.format({"efficiency vs baseline": "{:.0%}", "min_z": "{:.1f}"}),
                 use_container_width=True, hide_index=True)
    st.info("Recall and precision are both 1.0 on the synthetic fleet. That reflects "
            "a clean injected signal (~35% efficiency loss against ~5% noise), not "
            "method quality. Treat as an upper bound.")
