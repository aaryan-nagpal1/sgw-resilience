"""W2 pipeline: optimise dispatch, compare to baselines, Monte Carlo evaluate.

    python scripts/run_w2.py
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sgw_platform.w2_dispatch import baselines as B, optimise as O, simulate as S

DATA, ART = Path("data"), Path("artifacts")
BUDGET = 750_000        # USD available for pre-emptive action in the window
N_SIMS = 500


def main():
    print("=" * 74); print("W2  CONSTRAINED RESPONSE OPTIMISATION"); print("=" * 74)

    risk = pd.read_parquet(ART / "w1_risk_scores.parquet")
    crews = pd.read_csv(DATA / "crews.csv")

    # Demo scenario: the most damaging event in the test set.
    sid = risk.groupby("scenario_id").failed.sum().idxmax()
    r = risk[risk.scenario_id == sid].copy()
    print(f"\nscenario {sid}  ({r.hazard.iloc[0]}), {len(r):,} assets, "
          f"{int(r.failed.sum())} actual failures")
    print(f"budget ${BUDGET:,}  |  crews available: "
          f"{crews[crews.home_depot!='MUTUAL_AID'].crew_type.value_counts().to_dict()}")

    # ------------------------------------------------------------- optimise
    plan = O.build_and_solve(r, crews, budget_usd=BUDGET)
    print(f"\n--- optimiser -------------------------------------------------")
    print(f"status {plan.status} in {plan.solve_seconds:.2f}s  |  "
          f"{len(plan.actions)} actions  |  spend ${plan.spend:,.0f}")
    if len(plan.actions):
        print(plan.actions.action.value_counts().to_string())
    print(f"crews staged: {plan.staging.crews_staged.sum()} across "
          f"{(plan.staging.crews_staged>0).sum()} zones")

    plans = {
        "OPTIMISED (CP-SAT)": plan,
        "greedy rank-by-risk": B.rank_by_risk(r, crews, BUDGET),
        "rank-by-probability": B.rank_by_probability(r, crews, BUDGET),
        "rank-by-customers": B.rank_by_customers(r, crews, BUDGET),
        "random": B.random_policy(r, crews, BUDGET),
        "status quo (no action)": B.status_quo(r, crews, BUDGET),
    }

    print(f"\n--- plans (all held to the same budget and crew limits) --------")
    print(pd.DataFrame([{"plan": k, "actions": len(p.actions),
                         "spend_usd": f"${p.spend:,.0f}",
                         "crews staged": int(p.staging.crews_staged.sum())}
                        for k, p in plans.items()]).to_string(index=False))

    # ---------------------------------------------------- Monte Carlo replay
    print(f"\n--- Monte Carlo: {N_SIMS} sampled storms, common random numbers ---")
    losses = S.monte_carlo(r, plans, n=N_SIMS)
    summary = S.summarise(losses, reference="status quo (no action)", plans=plans)
    disp = summary.copy()
    for c in ["median realised loss", "mean", "p90 (bad storms)"]:
        disp[c] = disp[c].map(lambda v: f"{v/1e6:,.1f}M")
    for c in disp.columns:
        if c.startswith("median reduction") or c.startswith("win rate"):
            disp[c] = disp[c].map(lambda v: f"{v*100:.1f}%")
    disp["spend_usd"] = disp.spend_usd.map(lambda v: f"${v:,.0f}")
    print(disp.to_string(index=False))

    # Head-to-head against the strongest baseline.
    ref = "greedy rank-by-risk"
    opt, gre = losses["OPTIMISED (CP-SAT)"], losses[ref]
    print(f"\n--- head to head: optimiser vs {ref} --------------")
    print(f"  optimiser wins in {100*(opt<gre).mean():.1f}% of {N_SIMS} storms")
    print(f"  median reduction in realised loss: {100*(1-opt.median()/gre.median()):.1f}%")
    print(f"  it LOSES in {100*(opt>=gre).mean():.1f}% of storms; in those, "
          f"median excess {100*(opt[opt>=gre].median()/gre[opt>=gre].median()-1):.1f}%")

    from sgw_platform.w1_risk import figures as F
    F.monte_carlo_plot(losses, Path("figures") / "w2_monte_carlo.png")
    print("\nfigure written to figures/w2_monte_carlo.png")

    losses.to_parquet(ART / "w2_monte_carlo.parquet", index=False)
    summary.to_csv(ART / "w2_summary.csv", index=False)
    plan.actions.to_csv(ART / "w2_plan_actions.csv", index=False)
    plan.staging.to_csv(ART / "w2_plan_staging.csv", index=False)

    # ------------------------------------------------- human-in-the-loop demo
    if len(plan.actions):
        veto = plan.actions.nlargest(1, "avoided").asset_id.iloc[0]
        alt = O.build_and_solve(r, crews, budget_usd=BUDGET, forbidden={veto})
        cost = (alt.objective - plan.objective) / max(plan.objective, 1) * 100
        print(f"\n--- human override ---------------------------------------------")
        print(f"  dispatcher vetoes {veto} (the highest-value action)")
        print(f"  model re-solves in {alt.solve_seconds:.2f}s; expected consequence "
              f"rises {cost:.2f}%")
        print(f"  -> the override is allowed, and its cost is made visible")
    return r, plans, losses


if __name__ == "__main__":
    main()
