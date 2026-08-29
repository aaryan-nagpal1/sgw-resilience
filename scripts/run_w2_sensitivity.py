"""Sensitivity of the W2 conclusion to the invented mitigation effect sizes.

The effect of a pre-emptive inspection on failure probability is an assumption
with no source. The defensible claim is therefore not "the optimiser wins at
0.55" but "the optimiser wins across every value we consider plausible".
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from sgw_platform.w2_dispatch import actions as A, baselines as B, optimise as O, simulate as S

DATA, ART, FIG = Path("data"), Path("artifacts"), Path("figures")
BUDGET, N = 750_000, 300


def main():
    risk = pd.read_parquet(ART / "w1_risk_scores.parquet")
    crews = pd.read_csv(DATA / "crews.csv")
    sid = risk.groupby("scenario_id").failed.sum().idxmax()
    r = risk[risk.scenario_id == sid].copy()

    original = {k: v.risk_multiplier for k, v in A.CATALOGUE.items()}
    rows = []
    print(f"sweeping mitigation effectiveness ({N} storms per point)\n")
    for mult in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for k, act in A.CATALOGUE.items():
            if original[k] < 1.0:
                A.CATALOGUE[k] = A.Action(act.name, act.crew_type, act.crew_shifts,
                                          act.cost_usd, mult, act.severs_cascade)
        plan = O.build_and_solve(r, crews, budget_usd=BUDGET)
        greedy = B.rank_by_risk(r, crews, BUDGET)
        sq = B.status_quo(r, crews, BUDGET)
        losses = S.monte_carlo(r, {"opt": plan, "greedy": greedy, "sq": sq}, n=N)
        red_g = 1 - losses["opt"].median() / losses["greedy"].median()
        red_s = 1 - losses["opt"].median() / losses["sq"].median()
        win = float((losses["opt"] < losses["greedy"]).mean())
        rows.append({"risk_multiplier": mult, "actions": len(plan.actions),
                     "median_reduction_vs_greedy": red_g,
                     "median_reduction_vs_status_quo": red_s,
                     "win_rate_vs_greedy": win})
        print(f"  multiplier {mult:.2f}  ->  vs greedy {red_g*100:+5.1f}%  "
              f"(win {win*100:5.1f}%)   vs status quo {red_s*100:+5.1f}%")

    for k, v in original.items():
        act = A.CATALOGUE[k]
        A.CATALOGUE[k] = A.Action(act.name, act.crew_type, act.crew_shifts,
                                  act.cost_usd, v, act.severs_cascade)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(ART / "w2_sensitivity.csv", index=False)
    from sgw_platform.w1_risk import figures as F
    F.sensitivity_plot(sweep, FIG / "w2_sensitivity.png")
    print(f"\nadvantage over greedy across the whole range: "
          f"{sweep.median_reduction_vs_greedy.min()*100:+.1f}% to "
          f"{sweep.median_reduction_vs_greedy.max()*100:+.1f}%")
    print(f"win rate range: {sweep.win_rate_vs_greedy.min()*100:.1f}% to "
          f"{sweep.win_rate_vs_greedy.max()*100:.1f}%")


if __name__ == "__main__":
    main()
