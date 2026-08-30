"""Figures for the PRD, the executive briefing and the demo.

House style kept deliberately plain: no gridline clutter, no chartjunk, one
message per figure, and colours that survive greyscale printing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, BLUE, GREY, AMBER = "#10345C", "#1F6FB2", "#8A97A3", "#D98C1F"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A6570", "figure.dpi": 200,
})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def reliability_plot(rel_raw, rel_cal, path, ece_raw=None, ece_cal=None):
    """Calibration before and after. The diagonal is perfect calibration."""
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    hi = max(rel_raw.predicted.max(), rel_raw.observed.max(),
             rel_cal.predicted.max(), rel_cal.observed.max()) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=0.8, color=GREY, label="perfect calibration")
    ax.plot(rel_raw.predicted, rel_raw.observed, "o-", color=AMBER, ms=4, lw=1.2,
            label=f"uncalibrated" + (f"  (ECE {ece_raw:.5f})" if ece_raw else ""))
    ax.plot(rel_cal.predicted, rel_cal.observed, "s-", color=BLUE, ms=4, lw=1.2,
            label=f"Platt-scaled" + (f"  (ECE {ece_cal:.5f})" if ece_cal else ""))
    ax.set_xlabel("predicted probability of failure")
    ax.set_ylabel("observed failure rate")
    ax.set_title("Calibration, by decile of predicted risk")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _save(fig, path)


def baseline_bars(table, path, metric="loss capture@200"):
    """The operational metric: share of realised consequence captured in the
    top 200 ranked assets."""
    d = table.sort_values(metric)
    names = d.model.str.replace(" (published form)", "", regex=False)
    colors = [BLUE if "LightGBM" in n else GREY for n in d.model]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.barh(names, d[metric] * 100, color=colors, height=0.62)
    for b, v in zip(bars, d[metric] * 100):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=8, color=NAVY)
    ax.set_xlabel("share of realised impact captured (%)")
    ax.set_title("Impact Captured by Acting on Top 200 Assets")
    ax.set_xlim(0, max(d[metric] * 100) * 1.22)
    return _save(fig, path)


def per_hazard_bars(ph, path):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    d = ph.sort_values("PR-AUC")
    ax.barh(d.hazard.str.replace("_", " "), d["PR-AUC"], color=BLUE, height=0.6)
    for i, (v, n) in enumerate(zip(d["PR-AUC"], d.positives)):
        ax.text(v + 0.004, i, f"{v:.3f}  (n={n})", va="center", fontsize=7.5, color=NAVY)
    ax.set_xlabel("PR-AUC")
    ax.set_title("Model performance by hazard type")
    ax.set_xlim(0, d["PR-AUC"].max() * 1.45)
    return _save(fig, path)


def importance_plot(names, values, path, top=12):
    s = pd.Series(values, index=names).nlargest(top).sort_values()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.barh(s.index.str.replace("_", " "), s.values, color=BLUE, height=0.65)
    ax.set_xlabel("gain")
    ax.set_title("Most influential features")
    return _save(fig, path)


def cascade_plot(df, path):
    """Assets ranked by consequence, split into direct vs cascade impact.
    The point of the figure: substations whose largest impact is on water."""
    d = df.nlargest(12, "consequence_score").sort_values("consequence_score")
    lbl = d.asset_id + "  " + d.asset_class.str.replace("_", " ")
    direct = d.consequence_score - d.cascade_population_water
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.barh(lbl, direct / 1000, color=BLUE, height=0.65, label="direct impact")
    ax.barh(lbl, d.cascade_population_water / 1000, left=direct / 1000,
            color=AMBER, height=0.65, label="water lost to grid failure (cascade)")
    ax.set_xlabel("people-equivalent affected (thousands)")
    ax.set_title("People-Equivalent Affected by Asset Failure,\n"
                 "Including Power-to-Water Impacts")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right",
              bbox_to_anchor=(1.0, 0.02))
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(0, d.consequence_score.max() / 1000 * 1.06)
    return _save(fig, path)


def monte_carlo_plot(losses: pd.DataFrame, path, reference="status quo (no action)"):
    """Distribution of realised loss per plan. Shows spread, not just a median."""
    order = losses.median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    data = [losses[c].to_numpy() / 1e6 for c in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops=dict(color="white", lw=1.4))
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(BLUE if "OPTIMISED" in name else GREY)
        patch.set_edgecolor("none")
    ax.set_yticklabels([o.replace(" (no action)", "") for o in order], fontsize=8)
    ax.set_xlabel("realised loss (millions of people-equivalent hours)")
    ax.set_title(f"Realised loss across {len(losses)} simulated storms")
    return _save(fig, path)


def sensitivity_plot(sweep: pd.DataFrame, path):
    """Advantage over the best baseline across the plausible range of the
    mitigation effect sizes, which are assumptions rather than measurements."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.plot(sweep.risk_multiplier, sweep.median_reduction_vs_greedy * 100,
            "o-", color=BLUE, ms=4)
    ax.fill_between(sweep.risk_multiplier, 0,
                    sweep.median_reduction_vs_greedy * 100, alpha=0.12, color=BLUE)
    ax.set_xlabel("assumed effectiveness of pre-emptive action\n"
                  "(risk multiplier: lower = more effective)")
    ax.set_ylabel("advantage over best baseline (%)")
    ax.set_title("The conclusion holds across the plausible range")
    return _save(fig, path)


def value_sensitivity_plot(path, low=0.102, mid=0.140, high=0.191,
                           max_spend_m=120):
    """Executive value chart.

    Deliberately contains NO invented cost baseline. It is a straight
    transformation of the measured improvement range onto whatever annual
    response spend SGW actually has, so a reader locates their own number on
    the x-axis rather than being told one. That keeps the briefing's claim --
    "a financial ROI cannot yet be stated responsibly" -- intact while still
    giving leadership something quantitative to act on.
    """
    spend = np.linspace(0, max_spend_m, 200)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.fill_between(spend, spend * low, spend * high, color=BLUE, alpha=0.16,
                    label=f"range across tested assumptions ({low:.0%}–{high:.0%})")
    ax.plot(spend, spend * mid, color=BLUE, lw=1.8,
            label=f"measured improvement ({mid:.1%})")

    # One worked read-off so the chart teaches its own use.
    x = 40
    ax.plot([x, x], [0, x * high], ls=":", lw=0.9, color=GREY)
    ax.plot([0, x], [x * mid, x * mid], ls=":", lw=0.9, color=GREY)
    ax.annotate(f"at \\${x}M annual response spend,\n"
                f"\\${x*low:.1f}M–\\${x*high:.1f}M a year",
                xy=(x, x * mid), xytext=(x + 8, x * mid - 5.5),
                fontsize=7.5, color=NAVY,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))

    ax.set_xlabel("SGW annual weather-related response spend ($M)")
    ax.set_ylabel("indicative annual value ($M)")
    ax.set_title("Value scales with what response already costs SGW")
    ax.set_xlim(0, max_spend_m)
    ax.set_ylim(0, max_spend_m * high * 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(0.5, -0.30, "Phase 0 establishes SGW's actual spend. "
            "Programme cost is not shown and is not yet known.",
            transform=ax.transAxes, ha="center", fontsize=7, color=GREY)
    return _save(fig, path)
