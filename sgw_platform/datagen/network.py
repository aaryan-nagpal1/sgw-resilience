"""Network topology for both estates, plus the power->water dependency edges.

The dependency edges are the point of this module. SGW owns both networks,
so a substation failure does not only darken homes -- it silences the pump
stations fed from that substation, and a few hours later a different
population loses water pressure. That coupling is invisible to any siloed
tool and is the mechanism the platform exists to surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from . import config as C


def _nearest(src: pd.DataFrame, dst: pd.DataFrame, k: int = 1):
    """Index of the nearest dst row for each src row (planar approximation --
    fine at this scale and keeps the dependency free of a geo library)."""
    sl = src[["lat", "lon"]].to_numpy()
    dl = dst[["lat", "lon"]].to_numpy()
    d2 = ((sl[:, None, 0] - dl[None, :, 0]) ** 2
          + ((sl[:, None, 1] - dl[None, :, 1]) * 0.87) ** 2)
    if k == 1:
        return d2.argmin(axis=1)
    return np.argsort(d2, axis=1)[:, :k]


def build_network(rng: np.random.Generator, df: pd.DataFrame):
    """Return (edges DataFrame, networkx DiGraph). Edges flow supply -> load."""
    edges = []

    def add(u, v, kind):
        edges.extend({"source": a, "target": b, "edge_type": kind}
                     for a, b in zip(u, v))

    tss = df[df.asset_class == "transmission_substation"]
    dss = df[df.asset_class == "distribution_substation"]
    dfr = df[df.asset_class == "distribution_feeder"]
    rcl = df[df.asset_class == "recloser"]
    tls = df[df.asset_class == "transmission_span"]
    intk = df[df.asset_class == "raw_water_intake"]
    wtp = df[df.asset_class == "water_treatment_plant"]
    pmp = df[df.asset_class == "pump_station"]
    tnk = df[df.asset_class == "storage_tank"]
    wmn = df[df.asset_class == "water_main"]

    # ---------------------------------------------------------- electric
    # Transmission spans hang off the nearest transmission substation.
    add(tss.asset_id.to_numpy()[_nearest(tls, tss)], tls.asset_id, "transmission")
    # Distribution substations are fed from transmission substations.
    add(tss.asset_id.to_numpy()[_nearest(dss, tss)], dss.asset_id, "supply")
    # Feeders hang off distribution substations.
    feeder_parent = _nearest(dfr, dss)
    add(dss.asset_id.to_numpy()[feeder_parent], dfr.asset_id, "supply")
    # Reclosers sit on feeders.
    add(dfr.asset_id.to_numpy()[_nearest(rcl, dfr)], rcl.asset_id, "sectionalising")

    # ------------------------------------------------------------- water
    add(intk.asset_id.to_numpy()[_nearest(wtp, intk)], wtp.asset_id, "raw_water")
    add(wtp.asset_id.to_numpy()[_nearest(pmp, wtp)], pmp.asset_id, "treated_water")
    add(pmp.asset_id.to_numpy()[_nearest(tnk, pmp)], tnk.asset_id, "storage")
    add(pmp.asset_id.to_numpy()[_nearest(wmn, pmp)], wmn.asset_id, "distribution")

    # ------------------------------------------- power -> water dependency
    # Every pumping station and treatment plant is an electrical load on
    # SGW's own distribution network. This is the cross-domain cascade.
    # Assigned to one of the 3 nearest substations rather than strictly the
    # nearest: real feeder boundaries follow circuits and rights of way, not
    # straight-line distance, and strict nearest-assignment concentrates the
    # whole water estate onto a handful of urban substations.
    water_sites = pd.concat([pmp, wtp])
    cand = _nearest(water_sites, dss, k=3)
    choice = rng.integers(0, cand.shape[1], len(water_sites))
    add(dss.asset_id.to_numpy()[cand[np.arange(len(water_sites)), choice]],
        water_sites.asset_id, "power_dependency")

    edf = pd.DataFrame(edges).drop_duplicates()
    edf = edf[edf.source != edf.target]

    G = nx.DiGraph()
    G.add_nodes_from(df.asset_id)
    for r in edf.itertuples(index=False):
        G.add_edge(r.source, r.target, edge_type=r.edge_type)
    return edf, G


def compute_consequence(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """For every asset, propagate downstream to get the consequence of losing it.

    This is the 'consequence' term in risk = P(failure) x consequence. It is a
    deterministic graph reachability calculation, not a model -- which is
    exactly why it belongs in the graph and not in the feature vector.
    """
    cust = df.set_index("asset_id")["customers_served"].to_dict()
    pop = df.set_index("asset_id")["population_served_water"].to_dict()
    crit = df.set_index("asset_id")["critical_facilities"].to_dict()
    backup = df.set_index("asset_id")["has_backup_generation"].to_dict()
    runtime = df.set_index("asset_id")["backup_runtime_hours"].to_dict()

    out = []
    for aid in df.asset_id:
        # Everything downstream of this asset in the supply direction.
        try:
            down = nx.descendants(G, aid)
        except nx.NetworkXError:
            down = set()
        reach = down | {aid}

        direct_cust = sum(cust.get(a, 0.0) for a in reach)
        direct_pop = sum(pop.get(a, 0.0) for a in reach)
        n_crit = sum(crit.get(a, 0) for a in reach)

        # Water population reached *through* a power_dependency edge is the
        # cascade term -- population that loses water because power was lost.
        cascade_pop = 0.0
        cascade_delay_h = np.nan
        dep_children = [v for u, v, d in G.out_edges(aid, data=True)
                        if d.get("edge_type") == "power_dependency"]
        if dep_children:
            delays = []
            for c in dep_children:
                sub = nx.descendants(G, c) | {c}
                site_pop = sum(pop.get(a, 0.0) for a in sub)
                if backup.get(c, False):
                    # Backup generation buys time; it does not remove the risk.
                    delays.append(runtime.get(c, 0))
                    cascade_pop += site_pop * 0.35
                else:
                    delays.append(0)
                    cascade_pop += site_pop
            cascade_delay_h = float(np.mean(delays)) if delays else np.nan

        out.append({
            "asset_id": aid,
            "downstream_assets": len(down),
            "customers_affected": direct_cust,
            "population_water_affected": direct_pop,
            "critical_facilities_affected": n_crit,
            "cascade_population_water": round(cascade_pop, 1),
            "cascade_delay_hours": cascade_delay_h,
        })

    cons = pd.DataFrame(out)
    # A single headline consequence figure in "people-equivalent affected".
    #
    # Two different kinds of number are combined here, and the distinction
    # matters when defending the ranking:
    #   RESIDENTS_PER_ACCOUNT is DERIVED from the service territory (8.0M
    #     residents / 3.2M accounts). It is not a free parameter.
    #   CRITICAL_FACILITY_WEIGHT is a POLICY CHOICE with no empirical basis.
    #     It encodes how much more a hospital matters than a household, which
    #     is a question for the client, not the modeller. Exposed as a tunable
    #     and carried in the PRD assumptions register (A6).
    cons["consequence_score"] = (
        cons.customers_affected * C.RESIDENTS_PER_ACCOUNT
        + cons.population_water_affected * 1.0
        + cons.cascade_population_water * 1.0
        + cons.critical_facilities_affected * C.CRITICAL_FACILITY_WEIGHT
    ).round(1)
    return cons
