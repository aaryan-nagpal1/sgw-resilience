"""Multi-hazard scenario generation and per-asset exposure.

Four hazard types share one pipeline: generate hazard fields on a coarse grid
through time, then sample those fields at each asset's location to produce an
exposure vector. Keeping the four hazards in one pipeline is deliberate --
the literature's open problem is *compound* events, and you cannot model a
heatwave that coincides with a drought if each hazard owns its own codepath.

  hazard  -> the physical forcing on the grid   (this module, part 1)
  exposure-> that forcing sampled at the asset  (this module, part 2)
  vulnerability -> P(fail | exposure)           (history.py)
  consequence   -> what is lost if it fails     (network.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .geography import Territory, hazard_grid

BUCKET_HOURS = 6


@dataclass
class Scenario:
    scenario_id: str
    hazard: str
    start: pd.Timestamp
    duration_h: int
    params: dict = field(default_factory=dict)

    @property
    def n_buckets(self) -> int:
        return max(1, self.duration_h // BUCKET_HOURS)

    def timestamps(self):
        return pd.date_range(self.start, periods=self.n_buckets,
                             freq=f"{BUCKET_HOURS}h")


# --------------------------------------------------------------- scenarios
def make_scenario(rng, hazard: str, start: pd.Timestamp, idx: int) -> Scenario:
    sid = f"{hazard[:4].upper()}-{start:%Y%m%d}-{idx:03d}"
    if hazard == "hurricane":
        # Landfall point on the coast, track heading inland to the NNW.
        lf_lon = rng.uniform(C.LON_MIN + 0.4, C.LON_MAX - 0.4)
        p = dict(
            landfall_lon=lf_lon,
            max_wind=rng.uniform(*C.HURRICANE_MAX_WIND_RANGE),
            radius_max_wind_km=rng.uniform(28, 65),
            holland_b=rng.uniform(1.1, 1.7),
            forward_speed_kmh=rng.uniform(14, 32),
            track_bearing_deg=rng.uniform(-35, 10),     # 0 = due north
            rain_rate_mm_h=rng.uniform(6, 26),
            surge_peak_m=rng.uniform(0.8, 4.6),
            landfall_bucket=rng.integers(4, 8),
        )
        return Scenario(sid, hazard, start, 72, p)

    if hazard == "heatwave":
        p = dict(
            peak_temp_c=rng.uniform(*C.HEATWAVE_PEAK_RANGE),
            inland_amplification=rng.uniform(1.5, 4.0),  # deg C extra inland
            overnight_min_c=rng.uniform(22, 30),         # recovery, or lack of
            humidity=rng.uniform(0.45, 0.80),
            days=int(rng.integers(4, 8)),
            wind_ms=rng.uniform(1.0, 4.5),
        )
        return Scenario(sid, hazard, start, p["days"] * 24, p)

    if hazard == "inland_flood":
        p = dict(
            centre_lat=rng.uniform(C.LAT_MIN + 0.8, C.LAT_MAX - 0.3),
            centre_lon=rng.uniform(C.LON_MIN + 0.4, C.LON_MAX - 0.4),
            radius_deg=rng.uniform(0.35, 0.95),
            rain_total_mm=rng.uniform(90, 380),
            wind_ms=rng.uniform(8, 22),
        )
        return Scenario(sid, hazard, start, 36, p)

    if hazard == "wildfire":
        p = dict(
            centre_lat=rng.uniform(C.LAT_MIN + 1.2, C.LAT_MAX - 0.2),
            centre_lon=rng.uniform(C.LON_MIN + 0.4, C.LON_MAX - 0.4),
            radius_deg=rng.uniform(0.5, 1.3),
            temp_c=rng.uniform(31, 40),
            humidity=rng.uniform(0.08, 0.25),
            wind_ms=rng.uniform(9, 24),
            drought_index=rng.uniform(0.55, 0.95),
        )
        return Scenario(sid, hazard, start, 48, p)

    # baseline: an ordinary week. Assets still fail on ordinary days, and a
    # model trained only on disasters will badly overestimate blue-sky risk.
    p = dict(temp_c=rng.uniform(12, 33), wind_ms=rng.uniform(2, 11),
             rain_mm=rng.uniform(0, 14))
    return Scenario(sid, "baseline", start, 24, p)


# ------------------------------------------------------------ hazard fields
def _hurricane_wind(p, glat, glon, b_idx, n_buckets):
    """Holland-style radial wind profile around a moving centre."""
    hrs = b_idx * BUCKET_HOURS
    lf_hr = p["landfall_bucket"] * BUCKET_HOURS
    bearing = np.radians(p["track_bearing_deg"])
    dist_km = (hrs - lf_hr) * p["forward_speed_kmh"]
    clat = 29.6 + (dist_km / 111.0) * np.cos(bearing)
    clon = p["landfall_lon"] + (dist_km / 96.0) * np.sin(bearing)

    r_km = np.sqrt(((glat - clat) * 111.0) ** 2 + ((glon - clon) * 96.0) ** 2)
    r_km = np.maximum(r_km, 3.0)
    rmw, b = p["radius_max_wind_km"], p["holland_b"]
    # Overland decay after landfall: storms weaken fast once the fetch is cut.
    decay = 1.0 if hrs <= lf_hr else float(np.exp(-(hrs - lf_hr) / 26.0))
    vmax = p["max_wind"] * decay
    v = np.where(r_km <= rmw, vmax * (r_km / rmw) ** 0.6,
                 vmax * (rmw / r_km) ** b)
    return np.clip(v, 0, None), r_km


def hazard_fields(sc: Scenario, terr: Territory, glat, glon, b_idx: int) -> dict:
    """Physical forcing on the grid for one 6-hour bucket."""
    z = np.zeros_like(glat)
    f = dict(wind_gust_ms=z + 4.0, rainfall_mm=z.copy(), surge_m=z.copy(),
             temp_c=z + 24.0, humidity=z + 0.6, drought_index=z + 0.3)
    p = sc.params

    if sc.hazard == "hurricane":
        v, r_km = _hurricane_wind(p, glat, glon, b_idx, sc.n_buckets)
        f["wind_gust_ms"] = v * 1.28                      # gust factor
        # Rain banded around the core, heaviest just outside the eyewall.
        band = np.exp(-((r_km - p["radius_max_wind_km"] * 1.6) ** 2) / (2 * 90.0 ** 2))
        f["rainfall_mm"] = p["rain_rate_mm_h"] * BUCKET_HOURS * (0.35 + 0.9 * band)
        # Surge: coastal only, peaks around landfall, decays inland sharply.
        dcoast = terr.dist_to_coast_km(glat, glon)
        t_fac = np.exp(-((b_idx - p["landfall_bucket"]) ** 2) / 3.0)
        onshore = np.clip((v / max(p["max_wind"], 1e-6)), 0, 1)
        f["surge_m"] = (p["surge_peak_m"] * t_fac * onshore
                        * np.exp(-dcoast / 12.0))
        f["temp_c"] = z + 26.0
        f["humidity"] = z + 0.92

    elif sc.hazard == "heatwave":
        day = (b_idx * BUCKET_HOURS) / 24.0
        # Diurnal cycle: bucket 0 = 00:00. Peak mid-afternoon.
        hod = (b_idx * BUCKET_HOURS) % 24
        diurnal = np.sin((hod - 6) / 24.0 * 2 * np.pi)
        # Build-up then decline across the event.
        env = np.sin(np.pi * np.clip(day / max(sc.params["days"], 1), 0, 1)) ** 0.5
        dcoast = terr.dist_to_coast_km(glat, glon)
        inland = p["inland_amplification"] * np.clip(dcoast / 200.0, 0, 1.2)
        peak = p["peak_temp_c"] + inland
        mn = p["overnight_min_c"] + 0.4 * inland
        f["temp_c"] = mn + (peak - mn) * env * np.clip(0.5 + 0.5 * diurnal, 0, 1)
        f["humidity"] = z + p["humidity"]
        f["wind_gust_ms"] = z + p["wind_ms"]
        f["drought_index"] = z + np.clip(0.35 + 0.08 * day, 0, 1)

    elif sc.hazard == "inland_flood":
        d = np.sqrt((glat - p["centre_lat"]) ** 2 + (glon - p["centre_lon"]) ** 2)
        cell = np.exp(-(d ** 2) / (2 * p["radius_deg"] ** 2))
        # Rain front tracks across the cell over the event.
        t_fac = np.exp(-((b_idx - sc.n_buckets * 0.4) ** 2) / 2.5)
        f["rainfall_mm"] = p["rain_total_mm"] * cell * t_fac / 2.2
        f["wind_gust_ms"] = z + p["wind_ms"] * (0.4 + 0.6 * cell)
        f["temp_c"] = z + 21.0
        f["humidity"] = z + 0.9

    elif sc.hazard == "wildfire":
        d = np.sqrt((glat - p["centre_lat"]) ** 2 + (glon - p["centre_lon"]) ** 2)
        cell = np.exp(-(d ** 2) / (2 * p["radius_deg"] ** 2))
        f["temp_c"] = z + p["temp_c"]
        f["humidity"] = z + p["humidity"]
        f["wind_gust_ms"] = z + p["wind_ms"] * (0.5 + 0.7 * cell)
        f["drought_index"] = z + p["drought_index"] * (0.4 + 0.6 * cell)

    else:  # baseline
        f["temp_c"] = z + p["temp_c"]
        f["wind_gust_ms"] = z + p["wind_ms"]
        f["rainfall_mm"] = z + p["rain_mm"] / max(sc.n_buckets, 1)

    # Fire weather index: hot, dry, windy, over dry ground. Normalised 0-1.
    f["fire_weather_index"] = np.clip(
        0.34 * np.clip((f["temp_c"] - 20) / 22, 0, 1)
        + 0.30 * np.clip(1 - f["humidity"], 0, 1)
        + 0.22 * np.clip(f["wind_gust_ms"] / 26, 0, 1)
        + 0.14 * f["drought_index"], 0, 1)
    return f


# --------------------------------------------------------------- exposure
class GridSampler:
    """Nearest-cell lookup from asset coordinates into the hazard grid."""

    def __init__(self, lats, lons):
        self.lats, self.lons = lats, lons

    def index(self, lat, lon):
        i = np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 1)
        j = np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 1)
        return i, j


def asset_exposure(sc: Scenario, assets: pd.DataFrame, terr: Territory) -> pd.DataFrame:
    """Per asset, per 6h bucket exposure vector.

    Linear assets are sampled at both endpoints and the midpoint, and the
    *maximum* is taken: a line fails at its worst point, not its average one.
    Sampling a linear asset at its centroid is a common and consequential
    modelling error.
    """
    lats, lons, glat, glon = hazard_grid(terr)
    sampler = GridSampler(lats, lons)

    is_lin = assets.asset_class.isin(C.LINEAR_CLASSES).to_numpy()
    pts = [(assets.lat.to_numpy(), assets.lon.to_numpy())]
    pts.append((np.where(is_lin, assets.lat_end.to_numpy(), assets.lat.to_numpy()),
                np.where(is_lin, assets.lon_end.to_numpy(), assets.lon.to_numpy())))
    pts.append(((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2))
    idx = [sampler.index(la, lo) for la, lo in pts]

    site_elev = assets.site_elevation_m.to_numpy()
    defence = assets.flood_defence_m.to_numpy()
    drain = assets.drainage_quality.to_numpy()

    frames = []
    rain_trailing = np.zeros(len(assets))
    hot_hours = np.zeros(len(assets))

    for b, ts in enumerate(sc.timestamps()):
        f = hazard_fields(sc, terr, glat, glon, b)
        # max across the three sample points ~ 95th percentile along the run
        sampled = {k: np.maximum.reduce([v[i, j] for i, j in idx])
                   for k, v in f.items()}

        rain_trailing = rain_trailing * 0.72 + sampled["rainfall_mm"]
        hot_hours += np.where(sampled["temp_c"] > 35.0, BUCKET_HOURS, 0.0)

        # Pluvial depth: rain that the local drainage cannot take away.
        pluvial = np.clip(rain_trailing / 100.0 * (1.35 - drain), 0, None)
        # Inundation at the asset = water level minus ground and defences.
        inundation = np.clip(sampled["surge_m"] + pluvial
                             - site_elev - defence, 0, None)

        frames.append(pd.DataFrame({
            "scenario_id": sc.scenario_id,
            "hazard": sc.hazard,
            "asset_id": assets.asset_id.to_numpy(),
            "bucket": b,
            "timestamp": ts,
            "wind_gust_ms": sampled["wind_gust_ms"].round(2),
            "rainfall_mm": sampled["rainfall_mm"].round(2),
            "rainfall_trailing_mm": rain_trailing.round(2),
            "surge_m": sampled["surge_m"].round(3),
            "inundation_depth_m": inundation.round(3),
            "temp_c": sampled["temp_c"].round(2),
            "humidity": sampled["humidity"].round(3),
            "fire_weather_index": sampled["fire_weather_index"].round(3),
            "hours_above_35c": hot_hours,
            "soil_saturation": np.clip(rain_trailing / 160.0, 0, 1).round(3),
        }))

    ex = pd.concat(frames, ignore_index=True)
    # Duration-above-threshold features. Guikema's early hurricane models found
    # duration of winds above ~20 m/s as predictive as peak gust; it is cheap
    # to compute and consistently earns its place.
    ex["hours_wind_above_20"] = (
        ex.assign(h=np.where(ex.wind_gust_ms > 20, BUCKET_HOURS, 0))
          .groupby("asset_id")["h"].cumsum().to_numpy())
    return ex
