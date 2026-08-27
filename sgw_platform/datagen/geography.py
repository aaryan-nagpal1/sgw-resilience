"""Synthetic service territory: coastline, elevation, population, land cover.

The point of this module is that hazard exposure has to be *spatially
coherent*. Assets near the coast must share surge exposure; inland forested
assets must share fire weather. Sampling each asset's environment
independently would produce a dataset where no spatial model could ever
help, and would quietly flatter any risk model trained on it.
"""
from __future__ import annotations

import numpy as np

from . import config as C


class Territory:
    """Continuous scalar fields over the service territory."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # Coastline runs roughly SW-NE across the south of the territory.
        # Represented as lat = f(lon) with a couple of bays.
        self._coast_a = 29.95
        self._coast_b = 0.10
        self._bay_amp = self.rng.uniform(0.10, 0.18)
        self._bay_freq = self.rng.uniform(1.6, 2.4)
        # City centres drive customer density.
        self.cities = np.column_stack([
            self.rng.uniform(C.LAT_MIN + 0.2, C.LAT_MAX - 0.2, C.N_CITIES),
            self.rng.uniform(C.LON_MIN + 0.2, C.LON_MAX - 0.2, C.N_CITIES),
        ])
        self.city_pop = self.rng.dirichlet(np.ones(C.N_CITIES) * 1.4)
        # Zone seed points -> zones are the Voronoi cells of these.
        self.zone_seeds = np.column_stack([
            self.rng.uniform(C.LAT_MIN, C.LAT_MAX, C.N_ZONES),
            self.rng.uniform(C.LON_MIN, C.LON_MAX, C.N_ZONES),
        ])
        # Latent soil corrosivity field. Only *partially* observed in the
        # published dataset -- see assets.py. Real utilities do not have
        # complete soil surveys, and a model that assumes they do is lying.
        self._soil_phase = self.rng.uniform(0, 2 * np.pi, 3)

    # ------------------------------------------------------------ coastline
    def coast_lat(self, lon: np.ndarray) -> np.ndarray:
        x = (lon - C.LON_MIN) / (C.LON_MAX - C.LON_MIN)
        return (self._coast_a + self._coast_b * x
                + self._bay_amp * np.sin(self._bay_freq * np.pi * x))

    def dist_to_coast_km(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Positive inland, ~0 at the shoreline. Degrees latitude ~111 km."""
        return np.maximum(0.0, (lat - self.coast_lat(lon)) * 111.0)

    # ------------------------------------------------------------ elevation
    def elevation_m(self, lat, lon):
        """Coastal plain rising inland, with local relief."""
        d = self.dist_to_coast_km(lat, lon)
        base = 1.2 + 0.145 * d + 0.00042 * d ** 2
        relief = (7.0 * np.sin(0.9 * lat + 1.3) * np.cos(0.7 * lon - 0.4)
                  + 4.0 * np.sin(2.1 * lon + 0.9))
        return np.maximum(0.3, base + relief + self.rng.normal(0, 2.2, np.shape(lat)))

    # ----------------------------------------------------------- population
    def customer_density(self, lat, lon):
        """Relative customer density in [0, 1]-ish. Gaussian city kernels."""
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        dens = np.full(lat.shape, 0.02)
        for (clat, clon), w in zip(self.cities, self.city_pop):
            d2 = (lat - clat) ** 2 + (lon - clon) ** 2
            dens += w * np.exp(-d2 / (2 * 0.16 ** 2))
        return dens / dens.max()

    # ----------------------------------------------------------- land cover
    def vegetation_density(self, lat, lon):
        """0 = cleared/urban, 1 = dense forest. Rises inland, falls in cities."""
        d = self.dist_to_coast_km(lat, lon)
        veg = np.clip(0.18 + d / 260.0, 0, 0.92)
        veg *= (1.0 - 0.65 * self.customer_density(lat, lon))
        veg += self.rng.normal(0, 0.07, np.shape(lat))
        return np.clip(veg, 0.0, 1.0)

    def drought_susceptibility(self, lat, lon):
        """Drives fire weather. Inland + sandy soils burn."""
        d = self.dist_to_coast_km(lat, lon)
        return np.clip(0.15 + d / 320.0 + self.rng.normal(0, 0.06, np.shape(lat)), 0, 1)

    def soil_corrosivity(self, lat, lon):
        """Latent driver of water-main deterioration. Smooth, spatial."""
        s = (np.sin(1.4 * lat + self._soil_phase[0])
             + np.cos(1.1 * lon + self._soil_phase[1])
             + 0.6 * np.sin(2.3 * lat + 1.7 * lon + self._soil_phase[2]))
        s = (s - s.min()) / (np.ptp(s) + 1e-9) if np.ndim(s) and np.size(s) > 1 else 0.5
        return np.clip(s + self.rng.normal(0, 0.05, np.shape(lat)), 0, 1)

    def drainage_quality(self, lat, lon):
        """0 = ponds badly, 1 = drains freely. Urban areas drain worse."""
        return np.clip(0.72 - 0.35 * self.customer_density(lat, lon)
                       + self.rng.normal(0, 0.09, np.shape(lat)), 0.05, 1.0)

    # ---------------------------------------------------------------- zones
    def zone_of(self, lat, lon) -> np.ndarray:
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        d2 = ((lat[:, None] - self.zone_seeds[None, :, 0]) ** 2
              + (lon[:, None] - self.zone_seeds[None, :, 1]) ** 2)
        return d2.argmin(axis=1)

    # ----------------------------------------------------------- sampling
    def sample_points(self, n: int, coastal_bias: float = 0.0):
        """Draw n locations. coastal_bias in [0,1] pulls points toward the shore
        (water treatment plants and intakes sit low and near water)."""
        lon = self.rng.uniform(C.LON_MIN, C.LON_MAX, n)
        if coastal_bias > 0:
            # Beta skewed toward the coast end of the inland axis.
            u = self.rng.beta(1.0, 1.0 + 4.0 * coastal_bias, n)
        else:
            u = self.rng.uniform(0, 1, n)
        clat = self.coast_lat(lon)
        lat = clat + u * (C.LAT_MAX - clat)
        return lat, lon

    def sample_near_cities(self, n: int):
        """Draw n locations weighted toward population centres."""
        idx = self.rng.choice(C.N_CITIES, size=n, p=self.city_pop)
        lat = self.cities[idx, 0] + self.rng.normal(0, 0.22, n)
        lon = self.cities[idx, 1] + self.rng.normal(0, 0.22, n)
        return (np.clip(lat, C.LAT_MIN, C.LAT_MAX),
                np.clip(lon, C.LON_MIN, C.LON_MAX))


def hazard_grid(territory: Territory):
    """Coarse regular grid used as the hazard forecast raster."""
    lats = np.arange(C.LAT_MIN, C.LAT_MAX, C.GRID_RES_DEG)
    lons = np.arange(C.LON_MIN, C.LON_MAX, C.GRID_RES_DEG)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    return lats, lons, glat, glon
