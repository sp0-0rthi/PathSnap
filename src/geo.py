"""Geographical helpers.

Longitude/latitude are angular coordinates, so plain Euclidean distance on
degrees is not a physical distance (dataset brief, section 5). We use:

* haversine for point-to-point great-circle distances, and
* a local equirectangular projection (meters) centred on the data for
  point-to-segment geometry and bearings.

At the scale of a single city's road network the equirectangular
approximation introduces well under 0.1% error, which is far below GPS
noise, and it keeps all geometry cheap and vectorisable.
"""

from __future__ import annotations

import numpy as np

EARTH_R_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters (accepts scalars or arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalProjector:
    """Equirectangular projection to meters around a reference point."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._cos0 = np.cos(np.radians(self.lat0))

    def to_xy(self, lat, lon):
        x = np.radians(np.asarray(lon, dtype=float) - self.lon0) * self._cos0 * EARTH_R_M
        y = np.radians(np.asarray(lat, dtype=float) - self.lat0) * EARTH_R_M
        return x, y

    def to_latlon(self, x, y):
        lat = np.degrees(np.asarray(y, dtype=float) / EARTH_R_M) + self.lat0
        lon = np.degrees(np.asarray(x, dtype=float) / (EARTH_R_M * self._cos0)) + self.lon0
        return lat, lon


def point_to_segment_m(px, py, ax, ay, bx, by):
    """Distance (m) from point(s) P to segment AB in projected coordinates.

    All arguments are numpy arrays broadcastable to the same shape.
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    denom = np.where(denom == 0.0, 1e-12, denom)
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy)


def bearing_deg(ax, ay, bx, by):
    """Bearing in degrees clockwise from north, from A to B (projected m)."""
    return np.degrees(np.arctan2(bx - ax, by - ay)) % 360.0


def angle_diff_deg(h1, h2):
    """Smallest absolute difference between two headings, in [0, 180]."""
    return np.abs((np.asarray(h1) - np.asarray(h2) + 180.0) % 360.0 - 180.0)
