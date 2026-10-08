"""Route resampling, arc length, curvature and projection onto a polyline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from av_core.types import Route

SPACING = 0.5  # m, the Route contract


def densify(points: ArrayLike, speed_limit: float, spacing: float = SPACING) -> Route:
    """Resample a polyline at uniform arc-length spacing (the last point is kept)."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / spacing)), 1)
    grid = np.minimum(np.arange(n + 1) * spacing, s[-1])
    xy = np.column_stack([np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])])
    return Route(xy, speed_limit)


class RouteFrame:
    """Arc-length coordinates along a route: project points, read curvature."""

    def __init__(self, route: Route, curvature_step: int = 2):
        self.points = route.points
        seg = np.diff(self.points, axis=0)
        self._seg = seg
        self._seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if np.any(self._seg_len == 0):
            raise ValueError("route has repeated points")
        self.s = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self.curvature = _menger_curvature(self.points, curvature_step)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, x: float, y: float) -> tuple[float, float]:
        """(arc length, signed lateral offset) of the closest route point; left of travel is positive."""
        s, lateral = self.project_many(np.array([[x, y]]))
        return float(s[0]), float(lateral[0])

    def project_many(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ``project`` for (M, 2) points: arrays of arc length and signed lateral offset."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        a = self.points[:-1]
        rel = pts[:, None, :] - a[None, :, :]  # (M, N-1, 2)
        t = np.clip(np.einsum("mkj,kj->mk", rel, self._seg) / self._seg_len**2, 0.0, 1.0)
        off = rel - t[:, :, None] * self._seg[None, :, :]  # vector from the foot point to the query point
        dist = np.hypot(off[..., 0], off[..., 1])
        i = np.argmin(dist, axis=1)
        m = np.arange(len(pts))
        cross = self._seg[i, 0] * off[m, i, 1] - self._seg[i, 1] * off[m, i, 0]
        lateral = np.where(cross >= 0, 1.0, -1.0) * dist[m, i]
        return self.s[i] + t[m, i] * self._seg_len[i], lateral


def _menger_curvature(pts: np.ndarray, k: int) -> np.ndarray:
    n = len(pts)
    kappa = np.zeros(n)
    if n < 2 * k + 1:
        return kappa
    a, b, c = pts[:-2 * k], pts[k:-k], pts[2 * k:]
    ab, bc, ca = b - a, c - b, a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ca.T)
    kappa[k:-k] = 2.0 * cross / denom
    kappa[:k], kappa[-k:] = kappa[k], kappa[-k - 1]
    return kappa
