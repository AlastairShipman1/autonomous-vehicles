"""A closed track: a looped centreline with lanes either side, in arc-length (s) and lateral coordinates.

``s`` runs along the direction of travel and wraps at ``length``; ``lateral`` is the offset from the centreline,
left positive (so on a counter-clockwise circle, left is towards the middle). Lane 0 is the right-hand lane
(lateral 0), lane 1 the one to its left (lateral ``lane_width``). The right of lane 0 is the parking lane and the
sidewalk; the left of the last lane is an inner sidewalk.

Only a circle is built so far (``Track.circle``), but everything goes through ``Track.pose`` and ``Track.project``,
so another closed shape is a different set of points, not different code.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from av_core.geometry import RouteFrame, densify
from av_core.types import Route

LANE_WIDTH = 3.5
PARKING_LATERAL = -2.75  # centre of the parking lane, right of lane 0 (as in the straight-road sim)
PARKING_EDGE = -3.75  # where the parking lane ends and the sidewalk starts
SIDEWALK_RIGHT = -4.5  # where a pedestrian waits on the right
SPEED_LIMIT = 13.9


class Track:
    def __init__(self, points: ArrayLike, lane_width: float = LANE_WIDTH, n_lanes: int = 2,
                 speed_limit: float = SPEED_LIMIT):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
            raise ValueError("a track needs at least 8 centreline points, shape (N, 2)")
        if np.hypot(*(pts[0] - pts[-1])) < 1e-9:
            pts = pts[:-1]  # the closing point is added below; do not repeat it
        self.lane_width, self.n_lanes, self.speed_limit = lane_width, n_lanes, speed_limit
        route = densify(np.vstack([pts, pts[:1]]), speed_limit)  # closed: the last point is the first
        self.frame = RouteFrame(route)
        self.length = self.frame.length
        seg = np.diff(self.frame.points, axis=0)
        self._seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self._tangent = seg / self._seg_len[:, None]
        core, pad = self.frame.points[:-1], 6  # curvature from a periodic extension, so the seam is not special
        ext = RouteFrame(Route(np.vstack([core[-pad:], core, core[:pad + 1]]), speed_limit)).curvature
        self._curvature = np.append(ext[pad:pad + len(core)], ext[pad])

    @classmethod
    def circle(cls, radius: float = 100.0, centre: tuple[float, float] = (0.0, 0.0), **kw: float) -> Track:
        """Counter-clockwise circle of ``radius`` (lane 0's centreline), starting at (radius, 0) heading +y."""
        n = max(int(2 * math.pi * radius / 0.5), 16)  # a point every 0.5 m, so the polygon is the circle to well under a millimetre
        a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        pts = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
        return cls(pts, **kw)  # pyright: ignore[reportArgumentType]

    # --- coordinates ----------------------------------------------------------------------------------------

    def wrap(self, s: float) -> float:
        return s % self.length

    def gap_ahead(self, s_from: float, s_to: float) -> float:
        """Distance along the track from ``s_from`` forward to ``s_to``, in [0, length)."""
        return (s_to - s_from) % self.length

    def lane_lateral(self, lane: int) -> float:
        return lane * self.lane_width

    def xy_at(self, s: ArrayLike, lateral: float = 0.0) -> NDArray[np.float64]:
        """World points for arc lengths ``s`` (any shape (K,)) at one lateral offset: (K, 2)."""
        sw = np.asarray(s, dtype=np.float64) % self.length
        i = np.clip(np.searchsorted(self.frame.s, sw, side="right") - 1, 0, len(self._seg_len) - 1)
        t = self._tangent[i]
        base = self.frame.points[i] + (sw - self.frame.s[i])[:, None] * t
        return base + lateral * np.column_stack([-t[:, 1], t[:, 0]])

    def pose(self, s: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """(x, y, heading) at arc length ``s`` and lateral offset; the heading is the road's direction there."""
        xy = self.xy_at(np.array([s]), lateral)[0]
        i = int(min(np.searchsorted(self.frame.s, s % self.length, side="right") - 1, len(self._seg_len) - 1))
        t = self._tangent[i]
        return float(xy[0]), float(xy[1]), math.atan2(float(t[1]), float(t[0]))

    def project(self, x: float, y: float) -> tuple[float, float]:
        """(s in [0, length), lateral) of the closest centreline point."""
        s, lat = self.frame.project(x, y)
        return s % self.length, lat

    def curvature_at(self, s: float) -> float:
        """Signed curvature of the centreline (left turn positive), 1/m."""
        return float(np.interp(s % self.length, self.frame.s, self._curvature))

    def lane_scale(self, s: float, lateral: float) -> float:
        """Centreline arc length covered per metre driven in a lane at ``lateral``: 1 / (1 - curvature * lateral).

        Inside lanes of a left-curving track are shorter, so the same speed covers more of ``s``.
        """
        return 1.0 / max(1.0 - self.curvature_at(s) * lateral, 0.2)

    # --- routes ---------------------------------------------------------------------------------------------

    def window_route(self, s_start: float, length: float, lane: int = 0) -> Route:
        """An open Route (0.5 m spacing) along ``lane`` from ``s_start`` for ``length`` metres, wrapping the loop.

        Planners and the controller see a plain open route; the loop is endless because the sim hands out a fresh
        window as the ego advances. ``length`` must be shorter than the track, or the window would overlap itself.
        """
        if not 0.0 < length < 0.9 * self.length:
            raise ValueError(f"window length {length} must be in (0, 0.9 x track length {self.length:.0f})")
        grid = s_start + np.arange(0.0, length + 0.25, 0.25)
        return densify(self.xy_at(grid, self.lane_lateral(lane)), self.speed_limit)
