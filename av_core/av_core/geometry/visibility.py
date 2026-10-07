"""Line-of-sight against a rectangular occluder, and the shadow it casts.

A point is visible if it is within range and the segment from the sensor to it does not pass
through the occluder's interior; grazing an edge or corner counts as visible.
"""

from __future__ import annotations

import math

import numpy as np

SENSOR_RANGE = 50.0  # m
_EPS = 1e-9


def rect_corners(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
    """(4, 2) corners, counter-clockwise, of a rectangle centred on (cx, cy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def point_in_convex(poly: np.ndarray, p, strict: bool = False) -> bool:
    """Point-in-convex-polygon for a counter-clockwise polygon (boundary counts unless ``strict``)."""
    p = np.asarray(p, dtype=np.float64)
    e = np.roll(poly, -1, axis=0) - poly
    cross = e[:, 0] * (p[1] - poly[:, 1]) - e[:, 1] * (p[0] - poly[:, 0])
    return bool(np.all(cross > 0) if strict else np.all(cross >= 0))


def _crosses_interior(a: np.ndarray, b: np.ndarray, corners: np.ndarray) -> bool:
    """Does segment a->b pass through the interior of the convex CCW polygon? (Cyrus-Beck, strict.)"""
    d = b - a
    t0, t1 = 0.0, 1.0
    edges = np.roll(corners, -1, axis=0) - corners
    for (ex, ey), (px, py) in zip(edges, corners):
        # inside means cross(edge, p - corner) > 0; it is linear in t: num + t * den
        num = ex * (a[1] - py) - ey * (a[0] - px)
        den = ex * d[1] - ey * d[0]
        if abs(den) < 1e-12:  # parallel to this edge
            if num <= 0:  # on the edge line or outside it: never strictly inside
                return False
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t1 - t0 <= _EPS:
            return False
    return True


def is_visible(sensor, point, corners: np.ndarray, max_range: float = SENSOR_RANGE) -> bool:
    a, p = np.asarray(sensor, dtype=np.float64), np.asarray(point, dtype=np.float64)
    if math.hypot(*(p - a)) > max_range:
        return False
    return not _crosses_interior(a, p, corners)


def shadow_polygon(sensor, corners: np.ndarray, max_range: float = SENSOR_RANGE) -> np.ndarray | None:
    """Shadow quad: the two corners at the extreme bearings, each extended along its bearing ray.

    The far edge is placed tangent to the range circle (rays run to ``max_range / cos(half_angle)``),
    so the quad covers every in-range point hidden by the occluder; it also extends slightly past
    ``max_range`` toward its ends. Vertices are counter-clockwise: near corner, far point, far point,
    near corner. Returns ``None`` if the whole occluder is out of range.
    """
    a = np.asarray(sensor, dtype=np.float64)
    if point_in_convex(corners, a):
        raise ValueError("sensor is inside the occluder")
    rel = corners - a
    dist = np.hypot(rel[:, 0], rel[:, 1])
    if dist.min() >= max_range:
        return None
    centre_bearing = math.atan2(*(corners.mean(axis=0) - a)[::-1])
    ang = (np.arctan2(rel[:, 1], rel[:, 0]) - centre_bearing + math.pi) % (2 * math.pi) - math.pi
    lo, hi = int(np.argmin(ang)), int(np.argmax(ang))
    far = max_range / math.cos(0.5 * (ang[hi] - ang[lo]))
    far = max(far, dist[lo], dist[hi])
    dir_lo, dir_hi = rel[lo] / dist[lo], rel[hi] / dist[hi]
    return np.array([corners[lo], a + dir_lo * far, a + dir_hi * far, corners[hi]])  # counter-clockwise
