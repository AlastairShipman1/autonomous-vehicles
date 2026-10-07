"""Small shape helpers shared by the sim and the sweep metrics."""

from __future__ import annotations

import numpy as np

from av_core.geometry.visibility import point_in_convex


def distance_point_to_rect(corners: np.ndarray, p) -> float:
    """Distance from a point to a convex counter-clockwise polygon; 0 inside or on the boundary."""
    p = np.asarray(p, dtype=np.float64)
    if point_in_convex(corners, p):
        return 0.0
    a, b = corners, np.roll(corners, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return float(np.min(np.hypot(*(a + t[:, None] * ab - p).T)))
