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


def distance_points_to_convex(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorised distance from each of ``pts`` (M, 2) to a convex counter-clockwise polygon; 0 inside."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    ap = pts[:, None, :] - a[None, :, :]  # (M, K, 2)
    t = np.clip(np.einsum("mkj,kj->mk", ap, ab) / np.where(denom == 0, 1.0, denom), 0.0, 1.0)
    diff = ap - t[:, :, None] * ab[None, :, :]
    edge_dist = np.hypot(diff[..., 0], diff[..., 1]).min(axis=1)
    cross = ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]
    inside = np.all(cross >= 0, axis=1)
    return np.where(inside, 0.0, edge_dist)
