"""Per-episode metrics, computed from an ``EpisodeRecord``.

Pedestrian-only metrics return ``None`` when there is no pedestrian; metrics that cannot be defined
for an episode (e.g. no braking onset, never reached the finish) are ``nan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from av_core.sweep.record import EpisodeRecord

TTC_HORIZON = 5.0  # s
TTC_STEP = 0.01  # s
BRAKING_ONSET_DECEL = 1.0  # m/s^2
NEEDLESS_STOP_SPEED = 1.0  # m/s
FINISH_MARGIN = 30.0  # m past the occluder's far end, measured at the ego's rear bumper


def _ego_frame(rec: EpisodeRecord):
    """Per-row heading and speed, and every pedestrian's position in the ego frame, shape (N, P, 2)."""
    x, y, yaw, v = rec.ego.T
    cx, cy = x + 0.5 * rec.ego_wheelbase * np.cos(yaw), y + 0.5 * rec.ego_wheelbase * np.sin(yaw)
    dx, dy = rec.ped[:, :, 0] - cx[:, None], rec.ped[:, :, 1] - cy[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    local = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)
    return local, yaw, v


def _rect_distance(local: np.ndarray, rec: EpisodeRecord) -> np.ndarray:
    ex = np.maximum(np.abs(local[..., 0]) - 0.5 * rec.ego_length, 0.0)
    ey = np.maximum(np.abs(local[..., 1]) - 0.5 * rec.ego_width, 0.0)
    return np.hypot(ex, ey)


def pedestrian_distances(rec: EpisodeRecord) -> np.ndarray:
    """Distance from each pedestrian's centre to the ego rectangle at every row, (N, P); 0 when inside, NaN if absent."""
    local, _, _ = _ego_frame(rec)
    return _rect_distance(local, rec)


def collision(rec: EpisodeRecord) -> bool | None:
    """Whether the ego rectangle overlapped any pedestrian's disc at any row."""
    if not rec.ped_present:
        return None
    return bool(np.nanmin(pedestrian_distances(rec)) <= rec.ped_radius)


def vehicle_collision(rec: EpisodeRecord) -> bool:
    """Whether the ego ran into another vehicle, parked or moving. Defined for every episode."""
    return rec.hit_vehicle


def min_distance(rec: EpisodeRecord) -> float | None:
    """Smallest distance from any pedestrian's centre to the ego rectangle over the episode."""
    return float(np.nanmin(pedestrian_distances(rec))) if rec.ped_present else None


def min_ttc(rec: EpisodeRecord) -> float | None:
    """Smallest, over the episode and the pedestrians, time until overlap if both held their current velocities.

    Velocities are the displacement over the next row (the last row reuses the previous one), searched to
    5 s on a 0.01 s grid; ``inf`` if they never overlap. 0 when already overlapping.
    """
    if not rec.ped_present:
        return None
    local, yaw, v = _ego_frame(rec)
    vp = np.diff(rec.ped, axis=0) / np.diff(rec.t)[:, None, None]
    vp = np.concatenate([vp, vp[-1:]], axis=0)  # (N, P, 2)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    rel = np.stack([c * vp[..., 0] + s * vp[..., 1] - v[:, None], -s * vp[..., 0] + c * vp[..., 1]], axis=-1)
    t = np.arange(0.0, TTC_HORIZON + TTC_STEP / 2, TTC_STEP)
    best = math.inf
    for p in range(rec.ped.shape[1]):
        if np.isnan(rec.ped[:, p]).all():
            continue
        pos = local[:, p, None, :] + rel[:, p, None, :] * t[None, :, None]  # (N, T, 2)
        hit = _rect_distance(pos, rec) <= rec.ped_radius
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        ttcs = t[first[first >= 0]]
        if ttcs.size:
            best = min(best, float(ttcs.min()))
    return best


def braking_onset(rec: EpisodeRecord) -> float:
    """Ego front to the occluder's near end (m) when deceleration first exceeds 1 m/s^2.

    Negative if the front is already past the near end; ``nan`` if the ego never brakes that hard.
    """
    decel = -np.diff(rec.ego[:, 3]) / np.diff(rec.t)
    idx = np.flatnonzero(decel > BRAKING_ONSET_DECEL)
    if idx.size == 0:
        return math.nan
    k = idx[0]
    x, y, yaw, _ = rec.ego[k]
    front_x = x + 0.5 * (rec.ego_length + rec.ego_wheelbase) * math.cos(yaw)
    return float(rec.occluder_near_x - front_x)


def needless_stop(rec: EpisodeRecord) -> bool | None:
    if rec.ped_present:
        return None
    return bool(rec.ego[:, 3].min() < NEEDLESS_STOP_SPEED)


def time_penalty(rec: EpisodeRecord) -> float:
    """Time for the ego rear to reach 30 m past the occluder, minus the time at constant initial speed.

    ``nan`` if the finish is never reached (collision or timeout) or the initial speed is zero.
    """
    x, y, yaw, v = rec.ego.T
    rear = x - 0.5 * (rec.ego_length - rec.ego_wheelbase) * np.cos(yaw)
    target = rec.occluder_far_x + FINISH_MARGIN
    reached = np.flatnonzero(rear >= target)
    if reached.size == 0 or v[0] <= 0:
        return math.nan
    k = reached[0]
    if k == 0:
        t_reach = float(rec.t[0])
    else:
        f = (target - rear[k - 1]) / (rear[k] - rear[k - 1])
        t_reach = float(rec.t[k - 1] + f * (rec.t[k] - rec.t[k - 1]))
    baseline = (target - rear[0]) / v[0]
    return t_reach - rec.t[0] - baseline


@dataclass(frozen=True)
class EpisodeMetrics:
    seed: int
    ped_present: bool
    outcome: str
    duration: float
    collision: bool | None
    min_distance: float | None
    min_ttc: float | None
    braking_onset: float
    needless_stop: bool | None
    time_penalty: float
    vehicle_collision: bool = False


def compute_metrics(rec: EpisodeRecord) -> EpisodeMetrics:
    return EpisodeMetrics(
        seed=rec.seed, ped_present=rec.ped_present, outcome=rec.outcome,
        duration=float(rec.t[-1] - rec.t[0]), collision=collision(rec), min_distance=min_distance(rec),
        min_ttc=min_ttc(rec), braking_onset=braking_onset(rec), needless_stop=needless_stop(rec),
        time_penalty=time_penalty(rec), vehicle_collision=vehicle_collision(rec),
    )
