"""Pure-pursuit steering and PID speed control.

Constants are starting guesses, retuned in CARLA later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from av_core.types import ControlCommand, EgoState, Route

THROTTLE_CAP = 0.75


def lookahead_distance(speed: float) -> float:
    return float(np.clip(0.5 * speed + 3.0, 4.0, 12.0))


def pure_pursuit_steer(ego: EgoState, route: Route, max_steer_angle: float) -> float:
    """Normalised steer in [-1, 1] toward the point L_d ahead along the route."""
    pts = route.points
    i0 = int(np.argmin(np.hypot(pts[:, 0] - ego.x, pts[:, 1] - ego.y)))
    target = _point_along(pts, i0, lookahead_distance(ego.speed))
    dx, dy = target[0] - ego.x, target[1] - ego.y
    alpha = math.atan2(dy, dx) - ego.yaw
    ld = max(math.hypot(dx, dy), 1e-6)  # chord length to the target, as in the pure-pursuit derivation
    delta = math.atan2(2.0 * ego.wheelbase * math.sin(alpha), ld)
    return float(np.clip(delta / max_steer_angle, -1.0, 1.0))


def _point_along(pts: np.ndarray, i0: int, dist: float) -> np.ndarray:
    seg = np.hypot(*np.diff(pts[i0:], axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if dist >= cum[-1]:
        return pts[-1]
    j = int(np.searchsorted(cum, dist, side="right")) - 1
    f = (dist - cum[j]) / seg[j]
    return pts[i0 + j] + f * (pts[i0 + j + 1] - pts[i0 + j])


@dataclass
class SpeedPID:
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.0
    integral_limit: float = 2.0  # clamp on the integral *term* (output units)
    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._integral, self._prev_error = 0.0, None

    def step(self, target_speed: float, speed: float, dt: float) -> tuple[float, float]:
        """Returns (throttle, brake)."""
        e = target_speed - speed
        de = 0.0 if self._prev_error is None else (e - self._prev_error) / dt
        self._prev_error = e
        u = self.kp * e + self.ki * self._integral + self.kd * de
        # Conditional integration: freeze the integral while the output is saturated and the
        # error would push it further in, then clamp the integral term as a second guard.
        saturated = (u > THROTTLE_CAP and e > 0) or (u < -1.0 and e < 0)
        if self.ki > 0 and not saturated:
            lim = self.integral_limit / self.ki
            self._integral = float(np.clip(self._integral + e * dt, -lim, lim))
        if u >= 0:
            return min(u, THROTTLE_CAP), 0.0
        return 0.0, min(-u, 1.0)


@dataclass
class Controller:
    """Steering by pure pursuit, speed by PID. One instance per episode (PID has state)."""

    max_steer_angle: float = 0.6  # rad; on the desktop this comes from the vehicle physics control
    pid: SpeedPID = field(default_factory=SpeedPID)

    def step(self, stamp: float, ego: EgoState, route: Route, target_speed: float, dt: float) -> ControlCommand:
        steer = pure_pursuit_steer(ego, route, self.max_steer_angle)
        throttle, brake = self.pid.step(target_speed, ego.speed, dt)
        return ControlCommand(stamp=stamp, throttle=throttle, brake=brake, steer=steer)
