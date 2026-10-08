"""Kinematic bicycle model, reference point at the rear axle.

``EgoState.x, y`` is the rear-axle position. The rectangle centre sits ``wheelbase / 2``
ahead of it when the body is centred on the wheelbase (see ``ego_center``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from av_core.types import EgoState


# Stand-in for CARLA's dynamics: a = ACCEL_GAIN * throttle - BRAKE_GAIN * brake  [m/s^2]
ACCEL_GAIN = 4.0
BRAKE_GAIN = 8.0


@dataclass(frozen=True)
class BicycleState:
    x: float
    y: float
    yaw: float
    speed: float


def step(state: BicycleState, accel: float, steer_angle: float, wheelbase: float, dt: float) -> BicycleState:
    """One Euler step. Speed is clamped at 0: the vehicle never reverses."""
    v_new = max(0.0, state.speed + accel * dt)
    v_mid = 0.5 * (state.speed + v_new)  # trapezoidal speed keeps braking-to-stop distance accurate
    yaw_rate = v_mid / wheelbase * math.tan(steer_angle)
    yaw_mid = state.yaw + 0.5 * yaw_rate * dt
    return BicycleState(
        x=state.x + v_mid * math.cos(yaw_mid) * dt,
        y=state.y + v_mid * math.sin(yaw_mid) * dt,
        yaw=_wrap(state.yaw + yaw_rate * dt),
        speed=v_new,
    )


def accel_from_pedals(throttle: float, brake: float) -> float:
    return ACCEL_GAIN * throttle - BRAKE_GAIN * brake


def ego_center(state: BicycleState, wheelbase: float) -> tuple[float, float]:
    """Rectangle centre for a body centred on the wheelbase (rear overhang = front overhang)."""
    d = 0.5 * wheelbase
    return state.x + d * math.cos(state.yaw), state.y + d * math.sin(state.yaw)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def with_state(ego: EgoState, state: BicycleState) -> EgoState:
    """Copy an ``EgoState`` with the kinematic fields replaced."""
    return replace(ego, x=state.x, y=state.y, yaw=state.yaw, speed=state.speed)


__all__ = ["BicycleState", "step", "accel_from_pedals", "ego_center", "with_state"]
