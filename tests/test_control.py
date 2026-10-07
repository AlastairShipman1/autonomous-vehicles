import math

import numpy as np
import pytest

from av_core.control import (
    BicycleState,
    Controller,
    SpeedPID,
    accel_from_pedals,
    ego_center,
    lookahead_distance,
    pure_pursuit_steer,
    step,
)
from av_core.types import EgoState, Route

DT = 0.05
L = 2.9


def ego_of(s: BicycleState) -> EgoState:
    return EgoState(s.x, s.y, s.yaw, s.speed, 4.7, 1.9, L)


def straight_route(n=400):
    return Route(np.column_stack([np.arange(n) * 0.5, np.zeros(n)]), 13.9)


def circle_route(radius=30.0, laps=1.0):
    n = int(2 * math.pi * radius * laps / 0.5)
    th = np.arange(n) * 0.5 / radius
    return Route(np.column_stack([radius * np.sin(th), radius * (1 - np.cos(th))]), 8.0)


def simulate(route, state, speed_target, seconds, lateral_error):
    c = Controller()
    errs = []
    for k in range(int(seconds / DT)):
        cmd = c.step(k * DT, ego_of(state), route, speed_target, DT)
        state = step(state, accel_from_pedals(cmd.throttle, cmd.brake), cmd.steer * c.max_steer_angle, L, DT)
        errs.append(abs(lateral_error(state)))
    return np.array(errs), state


def test_bicycle_straight_and_turn():
    s = BicycleState(0, 0, 0, 10)
    s = step(s, 0.0, 0.0, L, 1.0)
    assert s.x == pytest.approx(10.0) and s.y == pytest.approx(0.0, abs=1e-12)
    # constant steer, constant speed: travels a circle of radius L / tan(delta)
    delta, r = 0.1, L / math.tan(0.1)
    s = BicycleState(0, 0, 0, 5)
    for _ in range(int(2 * math.pi * r / 5 / DT) + 1):
        s = step(s, 0.0, delta, L, DT)
    assert math.hypot(s.x, s.y - r) == pytest.approx(r, rel=2e-3)


def test_bicycle_speed_clamped_at_zero():
    s = step(BicycleState(0, 0, 0, 0.1), -8.0, 0.0, L, 1.0)
    assert s.speed == 0.0


def test_ego_center_is_half_wheelbase_ahead():
    assert ego_center(BicycleState(1, 2, math.pi / 2, 0), L) == pytest.approx((1.0, 2.0 + L / 2))


def test_lookahead_clipping():
    assert lookahead_distance(0.0) == 4.0
    assert lookahead_distance(10.0) == 8.0
    assert lookahead_distance(30.0) == 12.0


def test_steer_sign_and_clip():
    r = straight_route()
    left = ego_of(BicycleState(0, -1.0, 0, 8))  # right of the route -> steer left (positive)
    right = ego_of(BicycleState(0, 1.0, 0, 8))
    assert pure_pursuit_steer(left, r, 0.6) > 0 > pure_pursuit_steer(right, r, 0.6)
    assert pure_pursuit_steer(ego_of(BicycleState(0, 0, 0, 8)), r, 0.6) == pytest.approx(0.0, abs=1e-9)
    assert pure_pursuit_steer(ego_of(BicycleState(0, -20, 0, 8)), r, 0.1) == 1.0


def test_straight_route_lateral_error_after_2s():
    # starts 0.5 m off the route with a 5 degree heading error
    errs, _ = simulate(straight_route(), BicycleState(0, 0.5, math.radians(5), 8), 8.0, 15, lambda s: s.y)
    assert errs[int(2 / DT):].max() < 0.3


def test_circle_lateral_error_after_2s():
    r = 30.0
    errs, _ = simulate(circle_route(r), BicycleState(0, 0, 0, 8), 8.0, 18,
                       lambda s: math.hypot(s.x, s.y - r) - r)
    assert errs[int(2 / DT):].max() < 0.3


def test_speed_step_response():
    route = straight_route(1000)
    state, c = BicycleState(0, 0, 0, 0), Controller()
    speeds = []
    for k in range(int(10 / DT)):
        cmd = c.step(k * DT, ego_of(state), route, 10.0, DT)
        assert cmd.throttle <= 0.75
        state = step(state, accel_from_pedals(cmd.throttle, cmd.brake), cmd.steer * c.max_steer_angle, L, DT)
        speeds.append(state.speed)
    speeds = np.array(speeds)
    assert speeds.max() < 11.0  # overshoot under 1 m/s
    assert np.all(np.abs(speeds[int(6 / DT):] - 10.0) < 0.5)  # settled within 0.5 m/s in 6 s


def test_pid_brakes_when_too_fast_and_clamps_integral():
    pid = SpeedPID()
    throttle, brake = pid.step(0.0, 10.0, DT)
    assert throttle == 0.0 and brake > 0
    pid = SpeedPID(kp=0.01)  # never saturates, so the integral keeps growing until clamped
    for _ in range(10_000):
        pid.step(1.0, 0.0, DT)
    assert abs(pid.ki * pid._integral) <= pid.integral_limit + 1e-9
    assert pid.step(100.0, 0.0, DT)[0] == 0.75


def test_route_end_uses_last_point():
    r = straight_route(10)  # 4.5 m long
    steer = pure_pursuit_steer(ego_of(BicycleState(4.0, 0.0, 0, 8)), r, 0.6)
    assert math.isfinite(steer)
