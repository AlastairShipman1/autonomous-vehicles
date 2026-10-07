import math

import numpy as np
import pytest

from av_core.geometry import RouteFrame, densify
from av_core.plan import RuleBasedPlanner
from av_core.types import Agent, EgoState, Route, WorldModel

PLANNER = RuleBasedPlanner()
STRAIGHT = densify([[0, 0], [300, 0]], 13.9)


def world(speed=10.0, agents=(), light="none", stop_line=None, x=0.0):
    ego = EgoState(x, 0.0, 0.0, speed, 4.7, 1.9, 2.9)
    return WorldModel(1.0, ego, tuple(agents), (), light, stop_line)


def car(x, y=0.0, id=1, length=4.5, width=2.0, vx=0.0):
    return Agent(id, "vehicle", x, y, 0.0, vx, 0.0, length, width, vx == 0.0)


def test_densify_spacing_and_endpoint():
    r = densify([[0, 0], [10, 0], [10, 5]], 10.0)
    d = np.hypot(*np.diff(r.points, axis=0).T)
    assert np.allclose(d, 0.5)
    assert np.allclose(r.points[-1], [10, 5])
    assert np.allclose(densify([[0, 0], [1.3, 0]], 5.0).points[-1], [1.3, 0])


def test_projection_signs():
    f = RouteFrame(STRAIGHT)
    assert f.project(20, 1.5) == pytest.approx((20.0, 1.5))
    assert f.project(20, -1.5) == pytest.approx((20.0, -1.5))


def test_curvature_of_circle():
    th = np.linspace(0, math.pi / 2, 400)
    r = densify(np.column_stack([20 * np.sin(th), 20 * (1 - np.cos(th))]), 13.9)
    assert np.allclose(np.abs(RouteFrame(r).curvature[5:-5]), 1 / 20, rtol=2e-2)  # resampling a chord polyline cuts corners slightly


def test_empty_road_gives_route_speed():
    cmd = PLANNER.plan(world(), STRAIGHT)
    assert cmd.target_speed == 13.9 and cmd.reason == "route"
    assert cmd.stamp == 1.0


def test_curve_limits_speed_to_lateral_accel():
    th = np.linspace(0, math.pi, 600)
    r = densify(np.column_stack([20 * np.sin(th), 20 * (1 - np.cos(th))]), 13.9)
    cmd = PLANNER.plan(world(), r)
    assert cmd.reason == "route"
    assert cmd.target_speed == pytest.approx(math.sqrt(2.0 * 20.0), rel=0.01)


def test_curve_beyond_horizon_is_ignored():
    pts = [[0, 0], [100, 0]] + [[100 + 20 * math.sin(a), 20 - 20 * math.cos(a)] for a in np.linspace(0, 1.5, 50)]
    r = densify(pts, 13.9)
    assert PLANNER.plan(world(), r).target_speed == 13.9


def test_lead_car_gives_stopping_profile():
    speeds = []
    for gap in (20.0, 15.0, 10.0, 7.0, 6.0, 3.0):
        front = 0.5 * (4.7 + 2.9)
        lead = car(x=front + gap + 2.25)  # rear edge exactly `gap` ahead of the front bumper
        cmd = PLANNER.plan(world(agents=[lead]), STRAIGHT)
        assert cmd.reason == "lead"
        speeds.append(cmd.target_speed)
    assert speeds[0] == pytest.approx(math.sqrt(2 * 3 * 14))
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-2:] == pytest.approx([0.0, 0.0], abs=1e-6)  # at or inside the 6 m standoff


def test_far_lead_does_not_bind():
    cmd = PLANNER.plan(world(agents=[car(x=290)]), STRAIGHT)
    assert cmd.reason == "route"


def test_off_path_and_behind_agents_ignored():
    parked = car(x=30, y=-2.75)  # parking lane: 2.75 m off the centreline
    behind = car(x=-10, id=2)
    assert PLANNER.plan(world(agents=[parked, behind]), STRAIGHT).reason == "route"


def test_in_path_pedestrian_counts_as_lead():
    ped = Agent(3, "pedestrian", 25, 0.5, 0, 0, 0, 0.6, 0.6, False)
    assert PLANNER.plan(world(agents=[ped]), STRAIGHT).reason == "lead"


def test_nearest_lead_wins():
    cmd = PLANNER.plan(world(agents=[car(x=100, id=1), car(x=30, id=2)]), STRAIGHT)
    expected = math.sqrt(2 * 3 * (30 - 2.25 - 3.8 - 6))
    assert cmd.target_speed == pytest.approx(expected)


def test_red_light_gives_stopping_profile():
    speeds = []
    for x in (0.0, 5.0, 10.0, 15.0):
        cmd = PLANNER.plan(world(speed=5.0, x=x, light="red", stop_line=[25.0, 0.0]), STRAIGHT)
        assert cmd.reason == "light"
        speeds.append(cmd.target_speed)
    assert speeds == sorted(speeds, reverse=True)
    gap = 25 - 3.8
    assert speeds[0] == pytest.approx(math.sqrt(2 * 3 * (gap - 2)))


def test_yellow_stops_only_if_ego_can():
    stop = [30.0, 0.0]
    can = PLANNER.plan(world(speed=8.0, light="yellow", stop_line=stop), STRAIGHT)  # needs 10.7 m, has 26
    cannot = PLANNER.plan(world(speed=13.0, x=14.0, light="yellow", stop_line=stop), STRAIGHT)  # needs 28 m, has 12
    assert can.reason == "light"
    assert cannot.reason == "route" and cannot.target_speed == 13.9


def test_green_none_and_missing_stop_line_ignore_light():
    for kw in ({"light": "green", "stop_line": [30.0, 0.0]}, {"light": "none", "stop_line": [30.0, 0.0]},
               {"light": "red", "stop_line": None}):
        assert PLANNER.plan(world(**kw), STRAIGHT).reason == "route"


def test_past_the_stop_line_ignores_red():
    assert PLANNER.plan(world(x=40.0, light="red", stop_line=[30.0, 0.0]), STRAIGHT).reason == "route"


def test_minimum_of_limits_and_reason():
    w = world(speed=5.0, agents=[car(x=80)], light="red", stop_line=[40.0, 0.0])
    limits = PLANNER.limits(w, STRAIGHT)
    cmd = PLANNER.plan(w, STRAIGHT)
    assert cmd.target_speed == min(limits.values())
    assert cmd.reason == min(limits, key=limits.get)
    assert limits["light"] < limits["lead"]
