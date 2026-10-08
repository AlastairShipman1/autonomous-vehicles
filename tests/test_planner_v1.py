import math

import numpy as np
import pytest

from av_core.geometry import densify, distance_points_to_convex, rect_corners, shadow_polygon
from av_core.plan import RuleBasedPlanner, RuleBasedPlannerV1, stopping_speed
from av_core.predict import ConstantVelocityPredictor
from av_core.types import Agent, EgoState, OccludedRegion, WorldModel

V0, V1 = RuleBasedPlanner(), RuleBasedPlannerV1()
PRED = ConstantVelocityPredictor()
ROUTE = densify([[-10, 0], [300, 0]], 13.9)
FRONT = 0.5 * (4.7 + 2.9)


def world(speed=10.0, x=0.0, agents=(), occluded=()):
    return WorldModel(1.0, EgoState(x, 0.0, 0.0, speed, 4.7, 1.9, 2.9), tuple(agents), tuple(occluded), ())


def ped(x, y, vx=0.0, vy=0.0, id=5):
    return Agent(id, "pedestrian", x, y, 0.0, vx, vy, 0.6, 0.6, False)


def parked(x=50.0, length=6.0):
    return Agent(1, "vehicle", x, -2.75, 0.0, 0.0, 0.0, length, 2.0, True)


def shadow_of(x_ego_front, occ_x=50.0, length=6.0):
    corners = rect_corners(occ_x, -2.75, 0.0, length, 2.0)
    return OccludedRegion(1, shadow_polygon((x_ego_front, 0.0), corners))


def plan(planner, w):
    return planner.plan(w, ROUTE, PRED(w))


def test_distance_points_to_convex_matches_scalar_cases():
    sq = rect_corners(0, 0, 0, 2, 2)
    d = distance_points_to_convex(sq, np.array([[0, 0], [2, 0], [3, 3], [0, -1.5]]))
    assert d == pytest.approx([0.0, 1.0, math.hypot(2, 2), 0.5])


def test_no_occluder_leaves_v1_equal_to_v0():
    worlds = [
        world(),
        world(agents=[parked()]),  # parked car, nobody crossing
        world(agents=[ped(40, -6.0, vy=0.0)]),
        world(agents=[ped(60, -8.0, vy=1.0)]),  # walking, but arrives long after ego passed
        world(speed=3.0, agents=[Agent(2, "vehicle", 30, 0, 0, 0, 0, 4.5, 2.0, True)]),
    ]
    for w in worlds:
        assert plan(V1, w) == plan(V0, w)


def test_pedestrian_on_crossing_course_20m_ahead_stops_with_conflict():
    w = world(speed=12.0, agents=[ped(FRONT + 20.0, -3.0, vy=1.5)])
    cmd = plan(V1, w)
    assert cmd.reason == "conflict"
    assert cmd.target_speed == pytest.approx(stopping_speed(20.0, 3.0, 6.0))
    assert cmd.target_speed < w.ego.speed  # commands braking
    assert plan(V0, w).reason == "route"  # v0 can't see it coming


def test_conflict_needs_timing_overlap():
    # same crossing pedestrian, but ego arrives ~7 s before it is on the road
    slow_ped = world(speed=12.0, agents=[ped(FRONT + 20.0, -6.5, vy=0.8)])
    assert plan(V1, slow_ped).reason == "route"
    # pedestrian walking away from the lane
    away = world(speed=12.0, agents=[ped(FRONT + 20.0, -3.0, vy=-1.0)])
    assert plan(V1, away).reason == "route"


def test_conflict_ignores_entry_behind_the_bumper():
    w = world(speed=5.0, x=30.0, agents=[ped(20.0, -3.0, vy=1.5)])
    assert plan(V1, w).reason == "route"


def test_conflict_uses_first_entry_and_nearest_conflict():
    near, far = ped(FRONT + 15.0, -2.5, vy=1.5, id=5), ped(FRONT + 40.0, -2.5, vy=1.5, id=6)
    both = plan(V1, world(speed=10.0, agents=[far, near]))
    alone = plan(V1, world(speed=10.0, agents=[near]))
    assert both == alone and both.reason == "conflict"


def test_occlusion_cap_near_a_shadow_ahead():
    w = world(speed=13.0, x=25.0, agents=[parked()], occluded=[shadow_of(25.0 + FRONT)])
    cmd = plan(V1, w)
    assert cmd.reason == "occlusion"
    assert 4.0 <= cmd.target_speed < 13.9
    assert cmd.target_speed < plan(V0, w).target_speed


def test_occlusion_cap_profile_floors_at_v_floor_and_decreases_with_approach():
    speeds = []
    for x in (22.0, 30.0, 38.0, 45.0):
        w = world(speed=10.0, x=x, agents=[parked()], occluded=[shadow_of(x + FRONT)])
        speeds.append(plan(V1, w).target_speed)
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] == 4.0


def test_far_or_passed_shadow_does_not_cap():
    far = world(x=0.0, occluded=[OccludedRegion(1, rect_corners(200, -2.75, 0, 6, 2))])
    assert plan(V1, far).reason == "route"
    behind = world(x=70.0, occluded=[shadow_of(70.0 + FRONT)])  # ego is well past the occluder
    assert plan(V1, behind).reason == "route"


def test_shadow_away_from_the_route_does_not_cap():
    off = OccludedRegion(1, np.array([[30, -10], [40, -10], [40, -8], [30, -8]], float))  # > 2.5 m from y = 0
    assert plan(V1, world(occluded=[off])).reason == "route"


def test_target_is_minimum_of_all_limits():
    w = world(speed=12.0, agents=[parked(), ped(FRONT + 20.0, -3.0, vy=1.5)], occluded=[shadow_of(FRONT)])
    limits = V1.limits(w, ROUTE, predictions=PRED(w))
    cmd = plan(V1, w)
    assert cmd.target_speed == min(limits.values())
    assert cmd.reason == min(limits, key=limits.get)
    assert set(limits) == {"lead", "light", "conflict", "occlusion", "route"}


def test_works_without_predictions():
    w = world(x=25.0, agents=[parked()], occluded=[shadow_of(25.0 + FRONT)])
    assert V1.plan(w, ROUTE).reason == "occlusion"
