"""Several obstructions at once: rows of parked cars, either side of the road, curves, and other hazards.

The toy sim has one occluder; these build WorldModels by hand and test the geometry, planner v1 and the AIF
planner against more complicated streets.
"""

import math

import numpy as np
import pytest

from av_core.geometry import RouteFrame, densify, is_visible, rect_corners, shadow_polygon
from av_core.plan import AIFPlanner, RuleBasedPlanner, RuleBasedPlannerV1
from av_core.plan.aif.model import DIST_CENTRES, ZONE, PedObs
from av_core.plan.aif.observe import find_hazard, pedestrian_observation, visibility_profile
from av_core.predict import ConstantVelocityPredictor
from av_core.types import (
    Agent,
    AgentClass,
    EgoState,
    OccludedRegion,
    PlannerReason,
    TrafficLight,
    TrafficLightState,
    WorldModel,
)

ROUTE = densify([[0, 0], [300, 0]], 13.9)  # s == x
FRAME = RouteFrame(ROUTE)
FRONT = 0.5 * (4.7 + 2.9)
PRED = ConstantVelocityPredictor()


def car(id, x, y=-2.75, length=4.5, width=2.0, cls=AgentClass.VEHICLE):
    return Agent(id, cls, x, y, 0.0, 0.0, 0.0, length, width, True)


def ped(x, y, vx=0.0, vy=0.0, id=50, cls=AgentClass.PEDESTRIAN):
    return Agent(id, cls, x, y, 0.0, vx, vy, 0.6, 0.6, False)


def world(x_front, cars=(), others=(), speed=10.0, lights=(), regions_for=None):
    """WorldModel with an occluded region (as the toy sim would give) for each car in ``regions_for``."""
    ego = EgoState(x_front - FRONT, 0.0, 0.0, speed, 4.7, 1.9, 2.9)
    regions_for = [c.id for c in cars] if regions_for is None else regions_for
    regions = tuple(
        OccludedRegion(c.id, shadow_polygon((x_front, 0.0), rect_corners(c.x, c.y, c.yaw, c.length, c.width)))
        for c in cars if c.id in regions_for
        if shadow_polygon((x_front, 0.0), rect_corners(c.x, c.y, c.yaw, c.length, c.width)) is not None)
    return WorldModel(1.0, ego, (*cars, *others), regions, tuple(lights))


# --- geometry with several occluders ---------------------------------------------------------

def hidden_by_any(sensor, point, cars):
    return not all(is_visible(sensor, point, rect_corners(c.x, c.y, 0.0, c.length, c.width)) for c in cars)


def test_a_gap_between_two_parked_cars_lets_the_ego_see_through_only_at_the_right_angle():
    a, b = car(1, 40.0), car(2, 47.0)  # 4.5 m cars with a 2.5 m gap (x 42.25 to 44.75)
    behind_gap = (43.5, -4.5)
    assert hidden_by_any((38.0, 0.0), behind_gap, [a, b])  # oblique: sight line crosses the car in front
    assert not hidden_by_any((43.5, 0.0), behind_gap, [a, b])  # straight on, through the gap
    assert hidden_by_any((43.5, 0.0), (43.5, -4.5), [car(1, 40.0, length=4.5), car(2, 47.0), car(3, 43.5, length=2.0)])


def test_shadows_of_two_cars_hide_different_places():
    a, b = car(1, 30.0), car(2, 50.0)
    sensor = (20.0, 0.0)
    for c in (a, b):
        q = shadow_polygon(sensor, rect_corners(c.x, c.y, 0.0, c.length, c.width))
        assert q is not None
    behind_a, behind_b = (40.0, -5.5), (60.0, -5.0)
    assert hidden_by_any(sensor, behind_a, [a, b]) and hidden_by_any(sensor, behind_b, [a, b])
    assert not hidden_by_any(sensor, behind_b, [a]) and not hidden_by_any(sensor, behind_a, [b])  # each car alone


def test_shadow_is_none_for_a_car_beyond_sensor_range_but_not_for_the_nearer_one():
    near, far = car(1, 30.0), car(2, 120.0)
    assert shadow_polygon((10.0, 0.0), rect_corners(near.x, near.y, 0.0, near.length, near.width)) is not None
    assert shadow_polygon((10.0, 0.0), rect_corners(far.x, far.y, 0.0, far.length, far.width)) is None


# --- planner v1 with several occluded regions ------------------------------------------------

def v1(w):
    return RuleBasedPlannerV1().plan(w, ROUTE, PRED(w))


def test_v1_the_nearest_of_several_shadows_sets_the_cap():
    near, far = car(1, 60.0), car(2, 120.0)
    both, only_near, only_far = world(40.0, [near, far]), world(40.0, [near]), world(40.0, [far])
    assert v1(both).target_speed == pytest.approx(v1(only_near).target_speed)
    assert v1(only_far).target_speed >= v1(both).target_speed


def test_v1_parked_on_the_left_caps_like_parked_on_the_right():
    right, left = world(35.0, [car(1, 60.0, y=-2.75)]), world(35.0, [car(1, 60.0, y=2.75)])
    a, b = v1(right), v1(left)
    assert a.reason == b.reason == PlannerReason.OCCLUSION
    assert a.target_speed == pytest.approx(b.target_speed)


def test_v1_cars_on_both_sides_still_cap_once():
    both = world(35.0, [car(1, 60.0, y=-2.75), car(2, 62.0, y=2.75)])
    assert v1(both).reason == PlannerReason.OCCLUSION
    assert v1(both).target_speed <= v1(world(35.0, [car(1, 60.0)])).target_speed + 1e-9  # a second car never helps


def test_v1_a_long_row_of_cars_keeps_the_cap_until_the_last_is_passed():
    row = [car(i, 50.0 + 6.0 * i) for i in range(1, 6)]  # x 56 to 80, a car every 6 m
    for x_front in (40.0, 50.0, 62.0, 75.0):
        assert v1(world(x_front, row)).reason == PlannerReason.OCCLUSION, x_front
    assert v1(world(125.0, row)).reason == PlannerReason.ROUTE  # well past the last one


def test_v1_lead_car_and_shadow_take_the_lower_limit():
    lead = car(9, 70.0, y=0.0)  # in the lane, 30 m ahead
    parked = car(1, 55.0)
    w = world(40.0, [parked], [lead], speed=10.0)
    cmd, limits = v1(w), RuleBasedPlannerV1().limits(w, ROUTE, predictions=PRED(w))
    assert cmd.target_speed == min(limits.values())
    assert limits[PlannerReason.LEAD] < math.inf and limits[PlannerReason.OCCLUSION] < math.inf


def test_v1_pedestrian_stepping_out_from_behind_the_second_of_two_cars():
    cars = [car(1, 50.0), car(2, 56.0)]
    w = world(40.0, cars, [ped(62.5, -2.6, vy=1.4)], speed=10.0)
    assert v1(w).reason in (PlannerReason.CONFLICT, PlannerReason.OCCLUSION, PlannerReason.LEAD)
    assert v1(w).target_speed < RuleBasedPlanner().plan(w, ROUTE).target_speed


# --- the AIF planner's view of several obstructions ------------------------------------------

def test_hazard_is_the_nearest_car_ahead_of_several():
    cars = [car(3, 80.0), car(1, 60.0), car(2, 70.0)]  # all within the 50 m sensor range, so all have regions
    h = find_hazard(world(35.0, cars), FRAME, 35.0)
    assert h is not None and h.occluder.id == 1
    assert len(h.others) == 2  # the other two are kept as blockers
    out_of_range = find_hazard(world(35.0, [car(1, 60.0), car(2, 120.0)]), FRAME, 35.0)
    assert out_of_range is not None and out_of_range.others == ()  # no region beyond sensor range, so no blocker


def test_hazard_skips_cars_already_passed_and_picks_up_the_next():
    cars = [car(1, 60.0), car(2, 90.0)]
    first = find_hazard(world(40.0, cars), FRAME, 40.0)
    after = find_hazard(world(75.0, cars), FRAME, 75.0)  # past the first car's crossing point (64)
    assert first is not None and after is not None and (first.occluder.id, after.occluder.id) == (1, 2)
    assert find_hazard(world(130.0, cars), FRAME, 130.0) is None


def test_cars_without_an_occluded_region_are_not_hazards_but_orphan_regions_are_ignored():
    cars = [car(1, 60.0), car(2, 70.0)]
    only_second = world(35.0, cars, regions_for=[2])
    h = find_hazard(only_second, FRAME, 35.0)
    assert h is not None and h.occluder.id == 2 and h.others == ()  # car 1 has no region: not a hazard, not a blocker
    orphan = WorldModel(1.0, only_second.ego, (cars[1],), (OccludedRegion(7, np.array([[0, 0], [1, 0], [1, 1.0]])),), ())
    assert find_hazard(orphan, FRAME, 35.0) is None


def test_a_car_on_the_left_mirrors_the_visibility_profile_of_one_on_the_right():
    right = visibility_profile(find_hazard(world(30.0, [car(1, 60.0, y=-2.75, length=6.0)]), FRAME, 30.0), FRAME)
    left = visibility_profile(find_hazard(world(30.0, [car(1, 60.0, y=2.75, length=6.0)]), FRAME, 30.0), FRAME)
    assert np.array_equal(right, left)
    h = find_hazard(world(30.0, [car(1, 60.0, y=2.75)]), FRAME, 30.0)
    assert h is not None and h.ped_xy[1] == pytest.approx(4.5)  # waits on the far side of a left-hand car


def test_a_truck_in_the_lane_is_not_the_hazard_but_blocks_the_view():
    parked, bus = car(1, 60.0), car(9, 37.0, y=0.0, length=14.0, width=3.2)  # bus x 30 to 44, in the ego's lane
    w, alone = world(20.0, [parked, bus]), world(20.0, [parked])
    h, h_alone = find_hazard(w, FRAME, 20.0), find_hazard(alone, FRAME, 20.0)
    assert h is not None and h.occluder.id == 1 and len(h.others) == 1  # the parked car, not the truck
    assert h_alone is not None
    with_truck, without = visibility_profile(h, FRAME), visibility_profile(h_alone, FRAME)
    assert (with_truck <= without).all() and with_truck.sum() < without.sum()


def test_a_truck_in_the_lane_does_not_hide_the_pedestrian_once_the_ego_is_level_with_it():
    parked, bus = car(1, 60.0), car(9, 37.0, y=0.0, length=14.0, width=3.2)
    profile = visibility_profile(find_hazard(world(20.0, [parked, bus]), FRAME, 20.0), FRAME)
    assert profile[ZONE] == 1.0 and profile[ZONE + 1] == 1.0  # level with, and past, the crossing point


def test_truck_hides_longer_than_a_motorbike():
    truck = visibility_profile(find_hazard(world(30.0, [car(1, 60.0, length=12.0)]), FRAME, 30.0), FRAME)
    bike = visibility_profile(find_hazard(world(30.0, [car(1, 60.0, length=1.8, width=0.8)]), FRAME, 30.0), FRAME)
    assert truck.sum() < bike.sum()


def test_aif_with_several_cars_never_exceeds_the_rule_based_limits_and_runs_each_replan():
    cars = [car(i, 55.0 + 8.0 * i) for i in range(4)]
    pl = AIFPlanner(record=True)
    for step, x in enumerate(np.arange(20.0, 100.0, 4.0)):
        w = world(float(x), cars, speed=9.0)
        w = WorldModel(0.5 * step, w.ego, w.agents, w.occluded, w.traffic_lights)
        cmd = pl.plan(w, ROUTE)
        assert cmd.target_speed <= RuleBasedPlanner().plan(w, ROUTE).target_speed + 1e-9
    assert len(pl.diagnostics) >= 15


def test_aif_belief_resets_when_the_hazard_changes_to_the_next_car():
    cars = [car(1, 60.0), car(2, 95.0)]
    pl = AIFPlanner(record=True)
    seen = world(35.0, [*cars], [ped(64.0, -4.5)], speed=8.0)
    pl.plan(seen, ROUTE)
    assert pl.diagnostics[-1].ped_belief[1] > 0.85  # waiting pedestrian seen behind the first car
    nxt = world(80.0, cars, speed=8.0)
    nxt = WorldModel(5.0, nxt.ego, nxt.agents, nxt.occluded, ())
    pl.plan(nxt, ROUTE)
    assert pl.diagnostics[-1].ped_belief[1] == pytest.approx(0.5, abs=0.02)  # fresh prior for the second car


def test_two_pedestrians_the_crossing_one_wins_and_distant_or_other_classes_are_ignored():
    h = find_hazard(world(30.0, [car(1, 60.0)]), FRAME, 30.0)
    assert h is not None
    two = world(30.0, [car(1, 60.0)], [ped(64.0, -4.5, id=50), ped(64.0, -2.5, vy=1.4, id=51)])
    assert pedestrian_observation(two, FRAME, h) is PedObs.SEEN_CROSSING
    far = world(30.0, [car(1, 60.0)], [ped(150.0, -1.0, id=52)])
    assert pedestrian_observation(far, FRAME, h) is PedObs.NOT_SEEN
    cyclist = world(30.0, [car(1, 60.0)], [ped(64.0, -1.0, vx=6.0, cls=AgentClass.CYCLIST, id=53)])
    assert pedestrian_observation(cyclist, FRAME, h) is PedObs.NOT_SEEN


def test_red_light_beyond_an_occluder_the_light_still_binds_when_it_is_the_lowest_limit():
    light = TrafficLight(1, TrafficLightState.RED, np.array([[50.0, -1.5], [50.0, 1.5]]))
    w = world(30.0, [car(1, 60.0)], speed=8.0, lights=[light])  # light 20 m ahead, car beyond it
    cmd = AIFPlanner().plan(w, ROUTE)
    v0 = RuleBasedPlanner().plan(w, ROUTE)
    assert cmd.target_speed <= v0.target_speed + 1e-9
    assert v0.reason == PlannerReason.LIGHT
    assert cmd.reason in (PlannerReason.LIGHT, PlannerReason.OCCLUSION)


def test_lead_car_in_the_lane_bounds_the_aif_command():
    lead = car(9, 55.0, y=0.0)
    w = world(35.0, [car(1, 70.0)], [lead], speed=10.0)
    cmd = AIFPlanner().plan(w, ROUTE)
    assert cmd.target_speed <= RuleBasedPlanner().plan(w, ROUTE).target_speed + 1e-9


def test_a_curving_route_still_gives_a_hazard_a_visibility_profile_and_a_command():
    th = np.linspace(0.0, 0.5, 400)
    curved = densify(np.column_stack([150 * np.sin(th), 150 * (1 - np.cos(th))]), 13.9)
    frame = RouteFrame(curved)
    xy, tangent = frame.pose_at(60.0, -2.75)  # a parked car 2.75 m to the right of the route at s = 60
    yaw = math.atan2(tangent[1], tangent[0])
    parked = Agent(1, AgentClass.VEHICLE, float(xy[0]), float(xy[1]), yaw, 0.0, 0.0, 4.5, 2.0, True)
    front_xy, _ = frame.pose_at(30.0)
    corners = rect_corners(parked.x, parked.y, parked.yaw, parked.length, parked.width)
    region = OccludedRegion(1, shadow_polygon(front_xy, corners))
    ego_xy, ego_t = frame.pose_at(30.0 - FRONT)
    ego = EgoState(float(ego_xy[0]), float(ego_xy[1]), math.atan2(ego_t[1], ego_t[0]), 9.0, 4.7, 1.9, 2.9)
    w = WorldModel(1.0, ego, (parked,), (region,), ())
    h = find_hazard(w, frame, 30.0)
    assert h is not None and h.crossing_s == pytest.approx(60.0 + 2.25 + 1.0, abs=0.3)
    profile = visibility_profile(h, frame)
    assert profile.shape == DIST_CENTRES.shape and set(profile) <= {0.0, 1.0}
    cmd = AIFPlanner().plan(w, curved)
    assert math.isfinite(cmd.target_speed) and cmd.target_speed >= 0.0
