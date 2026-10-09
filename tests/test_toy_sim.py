import dataclasses
import math

import numpy as np
import pytest

from av_core.control import Controller
from av_core.geometry import distance_point_to_rect, rect_corners
from av_core.plan import RuleBasedPlanner
from av_core.types import ControlCommand, PlannerCommand, PlannerReason
from av_sim_toy import ScenarioParams, ToySim, run_episode

PLANNER = RuleBasedPlanner()


def test_distance_point_to_rect():
    r = rect_corners(0, 0, 0, 4, 2)
    assert distance_point_to_rect(r, (0, 0)) == 0.0
    assert distance_point_to_rect(r, (3, 0)) == pytest.approx(1.0)
    assert distance_point_to_rect(r, (3, 2)) == pytest.approx(math.sqrt(2))
    assert distance_point_to_rect(r, (2, 1)) == 0.0  # corner counts as touching


def test_same_params_bit_identical():
    p = ScenarioParams(ped_trigger_distance=25.0)
    a, b = run_episode(p, PLANNER), run_episode(p, PLANNER)
    for f in ("t", "ego", "ped", "ped_visible", "target_speed", "throttle", "brake", "steer"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True), f
    assert a.reason == b.reason and a.outcome == b.outcome


def test_initial_world_model():
    sim = ToySim(ScenarioParams(initial_speed=9.0, occluder_x=50.0, occluder_length=6.0))
    w = sim.world_model()
    assert w.ego.speed == 9.0 and w.stamp == 0.0 and (w.ego.x, w.ego.y) == (0.0, 0.0)
    occluder = next(a for a in w.agents if a.is_static)
    assert (occluder.x, occluder.y, occluder.length, occluder.width) == (50.0, -2.75, 6.0, 2.0)
    assert len(w.occluded) == 1 and w.occluded[0].occluder_id == occluder.id
    assert w.traffic_lights == ()


def visibility_transitions(params, speed=8.0):
    """(ego front x, visible) at each change of visibility while the ego approaches the pedestrian's x."""
    sim, ctl, prev, out = ToySim(params), Controller(), None, []
    while not sim.done and sim.ego_front()[0] < params.ped_x:
        w = sim.world_model()
        assert any(a.id == 2 for a in w.agents) == sim.ped_visible()  # in the WorldModel only while visible
        if sim.ped_visible() != prev:
            prev = sim.ped_visible()
            out.append((float(sim.ego_front()[0]), prev))
        sim.step(ctl.step(w.stamp, w.ego, sim.route, speed, sim.dt), ctl.max_steer_angle)
    return out


def test_short_occluder_parallax_visible_hidden_visible():
    # 4.5 m car (47.75 to 52.25), pedestrian 0.5 m past its far end: seen from afar, hidden while the car is in
    # the way, seen again only as the ego's front draws level with the far end
    (x0, v0), (x1, v1), (x2, v2) = visibility_transitions(
        ScenarioParams(occluder_length=4.5, ped_x=52.75, ped_trigger_distance=0.0))
    assert (v0, v1, v2) == (True, False, True)
    assert 15.0 < x1 < 30.0 and 50.0 < x2 < 53.0


def test_long_occluder_hides_the_pedestrian_until_the_ego_is_almost_alongside():
    transitions = visibility_transitions(ScenarioParams(occluder_length=10.0, ped_x=56.5, ped_trigger_distance=0.0))
    assert transitions[0][1] is False and transitions[-1][1] is True
    assert transitions[-1][0] > 53.0  # revealed with the far end (x = 55) almost level with the bumper


def test_no_pedestrian_never_collides_and_finishes():
    ep = run_episode(ScenarioParams(ped_present=False), PLANNER)
    assert ep.outcome == "finished"
    assert np.isnan(ep.ped).all() and not ep.ped_visible.any()
    assert ep.final_ego[0] > 50 + 3 + 30  # rear past the occluder's far end by the margin


def test_collision_when_pedestrian_steps_out_in_front():
    p = ScenarioParams(initial_speed=13.0, ped_x=56.0, ped_speed=1.5, ped_trigger_distance=35.0, occluder_length=10.0)
    ep = run_episode(p, PLANNER)
    assert ep.outcome == "collision"
    assert ep.t[-1] < 10.0


def test_no_collision_when_pedestrian_clears_the_lane_late():
    # ego passes the crossing before the pedestrian reaches the lane edge
    p = ScenarioParams(initial_speed=13.0, ped_x=54.0, ped_speed=0.8, ped_trigger_distance=10.0)
    assert run_episode(p, PLANNER).outcome == "finished"


def test_pedestrian_never_triggers_if_ego_stops_short():
    class Stopper:
        def plan(self, world, route, predictions=()):
            return PlannerCommand(world.stamp, 0.0, PlannerReason.ROUTE)

    p = ScenarioParams(initial_speed=8.0, ped_trigger_distance=5.0)
    ep = run_episode(p, Stopper())
    assert ep.outcome == "timeout"
    assert ep.final_ego[3] < 0.1 and ep.final_ego[0] < 50
    assert ep.t[-1] == pytest.approx(29.95)  # last step starts at 29.95, ends at 30.0


def test_stepping_after_end_raises():
    sim = ToySim(ScenarioParams(ped_present=False))
    sim.outcome = "finished"
    with pytest.raises(RuntimeError):
        sim.step(ControlCommand(0.0, 0.0, 0.0, 0.0), 0.6)


def test_ego_stays_in_lane_without_pedestrian():
    ep = run_episode(ScenarioParams(ped_present=False, initial_speed=12.0), PLANNER)
    assert np.abs(ep.ego[:, 1]).max() < 0.05
    assert dataclasses.is_dataclass(ep.params)
