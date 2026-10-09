import dataclasses

import numpy as np
import pytest

from av_core.control import Controller

from av_core.plan import AIFPlanner, RuleBasedPlanner, RuleBasedPlannerV1
from av_core.predict import ConstantVelocityPredictor
from av_sim_toy import SCENARIOS, ScenarioParams, ToySim, Vehicle, run_episode
from av_sim_toy.sweep import to_record
from tools import render_scenario


def test_every_scenario_is_constructible_described_and_runs_with_v0():
    assert len(SCENARIOS) >= 10
    for name, sc in SCENARIOS.items():
        assert sc.name == name and sc.description.endswith((".", ")")) and sc.params.seed is None
        ep = run_episode(sc.params, RuleBasedPlanner())
        assert ep.outcome in ("finished", "collision")


def test_params_expose_all_vehicles_and_their_extent():
    p = SCENARIOS["row_of_cars"].params
    assert len(p.vehicles) == 4 and p.vehicles[0].x == 44.0
    assert p.near_x == pytest.approx(44.0 - 2.25) and p.far_x == pytest.approx(62.0 + 2.25)
    assert ScenarioParams().vehicles[0].length == 6.0 and ScenarioParams().near_x == 47.0


def test_pedestrian_path_may_not_cross_any_vehicle_on_either_side():
    with pytest.raises(ValueError, match="through a parked vehicle"):
        ScenarioParams(extra_vehicles=(Vehicle(62.0),), ped_x=62.5)  # inside the extra car on the near side
    with pytest.raises(ValueError, match="through a parked vehicle"):
        ScenarioParams(extra_vehicles=(Vehicle(55.0, y=2.75),), ped_x=54.0)  # it would walk through the left-hand car
    ScenarioParams(extra_vehicles=(Vehicle(62.0),), ped_x=54.0)  # clear of both


def test_world_model_has_an_agent_and_a_shadow_for_every_vehicle_in_range():
    sim = ToySim(SCENARIOS["row_of_cars"].params)
    w = sim.world_model()
    assert {a.id for a in w.agents if a.is_static} == {1, 10, 11, 12}
    assert {r.occluder_id for r in w.occluded} == {1, 10, 11}  # the 4th car is beyond the 50 m sensor range
    assert all(a.y == -2.75 for a in w.agents)
    left = ToySim(SCENARIOS["cars_both_sides"].params).world_model()
    assert {round(a.y, 2) for a in left.agents} == {-2.75, 2.75}


def test_all_vehicles_block_sight_to_the_pedestrian():
    p = ScenarioParams(initial_speed=8.0, occluder_x=44.0, occluder_length=4.5, extra_vehicles=(Vehicle(50.0), Vehicle(56.0)),
                       ped_x=59.5, ped_trigger_distance=0.0)
    alone = ScenarioParams(initial_speed=8.0, occluder_x=56.0, occluder_length=4.5, ped_x=59.5, ped_trigger_distance=0.0)
    sim_row, sim_one = ToySim(p), ToySim(alone)
    assert not sim_row.ped_visible()
    # from 8 to 20 m the last car alone leaves the pedestrian in view; the cars in front of it block that view
    for x in (8.0, 14.0, 20.0):
        for sim in (sim_row, sim_one):
            sim.state = sim.state.__class__(x, 0.0, 0.0, 8.0)
        assert sim_one.ped_visible() and not sim_row.ped_visible(), x
    # level with, and past, the pedestrian it is seen in both
    for sim in (sim_row, sim_one):
        sim.state = sim.state.__class__(56.5, 0.0, 0.0, 8.0)
    assert sim_row.ped_visible() and sim_one.ped_visible()


def test_pedestrian_from_the_left_mirrors_the_right():
    right = run_episode(SCENARIOS["single_car"].params, RuleBasedPlanner())
    left = run_episode(SCENARIOS["ped_from_left"].params, RuleBasedPlanner())
    assert np.allclose(right.ego, left.ego) and right.outcome == left.outcome
    assert np.allclose(right.ped[:, 1], -left.ped[:, 1], equal_nan=True)
    assert np.array_equal(right.ped_visible, left.ped_visible)


def test_pedestrian_stops_on_the_far_sidewalk():
    sim = ToySim(ScenarioParams(ped_trigger_distance=60.0, ped_speed=2.0))
    sim.ped_trigger_step = 0
    sim.step_count = 400  # 20 s later
    assert sim.ped_xy()[1] == pytest.approx(4.5) and not sim.ped_moving()
    sim.step_count = 20
    assert sim.ped_moving()


def test_blind_planner_collides_in_the_hard_scenarios_and_planner_v1_does_not():
    for name in ("row_of_cars", "gap_between_cars", "dart_out"):
        params = SCENARIOS[name].params
        assert run_episode(params, RuleBasedPlanner()).outcome == "collision", name
        v1 = run_episode(params, RuleBasedPlannerV1(), predictor=ConstantVelocityPredictor())
        assert v1.outcome == "finished", name


def test_no_pedestrian_scenarios_have_no_collision_and_v0_does_not_slow():
    for name in ("row_no_pedestrian", "truck_no_pedestrian"):
        ep = run_episode(SCENARIOS[name].params, RuleBasedPlanner())
        assert ep.outcome == "finished" and ep.ego[:, 3].min() > 10.0 and np.isnan(ep.ped).all()


def test_sweep_record_uses_the_extent_of_all_vehicles():
    ep = run_episode(SCENARIOS["row_no_pedestrian"].params, RuleBasedPlanner())
    rec = to_record(ep)
    assert rec.occluder_near_x == pytest.approx(41.75) and rec.occluder_far_x == pytest.approx(64.25)
    assert rec.params["n_extra_vehicles"] == 3 and "extra_vehicles" not in rec.params


def test_scenario_tool_lists_and_tabulates(capsys):
    render_scenario.main(["--list"])
    assert "row_of_cars" in capsys.readouterr().out
    render_scenario.main(["gap_between_cars", "slow_ped", "--planner", "v0", "v1", "--no-video"])
    out = capsys.readouterr().out
    assert out.count("gap_between_cars") == 2 and "collision" in out and "finished" in out
    with pytest.raises(SystemExit):
        render_scenario.main(["nope", "--no-video"])


def test_scenario_tool_writes_a_video(tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    render_scenario.main(["slow_ped", "--planner", "v0", "--out", str(tmp_path), "--stride", "10"])
    assert (tmp_path / "slow_ped__v0.mp4").stat().st_size > 1000


# --- several pedestrians -----------------------------------------------------------------------

from av_sim_toy import Pedestrian  # noqa: E402
from av_core.sweep import compute_metrics  # noqa: E402
from av_core.types import AgentClass  # noqa: E402


def two_peds(**kw):
    base = dict(initial_speed=11.0, occluder_x=46.0, occluder_length=4.5, extra_vehicles=(Vehicle(53.0),), ped_x=49.5,
                ped_trigger_distance=29.0, extra_pedestrians=(Pedestrian(49.5, speed=1.2, trigger_distance=24.0),))
    return ScenarioParams(**{**base, **kw})


def test_params_list_every_pedestrian_primary_first():
    p = two_peds()
    assert [q.trigger_distance for q in p.pedestrians] == [29.0, 24.0]
    assert two_peds(ped_present=False).pedestrians == (Pedestrian(49.5, speed=1.2, trigger_distance=24.0),)
    assert ScenarioParams(ped_present=False).pedestrians == ()
    assert Pedestrian(1.0, from_left=True).y0 == 4.5 and Pedestrian(1.0).side == -1.0


def test_extra_pedestrians_are_validated_against_every_vehicle():
    with pytest.raises(ValueError, match="pedestrian 1"):
        two_peds(extra_pedestrians=(Pedestrian(46.0),))  # inside the first car
    with pytest.raises(ValueError, match="pedestrian 1"):
        two_peds(extra_vehicles=(Vehicle(53.0), Vehicle(60.0, 2.75)), extra_pedestrians=(Pedestrian(60.0, from_left=True),))


def test_each_pedestrian_triggers_walks_and_appears_independently():
    sim = ToySim(two_peds())
    assert sim.n_peds == 2 and [sim.ped_id(i) for i in range(2)] == [2, 21]
    seen_first_trigger = []
    ctl = Controller()
    while not sim.done and sim.ego_front()[0] < 50.0:
        w = sim.world_model()
        sim.step(ctl.step(w.stamp, w.ego, sim.route, 11.0, sim.dt), ctl.max_steer_angle)
        seen_first_trigger.append(tuple(t is not None for t in sim.ped_trigger_steps))
    assert (True, False) in seen_first_trigger and (True, True) in seen_first_trigger  # first goes before the second
    assert sim.ped_trigger_steps[0] < sim.ped_trigger_steps[1]


def test_visible_pedestrians_are_agents_with_their_own_ids():
    sim = ToySim(two_peds())
    sim.state = sim.state.__class__(45.7, 0.0, 0.0, 5.0)  # front-centre straight on to the gap: both in view
    peds = [a for a in sim.world_model().agents if a.cls is AgentClass.PEDESTRIAN]
    assert sorted(a.id for a in peds) == [2, 21] and sim.peds_visible().all()


def test_collision_with_either_pedestrian_ends_the_episode():
    only_second = two_peds(ped_present=False, extra_pedestrians=(Pedestrian(49.5, speed=1.6, trigger_distance=28.0),))
    ep = run_episode(only_second, RuleBasedPlanner())
    assert ep.outcome == "collision" and ep.peds.shape[1] == 1
    assert run_episode(two_peds(), RuleBasedPlanner()).outcome == "collision"


def test_episode_log_has_one_track_per_pedestrian_and_the_primary_view_is_unchanged():
    ep = run_episode(two_peds(), RuleBasedPlannerV1(), predictor=ConstantVelocityPredictor())
    assert ep.peds.shape == (len(ep.t), 2, 2) and ep.peds_visible.shape == (len(ep.t), 2) and ep.final_peds.shape == (2, 2)
    assert np.array_equal(ep.ped, ep.peds[:, 0]) and np.array_equal(ep.ped_visible, ep.peds_visible[:, 0])
    assert ep.peds[0, 0, 0] == ep.peds[0, 1, 0] == 49.5  # both wait at the gap
    none = run_episode(ScenarioParams(ped_present=False), RuleBasedPlanner())
    assert none.peds.shape == (len(none.t), 1, 2) and np.isnan(none.peds).all()


def test_metrics_take_the_minimum_over_pedestrians():
    rec = to_record(run_episode(two_peds(), RuleBasedPlannerV1(), predictor=ConstantVelocityPredictor()))
    both = compute_metrics(rec)
    singles = [compute_metrics(dataclasses.replace(rec, ped=rec.ped[:, k:k + 1])) for k in range(2)]
    assert both.min_distance == pytest.approx(min(m.min_distance for m in singles))
    assert both.min_ttc == min(m.min_ttc for m in singles)
    assert rec.params["n_pedestrians"] == 2


def test_collision_metric_sees_a_hit_on_the_second_pedestrian():
    ep = run_episode(two_peds(ped_present=False, extra_pedestrians=(Pedestrian(49.5, speed=1.6, trigger_distance=28.0),)),
                     RuleBasedPlanner())
    m = compute_metrics(to_record(ep))
    assert m.collision is True and m.min_ttc == 0.0 and m.min_distance <= 0.3
    assert compute_metrics(to_record(run_episode(ScenarioParams(ped_present=False), RuleBasedPlanner()))).collision is None


def test_record_accepts_a_single_track_for_backwards_compatibility():
    rec = to_record(run_episode(SCENARIOS["slow_ped"].params, RuleBasedPlanner()))
    assert rec.ped.ndim == 3 and rec.ped.shape[1] == 1
    old_style = dataclasses.replace(rec, ped=rec.ped[:, 0])  # (N, 2), as before this PR
    assert old_style.ped.shape == rec.ped.shape and compute_metrics(old_style) == compute_metrics(rec)


def test_multi_pedestrian_scenarios_collide_for_v0_and_not_for_v1():
    for name in ("two_peds_one_gap", "second_pedestrian_follows", "group_of_children", "peds_both_sides", "row_two_gaps"):
        params = SCENARIOS[name].params
        assert len(params.pedestrians) >= 2, name
        assert run_episode(params, RuleBasedPlanner()).outcome == "collision", name
        assert run_episode(params, RuleBasedPlannerV1(), predictor=ConstantVelocityPredictor()).outcome == "finished", name


def test_rendering_a_frame_with_several_pedestrians_draws_each():
    import matplotlib.pyplot as plt

    from av_sim_toy import render_frame

    ep = run_episode(two_peds(), RuleBasedPlannerV1(), predictor=ConstantVelocityPredictor())
    fig = render_frame(ep, len(ep.t) // 2)
    try:
        circles = [p for p in fig.axes[0].patches if p.__class__.__name__ == "Circle"]
        assert len(circles) == 2
    finally:
        plt.close(fig)


# --- moving vehicles -----------------------------------------------------------------------------

import math  # noqa: E402

from av_core.geometry import convex_polygons_overlap, rect_corners  # noqa: E402
from av_core.sweep import format_summary, run_sweep, summarize  # noqa: E402
from av_core.types import PlannerCommand  # noqa: E402
from av_sim_toy import MovingVehicle  # noqa: E402


class FullSpeed:
    """A planner that never slows: for provoking collisions."""

    def plan(self, world, route, predictions=()):
        return PlannerCommand(world.stamp, 13.9, "route")


def test_overlap_test_for_convex_polygons():
    a = rect_corners(0, 0, 0, 4, 2)
    assert convex_polygons_overlap(a, rect_corners(3, 0, 0, 4, 2))  # overlapping
    assert not convex_polygons_overlap(a, rect_corners(5, 0, 0, 4, 2))  # apart
    assert convex_polygons_overlap(a, rect_corners(4, 0, 0, 4, 2))  # sharing an edge: touching counts
    assert not convex_polygons_overlap(a, rect_corners(4, 0, 0, 4, 2), touching=False)
    assert convex_polygons_overlap(a, rect_corners(2.5, 1.5, 0.7, 4, 2))  # rotated, clipping a corner
    assert not convex_polygons_overlap(a, rect_corners(4.5, 3.5, 0.7, 4, 2))
    assert convex_polygons_overlap(a, rect_corners(0, 0, math.pi / 2, 1, 1))  # contained


def test_moving_vehicle_validation_and_heading():
    assert MovingVehicle(30.0, 8.0).yaw == 0.0 and MovingVehicle(30.0, -8.0).yaw == pytest.approx(math.pi)
    with pytest.raises(ValueError, match="on top of the ego"):
        ScenarioParams(moving_vehicles=(MovingVehicle(5.0, 8.0),))
    ScenarioParams(moving_vehicles=(MovingVehicle(5.0, -8.0, y=3.5),))  # oncoming, in the other lane
    ScenarioParams(moving_vehicles=(MovingVehicle(30.0, 8.0),))


def test_moving_vehicles_advance_at_their_speed_and_oncoming_ones_go_the_other_way():
    sim = ToySim(ScenarioParams(ped_present=False, moving_vehicles=(MovingVehicle(30.0, 8.0), MovingVehicle(60.0, -10.0, y=3.5))))
    ctl = Controller()
    for _ in range(40):  # 2 s
        w = sim.world_model()
        sim.step(ctl.step(w.stamp, w.ego, sim.route, 5.0, sim.dt), ctl.max_steer_angle)
    assert sim.moving_pose(0)[0] == pytest.approx(30.0 + 8.0 * 2.0, abs=1e-6)
    assert sim.moving_pose(1)[0] == pytest.approx(60.0 - 10.0 * 2.0, abs=1e-6)
    agents = {a.id: a for a in sim.world_model().agents}
    assert agents[30].vx == 8.0 and agents[31].vx == -10.0 and not agents[30].is_static
    assert agents[31].yaw == pytest.approx(math.pi) and agents[31].y == 3.5
    assert {30, 31} <= {r.occluder_id for r in sim.world_model().occluded}  # they cast shadows too


def test_a_braking_vehicle_slows_at_its_rate_to_brake_to_and_holds():
    sim = ToySim(ScenarioParams(ped_present=False, moving_vehicles=(MovingVehicle(60.0, 10.0, brake_time=1.0, brake_decel=4.0, brake_to=2.0),)))
    ctl, speeds = Controller(), []
    for _ in range(100):  # 5 s
        w = sim.world_model()
        sim.step(ctl.step(w.stamp, w.ego, sim.route, 3.0, sim.dt), ctl.max_steer_angle)
        speeds.append(sim.moving_state[0][1])
    assert speeds[18] == 10.0  # not yet braking at t = 0.95 s
    assert speeds[30] == pytest.approx(10.0 - 4.0 * (31 * 0.05 - 1.0), abs=0.05)  # braking at 4 m/s²
    assert speeds[-1] == 2.0 and speeds[-5] == 2.0  # held


def test_a_moving_vehicle_blocks_the_view_of_the_pedestrian():
    base = dict(ped_trigger_distance=0.0, ped_x=54.0, occluder_length=4.5, occluder_x=50.0)
    plain = ToySim(ScenarioParams(**base))
    truck = ToySim(ScenarioParams(**base, moving_vehicles=(MovingVehicle(30.0, 0.0, length=10.0, width=2.5),)))
    far_lane = ToySim(ScenarioParams(**base, moving_vehicles=(MovingVehicle(30.0, 0.0, y=3.5, length=10.0),)))
    for sim in (plain, truck, far_lane):
        sim.state = sim.state.__class__(12.0, 0.0, 0.0, 5.0)  # truck x 25 to 35, dead ahead of the ego
    assert plain.ped_visible() and not truck.ped_visible()  # the truck in the lane hides what the parked car lets through
    assert far_lane.ped_visible()  # the same truck in the far lane hides nothing on the near side


def test_ego_hitting_a_vehicle_ends_the_episode_and_is_reported():
    params = ScenarioParams(ped_present=False, moving_vehicles=(MovingVehicle(40.0, 0.0),))  # stopped in the lane
    ep = run_episode(params, FullSpeed())
    assert ep.outcome == "collision" and ep.hit_vehicle
    rec = to_record(ep)
    m = compute_metrics(rec)
    assert rec.hit_vehicle and m.vehicle_collision and m.collision is None  # no pedestrian: the pedestrian metric is silent
    ok = compute_metrics(to_record(run_episode(ScenarioParams(ped_present=False), RuleBasedPlanner())))
    assert ok.vehicle_collision is False


def test_a_planner_that_sees_the_lead_vehicle_does_not_hit_it():
    for planner in (RuleBasedPlanner(), RuleBasedPlannerV1()):
        ep = run_episode(SCENARIOS["lead_brakes_hard"].params, planner, predictor=ConstantVelocityPredictor())
        assert ep.outcome == "finished" and not ep.hit_vehicle


def test_vehicle_collision_rate_is_in_the_summary_and_csv(tmp_path):
    from av_core.sweep import read_csv, write_csv

    class Stopped:  # a scenario factory: one episode that rear-ends a stopped car, one that does not
        def __init__(self, seed):
            self.seed = seed

        def run(self, planner, predictor=None):
            moving = (MovingVehicle(40.0, 0.0),) if self.seed == 0 else ()
            return to_record(run_episode(ScenarioParams(ped_present=False, moving_vehicles=moving), planner))

    rows = run_sweep([0, 1], Stopped, FullSpeed)
    s = summarize(rows)["pedestrian absent"]
    assert s["vehicle_collision_rate"][0] == 0.5 and "vehicle_collision_rate" in format_summary(summarize(rows))
    csv_rows = read_csv(write_csv(rows, tmp_path / "e.csv"))
    assert [r["vehicle_collision"] for r in csv_rows] == ["1", "0"] and csv_rows[0]["param_n_moving_vehicles"] == "1"


def test_planners_ignore_oncoming_traffic_in_the_other_lane():
    with_traffic = SCENARIOS["oncoming_no_pedestrian"].params
    without = dataclasses.replace(with_traffic, moving_vehicles=())
    for planner_cls in (RuleBasedPlanner, RuleBasedPlannerV1):
        a = run_episode(with_traffic, planner_cls(), predictor=ConstantVelocityPredictor())
        b = run_episode(without, planner_cls(), predictor=ConstantVelocityPredictor())
        assert a.outcome == b.outcome == "finished" and not a.hit_vehicle
        assert a.ego[:, 3].min() == pytest.approx(b.ego[:, 3].min(), abs=0.3)


def test_the_pedestrian_never_overlaps_a_moving_vehicle_in_the_oncoming_scenario():
    for planner, predictor in ((RuleBasedPlanner(), None), (RuleBasedPlannerV1(), ConstantVelocityPredictor()), (AIFPlanner(), None)):
        ep = run_episode(SCENARIOS["oncoming_traffic"].params, planner, predictor=predictor)
        for k in range(len(ep.t)):
            for j, mv in enumerate(ep.params.moving_vehicles):
                rect = rect_corners(*ep.moving[k, j], mv.length, mv.width)
                assert distance_to(rect, ep.peds[k, 0]) > 0.3, (type(planner).__name__, k, j)


def distance_to(rect, p):
    from av_core.geometry import distance_point_to_rect

    return distance_point_to_rect(rect, p)


def test_every_moving_vehicle_scenario_runs_for_every_planner_without_hitting_a_vehicle():
    names = ("slow_lead_car", "lead_brakes_hard", "follow_the_leader", "truck_ahead_hides_view",
             "oncoming_traffic", "oncoming_no_pedestrian")
    for name in names:
        for planner in (RuleBasedPlanner(), RuleBasedPlannerV1()):
            ep = run_episode(SCENARIOS[name].params, planner, predictor=ConstantVelocityPredictor())
            assert not ep.hit_vehicle and ep.outcome == "finished", (name, type(planner).__name__)


def test_rendering_draws_moving_vehicles_and_an_oncoming_lane():
    import matplotlib.pyplot as plt

    from av_sim_toy import render_frame

    ep = run_episode(SCENARIOS["oncoming_traffic"].params, RuleBasedPlanner())
    fig = render_frame(ep, 20)
    try:
        polys = [p for p in fig.axes[0].patches if p.__class__.__name__ == "Polygon" and p.get_facecolor()[:3] != (0.0, 0.0, 0.0)]
        moving_faces = [p for p in polys if p.get_zorder() == 4 and np.allclose(p.get_facecolor()[:3], (0x7a / 255, 0x9e / 255, 0x7e / 255))]
        assert len(moving_faces) == 2
    finally:
        plt.close(fig)
