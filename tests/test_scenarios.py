import dataclasses

import numpy as np
import pytest

from av_core.control import Controller

from av_core.plan import RuleBasedPlanner, RuleBasedPlannerV1
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
