import numpy as np
import pytest

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
