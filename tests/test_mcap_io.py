import numpy as np
import pytest

from av_core.control import Controller
from av_core.plan import RuleBasedPlanner, RuleBasedPlannerV1
from av_core.types import (
    Agent,
    AgentClass,
    EgoState,
    PlannerCommand,
    PlannerReason,
    TrafficLight,
    TrafficLightState,
    WorldModel,
)
from av_sim_toy import ScenarioParams, ToySim
from tests.mcap_fixture import write_bag
from tools import mcap_io

PLANNER = RuleBasedPlanner()


def logged_run(params, steps=120):
    """Drive the toy sim with v0 and log (world model, command) like the desktop planner node would."""
    sim, ctl, worlds, cmds = ToySim(params), Controller(), [], []
    for _ in range(steps):
        if sim.done:
            break
        w = sim.world_model()
        cmd = PLANNER.plan(w, sim.route)
        worlds.append(w)
        cmds.append(cmd)
        sim.step(ctl.step(w.stamp, w.ego, sim.route, cmd.target_speed, sim.dt), ctl.max_steer_angle)
    return worlds, cmds, sim.route


@pytest.fixture
def bag(tmp_path):
    params = ScenarioParams(initial_speed=12.0, occluder_length=10.0, ped_x=50.0, ped_trigger_distance=25.0)
    worlds, cmds, route = logged_run(params)
    return write_bag(tmp_path / "run.mcap", worlds, cmds, route), worlds, cmds, route


def test_world_models_round_trip_through_the_bag(bag):
    path, worlds, _, _ = bag
    back = mcap_io.read_world_models(path)
    assert len(back) == len(worlds)
    for a, b in zip(back, worlds):
        assert a.stamp == pytest.approx(b.stamp, abs=1e-9)
        assert a.ego == b.ego and a.agents == b.agents and a.traffic_lights == b.traffic_lights
        assert [o.occluder_id for o in a.occluded] == [o.occluder_id for o in b.occluded]
        for oa, ob in zip(a.occluded, b.occluded):
            assert np.array_equal(oa.polygon, ob.polygon)
    assert any(len(w.agents) == 2 for w in back)  # includes steps with the pedestrian visible


def test_commands_and_route_round_trip(bag):
    path, _, cmds, route = bag
    back = mcap_io.read_planner_commands(path)
    assert [(c.target_speed, c.reason) for c in back] == [(c.target_speed, c.reason) for c in cmds]
    r = mcap_io.read_route(path)
    assert np.array_equal(r.points, route.points) and r.speed_limit == route.speed_limit


def test_replay_reproduces_logged_commands(bag):
    path, worlds, _, _ = bag
    result = mcap_io.replay_bag(path, RuleBasedPlanner())
    assert result.ok and result.n_compared == len(worlds)
    assert len({c.reason for c in mcap_io.read_planner_commands(path)}) >= 1


def test_replay_detects_a_different_planner(bag):
    path, _, _, _ = bag
    result = mcap_io.replay_bag(path, RuleBasedPlannerV1())  # adds the occlusion cap the log never had
    assert not result.ok and result.mismatches  # a different planner must not pass as matching


def test_replay_reports_unmatched_and_tampered_logs(bag):
    path, worlds, cmds, route = bag
    ok = mcap_io.replay(worlds, cmds, route, PLANNER)
    assert ok.ok
    tampered = list(cmds)
    tampered[3] = PlannerCommand(tampered[3].stamp, tampered[3].target_speed + 0.5, tampered[3].reason)
    r = mcap_io.replay(worlds, tampered, route, PLANNER)
    assert len(r.mismatches) == 1 and "stamp" in r.mismatches[0]
    r = mcap_io.replay(worlds, cmds[:-5], route, PLANNER)
    assert len(r.unmatched_worlds) == 5 and not r.ok
    r = mcap_io.replay(worlds[:-5], cmds, route, PLANNER)
    assert len(r.unmatched_commands) == 5 and not r.ok
    assert not mcap_io.replay([], [], route, PLANNER).ok  # nothing compared is not a pass


def test_bag_without_route_needs_one(tmp_path):
    params = ScenarioParams()
    worlds, cmds, route = logged_run(params, steps=10)
    path = write_bag(tmp_path / "noroute.mcap", worlds, cmds, None)
    assert mcap_io.read_route(path) is None
    with pytest.raises(ValueError, match="route"):
        mcap_io.replay_bag(path, PLANNER)
    assert mcap_io.replay_bag(path, PLANNER, route=route).ok


def test_cli_exit_codes(bag, tmp_path, capsys):
    path, _, _, route = bag
    assert mcap_io.main([str(path)]) == 0
    assert "MATCH" in capsys.readouterr().out
    import json
    rj = tmp_path / "route.json"
    rj.write_text(json.dumps({"points": route.points.tolist(), "speed_limit": route.speed_limit}))
    assert mcap_io.main([str(path), "--route", str(rj)]) == 0


def test_traffic_lights_round_trip(tmp_path):
    ego = EgoState(0, 0, 0, 5.0, 4.7, 1.9, 2.9)
    w = WorldModel(1.25, ego, (Agent(3, AgentClass.CYCLIST, 5, 1, 0, 2, 0, 1.8, 0.6, False),), (),
                   (TrafficLight(7, TrafficLightState.RED, [[20.0, -1.75], [20.0, 1.75]]),
                    TrafficLight(9, TrafficLightState.GREEN, [[50.0, 2.0], [50.0, 5.0]])))
    path = write_bag(tmp_path / "light.mcap", [w], [PlannerCommand(1.25, 3.0, PlannerReason.LIGHT)], None)
    (back,) = mcap_io.read_world_models(path)
    assert back.traffic_lights == w.traffic_lights and back.agents[0].cls is AgentClass.CYCLIST
