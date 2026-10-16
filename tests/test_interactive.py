"""The keyboard-driven pedestrian: its physics, the sim integration, recordings, pacing, and the window."""

import json
import math
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # no window on CI

import numpy as np
import pygame
import pytest

from av_core.plan import RuleBasedPlanner, RuleBasedPlannerV1
from av_core.predict import ConstantVelocityPredictor
from av_core.types import AgentClass, ControlCommand
from av_sim_toy import ScenarioParams, ToySim, run_episode
from av_sim_toy import scenario as sc
from av_sim_toy.live import SPEEDS, LiveSession, SessionConfig
from av_sim_toy.manual import ACCEL, MAX_SPEED, ManualPedestrian, direction
from av_sim_toy.pacing import Pacer
from av_sim_toy.play import HEIGHT, WIDTH, _Camera, _font, _rgb, draw_frame, run_app
from av_sim_toy.recording import Recording, build_params, default_start, replay
from av_sim_toy.render import COLORS
from av_sim_toy.sim import MANUAL_PED_ID, EpisodeRunner

DT = sc.DT


@pytest.fixture(scope="module", autouse=True)
def _shut_pygame_down():
    """SDL leaves threads behind; later tests fork worker processes, which is unsafe in a multi-threaded process."""
    yield
    pygame.quit()
    _font.cache_clear()  # a font from before pygame.quit() is invalid
NO_CMD = ControlCommand(0.0, 0.0, 0.0, 0.0)
CAR = np.array([[47.0, -3.75], [53.0, -3.75], [53.0, -1.75], [47.0, -1.75]])  # the default single_car occluder


def replay_with(rec, name):
    planner, predictor = make(name)
    return replay(rec, planner, predictor)


def make(name):
    planner = {"v0": RuleBasedPlanner, "v1": RuleBasedPlannerV1}[name]()
    return planner, (ConstantVelocityPredictor() if name == "v1" else None)


def walk(p, wanted, seconds, obstacles=()):
    for _ in range(round(seconds / DT)):
        p.step(wanted, DT, obstacles)


# --- the pedestrian's physics --------------------------------------------------------------------------------


def test_direction_opposite_keys_cancel():
    assert direction(False, True, False, False) == (1, 0)
    assert direction(True, True, True, False) == (0, 1)
    assert direction(False, False, False, False) == (0, 0)
    assert direction(True, False, False, True) == (-1, -1)


def test_accelerates_at_a_human_rate_and_caps_the_speed():
    p = ManualPedestrian(0.0, 0.0)
    p.step((1, 0), DT, [])
    assert p.speed == pytest.approx(ACCEL * DT)  # not instantly at walking pace
    walk(p, (1, 0), 2.0)
    assert p.speed == pytest.approx(MAX_SPEED)


def test_diagonal_is_no_faster_than_straight():
    p = ManualPedestrian(0.0, 0.0)
    walk(p, (1, 1), 2.0)
    assert p.speed == pytest.approx(MAX_SPEED)


def test_stops_when_keys_are_released_and_keeps_its_heading():
    p = ManualPedestrian(0.0, 0.0)
    walk(p, (0, -1), 1.0)
    walk(p, (0, 0), 1.0)
    assert p.speed == 0.0 and p.heading == pytest.approx(-math.pi / 2)


def test_cannot_walk_through_a_parked_vehicle():
    p = ManualPedestrian(50.0, -4.5)  # on the sidewalk, level with the middle of the car
    walk(p, (0, 1), 5.0, [CAR])
    assert p.y == pytest.approx(-3.75 - sc.PED_RADIUS - 0.02, abs=0.05)  # stopped at the car's near side
    assert p.speed == 0.0


def test_slides_along_a_vehicle_it_walks_into_diagonally():
    p = ManualPedestrian(48.0, -4.5)  # level with the car, on the sidewalk
    walk(p, (1, 1), 3.0, [CAR])
    assert p.x > 51.0  # carried along the car's side
    assert p.y == pytest.approx(-3.75 - sc.PED_RADIUS - 0.02, abs=0.05)  # and held off it
    walk(p, (1, 1), 4.0, [CAR])
    assert p.y > -3.0  # then free to cross once past its far end


def test_stays_inside_the_world():
    p = ManualPedestrian(0.0, 0.0)
    walk(p, (0, 1), 10.0)
    assert p.y <= 7.5
    walk(p, (-1, 0), 30.0)
    assert p.x >= -25.0


def test_starting_inside_a_vehicle_is_rejected():
    with pytest.raises(ValueError, match="starts inside"):
        ToySim(ScenarioParams(ped_present=False), manual_ped=ManualPedestrian(50.0, -2.75))


# --- in the sim ----------------------------------------------------------------------------------------------


def only_manual(**kw) -> ScenarioParams:
    return ScenarioParams(ped_present=False, **kw)


def test_the_driven_pedestrian_is_one_more_pedestrian():
    sim = ToySim(ScenarioParams(extra_pedestrians=()), manual_ped=ManualPedestrian(30.0, 4.5))
    assert sim.n_peds == 2 and sim.ped_id(1) == MANUAL_PED_ID and sim.ped_id(0) == 2
    assert sim.peds_xy().shape == (2, 2)
    solo = ToySim(only_manual(), manual_ped=ManualPedestrian(30.0, 4.5))
    assert solo.n_peds == 1 and np.allclose(solo.peds_xy(), [[30.0, 4.5]])


def test_without_a_driven_pedestrian_nothing_changes():
    assert ToySim(ScenarioParams()).manual is None and ToySim(ScenarioParams()).n_peds == 1
    assert ToySim(only_manual()).n_peds == 0 and np.isnan(ToySim(only_manual()).peds_xy()).all()


def test_world_model_reports_the_driven_pedestrian_only_while_visible():
    """Behind the parked cars it is absent from the WorldModel; once the ego has a line of sight it appears."""
    params = build_params("row_of_cars")
    runner = EpisodeRunner(params, RuleBasedPlanner(), manual_ped=ManualPedestrian(*default_start(params)))
    seen_flags = []
    while not runner.done and runner.sim.time < 6.0:
        world = runner.sim.world_model()
        in_world = any(a.id == MANUAL_PED_ID for a in world.agents)
        assert in_world == runner.sim.ped_visible_at(0)
        seen_flags.append(in_world)
        runner.advance()
    assert not seen_flags[0] and any(seen_flags)  # hidden at the start, visible later


def test_agent_carries_the_walking_velocity():
    sim = ToySim(only_manual(), manual_ped=ManualPedestrian(30.0, 4.5))
    for _ in range(20):
        sim.step(NO_CMD, 0.5, (0, -1))
    agent = next(a for a in sim.world_model().agents if a.id == MANUAL_PED_ID)
    assert agent.cls == AgentClass.PEDESTRIAN and agent.vy < -1.0 and agent.yaw == pytest.approx(-math.pi / 2)
    assert (agent.x, agent.y) == pytest.approx((sim.manual.x, sim.manual.y))


def test_ego_that_hits_the_driven_pedestrian_has_collided():
    """A pedestrian standing in the ego's path, with nothing slowing the ego: collision."""
    params = only_manual(initial_speed=10.0)
    sim = ToySim(params, manual_ped=ManualPedestrian(30.0, 0.0))
    while not sim.done:
        sim.step(NO_CMD, 0.5)  # no throttle or brake: the ego rolls (the bicycle model has no drag)
    assert sim.outcome == "collision"


def test_a_pedestrian_cannot_walk_into_a_stopped_ego():
    sim = ToySim(only_manual(initial_speed=0.0), manual_ped=ManualPedestrian(2.0, -3.0))
    for _ in range(200):
        sim.step(NO_CMD, 0.5, (0, 1))  # keep walking at the car's side
    assert sim.outcome is None
    assert sim.manual.y < -sc.EGO_WIDTH / 2  # held at the car's side, not overlapping it


def test_moving_ego_is_not_a_wall():
    """Only a stopped car is solid to a pedestrian; a moving one is the planner's job to avoid."""
    sim = ToySim(only_manual(initial_speed=5.0), manual_ped=ManualPedestrian(8.0, -3.0))
    for _ in range(40):
        sim.step(NO_CMD, 0.5, (0, 1))
        if sim.done:
            break
    assert sim.outcome == "collision"


# --- recordings ----------------------------------------------------------------------------------------------


def test_build_params_needs_one_of_scenario_or_seed():
    with pytest.raises(ValueError, match="exactly one"):
        build_params(None, None)
    with pytest.raises(ValueError, match="exactly one"):
        build_params("single_car", 3)
    with pytest.raises(ValueError, match="unknown scenario"):
        build_params("nope")


def test_build_params_removes_scripted_pedestrians_unless_asked():
    p = build_params("single_car")
    assert not p.ped_present and p.pedestrians == ()
    assert build_params("single_car", keep_scripted=True).ped_present
    assert build_params(None, 7).pedestrians == ()
    assert build_params("single_car", speed=7.5).initial_speed == 7.5


def test_recording_round_trips_through_json(tmp_path):
    rec = Recording("v1", (54.0, -4.5), [(0, 0)] * 30 + [(0, 1)] * 12 + [(1, 1)] * 3 + [(0, 0)], "collision",
                    scenario="single_car", speed=9.0)
    path = rec.save(tmp_path / "sub" / "r.json")
    back = Recording.load(path)
    assert back.inputs == rec.inputs and back.start == rec.start and back.outcome == "collision"
    assert (back.scenario, back.seed, back.speed, back.planner) == ("single_car", None, 9.0, "v1")
    runs = json.loads(path.read_text())["inputs"]
    assert runs == [[0, 0, 30], [0, 1, 12], [1, 1, 3], [0, 0, 1]]  # run-length encoded, so the file stays small


def test_recording_with_an_unknown_version_is_refused(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="version"):
        Recording.load(tmp_path / "r.json")


def scripted_inputs() -> list[tuple[int, int]]:
    """Stand, then walk out from behind the parked car into the road, then stand in it."""
    return [(0, 0)] * 80 + [(0, 1)] * 40 + [(0, 0)] * 40


def test_a_live_session_replays_to_the_identical_trajectory(tmp_path):
    session = LiveSession(SessionConfig("single_car", out_dir=tmp_path), make, ["v0", "v1"], "v1")
    session.toggle_pause()
    for wanted in scripted_inputs():
        session.step_once(wanted)
    while not session.done:
        session.step_once((0, 0))
    live = session.runner.episode()
    again = replay_with(session.recording(), "v1")
    assert again.outcome == live.outcome == session.recording().outcome
    assert np.array_equal(again.ego, live.ego) and np.array_equal(again.peds, live.peds)
    assert np.array_equal(again.target_speed, live.target_speed)
    reloaded = Recording.load(session.saved_path)  # and the file on disk replays the same
    third = replay_with(reloaded, "v1")
    assert np.array_equal(third.ego, live.ego)


def test_replay_matches_run_episode_with_the_same_inputs():
    params = build_params("single_car")
    start, inputs = default_start(params), scripted_inputs()
    a = run_episode(params, RuleBasedPlanner(), manual_ped=ManualPedestrian(*start), ped_inputs=inputs)
    b = replay(Recording("v0", start, inputs, scenario="single_car"), RuleBasedPlanner())
    assert np.array_equal(a.ego, b.ego) and a.outcome == b.outcome


def test_other_planners_meet_the_same_pedestrian():
    """Walking along the sidewalk (never near the road) the pedestrian's path cannot depend on the planner."""
    rec = Recording("v0", (30.0, 4.5), [(1, 0)] * 100, scenario="single_car")
    a, b = replay_with(rec, "v0"), replay_with(rec, "v1")
    n = min(len(a.t), len(b.t), 100)
    assert np.array_equal(a.peds[:n], b.peds[:n])


def test_run_episode_is_the_stepper_run_to_the_end():
    """run_episode and the live window's EpisodeRunner share one loop, so they give the same numbers."""
    p = build_params("single_car", keep_scripted=True)
    a = run_episode(p, RuleBasedPlanner())
    b = EpisodeRunner(p, RuleBasedPlanner())
    while not b.done:
        b.advance()
    assert np.array_equal(a.ego, b.episode().ego)


# --- pacing and session controls -----------------------------------------------------------------------------


def test_pacer_takes_whole_steps_and_carries_the_remainder():
    pc = Pacer(0.05)
    assert sum(pc.steps_for(1 / 60) for _ in range(60)) == 20  # one wall second is 20 steps, however it is sliced


def test_pacer_slow_motion_and_speed_up():
    slow, fast = Pacer(0.05, speed=0.25), Pacer(0.05, speed=2.0)
    assert sum(slow.steps_for(1 / 60) for _ in range(60)) == 5
    assert sum(fast.steps_for(1 / 60) for _ in range(60)) == 40


def test_pacer_drops_the_backlog_instead_of_spiralling():
    pc = Pacer(0.05, max_steps=4)
    assert pc.steps_for(1.0) == 4 and pc.dropped == 16
    assert pc.steps_for(0.0) == 0  # nothing owed afterwards


def test_pacer_validates():
    with pytest.raises(ValueError):
        Pacer(0.0)


def session(tmp_path, **kw) -> LiveSession:
    return LiveSession(SessionConfig(out_dir=tmp_path, **kw), make, ["v0", "v1"], "v1")


def test_session_starts_paused_and_does_nothing_until_started(tmp_path):
    s = session(tmp_path)
    assert s.paused and s.tick(1.0, (0, 0)) == 0 and s.sim.step_count == 0
    s.toggle_pause()
    assert s.tick(0.05, (0, 0)) == 1 and s.sim.step_count == 1 and len(s.inputs) == 1


def test_speed_keys_change_how_many_steps_a_frame_takes(tmp_path):
    s = session(tmp_path)
    s.toggle_pause()
    s.slower()
    s.slower()
    assert s.sim_speed == SPEEDS[SPEEDS.index(1.0) - 2] == 0.25
    assert s.tick(0.2, (0, 0)) == 1  # 0.05 s of sim in 0.2 s of wall time
    for _ in range(10):
        s.faster()
    assert s.sim_speed == SPEEDS[-1]


def test_single_step_works_while_paused_and_records_the_input(tmp_path):
    s = session(tmp_path)
    s.step_once((0, 1))
    assert s.sim.step_count == 1 and s.inputs == [(0, 1)] and s.paused


def test_a_finished_run_is_saved_once_and_reset_starts_afresh(tmp_path):
    s = session(tmp_path)
    s.toggle_pause()
    while not s.done:
        s.step_once((0, 0))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and Recording.load(files[0]).outcome == s.sim.outcome
    s.reset()
    assert len(list(tmp_path.glob("*.json"))) == 1  # nothing new to save
    assert s.sim.step_count == 0 and s.paused and s.inputs == []


def test_an_unfinished_run_is_saved_on_reset_and_an_empty_one_is_not(tmp_path):
    s = session(tmp_path)
    s.reset()
    assert list(tmp_path.glob("*.json")) == []
    s.step_once((0, 0))
    s.reset()
    (file,) = tmp_path.glob("*.json")
    assert Recording.load(file).outcome == "unfinished"


def test_next_planner_cycles_and_restarts_the_scenario(tmp_path):
    s = session(tmp_path)
    s.step_once((0, 0))
    s.next_planner()
    assert s.planner_name == "v0" and s.sim.step_count == 0
    s.next_planner()
    assert s.planner_name == "v1"
    assert len(list(tmp_path.glob("*.json"))) == 1  # the run that had steps was kept


def test_no_directory_means_nothing_is_written(tmp_path):
    s = LiveSession(SessionConfig(out_dir=None), make, ["v1"], "v1")
    s.step_once((0, 0))
    assert s.save() is None


def test_unknown_planner_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown planner"):
        LiveSession(SessionConfig(out_dir=None), make, ["v1"], "v9")


# --- the window ----------------------------------------------------------------------------------------------


def draw(s: LiveSession) -> pygame.Surface:
    pygame.font.init()
    surf = pygame.Surface((WIDTH, HEIGHT))
    draw_frame(surf, s)
    return surf


def ped_pixel_offsets(s: LiveSession):
    cam = _Camera(float(s.sim.ego_front()[0]))
    (px, py), = cam(np.array([s.sim.manual.x, s.sim.manual.y]))
    return int(px), int(py)


def test_a_visible_pedestrian_is_drawn_solid_and_a_hidden_one_as_an_outline(tmp_path):
    red = _rgb(COLORS["ped"])
    seen = session(tmp_path, scenario="single_car", start=(30.0, 4.5))
    assert seen.sim.ped_visible_at(0)
    px, py = ped_pixel_offsets(seen)
    assert tuple(draw(seen).get_at((px + 2, py + 2)))[:3] == red  # inside the disc, clear of the heading tick

    hidden = session(tmp_path, scenario="row_of_cars")
    assert not hidden.sim.ped_visible_at(0)
    px, py = ped_pixel_offsets(hidden)
    surf = draw(hidden)
    assert tuple(surf.get_at((px + 2, py + 2)))[:3] != red  # hollow
    assert any(tuple(surf.get_at((px + dx, py + dy)))[:3] == red for dx in range(-7, 8) for dy in range(-7, 8))  # ring


def test_the_frame_follows_the_ego_and_draws_it(tmp_path):
    s = session(tmp_path)
    before = pygame.image.tobytes(draw(s), "RGB")
    s.toggle_pause()
    for _ in range(30):
        s.step_once((0, 0))
    assert pygame.image.tobytes(draw(s), "RGB") != before  # the ego moved
    ego_px = _Camera(float(s.sim.ego_front()[0]))(np.array([s.sim.ego_front()[0] - 2.0, s.sim.ego_front()[1]]))[0]
    assert tuple(draw(s).get_at((int(ego_px[0]), int(ego_px[1]))))[:3] == _rgb(COLORS["ego"])


def test_the_loop_reads_keys_and_quits(tmp_path):
    """Post SPACE (start); the loop must step the sim over the next frames, return, and save the run."""
    s = session(tmp_path)
    pygame.display.init()
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    run_app(s, max_frames=30)  # about half a second at 60 fps
    assert not s.paused and s.sim.step_count > 0
    assert list(tmp_path.glob("*.json"))  # quitting saved the run
