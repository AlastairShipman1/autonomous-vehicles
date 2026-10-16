"""The loop track: geometry, the course, the sim, the session controls and the window."""

import math
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # no window on CI

import numpy as np
import pygame
import pytest

from av_core.plan import RuleBasedPlanner
from av_core.types import PlannerCommand, PlannerReason, TrafficLightState
from av_sim_toy.loop import (
    Course,
    Kind,
    LoopRunner,
    LoopSim,
    ParkedCar,
    Signal,
    Track,
    TrafficCar,
    Walker,
    default_course,
    empty_course,
    format_laps,
)
from av_sim_toy.loop.session import SPEEDS, LoopSession
from av_sim_toy.loop.sim import ROUTE_LENGTH, ROUTE_REFRESH, TRAFFIC_ID, WALKER_ID, _idm
from av_sim_toy.loop.view import HEIGHT, WIDTH, _font, _rgb, draw_frame, run_app
from av_sim_toy.render import COLORS
from tools.loop import main as loop_main
from tools.loop import make
from tools.planners import PLANNERS

R = 100.0
TRACK = Track.circle(R)


@pytest.fixture(scope="module", autouse=True)
def _shut_pygame_down():
    """SDL leaves threads behind; later tests fork worker processes, which is unsafe in a multi-threaded process."""
    yield
    pygame.quit()
    _font.cache_clear()


class Cruise:
    """A planner that ignores the world and asks for full speed: it finds every obstacle the hard way."""

    def plan(self, world, route, predictions=()):
        return PlannerCommand(world.stamp, 13.9, PlannerReason.ROUTE)


class Hold:
    """A planner that asks for a standstill."""

    def plan(self, world, route, predictions=()):
        return PlannerCommand(world.stamp, 0.0, PlannerReason.ROUTE)


def run(course, planner, seconds=None, laps=None):
    sim = LoopSim(TRACK, course)
    runner = LoopRunner(sim, planner)
    if laps is not None:
        runner.run_laps(laps, 600.0)
    else:
        while sim.time < seconds:
            runner.advance()
    return sim, runner


# --- the track -----------------------------------------------------------------------------------------------


def test_circle_has_the_right_length_and_starts_where_documented():
    assert TRACK.length == pytest.approx(2 * math.pi * R, rel=1e-3)
    x, y, yaw = TRACK.pose(0.0)
    assert (x, y) == pytest.approx((R, 0.0), abs=0.05) and yaw == pytest.approx(math.pi / 2, abs=0.02)


@pytest.mark.parametrize("lateral", [-4.5, -1.75, 0.0, 3.5, 6.5])
def test_pose_and_project_are_inverse_everywhere_including_the_seam(lateral):
    for s in list(np.linspace(1.0, TRACK.length - 1.0, 23)) + [TRACK.length - 0.3, 0.2]:
        x, y, _ = TRACK.pose(s, lateral)
        s2, lat2 = TRACK.project(x, y)
        assert lat2 == pytest.approx(lateral, abs=0.02)
        assert min(abs(s2 - s), TRACK.length - abs(s2 - s)) < 0.1


def test_left_is_towards_the_middle_and_lanes_get_shorter_inwards():
    x, y, _ = TRACK.pose(100.0, 3.5)
    assert math.hypot(x, y) == pytest.approx(R - 3.5, abs=0.02)
    assert TRACK.lane_scale(100.0, 3.5) == pytest.approx(R / (R - 3.5), rel=0.01)
    assert TRACK.lane_scale(100.0, -2.75) < 1.0  # outer lanes are longer


def test_wrap_and_gap_ahead():
    c = TRACK.length
    assert TRACK.wrap(c + 5.0) == pytest.approx(5.0)
    assert TRACK.gap_ahead(c - 10.0, 5.0) == pytest.approx(15.0)
    assert TRACK.gap_ahead(5.0, c - 10.0) == pytest.approx(c - 15.0)


def test_window_route_is_continuous_evenly_spaced_and_wraps_the_seam():
    route = TRACK.window_route(TRACK.length - 20.0, 100.0)
    steps = np.hypot(*np.diff(route.points, axis=0).T)
    assert steps[:-1] == pytest.approx(0.5, abs=0.01) and 0.0 < steps[-1] <= 0.51
    assert np.hypot(*route.points[0]) == pytest.approx(R, abs=0.05)
    assert np.hypot(*route.points[-1]) == pytest.approx(R, abs=0.05)  # still on the circle, 80 m round the seam
    for lane in (0, 1):
        r = TRACK.window_route(10.0, 50.0, lane)
        assert np.hypot(*r.points[0]) == pytest.approx(R - 3.5 * lane, abs=0.05)


def test_a_window_as_long_as_the_track_is_refused():
    with pytest.raises(ValueError, match="window length"):
        TRACK.window_route(0.0, TRACK.length)


def test_bad_tracks_are_refused():
    with pytest.raises(ValueError):
        Track(np.zeros((3, 2)))
    closed = Track(np.column_stack([R * np.cos(a := np.linspace(0, 2 * math.pi, 50)), R * np.sin(a)]))
    assert closed.length == pytest.approx(TRACK.length, rel=0.01)  # a repeated closing point is accepted


# --- the course ----------------------------------------------------------------------------------------------


def test_default_course_is_valid_and_covers_the_gauntlet_elements():
    c = default_course(TRACK)
    c.validate(TRACK)
    assert c.signals and len(c.parked) >= 3 and len(c.walkers) >= 2 and c.traffic


def test_a_walker_whose_path_runs_through_a_parked_car_is_refused():
    with pytest.raises(ValueError, match="through the parked car"):
        Course(parked=(ParkedCar(s=100.0),), walkers=(Walker(s=100.0),)).validate(TRACK)
    Course(parked=(ParkedCar(s=100.0),), walkers=(Walker(s=106.0),)).validate(TRACK)  # clear of its end: fine


def test_bad_traffic_is_refused():
    with pytest.raises(ValueError, match="lane 5"):
        Course(traffic=(TrafficCar(s=0.0, lane=5),)).validate(TRACK)
    with pytest.raises(ValueError, match="12 m apart"):
        Course(traffic=(TrafficCar(s=0.0), TrafficCar(s=8.0))).validate(TRACK)


# --- signals and walkers -------------------------------------------------------------------------------------


def test_signal_cycles_green_yellow_red_and_repeats():
    sim = LoopSim(TRACK, Course(signals=(Signal(s=100.0, green=10.0, yellow=2.0, red=5.0, offset=1.0),)))
    sig = sim.course.signals[0]
    states = [sim.signal_state(sig, t) for t in (0.0, 8.9, 9.1, 10.9, 11.1, 15.9, 16.1)]
    G, Y, Rd = TrafficLightState.GREEN, TrafficLightState.YELLOW, TrafficLightState.RED
    assert states == [G, G, Y, Y, Rd, Rd, G]


def test_stop_line_spans_every_lane():
    sim = LoopSim(TRACK, Course(signals=(Signal(s=100.0),)))
    a, b = sim.stop_line(sim.course.signals[0])
    assert math.hypot(*a) == pytest.approx(R + 1.75, abs=0.05) and math.hypot(*b) == pytest.approx(R - 5.25, abs=0.05)


def test_a_walker_waits_crosses_finishes_and_goes_back_to_waiting_for_the_next_lap():
    course = Course(walkers=(Walker(s=100.0, speed=2.0, trigger_distance=30.0),), ego_start_speed=8.0)
    sim, runner = LoopSim(TRACK, course), None
    runner = LoopRunner(sim, Cruise())
    phases = []
    while sim.time < 60.0:
        runner.advance()
        phase = sim.walkers[0].phase.value
        if not phases or phases[-1] != phase:
            phases.append(phase)
    assert phases[:4] == ["wait", "cross", "done", "wait"]


def test_a_walker_ends_on_the_far_sidewalk():
    course = Course(walkers=(Walker(s=100.0, speed=2.0, trigger_distance=60.0),), ego_start_speed=0.0, ego_start_s=60.0)
    sim, runner = LoopSim(TRACK, course), None
    runner = LoopRunner(sim, Hold())
    while sim.time < 12.0:
        runner.advance()
    w = sim.walkers[0]
    assert w.phase.value == "done" and w.lateral == pytest.approx(6.5)
    assert math.hypot(*sim.walker_xy(0)) == pytest.approx(R - 6.5, abs=0.05)


# --- the ego on the loop ---------------------------------------------------------------------------------------


def test_the_ego_laps_an_empty_loop_in_its_lane_at_the_speed_limit():
    sim = LoopSim(TRACK, empty_course())
    runner = LoopRunner(sim, RuleBasedPlanner())
    worst, jumps, last = 0.0, 0.0, 0.0
    while len(sim.laps) < 2:
        runner.advance()
        if sim.time > 15.0:
            worst = max(worst, abs(TRACK.project(sim.state.x, sim.state.y)[1]))
        jumps = max(jumps, abs(sim.progress - last))
        last = sim.progress
    assert worst < 0.3  # stays in its lane, across the seam too
    assert jumps < 1.0  # progress is unwrapped: no jump at the seam
    assert max(runner.speeds) < 13.9 + 0.6
    assert sim.laps[1].time == pytest.approx(TRACK.length / 13.9, rel=0.03)
    assert sim.laps[1].mean_speed == pytest.approx(13.9, abs=0.4)
    assert sim.incidents == []


def test_the_route_window_slides_and_always_has_road_ahead():
    sim = LoopSim(TRACK, empty_course())
    runner = LoopRunner(sim, RuleBasedPlanner())
    seen = set()
    for _ in range(1600):  # 80 s: nearly two laps
        runner.advance()
        route = sim.route()
        seen.add(tuple(route.points[0].round(3)))  # (not id(): freed routes hand their ids to new ones)
        start = TRACK.project(*route.points[0])[0]
        behind = TRACK.gap_ahead(start, sim.ego_s())
        assert behind < ROUTE_REFRESH + 1.0
        assert ROUTE_LENGTH - behind > 150.0  # plenty of road ahead
    assert len(seen) >= 20  # rebuilt as the ego moved (about every 45 m), not once


def test_the_run_is_deterministic():
    def lap_times():
        sim, _ = run(default_course(TRACK), RuleBasedPlanner(), laps=1)
        return [r.time for r in sim.laps], [(i.kind, round(i.time, 2)) for i in sim.incidents]

    assert lap_times() == lap_times()


def test_v0_obeys_the_light_where_cruise_runs_it():
    course = Course(signals=(Signal(s=250.0, green=2.0, yellow=2.0, red=40.0),), ego_start_speed=10.0)
    careful = LoopSim(TRACK, course)
    runner, stood = LoopRunner(careful, RuleBasedPlanner()), False
    while careful.time < 70.0:
        runner.advance()
        stood |= careful.state.speed < 0.1 and 235.0 < careful._front_s() < 250.0  # waited at the line
    reckless, _ = run(course, Cruise(), seconds=40.0)  # one pass of the line
    assert careful.totals()[Kind.RED_LIGHT] == 0 and stood and careful.progress > 255.0  # and went on at green
    assert reckless.totals()[Kind.RED_LIGHT] == 1  # logged once, not once per step spent over the line


def test_a_collision_is_logged_once_per_contact():
    course = Course(parked=(ParkedCar(s=80.0, lateral=0.0),), ego_start_speed=8.0)  # parked in the ego's lane
    sim, _ = run(course, Cruise(), seconds=20.0)
    hits = [i for i in sim.incidents if i.kind is Kind.COLLISION]
    assert len(hits) == 1 and "parked car" in hits[0].detail


def test_hitting_a_pedestrian_is_a_collision_and_passing_close_is_a_close_call():
    ped = Course(walkers=(Walker(s=100.0, speed=1.4, trigger_distance=30.0),), ego_start_speed=8.0)
    sim, _ = run(ped, Cruise(), seconds=30.0)
    kinds = [i.kind for i in sim.incidents]
    assert Kind.COLLISION in kinds and "pedestrian" in next(i for i in sim.incidents if i.kind is Kind.COLLISION).detail
    careful = LoopSim(TRACK, ped)
    LoopRunner(careful, *make("v1")).run_laps(1, 30.0)
    assert careful.totals()[Kind.COLLISION] == 0


def test_the_first_gauntlet_tells_the_planners_apart():
    """v0 ignores a hidden runner and hits it every lap; v1 gets through the same lap with no incidents."""
    v0, _ = run(default_course(TRACK), RuleBasedPlanner(), laps=2)
    assert v0.totals()[Kind.COLLISION] >= 2 and v0.totals()[Kind.RED_LIGHT] == 0  # stops at reds now
    sim = LoopSim(TRACK, default_course(TRACK))
    LoopRunner(sim, *make("v1")).run_laps(1, 600.0)
    assert sim.totals()[Kind.COLLISION] == 0 and len(sim.laps) == 1


def test_aif_drives_a_lap_without_trouble():
    sim = LoopSim(TRACK, default_course(TRACK))
    LoopRunner(sim, *make("aif")).run_laps(1, 600.0)
    assert len(sim.laps) == 1 and sim.laps[0].time < 120.0


# --- traffic and the world model -------------------------------------------------------------------------------


def test_traffic_keeps_circulating_without_overlapping():
    course = Course(traffic=tuple(TrafficCar(s=f * TRACK.length, lane=1, speed=v)
                                  for f, v in ((0.0, 12.0), (0.07, 8.0), (0.3, 10.0))), ego_start_speed=0.0)
    sim, _ = run(course, Hold(), seconds=0.0)
    runner = LoopRunner(sim, Hold())
    from av_core.geometry import convex_polygons_overlap

    for _ in range(4000):  # 200 s: several laps of the fast car catching the slow one
        runner.advance()
        rects = [sim.car_rect(i) for i in range(3)]
        assert not any(convex_polygons_overlap(rects[i], rects[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(c.v > 1.0 for c in sim.cars)  # nobody is stuck


def test_traffic_stops_behind_the_ego_when_it_is_stationary_in_its_lane():
    course = Course(traffic=(TrafficCar(s=TRACK.length - 60.0, lane=0, speed=12.0),), ego_start_speed=0.0)
    sim, runner = LoopSim(TRACK, course), None
    runner = LoopRunner(sim, Hold())
    for _ in range(1200):
        runner.advance()
    car = sim.cars[0]
    gap = TRACK.gap_ahead(car.s, TRACK.project(sim.state.x, sim.state.y)[0]) - 0.5 * (4.5 + 4.7)
    assert car.v < 0.1 and 0.5 < gap < 8.0 and not sim.incidents


def test_traffic_stops_at_a_red_light():
    course = Course(signals=(Signal(s=100.0, green=1.0, yellow=1.0, red=500.0),),
                    traffic=(TrafficCar(s=20.0, lane=1, speed=10.0),), ego_start_speed=0.0, ego_start_s=300.0)
    sim, runner = LoopSim(TRACK, course), None
    runner = LoopRunner(sim, Hold())
    for _ in range(1200):
        runner.advance()
    assert sim.cars[0].v < 0.1 and 80.0 < sim.cars[0].s < 100.0


def test_idm_free_road_approaches_desired_speed_and_brakes_for_a_stopped_leader():
    v = 0.0
    for _ in range(1200):
        v = _idm(v, 10.0, math.inf, 0.0)
    assert v == pytest.approx(10.0, abs=0.1)
    assert _idm(10.0, 10.0, 8.0, 0.0) < 10.0 - 0.3  # a stopped car 8 m ahead: brakes hard


def test_world_model_only_reports_what_the_ego_can_see():
    sim = LoopSim(TRACK, default_course(TRACK))
    runner = LoopRunner(sim, make("v1")[0])
    saw_hidden = saw_visible = False
    for _ in range(800):
        w = sim.world_model()
        ids = {a.id for a in w.agents}
        for i in range(len(sim.walkers)):
            assert (WALKER_ID + i in ids) == sim.walker_visible(i)
        assert {r.occluder_id for r in w.occluded} <= ids  # every shadow belongs to a reported vehicle
        for a in w.agents:
            assert math.hypot(a.x - sim.ego_front()[0], a.y - sim.ego_front()[1]) < 50.0 + 12.0
        for light in w.traffic_lights:
            assert math.hypot(*(light.stop_line.mean(axis=0) - sim.ego_front())) <= 80.0
        saw_hidden |= not sim.walker_visible(0) and sim.walkers[0].phase.value == "cross"
        saw_visible |= sim.walker_visible(0) and sim.walkers[0].phase.value == "cross"
        runner.advance()
    assert saw_hidden and saw_visible  # the runner is crossing unseen, then seen


def test_traffic_ids_are_stable_and_in_the_world():
    sim = LoopSim(TRACK, default_course(TRACK))
    ids = {a.id for a in sim.world_model().agents}
    assert all(TRAFFIC_ID <= i < WALKER_ID or i < TRAFFIC_ID or i >= WALKER_ID for i in ids)


def test_lap_table_formats_every_lap():
    sim, _ = run(empty_course(), RuleBasedPlanner(), laps=2)
    text = format_laps(sim.laps)
    assert len(text.splitlines()) == 3 and "collisions" in text.splitlines()[0]


# --- session controls ------------------------------------------------------------------------------------------


def session(planner="v0", course=None) -> LoopSession:
    return LoopSession(TRACK, course or default_course(TRACK), make, list(PLANNERS), planner)


def test_session_runs_from_the_start_and_pauses():
    s = session()
    assert not s.paused and s.tick(0.05) == 1 and s.sim.step_count == 1
    s.toggle_pause()
    assert s.tick(1.0) == 0 and s.sim.step_count == 1
    s.step_once()  # single-step works while paused
    assert s.sim.step_count == 2


def test_speed_keys_scale_the_steps_per_frame():
    s = session()
    s.slower()
    s.slower()
    assert s.sim_speed == SPEEDS[0] == 0.25
    assert s.tick(0.2) == 1
    for _ in range(10):
        s.faster()
    assert s.sim_speed == SPEEDS[-1] == 4.0


def test_reset_and_next_planner_start_a_fresh_run():
    s = session("v0")
    for _ in range(20):
        s.step_once()
    s.reset()
    assert s.sim.step_count == 0 and s.planner_name == "v0"
    s.next_planner()
    assert s.planner_name == "v1" and s.sim.step_count == 0
    s.toggle_view()
    assert s.overview


def test_unknown_planner_is_refused():
    with pytest.raises(ValueError, match="unknown planner"):
        session("v9")


# --- the window and the CLI --------------------------------------------------------------------------------------


def draw(s: LoopSession) -> pygame.Surface:
    pygame.font.init()
    surf = pygame.Surface((WIDTH, HEIGHT))
    draw_frame(surf, s)
    return surf


def test_the_follow_camera_keeps_the_ego_at_the_centre():
    s = session()
    for _ in range(100):
        s.step_once()
    assert tuple(draw(s).get_at((WIDTH // 2, HEIGHT // 2)))[:3] in (_rgb(COLORS["ego"]), _rgb(COLORS["edge"]))


def test_the_frame_changes_as_the_ego_moves_and_the_overview_differs_from_the_follow_view():
    s = session()
    a = pygame.image.tobytes(draw(s), "RGB")
    for _ in range(40):
        s.step_once()
    b = pygame.image.tobytes(draw(s), "RGB")
    s.toggle_view()
    c = pygame.image.tobytes(draw(s), "RGB")
    assert a != b and b != c


def test_the_window_draws_through_a_whole_lap_without_error():
    s = session("v1")
    surf = pygame.Surface((WIDTH, HEIGHT))
    pygame.font.init()
    for k in range(1000):
        s.step_once()
        if k % 100 == 0:
            draw_frame(surf, s)


def test_the_loop_reads_input_and_quits():
    s = session()
    pygame.display.init()
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET))
    run_app(s, max_frames=30)
    assert s.sim_speed == 2.0 and s.sim.step_count > 0


def test_cli_prints_a_lap_table(capsys):
    loop_main(["--headless", "--laps", "1", "--planner", "v0", "--course", "empty"])
    out = capsys.readouterr().out
    assert "v0: 1 laps" in out and "collisions" in out and "totals:" in out


def test_cli_writes_a_screenshot(tmp_path):
    path = tmp_path / "shot.png"
    loop_main(["--shot", str(path), "--at", "2"])
    assert path.stat().st_size > 5000
