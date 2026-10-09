import math

import numpy as np
import pytest
from pymdp import control

from av_core.geometry import RouteFrame, densify, rect_corners, shadow_polygon
from av_core.plan import AIFPlanner, RuleBasedPlanner
from av_core.plan.aif import Action, AIFParams, build_model, expected_free_energy
from av_core.plan.aif.efe import joint_likelihood
from av_core.plan.aif.model import (
    ACCEL,
    APPROACH,
    DIST_CENTRES,
    DT,
    NE,
    NP,
    NS,
    NV,
    SPEED_CENTRES,
    ZONE,
    Ped,
    PedObs,
    Safety,
    ego_index,
    split_distance,
    split_speed,
    state_index,
)
from av_core.plan.aif.observe import find_hazard, pedestrian_observation, visibility_profile
from av_core.types import Agent, AgentClass, EgoState, OccludedRegion, PlannerReason, WorldModel
from av_sim_toy import ScenarioParams, ToySim

ROUTE = densify([[-10, 0], [300, 0]], 13.9)
FRAME = RouteFrame(ROUTE)
FRONT = 0.5 * (4.7 + 2.9)
VIS = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0])  # hidden until level with the crossing point


# --- the generative model --------------------------------------------------------------------

def test_model_arrays_are_proper_distributions():
    m = build_model(VIS)
    for a in m.A:
        assert a.shape[1] == NS and np.allclose(a.sum(axis=0), 1.0) and (a >= 0).all()
    B = m.B[0]
    assert B.shape == (NS, NS, len(Action)) and np.allclose(B.sum(axis=0), 1.0) and (B >= 0).all()
    assert np.allclose(m.B_ped.sum(axis=1), 1.0)
    assert m.D_ped.sum() == pytest.approx(1.0)
    assert [len(c) for c in m.C] == [len(PedObs), NV, len(Safety)]


def test_band_splits_are_exact_at_centres_and_sum_to_one():
    for i, d in enumerate(DIST_CENTRES):
        assert {j: w for j, w in split_distance(float(d)) if w > 0} == {i: 1.0}
    for i, v in enumerate(SPEED_CENTRES):
        assert {j: w for j, w in split_speed(float(v)) if w > 0} == {i: 1.0}
    for x in (45.0, 13.0, 3.0, -2.0):
        assert sum(w for _, w in split_distance(x)) == pytest.approx(1.0)
    assert split_distance(100.0) == [(0, 1.0)] and split_distance(-50.0) == [(len(DIST_CENTRES) - 1, 1.0)]
    (i0, w0), (i1, w1) = split_distance(20.0)  # between the 24 m and 16 m bands
    assert {i0, i1} == {3, 4} and w0 == pytest.approx(0.5) and w1 == pytest.approx(0.5)


def expected_speed_and_distance(model, d, v, a):
    col = model.B[0][:, state_index(ego_index(d, v), Ped.WAITING), a].reshape(NE, NP).sum(axis=1)
    return float(col @ np.tile(SPEED_CENTRES, len(DIST_CENTRES))), float(col @ np.repeat(DIST_CENTRES, NV))


def test_actions_order_speed_and_everything_advances():
    m = build_model(VIS)
    d, v = 2, 5  # 32 m out, 10 m/s
    speeds = [expected_speed_and_distance(m, d, v, a)[0] for a in Action]
    assert speeds[Action.KEEP] > speeds[Action.EASE_OFF] > speeds[Action.BRAKE]
    assert speeds[Action.KEEP] == pytest.approx(SPEED_CENTRES[v] + ACCEL[Action.KEEP] * DT)
    assert speeds[Action.BRAKE] == pytest.approx(SPEED_CENTRES[v] + ACCEL[Action.BRAKE] * DT)
    dist_keep = expected_speed_and_distance(m, d, v, Action.KEEP)[1]
    dist_brake = expected_speed_and_distance(m, d, v, Action.BRAKE)[1]
    assert dist_brake > dist_keep  # braking closes the distance more slowly
    assert dist_keep == pytest.approx(DIST_CENTRES[d] - 0.5 * (10.0 + 10.5) * DT)


def test_stopped_ego_stays_put_and_past_the_crossing_is_absorbing():
    m = build_model(VIS)
    s, d = expected_speed_and_distance(m, 4, 0, Action.BRAKE)
    assert s == 0.0 and d == DIST_CENTRES[4]
    _, d_past = expected_speed_and_distance(m, len(DIST_CENTRES) - 1, 6, Action.KEEP)
    assert d_past == DIST_CENTRES[-1]


def test_pedestrian_dynamics_depend_on_the_distance_band():
    m = build_model(VIS)
    far, mid, zone = 0, 3, ZONE
    assert m.B_ped[far, Ped.CROSSING, Ped.WAITING] < m.B_ped[mid, Ped.CROSSING, Ped.WAITING]
    assert m.B_ped[zone, Ped.CROSSING, Ped.WAITING] == 0.0  # nothing new starts once the ego is on top of it
    for d in range(len(DIST_CENTRES)):
        assert m.B_ped[d, Ped.NONE, Ped.NONE] == 1.0 and m.B_ped[d, Ped.CLEARED, Ped.CLEARED] == 1.0


def test_observation_model():
    m = build_model(VIS)
    far, zone = ego_index(0, 3), ego_index(ZONE, 3)
    seen_waiting = lambda e: m.A[0][PedObs.SEEN_WAITING, state_index(e, Ped.WAITING)]  # noqa: E731
    assert seen_waiting(far) == 0.0 and seen_waiting(zone) == pytest.approx(0.95)  # only in line of sight
    for e in (far, zone):  # a crossing pedestrian is seen from anywhere; nothing is seen when none is there
        assert m.A[0][PedObs.SEEN_CROSSING, state_index(e, Ped.CROSSING)] == pytest.approx(0.95)
        assert m.A[0][PedObs.NOT_SEEN, state_index(e, Ped.NONE)] == 1.0


def test_collision_only_in_the_zone_with_a_crossing_pedestrian_and_worse_when_fast():
    m = build_model(VIS)
    hit = lambda d, v, ped: m.A[2][Safety.COLLISION, state_index(ego_index(d, v), ped)]  # noqa: E731
    assert hit(ZONE, 5, Ped.CROSSING) > hit(ZONE, 1, Ped.CROSSING) > 0
    assert hit(ZONE, 5, Ped.WAITING) == 0.0 and hit(APPROACH, 5, Ped.CROSSING) == 0.0
    assert m.A[2][Safety.NEAR, state_index(ego_index(APPROACH, 5), Ped.CROSSING)] > 0


# --- expected free energy --------------------------------------------------------------------

def random_belief(seed):
    q = np.random.default_rng(seed).random(NS) ** 4
    return q / q.sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_efe_equals_pymdp(seed):
    m = build_model(VIS, AIFParams(policy_len=3))
    qs0 = random_belief(seed)
    qs = np.empty(1, dtype=object)
    qs[0] = qs0
    policies, risk, ambiguity = expected_free_energy(qs0, m, 3)
    ref_pols = control.construct_policies(m.num_states, m.num_controls, policy_len=3, control_fac_idx=[0])
    _, G_ref = control.update_posterior_policies(
        qs, m.A, m.B, m.C, ref_pols, use_utility=True, use_states_info_gain=True, gamma=16.0)
    ref = {tuple(int(x) for x in p[:, 0]): g for p, g in zip(ref_pols, G_ref)}
    assert len(policies) == len(ref) == 27
    for pol, r, a in zip(policies, risk, ambiguity):
        assert -(r + a) == pytest.approx(ref[tuple(int(x) for x in pol)], abs=1e-5)


def test_efe_enumerates_all_policies_once():
    m = build_model(VIS, AIFParams(policy_len=4))
    policies, risk, ambiguity = expected_free_energy(random_belief(3), m, 4)
    assert policies.shape == (81, 4) and len({tuple(p) for p in policies}) == 81
    assert (risk >= -1e-9).all() and (ambiguity >= -1e-9).all()  # KL and entropy are non-negative


def test_joint_likelihood_is_a_distribution_over_joint_outcomes():
    m = build_model(VIS)
    J = joint_likelihood([np.asarray(a) for a in m.A])
    assert J.shape == (len(PedObs) * NV * len(Safety), NS) and np.allclose(J.sum(axis=0), 1.0)


def test_approaching_the_visible_zone_resolves_ambiguity_that_keeping_away_does_not():
    """The epistemic term: from outside the visible bands, only policies that reach them can tell
    'waiting' from 'none'; pymdp's state information gain is higher for them."""
    m = build_model(VIS, AIFParams(policy_len=4))
    qs0 = np.zeros(NS)
    for p, w in ((Ped.NONE, 0.5), (Ped.WAITING, 0.5)):
        qs0[state_index(ego_index(APPROACH, 3), p)] = w  # 8 m out, 6 m/s, unsure whether anyone is there
    qs = np.empty(1, dtype=object)
    qs[0] = qs0
    gain = {}
    for name, a in (("keep", Action.KEEP), ("brake", Action.BRAKE)):
        policy = np.full((4, 1), int(a))
        gain[name] = control.calc_states_info_gain(m.A, control.get_expected_states(qs, m.B, policy))
    assert gain["keep"] > gain["brake"]


# --- observing the world ---------------------------------------------------------------------

def parked(x=50.0, length=6.0, id=1):
    return Agent(id, AgentClass.VEHICLE, x, -2.75, 0.0, 0.0, 0.0, length, 2.0, True)


def ped(x, y, vx=0.0, vy=0.0, id=2):
    return Agent(id, AgentClass.PEDESTRIAN, x, y, 0.0, vx, vy, 0.6, 0.6, False)


def world(x_front=0.0, speed=10.0, agents=None, length=6.0, occluder_x=50.0):
    ego = EgoState(x_front - FRONT, 0.0, 0.0, speed, 4.7, 1.9, 2.9)
    agents = [parked(occluder_x, length)] if agents is None else agents
    corners = rect_corners(occluder_x, -2.75, 0.0, length, 2.0)
    regions = (OccludedRegion(1, shadow_polygon((x_front, 0.0), corners)),) if any(a.id == 1 for a in agents) else ()
    return WorldModel(1.0, ego, tuple(agents), regions, ())


def test_hazard_is_found_beside_the_route_and_none_without_an_occluder():
    w = world()
    h = find_hazard(w, FRAME, 0.0)
    assert h is not None and FRAME.pose_at(h.crossing_s)[0][0] == pytest.approx(50.0 + 3.0 + 1.0, abs=0.1)
    assert h.ped_xy == pytest.approx([54.0, -4.5], abs=0.05)  # 1 m past the far end, 0.75 m beyond the car
    assert find_hazard(world(agents=[]), FRAME, 0.0) is None
    assert find_hazard(world(x_front=80.0, occluder_x=50.0), FRAME, 80.0) is None  # already well past it


def test_visibility_profile_long_car_hides_until_alongside_and_short_car_shows_from_afar():
    long_car = visibility_profile(find_hazard(world(length=10.0), FRAME, 0.0), FRAME)
    assert list(long_car[:ZONE]) == [0.0] * ZONE and long_car[ZONE] == 1.0
    short_car = visibility_profile(find_hazard(world(length=4.5), FRAME, 0.0), FRAME)
    assert short_car[0] == 1.0 and short_car[ZONE] == 1.0 and 0.0 in short_car[1:ZONE]  # parallax


def test_pedestrian_observation_classes():
    h = find_hazard(world(), FRAME, 0.0)
    assert h is not None
    assert pedestrian_observation(world(), FRAME, h) is PedObs.NOT_SEEN
    waiting = world(agents=[parked(), ped(54.0, -4.5)])
    assert pedestrian_observation(waiting, FRAME, h) is PedObs.SEEN_WAITING
    walking = world(agents=[parked(), ped(54.0, -3.5, vy=1.4)])
    assert pedestrian_observation(walking, FRAME, h) is PedObs.SEEN_CROSSING
    in_lane = world(agents=[parked(), ped(54.0, -1.0)])
    assert pedestrian_observation(in_lane, FRAME, h) is PedObs.SEEN_CROSSING
    elsewhere = world(agents=[parked(), ped(150.0, -1.0)])
    assert pedestrian_observation(elsewhere, FRAME, h) is PedObs.NOT_SEEN


# --- the planner -----------------------------------------------------------------------------

def test_without_an_occluder_the_command_is_exactly_planner_v0s():
    w = world(agents=[])
    assert AIFPlanner().plan(w, ROUTE) == RuleBasedPlanner().plan(w, ROUTE)
    w = world(agents=[ped(30.0, 0.0)])  # a lead pedestrian in the lane, but no occluder
    assert AIFPlanner().plan(w, ROUTE) == RuleBasedPlanner().plan(w, ROUTE)


def test_command_never_exceeds_the_rule_based_limits():
    w = world(x_front=20.0, speed=12.0, agents=[parked(), ped(30.0, 0.0, id=5)])  # lead pedestrian close ahead
    cmd = AIFPlanner().plan(w, ROUTE)
    assert cmd.target_speed <= RuleBasedPlanner().plan(w, ROUTE).target_speed


def test_target_is_held_between_replans_and_replanned_after_dt():
    pl = AIFPlanner(record=True)
    w = world(x_front=25.0, speed=11.0)
    c1 = pl.plan(w, ROUTE)
    again = WorldModel(w.stamp + 0.2, w.ego, w.agents, w.occluded, w.traffic_lights)
    assert pl.plan(again, ROUTE).target_speed == c1.target_speed and len(pl.diagnostics) == 1
    later = WorldModel(w.stamp + DT, w.ego, w.agents, w.occluded, w.traffic_lights)
    pl.plan(later, ROUTE)
    assert len(pl.diagnostics) == 2


def test_diagnostics_record_the_efe_breakdown():
    pl = AIFPlanner(record=True)
    pl.plan(world(x_front=25.0, speed=11.0), ROUTE)
    (d,) = pl.diagnostics
    n = 3 ** pl.params.policy_len
    assert d.policies.shape == (n, pl.params.policy_len) and d.G.shape == d.risk.shape == (n,)
    assert np.allclose(d.G, -(d.risk + d.ambiguity)) and d.q_pi.sum() == pytest.approx(1.0)
    assert d.ped_belief.sum() == pytest.approx(1.0) and d.ped_obs is PedObs.NOT_SEEN
    assert d.distance == pytest.approx(54.0 - 25.0, abs=0.2)


def test_belief_remembers_a_pedestrian_after_it_goes_out_of_sight():
    pl = AIFPlanner(record=True)
    seen = world(x_front=5.0, speed=8.0, agents=[parked(), ped(54.0, -4.5)])
    pl.plan(seen, ROUTE)
    assert pl.diagnostics[-1].ped_obs is PedObs.SEEN_WAITING and pl.diagnostics[-1].ped_belief[Ped.WAITING] > 0.9
    hidden = world(x_front=9.0, speed=8.0)
    hidden = WorldModel(seen.stamp + DT, hidden.ego, hidden.agents, hidden.occluded, ())
    pl.plan(hidden, ROUTE)
    assert pl.diagnostics[-1].ped_obs is PedObs.NOT_SEEN
    assert pl.diagnostics[-1].ped_belief[Ped.WAITING] > 0.8  # still believed to be waiting


def test_a_crossing_pedestrian_makes_the_planner_act():
    pl = AIFPlanner(record=True)
    w = world(x_front=40.0, speed=11.0, agents=[parked(), ped(54.0, -2.6, vy=1.4)])  # 14 m from the crossing
    cmd = pl.plan(w, ROUTE)
    assert pl.diagnostics[-1].ped_belief[Ped.CROSSING] > 0.9
    assert cmd.target_speed < 11.0 and cmd.reason in (PlannerReason.OCCLUSION, PlannerReason.LEAD, PlannerReason.CONFLICT)


def test_new_hazard_resets_the_belief_and_leaving_it_clears_state():
    pl = AIFPlanner(record=True)
    pl.plan(world(x_front=25.0, speed=11.0), ROUTE)
    pl.plan(world(agents=[]), ROUTE)  # occluder gone
    assert pl._belief is None and pl._hazard_id is None  # pyright: ignore[reportPrivateUsage]


def test_runs_in_the_toy_sim_and_is_deterministic():
    from av_sim_toy import run_episode

    p = ScenarioParams(initial_speed=11.0, occluder_length=6.0, ped_x=54.0, ped_trigger_distance=22.0)
    a, b = run_episode(p, AIFPlanner()), run_episode(p, AIFPlanner())
    assert np.array_equal(a.ego, b.ego) and a.outcome == b.outcome
    sim = ToySim(p)
    assert sim.world_model().occluded  # the toy provides the occluded region the planner needs


def test_trace_tool_writes_a_plot(tmp_path):
    from tools import aif_trace

    out = tmp_path / "trace.png"
    aif_trace.main(["2", "--out", str(out)])
    assert out.exists() and out.stat().st_size > 10_000


def test_planner_is_registered_for_the_run_scripts():
    from tools.planners import PLANNERS, make_planner

    assert "aif" in PLANNERS and isinstance(make_planner("aif"), AIFPlanner)
