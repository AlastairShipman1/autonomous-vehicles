import math

import numpy as np
import pytest

from av_core.sweep import (
    EpisodeRecord,
    bootstrap_mean,
    compute_metrics,
    format_summary,
    read_csv,
    run_sweep,
    summarize,
    wilson_interval,
    write_csv,
)
from av_core.sweep import metrics as M
from av_sim_toy.sweep import toy_scenario_factory
from tools.planners import PLANNERS, PREDICTORS

DT = 0.05
L, W, WB = 4.7, 1.9, 2.9


def record(speeds, ped=None, x0=0.0, y=0.0, near=40.0, far=46.0, outcome="finished", ped_radius=0.3, dt=DT):
    """Straight-line ego driven by a speed profile; ``ped`` is an (N, 2) array or None."""
    v = np.asarray(speeds, dtype=float)
    x = x0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * dt)])
    ego = np.column_stack([x, np.full(len(v), y), np.zeros(len(v)), v])
    pedarr = np.full((len(v), 2), np.nan) if ped is None else np.asarray(ped, dtype=float)
    return EpisodeRecord(seed=1, dt=dt, t=np.arange(len(v)) * dt, ego=ego, ego_length=L, ego_width=W,
                         ego_wheelbase=WB, ped=pedarr, ped_radius=ped_radius, occluder_near_x=near,
                         occluder_far_x=far, outcome=outcome)


def static_ped(n, x, y):
    return np.tile([x, y], (n, 1))


# --- stats ----------------------------------------------------------------------------------

def test_wilson_known_values():
    p, lo, hi = wilson_interval(5, 10)
    assert (p, lo, hi) == pytest.approx((0.5, 0.2366, 0.7634), abs=1e-4)
    p, lo, hi = wilson_interval(0, 20)
    assert (p, lo, hi) == pytest.approx((0.0, 0.0, 0.1611), abs=1e-4)
    assert wilson_interval(20, 20)[1:] == pytest.approx((0.8389, 1.0), abs=1e-4)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_bootstrap_mean_properties():
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 2.0, 400)
    m, lo, hi = bootstrap_mean(data)
    assert m == pytest.approx(data.mean()) and lo < m < hi
    assert hi - lo == pytest.approx(2 * 1.96 * 2.0 / math.sqrt(400), rel=0.2)
    assert bootstrap_mean(data) == (m, lo, hi)  # deterministic
    assert bootstrap_mean([3.0] * 10) == (3.0, 3.0, 3.0)
    assert bootstrap_mean([1.0, math.nan, math.inf, 3.0])[0] == 2.0  # non-finite values are dropped
    assert all(math.isnan(x) for x in bootstrap_mean([math.nan]))


# --- metrics --------------------------------------------------------------------------------

def test_no_pedestrian_gives_none_for_pedestrian_metrics():
    m = compute_metrics(record([10.0] * 20))
    assert m.collision is None and m.min_distance is None and m.min_ttc is None
    assert m.needless_stop is False and not m.ped_present


def test_collision_and_min_distance():
    # ego rectangle centre starts at x = 1.45, 10 m/s: x in [~-0.9, 3.8] after 0 steps
    n = 60
    near = record([10.0] * n, static_ped(n, 30.0, 0.95 + 0.2))  # 0.2 m beside the ego's side on passing
    far = record([10.0] * n, static_ped(n, 30.0, 3.0))
    assert M.collision(far) is False
    assert M.min_distance(far) == pytest.approx(3.0 - 0.95, abs=0.02)
    assert M.collision(near) is True  # 0.2 < radius 0.3
    assert M.min_distance(near) == pytest.approx(0.2, abs=0.02)


def test_min_distance_zero_when_ped_inside_rectangle():
    n = 5
    rec = record([0.0] * n, static_ped(n, 1.45, 0.0))
    assert M.min_distance(rec) == 0.0 and M.collision(rec) is True


def test_ttc_head_on_with_stationary_pedestrian():
    # ego front at ~3.8 m, ped stationary in lane 20 m ahead of the front bumper, ego at 10 m/s
    n = 3
    rec = record([10.0] * n, static_ped(n, 3.8 + 20.0 + 0.3, 0.0))
    # the front bumper is within r = 0.3 of the ped centre after 20 m / 10 m/s = 2 s; the minimum over the
    # episode comes from the last row, 0.1 s later, when 1 m has already been covered
    assert M.min_ttc(rec) == pytest.approx(1.9, abs=0.02)


def test_ttc_infinite_when_paths_never_meet_and_zero_when_overlapping():
    n = 4
    assert M.min_ttc(record([10.0] * n, static_ped(n, 30.0, 4.5))) == math.inf
    assert M.min_ttc(record([0.0] * n, static_ped(n, 1.45, 0.0))) == 0.0
    # stationary ego, pedestrian 0.2 m past the front bumper walking toward the lane at 1 m/s from y = -2.5:
    # contact once hypot(0.2, y + 0.95) <= 0.3, i.e. at y = -1.17 (1.33 s); the last row is 0.1 s later
    t = np.arange(3) * DT
    walker = np.column_stack([np.full(3, 4.0), -2.5 + 1.0 * t])
    assert M.min_ttc(record([0.0] * 3, walker)) == pytest.approx(1.23, abs=0.02)


def test_ttc_takes_minimum_over_the_episode():
    n = 40
    t = np.arange(n) * DT
    walker = np.column_stack([np.full(n, 30.0), -4.5 + 1.5 * t])
    rec = record([12.0] * n, walker)
    assert M.min_ttc(rec) <= M.min_ttc(record([12.0] * 2, walker[:2]))


def test_braking_onset_distance_and_sign():
    # cruise at 10 m/s, then decelerate at 2 m/s^2 from step 40
    v = [10.0] * 40 + [10.0 - 2.0 * DT * k for k in range(1, 60)]
    rec = record(v, near=60.0)
    onset = M.braking_onset(rec)
    front = rec.ego[39, 0] + 0.5 * (L + WB)  # decel over rows 39 -> 40 is the first above 1 m/s^2
    assert onset == pytest.approx(60.0 - front, abs=0.01)
    assert math.isnan(M.braking_onset(record([10.0] * 50)))  # never brakes hard
    # braking only after the front passed the near end -> negative
    late = record([10.0] * 100 + [10.0 - 3.0 * DT * k for k in range(1, 30)], near=10.0)
    assert M.braking_onset(late) < 0


def test_needless_stop():
    assert M.needless_stop(record([10.0, 5.0, 0.5, 5.0])) is True
    assert M.needless_stop(record([10.0, 5.0, 1.5, 5.0])) is False


def test_time_penalty_zero_at_constant_speed_and_positive_when_slow():
    v0, far = 10.0, 46.0
    target = far + 30.0
    # rear starts at x0 - overhang; run long enough to reach the target
    n = int((target / v0 + 5) / DT)
    steady = record([v0] * n, far=far)
    assert M.time_penalty(steady) == pytest.approx(0.0, abs=0.02)
    slow = record([v0] * 20 + [5.0] * (3 * n), far=far)
    assert M.time_penalty(slow) > 1.0
    assert math.isnan(M.time_penalty(record([v0] * 10, far=far)))  # never reaches the finish
    # a faster-than-initial ego has a negative penalty
    fast = record([v0] + [14.0] * n, far=far)
    assert M.time_penalty(fast) < 0


# --- harness --------------------------------------------------------------------------------

SEEDS = range(0, 12)


def run(planner="v0", workers=1, seeds=SEEDS):
    return run_sweep(seeds, toy_scenario_factory, PLANNERS[planner], PREDICTORS.get(planner), workers=workers)


def test_sweep_is_deterministic_and_seed_ordered():
    a, b = run(), run()
    assert [repr(m) for m, _ in a] == [repr(m) for m, _ in b]  # repr: nan != nan under ==
    assert [m.seed for m, _ in a] == list(SEEDS)


def test_workers_give_identical_results():
    serial = run(seeds=range(8))
    parallel = run(workers=2, seeds=range(8))
    assert [repr(m) for m, _ in serial] == [repr(m) for m, _ in parallel]


def test_metrics_agree_with_sim_outcomes():
    for m, _ in run():
        if m.ped_present:
            assert m.collision == (m.outcome == "collision")
        else:
            assert m.outcome == "finished" and m.collision is None


def test_csv_one_row_per_episode_and_round_trips(tmp_path):
    rows = run()
    path = write_csv(rows, tmp_path / "out" / "episodes.csv")
    back = read_csv(path)
    assert len(back) == len(rows)
    for (m, params), r in zip(rows, back):
        assert int(r["seed"]) == m.seed and r["outcome"] == m.outcome
        assert float(r["duration"]) == m.duration
        assert float(r["param_initial_speed"]) == params["initial_speed"]
        assert r["param_split"] == "tuning"
        assert (r["collision"] == "") == (m.collision is None)
        assert float(r["time_penalty"]) == pytest.approx(m.time_penalty, nan_ok=True)


def test_summary_splits_and_intervals():
    rows = run()
    s = summarize(rows)
    present, absent = s["pedestrian present"], s["pedestrian absent"]
    assert present["n"] + absent["n"] == len(rows)
    rate, lo, hi = present["collision_rate"]
    n_coll = sum(1 for m, _ in rows if m.collision)
    assert rate == pytest.approx(n_coll / present["n"]) and lo <= rate <= hi
    assert "collision_rate" not in absent and "needless_stop_rate" in absent
    text = format_summary(s)
    assert "pedestrian present" in text and "95% CI" in text


@pytest.mark.slow
def test_200_episodes_in_under_5_minutes():
    import time

    from av_sim_toy import REPORTING_SEEDS

    t0 = time.time()
    rows = run("v1", seeds=REPORTING_SEEDS[:200])
    assert len(rows) == 200
    assert time.time() - t0 < 300
