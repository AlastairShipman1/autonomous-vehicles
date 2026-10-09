import numpy as np
import pytest

from av_core.plan import RuleBasedPlanner
from av_sim_toy import REPORTING_SEEDS, TUNING_SEEDS, ScenarioParams, run_episode, sample_scenario, split_of
from av_sim_toy.sampler import OCCLUDER_LENGTHS

ALL = [sample_scenario(s) for s in list(TUNING_SEEDS) + list(REPORTING_SEEDS)]


def test_seed_ranges():
    assert (TUNING_SEEDS[0], TUNING_SEEDS[-1]) == (0, 999)
    assert (REPORTING_SEEDS[0], REPORTING_SEEDS[-1]) == (10000, 10999)
    assert [split_of(s) for s in (0, 999, 1000, 10000, 10999, 11000)] == [
        "tuning", "tuning", "other", "reporting", "reporting", "other"]


def test_reproducible_from_seed_alone():
    assert sample_scenario(123) == sample_scenario(123)
    assert sample_scenario(123) != sample_scenario(124)
    assert sample_scenario(np.int64(123)) == sample_scenario(123)
    assert sample_scenario(5).seed == 5


def test_reproducible_across_fresh_interpreter_state():
    # no global RNG involved: reseeding numpy's legacy global state changes nothing
    a = sample_scenario(7)
    np.random.seed(999)
    np.random.random(10)
    assert sample_scenario(7) == a


def test_pedestrian_path_never_crosses_the_occluder():
    from av_sim_toy import ScenarioParams

    with pytest.raises(ValueError, match="through the occluder"):
        ScenarioParams(occluder_x=50.0, occluder_length=6.0, ped_x=52.0)
    ScenarioParams(occluder_x=50.0, occluder_length=6.0, ped_x=53.4)  # clears the car and the 0.3 m radius
    ScenarioParams(occluder_x=50.0, occluder_length=6.0, ped_x=52.0, ped_present=False)  # no pedestrian, no path


def test_pinned_values_for_seed_0():
    # guards against accidental changes to draw order or distributions
    p = sample_scenario(0)
    assert p.initial_speed == pytest.approx(11.184808436607272, abs=1e-9)
    assert p.occluder_length == 4.5 and p.ped_present
    assert p.ped_trigger_distance == pytest.approx(25.165894394179496, abs=1e-9)


def test_ranges_match_spec():
    for p in ALL:
        assert 8.0 <= p.initial_speed <= 13.0
        assert 40.0 <= p.occluder_x <= 60.0
        assert p.occluder_length in OCCLUDER_LENGTHS
        assert 0.8 <= p.ped_speed <= 2.0
        assert 10.0 <= p.ped_trigger_distance <= 35.0
        far_end = p.occluder_x + 0.5 * p.occluder_length
        assert far_end + 0.5 <= p.ped_x <= far_end + 1.5  # steps out past the far end, never through the car


def test_distributions_are_sensible():
    present = np.mean([p.ped_present for p in ALL])
    assert 0.46 < present < 0.54
    lengths = [p.occluder_length for p in ALL]
    for L in OCCLUDER_LENGTHS:
        assert 0.28 < lengths.count(L) / len(lengths) < 0.39


def test_tuning_and_reporting_sets_differ():
    tuning = {p.initial_speed for p in ALL[:1000]}
    reporting = {p.initial_speed for p in ALL[1000:]}
    assert not tuning & reporting


def test_bad_seeds_rejected():
    for bad in (-1, 1.5, "0", None, True):
        with pytest.raises(ValueError):
            sample_scenario(bad)


def test_same_seed_gives_bit_identical_trajectory():
    planner = RuleBasedPlanner()
    for seed in (0, 1, 10000):
        a = run_episode(sample_scenario(seed), planner)
        b = run_episode(sample_scenario(seed), planner)
        assert np.array_equal(a.ego, b.ego) and np.array_equal(a.ped, b.ped, equal_nan=True)
        assert a.outcome == b.outcome


def test_params_type():
    assert isinstance(sample_scenario(3), ScenarioParams)
