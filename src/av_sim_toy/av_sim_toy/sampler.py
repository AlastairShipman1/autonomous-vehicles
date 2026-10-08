"""Scenario sampler: every parameter is a pure function of the seed.

Tuning seeds are 0-999 (tune on these only); reporting seeds are 10000-10999 (report on these).
Every parameter is drawn on every call, in a fixed order, so changing one option never shifts the
others for a given seed.
"""

from __future__ import annotations

import numpy as np

from av_sim_toy.scenario import ScenarioParams

TUNING_SEEDS = range(0, 1000)
REPORTING_SEEDS = range(10_000, 11_000)

OCCLUDER_LENGTHS = (4.5, 6.0, 10.0)
PED_PRESENT_PROB = 0.5
PED_X_MARGIN = 1.0  # m beyond each end of the occluder, per the spec ("occluder span + 1 m")


def sample_scenario(seed: int) -> ScenarioParams:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    rng = np.random.default_rng(int(seed))
    initial_speed = rng.uniform(8.0, 13.0)
    occluder_x = rng.uniform(40.0, 60.0)
    occluder_length = float(rng.choice(OCCLUDER_LENGTHS))
    ped_present = bool(rng.random() < PED_PRESENT_PROB)
    ped_x_u = rng.random()
    ped_speed = rng.uniform(0.8, 2.0)
    trigger = rng.uniform(10.0, 35.0)
    half = 0.5 * occluder_length + PED_X_MARGIN
    return ScenarioParams(
        initial_speed=float(initial_speed),
        occluder_x=float(occluder_x),
        occluder_length=occluder_length,
        ped_present=ped_present,
        ped_x=float(occluder_x - half + ped_x_u * 2 * half),
        ped_speed=float(ped_speed),
        ped_trigger_distance=float(trigger),
        seed=int(seed),
    )


def split_of(seed: int) -> str:
    if seed in TUNING_SEEDS:
        return "tuning"
    if seed in REPORTING_SEEDS:
        return "reporting"
    return "other"
