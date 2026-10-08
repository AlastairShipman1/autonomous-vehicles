"""Adapter from the toy sim to the sweep harness (``Scenario`` / ``EpisodeRecord``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from av_core.protocols import Planner, Predictor
from av_core.sweep import EpisodeRecord
from av_sim_toy import scenario as sc
from av_sim_toy.sampler import sample_scenario, split_of
from av_sim_toy.sim import Episode, run_episode


def to_record(ep: Episode) -> EpisodeRecord:
    p = ep.params
    half = 0.5 * p.occluder_length
    params = {k: v for k, v in asdict(p).items() if k != "seed"}
    params["split"] = split_of(p.seed) if p.seed is not None else ""
    return EpisodeRecord(
        seed=-1 if p.seed is None else p.seed,
        dt=ep.dt,
        t=np.append(ep.t, ep.t[-1] + ep.dt),
        ego=np.vstack([ep.ego, ep.final_ego]),
        ego_length=sc.EGO_LENGTH, ego_width=sc.EGO_WIDTH, ego_wheelbase=sc.EGO_WHEELBASE,
        ped=np.vstack([ep.ped, ep.final_ped]),
        ped_radius=sc.PED_RADIUS,
        occluder_near_x=p.occluder_x - half,
        occluder_far_x=p.occluder_x + half,
        outcome=ep.outcome,
        params=params,
    )


@dataclass(frozen=True)
class ToyScenario:
    seed: int

    def run(self, planner: Planner, predictor: Predictor | None = None) -> EpisodeRecord:
        return to_record(run_episode(sample_scenario(self.seed), planner, predictor=predictor))


def toy_scenario_factory(seed: int) -> ToyScenario:
    return ToyScenario(seed)
