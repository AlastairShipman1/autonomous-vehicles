"""Save a live session as a small JSON file and replay it against any planner.

A recording is the scenario, where the pedestrian started, and the key direction held on every step. The sim and the
planner are deterministic, so the same recording replays to the identical trajectory, and a different planner meets
exactly the same pedestrian. A person reacting live to each planner would not give a fair comparison; this does.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from av_core.control import Controller
from av_core.protocols import Planner, Predictor
from av_sim_toy import scenario as sc
from av_sim_toy.manual import Direction, ManualPedestrian
from av_sim_toy.sampler import sample_scenario
from av_sim_toy.scenarios import SCENARIOS
from av_sim_toy.scenario import ScenarioParams
from av_sim_toy.sim import Episode, run_episode

VERSION = 1


def build_params(scenario: str | None = None, seed: int | None = None, speed: float | None = None,
                 keep_scripted: bool = False) -> ScenarioParams:
    """The scenario a live session runs on: a named one or a sampled seed, the scripted pedestrians removed by default
    (so the only pedestrian is the one you drive), and optionally a different starting speed for the ego."""
    if (scenario is None) == (seed is None):
        raise ValueError("give exactly one of a scenario name or a seed")
    if scenario is not None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        params = SCENARIOS[scenario].params
    else:
        assert seed is not None
        params = sample_scenario(seed)
    if not keep_scripted:
        params = replace(params, ped_present=False, extra_pedestrians=())
    if speed is not None:
        params = replace(params, initial_speed=speed)
    return params


def default_start(params: ScenarioParams) -> tuple[float, float]:
    """On the right-hand sidewalk, a metre past the last parked vehicle."""
    return params.far_x + 1.0, sc.PED_START_Y


def _encode(inputs: Sequence[Direction]) -> list[list[int]]:
    """Run-length encoding: [[dx, dy, count], ...]."""
    out: list[list[int]] = []
    for dx, dy in inputs:
        if out and out[-1][0] == dx and out[-1][1] == dy:
            out[-1][2] += 1
        else:
            out.append([dx, dy, 1])
    return out


def _decode(runs: Sequence[Sequence[int]]) -> list[Direction]:
    return [(int(dx), int(dy)) for dx, dy, n in runs for _ in range(int(n))]


@dataclass
class Recording:
    planner: str
    start: tuple[float, float]
    inputs: list[Direction]
    outcome: str = "unfinished"
    scenario: str | None = None
    seed: int | None = None
    speed: float | None = None
    keep_scripted: bool = False
    dt: float = sc.DT
    created: str = ""

    def params(self) -> ScenarioParams:
        return build_params(self.scenario, self.seed, self.speed, self.keep_scripted)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": VERSION, "scenario": self.scenario, "seed": self.seed, "speed": self.speed,
            "keep_scripted": self.keep_scripted, "planner": self.planner, "dt": self.dt, "start": list(self.start),
            "outcome": self.outcome, "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "steps": len(self.inputs), "inputs": _encode(self.inputs),
        }
        path.write_text(json.dumps(doc, indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        doc = json.loads(Path(path).read_text())
        if doc.get("version") != VERSION:
            raise ValueError(f"{path}: unsupported recording version {doc.get('version')!r}")
        return cls(planner=doc["planner"], start=(float(doc["start"][0]), float(doc["start"][1])),
                   inputs=_decode(doc["inputs"]), outcome=doc["outcome"], scenario=doc["scenario"], seed=doc["seed"],
                   speed=doc["speed"], keep_scripted=doc["keep_scripted"], dt=float(doc["dt"]), created=doc["created"])


def replay(rec: Recording, planner: Planner, predictor: Predictor | None = None,
           controller: Controller | None = None) -> Episode:
    """Run the recorded pedestrian against ``planner`` (not necessarily the one that was live)."""
    return run_episode(rec.params(), planner, controller, predictor, rec.dt,
                       manual_ped=ManualPedestrian(*rec.start), ped_inputs=rec.inputs)
