"""State of one live session: the episode in progress, the inputs so far, and the controls (no display needed).

The window in ``play.py`` is a thin shell over this, so everything that matters is testable without a screen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from av_core.protocols import Planner, Predictor
from av_sim_toy import scenario as sc
from av_sim_toy.manual import Direction, ManualPedestrian
from av_sim_toy.pacing import Pacer
from av_sim_toy.recording import Recording, build_params, default_start
from av_sim_toy.sim import EpisodeRunner, ToySim

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)  # sim seconds per wall second
PlannerFactory = Callable[[str], tuple[Planner, Predictor | None]]


@dataclass(frozen=True)
class SessionConfig:
    scenario: str | None = "single_car"
    seed: int | None = None
    speed: float | None = None  # ego's starting speed, m/s; None keeps the scenario's own
    keep_scripted: bool = False
    start: tuple[float, float] | None = None  # where the pedestrian starts; None is the sidewalk past the last vehicle
    out_dir: Path | None = Path("out/play")  # where recordings are written; None writes nothing


class LiveSession:
    def __init__(self, config: SessionConfig, make: PlannerFactory, planner_names: Sequence[str], planner: str):
        if planner not in planner_names:
            raise ValueError(f"unknown planner {planner!r}; choose from {list(planner_names)}")
        self.cfg, self._make, self.names = config, make, list(planner_names)
        self.planner_name = planner
        self.params = build_params(config.scenario, config.seed, config.speed, config.keep_scripted)
        self.start = config.start or default_start(self.params)
        self.pacer = Pacer(sc.DT)
        self.paused = True
        self.saved_path: Path | None = None
        self.sim_speed = 1.0
        self._new_episode()

    # --- episode lifecycle ------------------------------------------------------------------

    def _new_episode(self) -> None:
        planner, predictor = self._make(self.planner_name)
        self.runner = EpisodeRunner(self.params, planner, predictor=predictor, manual_ped=ManualPedestrian(*self.start))
        self.inputs: list[Direction] = []
        self.paused = True
        self._saved = False
        self.pacer.reset()

    @property
    def sim(self) -> ToySim:
        return self.runner.sim

    @property
    def done(self) -> bool:
        return self.runner.done

    def recording(self) -> Recording:
        outcome = self.sim.outcome or "unfinished"
        return Recording(planner=self.planner_name, start=self.start, inputs=list(self.inputs), outcome=outcome,
                         scenario=self.cfg.scenario, seed=self.cfg.seed, speed=self.cfg.speed,
                         keep_scripted=self.cfg.keep_scripted, dt=sc.DT,
                         created=datetime.now().isoformat(timespec="seconds"))

    def save(self) -> Path | None:
        """Write the recording once per episode, if there is a directory and anything happened."""
        if self._saved or self.cfg.out_dir is None or not self.inputs:
            return None
        rec = self.recording()
        label = rec.scenario if rec.scenario else f"seed{rec.seed}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saved_path = rec.save(self.cfg.out_dir / f"{label}__{rec.planner}__{stamp}.json")
        self._saved = True
        return self.saved_path

    def reset(self) -> None:
        self.save()
        self._new_episode()

    def next_planner(self) -> None:
        """Save this run and restart the same scenario under the next planner."""
        self.save()
        self.planner_name = self.names[(self.names.index(self.planner_name) + 1) % len(self.names)]
        self._new_episode()

    def finish(self) -> None:
        self.save()

    # --- controls ---------------------------------------------------------------------------

    def toggle_pause(self) -> None:
        if not self.done:
            self.paused = not self.paused

    def faster(self) -> None:
        self._set_speed(min(SPEEDS.index(self.sim_speed) + 1, len(SPEEDS) - 1))

    def slower(self) -> None:
        self._set_speed(max(SPEEDS.index(self.sim_speed) - 1, 0))

    def _set_speed(self, i: int) -> None:
        self.sim_speed = SPEEDS[i]
        self.pacer.speed = self.sim_speed

    # --- stepping ---------------------------------------------------------------------------

    def step_once(self, wanted: Direction) -> None:
        if self.done:
            return
        self.runner.advance(wanted)
        self.inputs.append(wanted)
        if self.done:
            self.save()

    def tick(self, wall_seconds: float, wanted: Direction) -> int:
        """Advance by however many whole steps the elapsed wall time allows; returns how many were taken."""
        if self.paused or self.done:
            return 0
        n = self.pacer.steps_for(wall_seconds)
        for taken in range(n):
            self.step_once(wanted)
            if self.done:
                return taken + 1
        return n
