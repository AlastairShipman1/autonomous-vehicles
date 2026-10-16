"""State of a live loop session and its controls (no display needed): the window in ``view.py`` is a shell over this."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from av_core.protocols import Planner, Predictor
from av_sim_toy.loop.course import Course
from av_sim_toy.loop.sim import LoopRunner, LoopSim
from av_sim_toy.loop.track import Track
from av_sim_toy.pacing import Pacer

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)  # sim seconds per wall second
PlannerFactory = Callable[[str], tuple[Planner, Predictor | None]]


class LoopSession:
    def __init__(self, track: Track, course: Course, make: PlannerFactory, planner_names: Sequence[str], planner: str):
        if planner not in planner_names:
            raise ValueError(f"unknown planner {planner!r}; choose from {list(planner_names)}")
        self.track, self.course, self._make, self.names = track, course, make, list(planner_names)
        self.planner_name = planner
        self.paused = False
        self.overview = False
        self.sim_speed = 1.0
        self.pacer = Pacer(0.05, max_steps=10)  # up to 4x speed at 60 fps is 12 steps a frame; the cap drops the rest
        self._new_run()

    def _new_run(self) -> None:
        planner, predictor = self._make(self.planner_name)
        self.runner = LoopRunner(LoopSim(self.track, self.course), planner, predictor)
        self.pacer.reset()

    @property
    def sim(self) -> LoopSim:
        return self.runner.sim

    # --- controls -------------------------------------------------------------------------------------------

    def reset(self) -> None:
        self._new_run()

    def next_planner(self) -> None:
        self.planner_name = self.names[(self.names.index(self.planner_name) + 1) % len(self.names)]
        self._new_run()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_view(self) -> None:
        self.overview = not self.overview

    def faster(self) -> None:
        self._set_speed(min(SPEEDS.index(self.sim_speed) + 1, len(SPEEDS) - 1))

    def slower(self) -> None:
        self._set_speed(max(SPEEDS.index(self.sim_speed) - 1, 0))

    def _set_speed(self, i: int) -> None:
        self.sim_speed = SPEEDS[i]
        self.pacer.speed = self.sim_speed

    # --- stepping -------------------------------------------------------------------------------------------

    def step_once(self) -> None:
        self.runner.advance()

    def tick(self, wall_seconds: float) -> int:
        """Advance by however many whole steps the elapsed wall time allows; returns how many were taken."""
        if self.paused:
            return 0
        n = self.pacer.steps_for(wall_seconds)
        for _ in range(n):
            self.runner.advance()
        return n
