"""Structural interfaces (the Python analogue of a TypeScript ``interface``).

Anything with a matching method is accepted; no inheritance needed.
"""

from __future__ import annotations

from typing import Protocol

from av_core.types import PlannerCommand, PredictedTrajectory, Route, WorldModel


class Planner(Protocol):
    def plan(self, world: WorldModel, route: Route,
             predictions: tuple[PredictedTrajectory, ...] = ()) -> PlannerCommand: ...


class Predictor(Protocol):
    def __call__(self, world: WorldModel) -> tuple[PredictedTrajectory, ...]: ...


__all__ = ["Planner", "Predictor"]
