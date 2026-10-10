"""Planner registry for the run scripts. Add new planners here."""

from __future__ import annotations

from av_core.plan import AIFPlanner, RuleBasedPlanner, RuleBasedPlannerV1
from av_core.predict import ConstantVelocityPredictor
from av_core.protocols import Planner, Predictor
from av_sim_toy import Run, ScenarioParams, run_episode

PLANNERS = {"v0": RuleBasedPlanner, "v1": RuleBasedPlannerV1, "aif": AIFPlanner}

# Planners that consume predictions get this predictor; others get none.
PREDICTORS = {"v1": ConstantVelocityPredictor}


def make_predictor(name: str) -> Predictor | None:
    return PREDICTORS[name]() if name in PREDICTORS else None


def make_planner(name: str) -> Planner:
    try:
        return PLANNERS[name]()
    except KeyError:
        raise SystemExit(f"unknown planner {name!r}; choose from {sorted(PLANNERS)}") from None


def run_planners(params: ScenarioParams, names: list[str]) -> list[Run]:
    """Run each named planner on the same scenario, for the table and the overlay."""
    return [Run(n, run_episode(params, make_planner(n), predictor=make_predictor(n))) for n in names]


def expand(names: list[str]) -> list[str]:
    """``["all"]`` means every planner, in registry order; otherwise the names as given."""
    return list(PLANNERS) if "all" in names else names
