"""Planner registry for the run scripts. Add new planners here."""

from __future__ import annotations

from av_core.plan import RuleBasedPlanner, RuleBasedPlannerV1
from av_core.predict import ConstantVelocityPredictor

PLANNERS = {"v0": RuleBasedPlanner, "v1": RuleBasedPlannerV1}

# Planners that consume predictions get this predictor; others get none.
PREDICTORS = {"v1": ConstantVelocityPredictor}


def make_predictor(name: str):
    return PREDICTORS[name]() if name in PREDICTORS else None


def make_planner(name: str):
    try:
        return PLANNERS[name]()
    except KeyError:
        raise SystemExit(f"unknown planner {name!r}; choose from {sorted(PLANNERS)}") from None
