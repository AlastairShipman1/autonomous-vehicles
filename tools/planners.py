"""Planner registry for the run scripts. Add new planners here."""

from __future__ import annotations

from av_core.plan import RuleBasedPlanner

PLANNERS = {"v0": RuleBasedPlanner}


def make_planner(name: str):
    try:
        return PLANNERS[name]()
    except KeyError:
        raise SystemExit(f"unknown planner {name!r}; choose from {sorted(PLANNERS)}") from None
