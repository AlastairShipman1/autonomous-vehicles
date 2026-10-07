"""Planner v0: target speed is the minimum of route, lead-vehicle and traffic-light limits.

Constants are starting guesses to tune. ``reason`` names the limit that bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from av_core.geometry import RouteFrame
from av_core.types import PlannerCommand, PredictedTrajectory, Route, WorldModel


def stopping_speed(gap: float, decel: float, standoff: float) -> float:
    """Highest speed from which ``decel`` stops ``standoff`` short of an obstacle ``gap`` ahead."""
    return math.sqrt(2.0 * decel * max(0.0, gap - standoff))


@dataclass
class RuleBasedPlanner:
    a_lat: float = 2.0  # m/s^2, lateral acceleration limit in curves
    curve_horizon: float = 30.0  # m
    b: float = 3.0  # m/s^2, comfortable braking
    lead_standoff: float = 6.0  # m
    light_standoff: float = 2.0  # m
    lane_margin: float = 0.3  # m, added to the half-widths for the "in my path" test
    _frame_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def frame_for(self, route: Route) -> RouteFrame:
        """RouteFrame for ``route``, reused while the same Route object is passed in (one per episode)."""
        if self._frame_cache is None or self._frame_cache[0] is not route:
            self._frame_cache = (route, RouteFrame(route))
        return self._frame_cache[1]

    def plan(self, world: WorldModel, route: Route,
             predictions: tuple[PredictedTrajectory, ...] = ()) -> PlannerCommand:
        """``predictions`` is ignored here; it keeps the signature shared with planner v1."""
        frame = self.frame_for(route)
        limits = self.limits(world, route, frame, predictions)
        reason = min(limits, key=limits.get)  # dict order breaks ties: safety limits first
        return PlannerCommand(world.stamp, limits[reason], reason)

    def limits(self, world: WorldModel, route: Route, frame: RouteFrame | None = None,
               predictions: tuple[PredictedTrajectory, ...] = ()) -> dict[str, float]:
        frame = frame or self.frame_for(route)
        ego = world.ego
        s0, _ = frame.project(ego.x, ego.y)
        # Gaps are measured from the front bumper, which is (length + wheelbase) / 2 ahead of the rear axle.
        s_front = s0 + 0.5 * (ego.length + ego.wheelbase)
        return {
            "lead": self._lead_limit(world, frame, s_front),
            "light": self._light_limit(world, frame, s_front, ego.speed),
            "route": self._route_limit(route, frame, s0),
        }

    def _route_limit(self, route: Route, frame: RouteFrame, s0: float) -> float:
        window = (frame.s >= s0) & (frame.s <= s0 + self.curve_horizon)
        kappa = np.abs(frame.curvature[window])
        kappa = kappa[kappa > 1e-9]
        v_curve = math.sqrt(self.a_lat / kappa.max()) if kappa.size else math.inf
        return min(route.speed_limit, v_curve)

    def _lead_limit(self, world: WorldModel, frame: RouteFrame, s_front: float) -> float:
        ego, gap = world.ego, math.inf
        for a in world.agents:
            s_a, lateral = frame.project(a.x, a.y)
            if abs(lateral) >= 0.5 * ego.width + 0.5 * a.width + self.lane_margin:
                continue
            gap_a = (s_a - 0.5 * a.length) - s_front  # to the agent's rear edge
            if s_a > s_front - 0.5 * ego.length and gap_a >= -0.5 * a.length:
                gap = min(gap, max(gap_a, 0.0))
        return stopping_speed(gap, self.b, self.lead_standoff) if math.isfinite(gap) else math.inf

    def _light_limit(self, world: WorldModel, frame: RouteFrame, s_front: float, speed: float) -> float:
        if world.light not in ("red", "yellow") or world.stop_line is None:
            return math.inf
        gap = frame.project(*world.stop_line)[0] - s_front
        if gap < 0.0 or speed**2 / (2.0 * self.b) > gap:  # already past it, or can no longer stop at b
            return math.inf
        return stopping_speed(gap, self.b, self.light_standoff)
