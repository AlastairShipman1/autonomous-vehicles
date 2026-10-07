"""Planner v1: v0 plus a predicted-conflict stop and an occlusion speed cap.

This is the baseline the active-inference planner is compared against, so it should be sensible,
not a strawman. Constants are starting points; tune ``v_floor`` and ``b_occ`` on tuning seeds only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from av_core.geometry import RouteFrame, distance_points_to_convex
from av_core.plan.rule_based import RuleBasedPlanner, stopping_speed
from av_core.types import PredictedTrajectory, Route, WorldModel


@dataclass
class RuleBasedPlannerV1(RuleBasedPlanner):
    corridor_margin: float = 0.5  # m, added to half the ego width
    conflict_window: float = 2.0  # s, |t_ego - t_agent| below this is a conflict
    occlusion_range: float = 2.5  # m, route points this close to an occluded region count
    v_floor: float = 4.0  # m/s
    b_occ: float = 4.0  # m/s^2

    def limits(self, world: WorldModel, route: Route, frame: RouteFrame | None = None,
               predictions: tuple[PredictedTrajectory, ...] = ()) -> dict[str, float]:
        frame = frame or self.frame_for(route)
        base = super().limits(world, route, frame, predictions)
        ego = world.ego
        s0, _ = frame.project(ego.x, ego.y)
        s_front = s0 + 0.5 * (ego.length + ego.wheelbase)
        # key order is the tie-break order: safety limits before the route limit
        return {
            "lead": base["lead"],
            "light": base["light"],
            "conflict": self._conflict_limit(world, frame, s_front, predictions),
            "occlusion": self._occlusion_limit(world, frame, s0),
            "route": base["route"],
        }

    def _conflict_limit(self, world: WorldModel, frame: RouteFrame, s_front: float,
                        predictions: tuple[PredictedTrajectory, ...]) -> float:
        ego = world.ego
        half_corridor = 0.5 * ego.width + self.corridor_margin
        v_ego = max(ego.speed, 0.1)
        gap = math.inf
        for traj in predictions:
            s_all, lat_all = frame.project_many(traj.xy)
            for t_c, s_c, lateral in zip(traj.t, s_all, lat_all):
                if abs(lateral) > half_corridor:
                    continue
                if s_c >= s_front:  # first entry into the corridor, if it is ahead of the bumper
                    t_e = (s_c - s_front) / v_ego  # ego's front bumper reaches s_c
                    if abs(t_e - t_c) < self.conflict_window:
                        gap = min(gap, s_c - s_front)
                break  # only the first corridor entry counts
        return stopping_speed(gap, self.b, self.lead_standoff) if math.isfinite(gap) else math.inf

    def _occlusion_limit(self, world: WorldModel, frame: RouteFrame, s0: float) -> float:
        ahead = frame.s >= s0
        pts, s = frame.points[ahead], frame.s[ahead]
        s_occ = math.inf
        for region in world.occluded:
            near = distance_points_to_convex(region.polygon, pts) <= self.occlusion_range
            if near.any():
                s_occ = min(s_occ, float(s[np.argmax(near)]))
        if not math.isfinite(s_occ):
            return math.inf
        return max(self.v_floor, math.sqrt(2.0 * self.b_occ * max(0.0, s_occ - s0 - self.lead_standoff)))
