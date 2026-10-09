"""From a ``WorldModel`` to the discrete quantities the generative model works in.

The model is specific to one situation: a parked vehicle beside the route with a pedestrian who may step out
just past its far end. Two assumptions, both true in the toy and to be revisited for CARLA: the parked vehicle
is parallel to the route, and a hidden pedestrian would be waiting about 1 m past the far end and 0.75 m beyond
the vehicle's outer side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from av_core.geometry import SENSOR_RANGE, RouteFrame, is_visible, rect_corners
from av_core.plan.aif.model import DIST_CENTRES, ND, PedObs
from av_core.types import Agent, AgentClass, WorldModel

GAP_PAST_FAR_END = 1.0  # m, where a hidden pedestrian is assumed to wait, along the route
OUTSIDE_OFFSET = 0.75  # m beyond the occluder's outer side
NEAR_HAZARD = 15.0  # m along the route: pedestrians farther than this from the crossing point are not ours
CROSSING_SPEED = 0.3  # m/s toward the route that makes a pedestrian "crossing"
IN_LANE = 1.75  # m from the route centreline: inside the lane counts as crossing


@dataclass(frozen=True)
class Hazard:
    """The occluder and the crossing point it hides, in route coordinates."""

    occluder: Agent
    crossing_s: float  # arc length of the hypothesised pedestrian position (the crossing point)
    ped_xy: NDArray[np.float64]  # world position of a pedestrian waiting there
    corners: NDArray[np.float64]  # occluder rectangle


def find_hazard(world: WorldModel, frame: RouteFrame, s_front: float) -> Hazard | None:
    """The nearest occluder ahead of the ego that has an occluded region, or ``None``."""
    by_id = {a.id: a for a in world.agents}
    best: Hazard | None = None
    for region in world.occluded:
        a = by_id.get(region.occluder_id)
        if a is None:
            continue
        s_c, lat_c = frame.project(a.x, a.y)
        crossing_s = s_c + 0.5 * a.length + GAP_PAST_FAR_END
        if crossing_s < s_front - 2.0:  # already well past it
            continue
        side = -1.0 if lat_c < 0 else 1.0
        xy, _ = frame.pose_at(crossing_s, lat_c + side * (0.5 * a.width + OUTSIDE_OFFSET))
        corners = rect_corners(a.x, a.y, a.yaw, a.length, a.width)
        if best is None or crossing_s < best.crossing_s:
            best = Hazard(a, crossing_s, xy, corners)
    return best


def visibility_profile(hazard: Hazard, frame: RouteFrame) -> NDArray[np.float64]:
    """1 where a pedestrian waiting at the crossing point is in line of sight from each distance band, else 0.

    The sensor is the ego's front-centre, ``d`` metres before the crossing point on the route centreline.
    """
    vis = np.zeros(ND)
    for i, d in enumerate(DIST_CENTRES):
        sensor, _ = frame.pose_at(hazard.crossing_s - d, 0.0)
        vis[i] = float(is_visible(sensor, hazard.ped_xy, hazard.corners, SENSOR_RANGE))
    return vis


def ego_distance(hazard: Hazard, s_front: float) -> float:
    """Metres from the ego's front bumper to the crossing point (negative once past it)."""
    return hazard.crossing_s - s_front


def pedestrian_observation(world: WorldModel, frame: RouteFrame, hazard: Hazard) -> PedObs:
    """What the ego currently sees of the pedestrian: nothing, one waiting, or one crossing."""
    seen = PedObs.NOT_SEEN
    for a in world.agents:
        if a.cls is not AgentClass.PEDESTRIAN:
            continue
        s, lateral = frame.project(a.x, a.y)
        if abs(s - hazard.crossing_s) > NEAR_HAZARD:
            continue
        _, tangent = frame.pose_at(s)
        left = np.array([-tangent[1], tangent[0]])
        toward_route = -np.sign(lateral) * float(np.dot((a.vx, a.vy), left)) if lateral != 0 else 0.0
        if abs(lateral) < IN_LANE or toward_route > CROSSING_SPEED:
            return PedObs.SEEN_CROSSING
        seen = PedObs.SEEN_WAITING
    return seen
