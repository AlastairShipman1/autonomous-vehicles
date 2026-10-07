"""The simulator-agnostic episode log the sweep metrics are computed from.

Any backend (the toy sim now, CARLA later) provides a scenario object with
``run(planner, predictor) -> EpisodeRecord``; the harness and metrics never see the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class EpisodeRecord:
    """Row k is the state at time ``t[k]``; the final row is the state after the last step."""

    seed: int
    dt: float
    t: np.ndarray  # (N,)
    ego: np.ndarray  # (N, 4): rear-axle x, y, yaw, speed
    ego_length: float
    ego_width: float
    ego_wheelbase: float
    ped: np.ndarray  # (N, 2) pedestrian centre, NaN rows when there is no pedestrian
    ped_radius: float
    occluder_near_x: float  # x of the occluder's near end
    occluder_far_x: float
    outcome: str  # 'collision' | 'finished' | 'timeout'
    params: dict[str, Any] = field(default_factory=dict)  # scenario parameters, copied into the CSV

    @property
    def ped_present(self) -> bool:
        return not bool(np.isnan(self.ped).all())


class Scenario(Protocol):
    def run(self, planner: Any, predictor: Any | None) -> EpisodeRecord: ...
