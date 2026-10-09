"""The simulator-agnostic episode log the sweep metrics are computed from.

Any backend (the toy sim now, CARLA later) provides a scenario object with
``run(planner, predictor) -> EpisodeRecord``; the harness and metrics never see the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from av_core.protocols import Planner, Predictor

Outcome = Literal["collision", "finished", "timeout"]
# A scenario parameter as it lands in the CSV.
ParamValue = float | int | str | bool | None


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
    ped: np.ndarray  # (N, P, 2) centre of each of P pedestrians; (N, 2) is accepted for one. NaN when absent
    ped_radius: float
    occluder_near_x: float  # x of the occluder's near end
    occluder_far_x: float
    outcome: Outcome
    params: dict[str, ParamValue] = field(default_factory=dict)  # scenario parameters, copied into the CSV

    def __post_init__(self) -> None:
        if self.ped.ndim == 2:  # a single pedestrian track
            self.ped = self.ped[:, None, :]

    @property
    def ped_present(self) -> bool:
        return not bool(np.isnan(self.ped).all())


class Scenario(Protocol):
    def run(self, planner: Planner, predictor: Predictor | None) -> EpisodeRecord: ...
