"""Constant-velocity predictor: every agent keeps its current (vx, vy) for the whole horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from av_core.types import PredictedTrajectory, WorldModel


@dataclass(frozen=True)
class ConstantVelocityPredictor:
    horizon: float = 3.0  # s
    dt: float = 0.1  # s

    def __call__(self, world: WorldModel) -> tuple[PredictedTrajectory, ...]:
        """One trajectory per agent, in the order of ``world.agents``.

        ``t`` runs 0, dt, ..., horizon in seconds from ``world.stamp`` (t = 0 is the current position).
        Static agents are included; their trajectory is a constant position.
        """
        n = int(round(self.horizon / self.dt))
        t = np.arange(n + 1) * self.dt
        out = []
        for a in world.agents:
            xy = np.column_stack([a.x + a.vx * t, a.y + a.vy * t])
            out.append(PredictedTrajectory(a.id, t, xy, 1.0))
        return tuple(out)
