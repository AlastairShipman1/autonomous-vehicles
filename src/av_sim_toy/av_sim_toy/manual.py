"""A pedestrian a person drives: key directions in, a walking pedestrian out.

The sim, not the input, decides what is possible: the pedestrian accelerates at a human rate, cannot walk through a
parked vehicle (it slides along the side instead) and cannot walk into a car that has stopped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from av_core.geometry import distance_point_to_rect
from av_sim_toy import scenario as sc

Direction = tuple[int, int]  # (dx, dy) each in {-1, 0, 1}: +x along the road, +y to the left of it
Y_BOUND = 7.5  # the pedestrian stays inside the drawn world
X_MIN = -25.0
MAX_SPEED = 2.0  # m/s, a brisk walk; the scripted pedestrians walk at 0.8 to 2.0
ACCEL = 4.0  # m/s^2: from standing to the 1.4 m/s default pace in about a third of a second
STOPPED_EGO_SPEED = 0.3  # a pedestrian does not walk into a car that is standing still
MARGIN = 0.02  # m kept between the pedestrian's disc and an obstacle, so touching is never a collision


def direction(left: bool, right: bool, up: bool, down: bool) -> Direction:
    """Held keys to a direction; opposite keys cancel. ``up`` is +y (screen up), ``right`` is +x (the way the ego drives)."""
    return int(right) - int(left), int(up) - int(down)


@dataclass
class ManualPedestrian:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = math.pi / 2  # kept while standing still
    max_speed: float = MAX_SPEED
    accel: float = ACCEL

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def copy(self) -> ManualPedestrian:
        return ManualPedestrian(self.x, self.y, self.vx, self.vy, self.heading, self.max_speed, self.accel)

    def blocked(self, x: float, y: float, obstacles: Sequence[np.ndarray]) -> bool:
        """Whether a pedestrian centred at (x, y) would overlap an obstacle rectangle or leave the world."""
        if not (X_MIN <= x and -Y_BOUND <= y <= Y_BOUND):
            return True
        p = np.array([x, y])
        return any(distance_point_to_rect(rect, p) < sc.PED_RADIUS + MARGIN for rect in obstacles)

    def step(self, wanted: Direction, dt: float, obstacles: Sequence[np.ndarray]) -> None:
        """Advance ``dt`` seconds towards the wanted direction. ``obstacles`` are rectangles (4, 2) it cannot enter."""
        dx, dy = wanted
        norm = math.hypot(dx, dy)
        tx, ty = (dx / norm * self.max_speed, dy / norm * self.max_speed) if norm else (0.0, 0.0)
        ax, ay = tx - self.vx, ty - self.vy
        gap, limit = math.hypot(ax, ay), self.accel * dt
        if gap > limit:  # accelerate at a human rate, not instantly
            ax, ay = ax / gap * limit, ay / gap * limit
        self.vx, self.vy = self.vx + ax, self.vy + ay
        nx, ny = self.x + self.vx * dt, self.y + self.vy * dt
        if self.blocked(nx, ny, obstacles):  # slide along whichever axis is free
            if not self.blocked(nx, self.y, obstacles):
                ny, self.vy = self.y, 0.0
            elif not self.blocked(self.x, ny, obstacles):
                nx, self.vx = self.x, 0.0
            else:
                nx, ny, self.vx, self.vy = self.x, self.y, 0.0, 0.0
        self.x, self.y = nx, ny
        if self.speed > 0.05:
            self.heading = math.atan2(self.vy, self.vx)
