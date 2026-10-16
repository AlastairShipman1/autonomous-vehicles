"""Turn wall-clock time into whole simulation steps for the live window."""

from __future__ import annotations


class Pacer:
    """Fixed-step accumulator: the sim always advances in whole ``dt`` steps, however long a frame took.

    ``speed`` scales sim time against wall time (0.25 is slow motion). If frames run too slow to keep up, at most
    ``max_steps`` are taken per frame and the backlog is dropped, so the sim slows down instead of spiralling.
    """

    def __init__(self, dt: float, speed: float = 1.0, max_steps: int = 4):
        if dt <= 0 or speed <= 0 or max_steps < 1:
            raise ValueError("dt, speed and max_steps must be positive")
        self.dt, self.speed, self.max_steps = dt, speed, max_steps
        self._acc = 0.0
        self.dropped = 0  # steps' worth of time discarded because the frame was too slow

    def steps_for(self, wall_seconds: float) -> int:
        self._acc += max(0.0, wall_seconds) * self.speed
        n = int(self._acc / self.dt + 1e-9)
        if n > self.max_steps:
            self.dropped += n - self.max_steps
            n = self.max_steps
            self._acc = 0.0
        else:
            self._acc -= n * self.dt
        return n

    def reset(self) -> None:
        self._acc = 0.0
