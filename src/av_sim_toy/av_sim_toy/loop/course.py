"""What is on the track: the gauntlet course.

A ``Course`` is plain data, so a new obstacle is a new dataclass here plus a few lines in ``LoopSim``. Positions are
arc lengths ``s`` along the track (lateral offsets as in ``track.py``: left positive, lane 0 at 0, parking at -2.75).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from av_sim_toy.loop.track import LANE_WIDTH, PARKING_LATERAL, SIDEWALK_RIGHT, Track

PED_RADIUS = 0.3
INNER_SIDEWALK = 6.5  # lateral where a pedestrian crossing from the right finishes (beyond lane 1)


@dataclass(frozen=True)
class ParkedCar:
    """A parked vehicle: a static obstacle that also blocks sight."""

    s: float
    lateral: float = PARKING_LATERAL
    length: float = 4.5
    width: float = 2.0


@dataclass(frozen=True)
class Walker:
    """A pedestrian who waits at ``from_lateral`` and crosses to ``to_lateral`` when the ego is ``trigger_distance`` short
    of ``s`` (measured to the ego's front). After the ego has gone by, the walker goes back to waiting for the next lap."""

    s: float
    speed: float = 1.4
    trigger_distance: float = 25.0
    from_lateral: float = SIDEWALK_RIGHT
    to_lateral: float = INNER_SIDEWALK


@dataclass(frozen=True)
class Signal:
    """A traffic light with a stop line across every lane. Cycles green, yellow, red from ``offset`` seconds in."""

    s: float
    green: float = 15.0
    yellow: float = 3.0
    red: float = 8.0
    offset: float = 0.0

    @property
    def cycle(self) -> float:
        return self.green + self.yellow + self.red


@dataclass(frozen=True)
class TrafficCar:
    """A car that circulates for ever, following whatever is ahead of it in its lane (and stopping at red lights).
    ``speed`` is the speed it would like to drive at."""

    s: float
    lane: int = 1
    speed: float = 10.0
    length: float = 4.5
    width: float = 1.9


@dataclass(frozen=True)
class Course:
    name: str = "empty"
    description: str = "An empty loop."
    parked: tuple[ParkedCar, ...] = ()
    walkers: tuple[Walker, ...] = ()
    signals: tuple[Signal, ...] = ()
    traffic: tuple[TrafficCar, ...] = ()
    ego_start_s: float = 0.0
    ego_start_speed: float = 6.0
    extras: dict[str, float] = field(default_factory=dict)  # room for later: tuning knobs for new obstacle types

    def validate(self, track: Track) -> None:
        """Reject layouts that cannot work: a walker whose path would run through a parked car, overlapping cars."""
        n = track.n_lanes
        for car in self.traffic:
            if not 0 <= car.lane < n:
                raise ValueError(f"traffic car at s={car.s} is in lane {car.lane}; the track has {n} lanes")
        for w in self.walkers:
            lo, hi = sorted((w.from_lateral, w.to_lateral))
            for p in self.parked:
                gap = abs((p.s - w.s + 0.5 * track.length) % track.length - 0.5 * track.length)  # along the track
                if gap < 0.5 * p.length + PED_RADIUS and lo - 0.5 * p.width <= p.lateral <= hi + 0.5 * p.width:
                    raise ValueError(f"walker at s={w.s} would cross through the parked car at s={p.s}")
        starts = sorted(c.s % track.length for c in self.traffic)
        for a, b in zip(starts, starts[1:] + [starts[0] + track.length] if starts else []):
            if b - a < 12.0:
                raise ValueError("traffic cars start less than 12 m apart")


def default_course(track: Track) -> Course:
    """The first gauntlet, laid out as fractions of the lap so it fits any track length.

    In lap order: a traffic light, a row of parked cars with a pedestrian who runs out from behind the last one,
    an open crosswalk where a pedestrian crosses in plain sight, and a few cars circulating in the left lane.
    """
    c = track.length
    row = (0.40 * c, 0.40 * c + 6.0, 0.40 * c + 12.0)
    return Course(
        name="gauntlet_1",
        description="Traffic light, parked row with a hidden runner, open crosswalk, traffic in the left lane.",
        signals=(Signal(s=0.20 * c),),
        parked=tuple(ParkedCar(s) for s in row),
        walkers=(Walker(s=row[-1] + 4.5 / 2 + 1.0, speed=1.8, trigger_distance=22.0),  # hidden: just past the row
                 Walker(s=0.72 * c, speed=1.4, trigger_distance=25.0)),  # in the open
        traffic=tuple(TrafficCar(s=f * c, lane=1, speed=v) for f, v in ((0.05, 9.0), (0.33, 10.5), (0.62, 8.5), (0.88, 10.0))),
    )


def empty_course() -> Course:
    return Course()


__all__ = ["Course", "ParkedCar", "Walker", "Signal", "TrafficCar", "default_course", "empty_course", "LANE_WIDTH"]
