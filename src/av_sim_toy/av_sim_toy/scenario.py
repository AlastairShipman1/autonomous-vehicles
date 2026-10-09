"""Scenario parameters: parked vehicles beside a straight road and a pedestrian who may step out from behind them."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Fixed road geometry (metres, map frame; the road runs along +x).
LANE_CENTER_Y = 0.0
LANE_WIDTH = 3.5
PARKING_CENTER_Y = -2.75
SIDEWALK_Y = -3.75
SPEED_LIMIT = 13.9
OCCLUDER_WIDTH = 2.0
PED_RADIUS = 0.3
PED_START_Y = -4.5

# Ego body, as in the spec (the kinematic reference point is the rear axle).
EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE = 4.7, 1.9, 2.9

DT = 0.05  # 20 Hz
MAX_TIME = 30.0
END_MARGIN = 30.0  # episode ends when ego's rear is this far past the occluder


@dataclass(frozen=True)
class Vehicle:
    """A parked vehicle: centre, length, width. ``y`` is -2.75 in the right-hand parking lane, +2.75 on the left."""

    x: float
    y: float = PARKING_CENTER_Y
    length: float = 4.5
    width: float = OCCLUDER_WIDTH

    @property
    def near_x(self) -> float:
        return self.x - 0.5 * self.length

    @property
    def far_x(self) -> float:
        return self.x + 0.5 * self.length


@dataclass(frozen=True)
class MovingVehicle:
    """A vehicle driving along the road at constant speed, optionally braking to a stop.

    ``speed`` is signed along +x: positive goes the ego's way, negative is oncoming (it then faces -x). ``y`` is its lane
    centre: 0 for the ego's lane (a lead vehicle), +3.5 for an oncoming lane to its left. From ``brake_time`` (s) it
    decelerates at ``brake_decel`` until it reaches ``brake_to``. It occludes while it moves, and the ego hitting it ends the episode;
    pedestrians do not react to it and it does not react to anything.
    """

    x: float
    speed: float
    y: float = 0.0
    length: float = 4.5
    width: float = 1.9
    brake_time: float | None = None
    brake_decel: float = 4.0
    brake_to: float = 0.0  # speed (magnitude) it brakes down to; it then holds that speed

    @property
    def yaw(self) -> float:
        return 0.0 if self.speed >= 0 else math.pi


@dataclass(frozen=True)
class Pedestrian:
    """An extra pedestrian: waits on the sidewalk at ``x`` and walks straight across once the ego front is within
    ``trigger_distance`` of ``x``. ``from_left`` starts it on the left sidewalk."""

    x: float
    speed: float = 1.4
    trigger_distance: float = 20.0
    from_left: bool = False

    @property
    def side(self) -> float:
        """+1 if it starts on the left of the road, -1 on the right."""
        return 1.0 if self.from_left else -1.0

    @property
    def y0(self) -> float:
        return self.side * abs(PED_START_Y)


@dataclass(frozen=True)
class ScenarioParams:
    """Everything that varies between episodes; the sampler draws the single-car ones from a seed.

    The *primary* occluder (``occluder_x``, ``occluder_length``) is the car the pedestrian is hidden behind; any
    ``extra_vehicles`` are further parked vehicles, on either side. All of them occlude. The pedestrian waits on
    the sidewalk beside the primary occluder (on the left if ``ped_from_left``), walks straight across the road once
    triggered, and stops on the far sidewalk.
    """

    initial_speed: float = 10.0
    occluder_x: float = 50.0  # centre
    occluder_length: float = 6.0
    ped_present: bool = True
    ped_x: float = 54.0  # past the default occluder's far end (53)
    ped_speed: float = 1.4
    ped_trigger_distance: float = 20.0  # ego front this far short of ped_x starts the walk
    seed: int | None = None
    extra_vehicles: tuple[Vehicle, ...] = ()
    ped_from_left: bool = False
    extra_pedestrians: tuple[Pedestrian, ...] = ()  # more pedestrians, in addition to the primary one
    moving_vehicles: tuple[MovingVehicle, ...] = ()

    @property
    def side(self) -> float:
        """+1 if the pedestrian starts on the left of the road, -1 on the right."""
        return 1.0 if self.ped_from_left else -1.0

    @property
    def occluder_y(self) -> float:
        return self.side * abs(PARKING_CENTER_Y)

    @property
    def ped_y0(self) -> float:
        return self.side * abs(PED_START_Y)

    @property
    def pedestrians(self) -> tuple[Pedestrian, ...]:
        """Every pedestrian in the scenario: the primary one (if present) first, then the extras."""
        primary = Pedestrian(self.ped_x, self.ped_speed, self.ped_trigger_distance, self.ped_from_left)
        return ((primary,) if self.ped_present else ()) + self.extra_pedestrians

    @property
    def vehicles(self) -> tuple[Vehicle, ...]:
        """The primary occluder first, then the extras."""
        return (Vehicle(self.occluder_x, self.occluder_y, self.occluder_length), *self.extra_vehicles)

    @property
    def near_x(self) -> float:
        return min(v.near_x for v in self.vehicles)

    @property
    def far_x(self) -> float:
        return max(v.far_x for v in self.vehicles)

    def __post_init__(self) -> None:
        for i, mv in enumerate(self.moving_vehicles):
            # must not start overlapping the ego (rear axle at x = 0, body from -0.9 to 3.8 m, 1.9 m wide)
            if mv.x - 0.5 * mv.length < 3.8 + 1.0 and mv.x + 0.5 * mv.length > -0.9 and abs(mv.y) < 0.95 + 0.5 * mv.width:
                raise ValueError(f"moving vehicle {i} starts on top of the ego (x {mv.x - 0.5 * mv.length:.1f} "
                                 f"to {mv.x + 0.5 * mv.length:.1f})")
        # A pedestrian walks straight across; its path must not run through any parked vehicle.
        for i, ped in enumerate(self.pedestrians):
            for v in self.vehicles:
                if v.near_x - PED_RADIUS < ped.x < v.far_x + PED_RADIUS:
                    raise ValueError(f"ped_x={ped.x} puts pedestrian {i}'s path through a parked vehicle "
                                     f"(x {v.near_x:.2f} to {v.far_x:.2f})")
