"""Scenario parameters: parked vehicles beside a straight road and a pedestrian who may step out from behind them."""

from __future__ import annotations

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
        # The pedestrian walks straight across; its path must not run through any parked vehicle.
        if not self.ped_present:
            return
        for v in self.vehicles:
            if v.near_x - PED_RADIUS < self.ped_x < v.far_x + PED_RADIUS:
                raise ValueError(f"ped_x={self.ped_x} puts the pedestrian's path through a parked vehicle "
                                 f"(x {v.near_x:.2f} to {v.far_x:.2f})")
