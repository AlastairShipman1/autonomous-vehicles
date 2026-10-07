"""Scenario parameters for the one toy scenario: a pedestrian hidden behind a parked vehicle."""

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
class ScenarioParams:
    """Everything that varies between episodes; the sampler (PR 2.3) draws these from a seed."""

    initial_speed: float = 10.0
    occluder_x: float = 50.0  # centre
    occluder_length: float = 6.0
    ped_present: bool = True
    ped_x: float = 50.0
    ped_speed: float = 1.4
    ped_trigger_distance: float = 20.0  # ego front this far short of ped_x starts the walk
    seed: int | None = None
