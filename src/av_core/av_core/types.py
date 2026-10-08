"""Shared contracts for the laptop and desktop code paths.

One ``map`` frame: right-handed, x and y in metres, yaw in radians counter-clockwise
from +x, speeds in m/s, time in seconds. Arrays are numpy float64. Every type validates
on construction (finite values, valid ranges) and round-trips through ``to_dict`` /
``from_dict``. The desktop's ROS messages mirror these field for field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from collections.abc import Mapping
from enum import StrEnum
from typing import SupportsFloat, cast

import numpy as np
from numpy.typing import ArrayLike

# What ``to_dict`` produces: plain data a JSON encoder accepts.
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]

# The values are the lowercase strings used in dicts and ROS messages, so the wire format is plain text.
class AgentClass(StrEnum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"


class TrafficLightState(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"  # the light exists (it is on the map) but its state is not observed


class PlannerReason(StrEnum):
    ROUTE = "route"
    LEAD = "lead"
    LIGHT = "light"
    CONFLICT = "conflict"
    OCCLUSION = "occlusion"


def _scalar(name: str, value: SupportsFloat | str, *, lo: float = -math.inf, hi: float = math.inf,
            lo_open: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a real number, got {value!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v < lo or v > hi or (lo_open and v == lo):
        raise ValueError(f"{name}={v} outside {'(' if lo_open else '['}{lo}, {hi}]")
    return v


def _array(name: str, value: ArrayLike, shape: tuple[int | None, ...]) -> np.ndarray:
    a = np.array(value, dtype=np.float64)  # copies, so callers can't mutate our state
    ok = a.ndim == len(shape) and all(s is None or s == d for s, d in zip(shape, a.shape))
    if not ok:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    a.setflags(write=False)
    return a


def _enum[E: StrEnum](name: str, value: str, kind: type[E]) -> E:
    """``value`` as a member of ``kind``; accepts the member itself or its string value."""
    try:
        return kind(value)
    except ValueError:
        raise ValueError(f"{name} must be one of {[m.value for m in kind]}, got {value!r}") from None


def _set(obj: object, **kw: object) -> None:
    for k, v in kw.items():
        object.__setattr__(obj, k, v)


class _Contract:
    """Equality and dict round-trip shared by all contract types."""

    def to_dict(self) -> dict[str, Json]:
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # arrays are unhashable


def _to_plain(v: object) -> Json:
    if isinstance(v, StrEnum):
        return v.value
    if isinstance(v, _Contract):
        return v.to_dict()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return [_to_plain(x) for x in cast(tuple[object, ...], v)]
    return cast(Json, v)


@dataclass(frozen=True, eq=False)
class EgoState(_Contract):
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float
    wheelbase: float

    def __post_init__(self) -> None:
        _set(
            self,
            x=_scalar("x", self.x),
            y=_scalar("y", self.y),
            yaw=_scalar("yaw", self.yaw),
            speed=_scalar("speed", self.speed, lo=0.0),
            length=_scalar("length", self.length, lo=0.0, lo_open=True),
            width=_scalar("width", self.width, lo=0.0, lo_open=True),
            wheelbase=_scalar("wheelbase", self.wheelbase, lo=0.0, lo_open=True),
        )

    @property
    def velocity(self) -> np.ndarray:
        """(vx, vy) in the map frame, assuming the velocity points along ``yaw`` (no sideslip).

        Exact for the kinematic bicycle model; an approximation once tyre slip matters. Keep that
        assumption here so a better model changes one property. ``speed`` is validated >= 0 today
        (no reversing); if reversing is added, ``speed`` becomes signed longitudinal velocity and this
        property stays correct unchanged.
        """
        return self.speed * np.array([math.cos(self.yaw), math.sin(self.yaw)])

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> EgoState:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class Agent(_Contract):
    id: int
    cls: AgentClass
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float
    is_static: bool

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, (int, np.integer)):
            raise ValueError(f"id must be an int, got {self.id!r}")
        _set(
            self,
            id=int(self.id),
            cls=_enum("cls", self.cls, AgentClass),
            x=_scalar("x", self.x),
            y=_scalar("y", self.y),
            yaw=_scalar("yaw", self.yaw),
            vx=_scalar("vx", self.vx),
            vy=_scalar("vy", self.vy),
            length=_scalar("length", self.length, lo=0.0, lo_open=True),
            width=_scalar("width", self.width, lo=0.0, lo_open=True),
            is_static=bool(self.is_static),
        )

    @property
    def velocity(self) -> np.ndarray:
        """(vx, vy) in the map frame. Independent of ``yaw``: a tracked object need not move the way it faces."""
        return np.array([self.vx, self.vy])

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Agent:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class OccludedRegion(_Contract):
    occluder_id: int
    polygon: np.ndarray  # (K, 2), K >= 3

    def __post_init__(self) -> None:
        poly = _array("polygon", self.polygon, (None, 2))
        if len(poly) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        _set(self, occluder_id=int(self.occluder_id), polygon=poly)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> OccludedRegion:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class TrafficLight(_Contract):
    """One signal: its id (from the map), current state, and the stop line it controls.

    ``stop_line`` is the segment's two endpoints. Whether a light applies to ego is the consumer's call
    (planner v0 checks the stop line against the route); the producer reports every light it knows about.
    """

    id: int
    state: TrafficLightState
    stop_line: np.ndarray  # (2, 2): endpoints of the stop line

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, (int, np.integer)):
            raise ValueError(f"id must be an int, got {self.id!r}")
        line = _array("stop_line", self.stop_line, (2, 2))
        if np.array_equal(line[0], line[1]):
            raise ValueError("stop_line endpoints must differ")
        _set(self, id=int(self.id), state=_enum("state", self.state, TrafficLightState), stop_line=line)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> TrafficLight:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class WorldModel(_Contract):
    stamp: float
    ego: EgoState
    agents: tuple[Agent, ...]
    occluded: tuple[OccludedRegion, ...]
    traffic_lights: tuple[TrafficLight, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ego, EgoState):
            raise ValueError("ego must be an EgoState")
        agents, occluded = tuple(self.agents), tuple(self.occluded)
        lights = tuple(self.traffic_lights)
        if not all(isinstance(a, Agent) for a in agents):
            raise ValueError("agents must all be Agent")
        if not all(isinstance(o, OccludedRegion) for o in occluded):
            raise ValueError("occluded must all be OccludedRegion")
        if not all(isinstance(t, TrafficLight) for t in lights):
            raise ValueError("traffic_lights must all be TrafficLight")
        ids = [a.id for a in agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        light_ids = [t.id for t in lights]
        if len(set(light_ids)) != len(light_ids):
            raise ValueError("traffic light ids must be unique")
        _set(
            self,
            stamp=_scalar("stamp", self.stamp),
            agents=agents,
            occluded=occluded,
            traffic_lights=lights,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> WorldModel:
        return cls(
            stamp=cast(float, d["stamp"]),
            ego=EgoState.from_dict(cast(Mapping[str, object], d["ego"])),
            agents=tuple(Agent.from_dict(a) for a in cast(list[Mapping[str, object]], d["agents"])),
            occluded=tuple(OccludedRegion.from_dict(o) for o in cast(list[Mapping[str, object]], d["occluded"])),
            traffic_lights=tuple(
                TrafficLight.from_dict(t) for t in cast(list[Mapping[str, object]], d["traffic_lights"])),
        )


@dataclass(frozen=True, eq=False)
class Route(_Contract):
    points: np.ndarray  # (N, 2), N >= 2, densified to 0.5 m spacing by the producer
    speed_limit: float

    def __post_init__(self) -> None:
        pts = _array("points", self.points, (None, 2))
        if len(pts) < 2:
            raise ValueError("route needs at least 2 points")
        _set(self, points=pts, speed_limit=_scalar("speed_limit", self.speed_limit, lo=0.0, lo_open=True))

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Route:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class PredictedTrajectory(_Contract):
    agent_id: int
    t: np.ndarray  # (H,), strictly increasing, seconds from the WorldModel stamp
    xy: np.ndarray  # (H, 2)
    prob: float = 1.0

    def __post_init__(self) -> None:
        t = _array("t", self.t, (None,))
        xy = _array("xy", self.xy, (len(t), 2))
        if len(t) < 1:
            raise ValueError("trajectory needs at least 1 step")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        _set(self, agent_id=int(self.agent_id), t=t, xy=xy, prob=_scalar("prob", self.prob, lo=0.0, hi=1.0))

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> PredictedTrajectory:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class PlannerCommand(_Contract):
    stamp: float
    target_speed: float
    reason: PlannerReason

    def __post_init__(self) -> None:
        _set(
            self,
            stamp=_scalar("stamp", self.stamp),
            target_speed=_scalar("target_speed", self.target_speed, lo=0.0),
            reason=_enum("reason", self.reason, PlannerReason),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> PlannerCommand:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__


@dataclass(frozen=True, eq=False)
class ControlCommand(_Contract):
    stamp: float
    throttle: float
    brake: float
    steer: float

    def __post_init__(self) -> None:
        _set(
            self,
            stamp=_scalar("stamp", self.stamp),
            throttle=_scalar("throttle", self.throttle, lo=0.0, hi=1.0),
            brake=_scalar("brake", self.brake, lo=0.0, hi=1.0),
            steer=_scalar("steer", self.steer, lo=-1.0, hi=1.0),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> ControlCommand:
        return cls(**d)  # pyright: ignore[reportArgumentType]  # validated in __post_init__
