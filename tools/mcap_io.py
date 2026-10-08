"""Read desktop MCAP bags and replay the logged WorldModels through an ``av_core`` planner.

Uses ``mcap`` and ``mcap-ros2-support``, which decode the message definitions stored in the bag, so no ROS
install is needed.

**Assumed message layout.** The ``av_interfaces`` messages are not in this repo yet, so the converters assume they
mirror ``av_core.types`` field for field (names as in the spec), with ``stamp`` a ``builtin_interfaces/Time`` (or
under ``header``). Polygons, routes and the stop line may be flat ``float64[]`` (x0, y0, x1, y1, ...), a list of
``[x, y]`` pairs, or a list of objects with ``x`` and ``y``. ``traffic_lights`` is an array of
``{id, state, stop_line}`` where ``stop_line`` holds the segment's two endpoints. If the real
messages differ, the places to change are the ``*_from_msg`` functions below.

Topics: ``/world_model`` and ``/planner_cmd`` as specified, and ``/route`` for the route. Planner v0 needs a route
and the spec does not list a topic for it; pass ``route=`` explicitly when the bag has none.

    uv run python -m tools.mcap_io bag.mcap --planner v0 [--route route.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, cast, runtime_checkable

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from av_core.protocols import Planner
from av_core.types import (
    Agent,
    AgentClass,
    EgoState,
    OccludedRegion,
    PlannerCommand,
    PlannerReason,
    Route,
    TrafficLight,
    TrafficLightState,
    WorldModel,
)

WORLD_MODEL_TOPIC = "/world_model"
PLANNER_CMD_TOPIC = "/planner_cmd"
ROUTE_TOPIC = "/route"


# --- assumed message shapes (structural types; see the module docstring) -----------------------

@runtime_checkable
class _Time(Protocol):
    sec: int
    nanosec: int


@runtime_checkable
class _XY(Protocol):
    x: float
    y: float


# A polygon / route / stop line: flat floats, a list of pairs, or a list of {x, y} objects.
_Points = Sequence[float] | Sequence[Sequence[float]] | Sequence[_XY]


@runtime_checkable
class _HasStamp(Protocol):
    stamp: _Time | float


@runtime_checkable
class _HasHeader(Protocol):
    header: _HasStamp


class _EgoMsg(Protocol):
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float
    wheelbase: float


class _AgentMsg(Protocol):
    id: int
    cls: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float
    is_static: bool


class _OccludedMsg(Protocol):
    occluder_id: int
    polygon: _Points


class _TrafficLightMsg(Protocol):
    id: int
    state: str
    stop_line: _Points


class _WorldModelMsg(Protocol):
    ego: _EgoMsg
    agents: Sequence[_AgentMsg]
    occluded: Sequence[_OccludedMsg]
    traffic_lights: Sequence[_TrafficLightMsg]


class _PlannerCommandMsg(Protocol):
    target_speed: float
    reason: str


class _RouteMsg(Protocol):
    points: _Points
    speed_limit: float


# --- message -> av_core ------------------------------------------------------------------------

def _stamp(msg: object) -> float:
    """Seconds from ``msg.stamp`` or ``msg.header.stamp`` (a Time struct or a plain number)."""
    if isinstance(msg, _HasStamp):
        s = msg.stamp
    elif isinstance(msg, _HasHeader):
        s = msg.header.stamp
    else:
        raise ValueError(f"message has neither stamp nor header.stamp: {msg!r}")
    if isinstance(s, _Time):
        return float(s.sec) + 1e-9 * float(s.nanosec)
    return float(s)


def _xy(seq: _Points) -> np.ndarray:
    """(K, 2) array from a flat float list, a list of pairs, or a list of objects with x and y."""
    items = list(seq)
    if not items:
        return np.zeros((0, 2))
    if isinstance(items[0], _XY):
        return np.array([[p.x, p.y] for p in cast(Sequence[_XY], items)], dtype=np.float64)
    arr = np.asarray(cast(Sequence[float], items), dtype=np.float64)
    return arr.reshape(-1, 2)


def world_model_from_msg(msg: _WorldModelMsg) -> WorldModel:
    e = msg.ego
    ego = EgoState(e.x, e.y, e.yaw, e.speed, e.length, e.width, e.wheelbase)
    agents = tuple(
        Agent(int(a.id), AgentClass(a.cls), a.x, a.y, a.yaw, a.vx, a.vy, a.length, a.width, bool(a.is_static))
        for a in msg.agents
    )
    occluded = tuple(OccludedRegion(int(o.occluder_id), _xy(o.polygon)) for o in msg.occluded)
    lights = tuple(TrafficLight(int(t.id), TrafficLightState(t.state), _xy(t.stop_line)) for t in msg.traffic_lights)
    return WorldModel(_stamp(msg), ego, agents, occluded, lights)


def planner_command_from_msg(msg: _PlannerCommandMsg) -> PlannerCommand:
    return PlannerCommand(_stamp(msg), float(msg.target_speed), PlannerReason(msg.reason))


def route_from_msg(msg: _RouteMsg) -> Route:
    return Route(_xy(msg.points), float(msg.speed_limit))


# --- bag readers -------------------------------------------------------------------------------

def _read[M, T](path: str | Path, topic: str, convert: Callable[[M], T]) -> list[T]:
    """Decode ``topic``; the caller's ``convert`` names the message shape ``M`` it expects (unchecked here)."""
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        return [convert(cast(M, m.decoded_message)) for m in reader.iter_decoded_messages(topics=[topic])]


def read_world_models(path: str | Path) -> list[WorldModel]:
    return _read(path, WORLD_MODEL_TOPIC, world_model_from_msg)


def read_planner_commands(path: str | Path) -> list[PlannerCommand]:
    return _read(path, PLANNER_CMD_TOPIC, planner_command_from_msg)


def read_route(path: str | Path) -> Route | None:
    """The first ``/route`` message, or ``None`` if the bag has none."""
    routes = _read(path, ROUTE_TOPIC, route_from_msg)
    return routes[0] if routes else None


def load_route_json(path: str | Path) -> Route:
    """``{"points": [[x, y], ...], "speed_limit": 13.9}``"""
    d = json.loads(Path(path).read_text())
    return Route(np.asarray(d["points"], dtype=np.float64), d["speed_limit"])


# --- replay ------------------------------------------------------------------------------------

@dataclass
class ReplayResult:
    n_worlds: int
    n_compared: int = 0
    mismatches: list[str] = field(default_factory=list)
    unmatched_worlds: list[float] = field(default_factory=list)  # stamps with no logged command
    unmatched_commands: list[float] = field(default_factory=list)  # logged commands with no world model

    @property
    def ok(self) -> bool:
        return self.n_compared > 0 and not (self.mismatches or self.unmatched_worlds or self.unmatched_commands)

    def summary(self) -> str:
        lines = [f"{self.n_compared} of {self.n_worlds} world models compared; "
                 f"{len(self.mismatches)} mismatches, {len(self.unmatched_worlds)} worlds without a command, "
                 f"{len(self.unmatched_commands)} commands without a world"]
        lines += [f"  {m}" for m in self.mismatches[:10]]
        if len(self.mismatches) > 10:
            lines.append(f"  ... and {len(self.mismatches) - 10} more")
        return "\n".join(lines)


def replay(worlds: Iterable[WorldModel], commands: Iterable[PlannerCommand], route: Route, planner: Planner,
           atol: float = 1e-6, stamp_tol: float = 1e-6) -> ReplayResult:
    """Run ``planner`` over the recorded WorldModels and compare with the logged commands.

    Commands are matched to world models by stamp (each output carries the stamp of the WorldModel it was
    computed from). Planner v0 is stateless; a stateful planner would need the worlds in order, which they are.
    """
    worlds, commands = list(worlds), list(commands)
    result = ReplayResult(n_worlds=len(worlds))
    by_stamp = {round(c.stamp / stamp_tol): c for c in commands}
    used: set[int] = set()
    for w in worlds:
        key = round(w.stamp / stamp_tol)
        logged = by_stamp.get(key)
        if logged is None:
            result.unmatched_worlds.append(w.stamp)
            continue
        used.add(key)
        out = planner.plan(w, route)
        result.n_compared += 1
        if out.reason != logged.reason or not math.isclose(out.target_speed, logged.target_speed,
                                                           rel_tol=0.0, abs_tol=atol):
            result.mismatches.append(
                f"stamp {w.stamp:.3f}: replay ({out.target_speed:.4f}, {out.reason}) "
                f"!= logged ({logged.target_speed:.4f}, {logged.reason})")
    result.unmatched_commands = [c.stamp for k, c in by_stamp.items() if k not in used]
    return result


def replay_bag(path: str | Path, planner: Planner, route: Route | None = None, atol: float = 1e-6) -> ReplayResult:
    route = route or read_route(path)
    if route is None:
        raise ValueError("the bag has no /route topic; pass route=")
    return replay(read_world_models(path), read_planner_commands(path), route, planner, atol)


def main(argv: list[str] | None = None) -> int:
    from tools.planners import make_planner

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--planner", default="v0")
    ap.add_argument("--route", type=Path, default=None, help="route JSON, if the bag has no /route topic")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args(argv)
    route = load_route_json(args.route) if args.route else None
    result = replay_bag(args.bag, make_planner(args.planner), route, args.atol)
    print(result.summary())
    print("MATCH" if result.ok else "MISMATCH")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
