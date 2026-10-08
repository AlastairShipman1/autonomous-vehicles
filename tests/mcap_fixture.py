"""Write a synthetic bag in the *assumed* av_interfaces layout (see tools/mcap_io.py).

This validates the plumbing (decode -> av_core types -> replay), not the real desktop message definitions,
which do not exist in this repo yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mcap_ros2.writer import Writer as Ros2Writer

from av_core.types import PlannerCommand, Route, WorldModel

SEP = "=" * 80
DEFS = {
    "builtin_interfaces/msg/Time": "int32 sec\nuint32 nanosec\n",
    "av_interfaces/msg/EgoState": "float64 x\nfloat64 y\nfloat64 yaw\nfloat64 speed\nfloat64 length\nfloat64 width\nfloat64 wheelbase\n",
    "av_interfaces/msg/Agent": ("int32 id\nstring cls\nfloat64 x\nfloat64 y\nfloat64 yaw\nfloat64 vx\nfloat64 vy\n"
                                "float64 length\nfloat64 width\nbool is_static\n"),
    "av_interfaces/msg/OccludedRegion": "int32 occluder_id\nfloat64[] polygon\n",
    "av_interfaces/msg/WorldModel": ("builtin_interfaces/Time stamp\nav_interfaces/EgoState ego\nav_interfaces/Agent[] agents\n"
                                     "av_interfaces/OccludedRegion[] occluded\nav_interfaces/TrafficLight[] traffic_lights\n"),
    "av_interfaces/msg/TrafficLight": "int32 id\nstring state\nfloat64[] stop_line\n",
    "av_interfaces/msg/PlannerCommand": "builtin_interfaces/Time stamp\nfloat64 target_speed\nstring reason\n",
    "av_interfaces/msg/Route": "float64[] points\nfloat64 speed_limit\n",
}
DEPS = {
    "av_interfaces/msg/WorldModel": ["builtin_interfaces/msg/Time", "av_interfaces/msg/EgoState",
                                     "av_interfaces/msg/Agent", "av_interfaces/msg/OccludedRegion",
                                     "av_interfaces/msg/TrafficLight"],
    "av_interfaces/msg/PlannerCommand": ["builtin_interfaces/msg/Time"],
}


def _full(name: str) -> str:
    parts = [DEFS[name]]
    for dep in DEPS.get(name, []):
        parts.append(f"{SEP}\nMSG: {dep.replace('/msg/', '/')}\n{DEFS[dep]}")
    return "".join(parts)


def _time(stamp: float) -> dict:
    sec = int(stamp)
    return {"sec": sec, "nanosec": int(round((stamp - sec) * 1e9))}


def world_msg(w: WorldModel) -> dict:
    e = w.ego
    return {
        "stamp": _time(w.stamp),
        "ego": dict(x=e.x, y=e.y, yaw=e.yaw, speed=e.speed, length=e.length, width=e.width, wheelbase=e.wheelbase),
        "agents": [dict(id=a.id, cls=a.cls, x=a.x, y=a.y, yaw=a.yaw, vx=a.vx, vy=a.vy, length=a.length,
                        width=a.width, is_static=a.is_static) for a in w.agents],
        "occluded": [dict(occluder_id=o.occluder_id, polygon=o.polygon.reshape(-1).tolist()) for o in w.occluded],
        "traffic_lights": [dict(id=t.id, state=t.state, stop_line=t.stop_line.reshape(-1).tolist())
                           for t in w.traffic_lights],
    }


def write_bag(path: Path, worlds: list[WorldModel], commands: list[PlannerCommand], route: Route | None) -> Path:
    with open(path, "wb") as fh:
        w = Ros2Writer(fh)
        schemas = {n: w.register_msgdef(n, _full(n)) for n in
                   ("av_interfaces/msg/WorldModel", "av_interfaces/msg/PlannerCommand", "av_interfaces/msg/Route")}
        if route is not None:
            w.write_message("/route", schemas["av_interfaces/msg/Route"],
                            {"points": route.points.reshape(-1).tolist(), "speed_limit": route.speed_limit}, log_time=0)
        for wm in worlds:
            w.write_message("/world_model", schemas["av_interfaces/msg/WorldModel"], world_msg(wm),
                            log_time=int(wm.stamp * 1e9))
        for c in commands:
            w.write_message("/planner_cmd", schemas["av_interfaces/msg/PlannerCommand"],
                            {"stamp": _time(c.stamp), "target_speed": c.target_speed, "reason": c.reason},
                            log_time=int(c.stamp * 1e9) + 1)
        w.finish()
    return path
