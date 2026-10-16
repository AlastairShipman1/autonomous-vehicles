"""The loop sim: the ego drives round a closed track for ever, past the obstacles of a ``Course``.

Same ego, controller and planner interface as the straight-road toy sim. What is new is that nothing ends the run:
the ego's route is a window that slides round the loop, contacts are *logged* (once each) and the ego carries on,
and a lap counter turns the run into rows of a table.

Contacts have no physical effect: the ego does not bounce off what it hits. They are incidents, not crashes.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from av_core.control import BicycleState, Controller, accel_from_pedals, ego_center, step
from av_core.geometry import (
    SENSOR_RANGE,
    convex_polygons_overlap,
    distance_point_to_rect,
    is_visible,
    point_in_convex,
    rect_corners,
    shadow_polygon,
)
from av_core.protocols import Planner, Predictor
from av_core.types import (
    Agent,
    AgentClass,
    ControlCommand,
    EgoState,
    OccludedRegion,
    PlannerCommand,
    PredictedTrajectory,
    Route,
    TrafficLight,
    TrafficLightState,
    WorldModel,
)
from av_sim_toy import scenario as sc
from av_sim_toy.loop.course import PED_RADIUS, Course, Signal, TrafficCar, Walker
from av_sim_toy.loop.track import Track

DT = sc.DT
PARKED_ID, TRAFFIC_ID, WALKER_ID = 1000, 2000, 3000
ROUTE_LENGTH = 250.0  # m of route handed to the planner
ROUTE_BEHIND = 15.0  # m of it behind the ego at the moment a window is made
ROUTE_REFRESH = 60.0  # make a new window once the ego is this far into the current one
CLOSE_CALL = 1.0  # m, ego body to a pedestrian, while the ego is moving
CLOSE_CALL_RESET = 3.0  # the pedestrian must get this far away before the next close call counts
LIGHT_RANGE = 80.0  # m, lights farther than this from the ego are not reported


class Kind(StrEnum):
    COLLISION = "collision"  # ego body overlaps a vehicle or a pedestrian
    RED_LIGHT = "red light"  # ego front crossed a stop line while the light was red
    CLOSE_CALL = "close call"  # within ``CLOSE_CALL`` of a pedestrian, still moving


@dataclass(frozen=True)
class Incident:
    kind: Kind
    time: float
    lap: int  # 1-based
    detail: str


@dataclass(frozen=True)
class LapRecord:
    lap: int
    time: float  # seconds this lap took
    mean_speed: float
    collisions: int
    red_lights: int
    close_calls: int


class _Phase(StrEnum):
    WAIT = "wait"
    CROSS = "cross"
    DONE = "done"


@dataclass
class _WalkerState:
    spec: Walker
    phase: _Phase = _Phase.WAIT
    progress: float = 0.0  # metres walked across

    @property
    def lateral(self) -> float:
        sign = 1.0 if self.spec.to_lateral > self.spec.from_lateral else -1.0
        return self.spec.from_lateral + sign * self.progress

    @property
    def width(self) -> float:
        return abs(self.spec.to_lateral - self.spec.from_lateral)


@dataclass
class _CarState:
    spec: TrafficCar
    s: float
    v: float


@dataclass(frozen=True)
class _Obstacle:
    """What a traffic car must not run into: position, extent and speed along the track."""

    s: float
    lateral: float
    length: float
    width: float
    speed: float


@dataclass
class LoopSim:
    track: Track
    course: Course
    dt: float = DT
    time: float = 0.0
    incidents: list[Incident] = field(default_factory=list)
    laps: list[LapRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.course.validate(self.track)
        x, y, yaw = self.track.pose(self.course.ego_start_s, 0.0)
        self.state = BicycleState(x, y, yaw, self.course.ego_start_speed)
        self.cars = [_CarState(c, c.s, c.speed) for c in self.course.traffic]
        self.walkers = [_WalkerState(w) for w in self.course.walkers]
        self.step_count = 0
        self.progress = 0.0  # metres of track the ego has covered since the start (unwrapped, rear axle)
        s0, _ = self.track.project(x, y)
        self._s_prev = s0
        self._lap_start_time, self._lap_start_progress = 0.0, 0.0
        self._lap_counts = {k: 0 for k in Kind}
        self._route: Route | None = None
        self._route_start = 0.0
        self._in_contact: set[int] = set()
        self._near_ped: set[int] = set()
        self._parked_rects = [self._rect(*self.track.pose(p.s, p.lateral), p.length, p.width) for p in self.course.parked]
        self._front_prev = self._front_s()

    # --- ego geometry ---------------------------------------------------------------------------------------

    @staticmethod
    def _rect(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
        return rect_corners(x, y, yaw, length, width)

    def ego_corners(self) -> np.ndarray:
        cx, cy = ego_center(self.state, sc.EGO_WHEELBASE)
        return rect_corners(cx, cy, self.state.yaw, sc.EGO_LENGTH, sc.EGO_WIDTH)

    def ego_front(self) -> np.ndarray:
        d = 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE)
        return np.array([self.state.x + d * math.cos(self.state.yaw), self.state.y + d * math.sin(self.state.yaw)])

    def ego_s(self) -> float:
        return self.track.project(self.state.x, self.state.y)[0]

    def _front_s(self) -> float:
        return self.track.project(*self.ego_front())[0]

    @property
    def lap(self) -> int:
        """The lap in progress, 1-based."""
        return int(self.progress // self.track.length) + 1

    # --- entities in the world ------------------------------------------------------------------------------

    def car_pose(self, i: int) -> tuple[float, float, float]:
        c = self.cars[i]
        return self.track.pose(c.s, self.track.lane_lateral(c.spec.lane))

    def car_rect(self, i: int) -> np.ndarray:
        x, y, yaw = self.car_pose(i)
        return rect_corners(x, y, yaw, self.cars[i].spec.length, self.cars[i].spec.width)

    def walker_xy(self, i: int) -> np.ndarray:
        w = self.walkers[i]
        x, y, _ = self.track.pose(w.spec.s, w.lateral)
        return np.array([x, y])

    def walker_velocity(self, i: int) -> tuple[float, float]:
        w = self.walkers[i]
        if w.phase is not _Phase.CROSS:
            return 0.0, 0.0
        _, _, yaw = self.track.pose(w.spec.s, w.lateral)
        sign = 1.0 if w.spec.to_lateral > w.spec.from_lateral else -1.0
        return -math.sin(yaw) * sign * w.spec.speed, math.cos(yaw) * sign * w.spec.speed

    def signal_state(self, sig: Signal, t: float | None = None) -> TrafficLightState:
        phase = ((self.time if t is None else t) + sig.offset) % sig.cycle
        if phase < sig.green:
            return TrafficLightState.GREEN
        return TrafficLightState.YELLOW if phase < sig.green + sig.yellow else TrafficLightState.RED

    def stop_line(self, sig: Signal) -> np.ndarray:
        """Endpoints of the stop line, across all lanes."""
        right = self.track.xy_at(np.array([sig.s]), -0.5 * self.track.lane_width)[0]
        left = self.track.xy_at(np.array([sig.s]), (self.track.n_lanes - 0.5) * self.track.lane_width)[0]
        return np.array([right, left])

    def occluder_rects(self) -> list[tuple[int, np.ndarray]]:
        """(agent id, rectangle) of everything that blocks sight: parked cars and traffic."""
        return [(PARKED_ID + i, r) for i, r in enumerate(self._parked_rects)] + \
               [(TRAFFIC_ID + i, self.car_rect(i)) for i in range(len(self.cars))]

    def blocking_rects(self, sensor: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """``occluder_rects`` without any the sensor is inside: after a contact the ego can be overlapping a vehicle,
        and a sensor inside an occluder has no shadow to cast."""
        return [(i, r) for i, r in self.occluder_rects() if not point_in_convex(r, sensor)]

    def walker_visible(self, i: int, front: np.ndarray | None = None) -> bool:
        front = self.ego_front() if front is None else front
        xy = self.walker_xy(i)
        return all(is_visible(front, xy, rect, SENSOR_RANGE) for _, rect in self.blocking_rects(front))

    # --- what the planner sees ------------------------------------------------------------------------------

    def route(self) -> Route:
        """The ego's route: a window of lane 0 that moves round the loop with it (rebuilt every ``ROUTE_REFRESH`` m)."""
        s = self.ego_s()
        if self._route is None or self.track.gap_ahead(self._route_start, s) > ROUTE_REFRESH:
            self._route_start = self.track.wrap(s - ROUTE_BEHIND)
            self._route = self.track.window_route(self._route_start, ROUTE_LENGTH, lane=0)
        return self._route

    def world_model(self) -> WorldModel:
        s = self.state
        ego = EgoState(s.x, s.y, s.yaw, s.speed, sc.EGO_LENGTH, sc.EGO_WIDTH, sc.EGO_WHEELBASE)
        front = self.ego_front()
        agents: list[Agent] = []
        occluded: list[OccludedRegion] = []

        def add_vehicle(aid: int, rect: np.ndarray, x: float, y: float, yaw: float, length: float, width: float,
                        speed: float, static: bool) -> None:
            if math.hypot(x - front[0], y - front[1]) > SENSOR_RANGE + length:
                return
            agents.append(Agent(aid, AgentClass.VEHICLE, x, y, yaw, speed * math.cos(yaw), speed * math.sin(yaw),
                                length, width, static))
            shadow = None if point_in_convex(rect, front) else shadow_polygon(front, rect, SENSOR_RANGE)
            if shadow is not None:
                occluded.append(OccludedRegion(aid, shadow))

        for i, p in enumerate(self.course.parked):
            x, y, yaw = self.track.pose(p.s, p.lateral)
            add_vehicle(PARKED_ID + i, self._parked_rects[i], x, y, yaw, p.length, p.width, 0.0, True)
        for i, c in enumerate(self.cars):
            x, y, yaw = self.car_pose(i)
            add_vehicle(TRAFFIC_ID + i, self.car_rect(i), x, y, yaw, c.spec.length, c.spec.width, c.v, False)
        for i in range(len(self.walkers)):
            if self.walker_visible(i, front):
                xy, (vx, vy) = self.walker_xy(i), self.walker_velocity(i)
                yaw = math.atan2(vy, vx) if (vx or vy) else 0.0
                agents.append(Agent(WALKER_ID + i, AgentClass.PEDESTRIAN, float(xy[0]), float(xy[1]), yaw, vx, vy,
                                    2 * PED_RADIUS, 2 * PED_RADIUS, False))
        lights = tuple(TrafficLight(i, self.signal_state(sig), self.stop_line(sig))
                       for i, sig in enumerate(self.course.signals)
                       if math.hypot(*(self.stop_line(sig).mean(axis=0) - front)) <= LIGHT_RANGE)
        return WorldModel(self.time, ego, tuple(agents), tuple(occluded), lights)

    # --- stepping -------------------------------------------------------------------------------------------

    def step(self, cmd: ControlCommand, max_steer_angle: float) -> None:
        """Advance one ``dt``: the ego under ``cmd``, then the world, then log what happened."""
        self.state = step(self.state, accel_from_pedals(cmd.throttle, cmd.brake),
                          cmd.steer * max_steer_angle, sc.EGO_WHEELBASE, self.dt)
        self.time += self.dt
        self.step_count += 1
        self._advance_traffic()
        self._advance_walkers()
        self._update_progress()
        self._log_contacts()
        self._log_red_lights()

    def _advance_walkers(self) -> None:
        front_s = self._front_s()
        c = self.track.length
        for w in self.walkers:
            ahead = self.track.gap_ahead(front_s, w.spec.s)
            if w.phase is _Phase.WAIT and ahead <= w.spec.trigger_distance:
                w.phase = _Phase.CROSS
            elif w.phase is _Phase.CROSS:
                w.progress = min(w.progress + w.spec.speed * self.dt, w.width)
                if w.progress >= w.width:
                    w.phase = _Phase.DONE
            elif w.phase is _Phase.DONE and w.spec.trigger_distance + 60.0 < ahead < c - 5.0:
                w.phase, w.progress = _Phase.WAIT, 0.0  # the ego has gone by: back to the start for the next lap

    def _obstacles_for_traffic(self, skip: int) -> list[_Obstacle]:
        track = self.track
        ego_s, ego_lat = track.project(*ego_center(self.state, sc.EGO_WHEELBASE))
        obs = [_Obstacle(ego_s, ego_lat, sc.EGO_LENGTH, sc.EGO_WIDTH, self.state.speed)]
        obs += [_Obstacle(p.s, p.lateral, p.length, p.width, 0.0) for p in self.course.parked]
        obs += [_Obstacle(o.s, track.lane_lateral(o.spec.lane), o.spec.length, o.spec.width, o.v)
                for j, o in enumerate(self.cars) if j != skip]
        for w in self.walkers:
            if w.phase is not _Phase.WAIT:
                obs.append(_Obstacle(w.spec.s, w.lateral, 2 * PED_RADIUS, 2 * PED_RADIUS, 0.0))
        return obs

    def _advance_traffic(self) -> None:
        track = self.track
        new_v: list[float] = []
        for i, car in enumerate(self.cars):
            lat = track.lane_lateral(car.spec.lane)
            gap, lead_v = math.inf, car.v
            for o in self._obstacles_for_traffic(i):
                if abs(o.lateral - lat) >= 0.5 * (car.spec.width + o.width) + 0.3:
                    continue  # not in this car's way
                d = track.gap_ahead(car.s, o.s)
                if d > 150.0:
                    continue
                g = d - 0.5 * (car.spec.length + o.length)
                if g < gap:
                    gap, lead_v = g, o.speed
            for sig in self.course.signals:  # a red or yellow light is a stopped car at the stop line
                if self.signal_state(sig) is TrafficLightState.GREEN:
                    continue
                d = track.gap_ahead(car.s + 0.5 * car.spec.length, sig.s)
                if 0.0 < d < 150.0 and d < gap and car.v**2 / (2 * 4.0) < d:  # still able to stop comfortably
                    gap, lead_v = d, 0.0
            new_v.append(_idm(car.v, car.spec.speed, gap, lead_v))
        for car, v in zip(self.cars, new_v):
            car.v = v
            lat = track.lane_lateral(car.spec.lane)
            car.s = track.wrap(car.s + v * self.dt * track.lane_scale(car.s, lat))

    def _update_progress(self) -> None:
        s = self.ego_s()
        c = self.track.length
        self.progress += (s - self._s_prev + 0.5 * c) % c - 0.5 * c
        self._s_prev = s
        while self.progress >= (len(self.laps) + 1) * c:
            self._finish_lap()

    def _finish_lap(self) -> None:
        lap = len(self.laps) + 1
        dist = lap * self.track.length - self._lap_start_progress
        took = self.time - self._lap_start_time
        counts = self._lap_counts
        self.laps.append(LapRecord(lap, took, dist / took if took > 0 else 0.0, counts[Kind.COLLISION],
                                   counts[Kind.RED_LIGHT], counts[Kind.CLOSE_CALL]))
        self._lap_start_time, self._lap_start_progress = self.time, lap * self.track.length
        self._lap_counts = {k: 0 for k in Kind}

    def _log(self, kind: Kind, detail: str) -> None:
        self.incidents.append(Incident(kind, self.time, self.lap, detail))
        self._lap_counts[kind] += 1

    def _log_contacts(self) -> None:
        ego = self.ego_corners()
        touching: set[int] = set()
        for aid, rect in self.occluder_rects():
            if convex_polygons_overlap(ego, rect):
                touching.add(aid)
        near: set[int] = set()
        for i in range(len(self.walkers)):
            wid, d = WALKER_ID + i, distance_point_to_rect(ego, self.walker_xy(i))
            if d <= PED_RADIUS:
                touching.add(wid)
            if d < (CLOSE_CALL_RESET if wid in self._near_ped else CLOSE_CALL):
                near.add(wid)
        for aid in sorted(touching - self._in_contact):
            what = "pedestrian" if aid >= WALKER_ID else "parked car" if aid < TRAFFIC_ID else "traffic car"
            self._log(Kind.COLLISION, f"ego hit a {what} (id {aid})")
        for wid in sorted(near - self._near_ped - touching):
            if self.state.speed > 1.0:
                self._log(Kind.CLOSE_CALL, f"within {CLOSE_CALL:.0f} m of pedestrian {wid}")
        self._in_contact = touching
        self._near_ped = near

    def _log_red_lights(self) -> None:
        front = self._front_s()
        c = self.track.length
        for i, sig in enumerate(self.course.signals):
            before = (sig.s - self._front_prev + 0.5 * c) % c - 0.5 * c  # signed distance to the line, - once past it
            after = (sig.s - front + 0.5 * c) % c - 0.5 * c
            if before > 0.0 >= after and self.signal_state(sig) is TrafficLightState.RED:
                self._log(Kind.RED_LIGHT, f"ran the red light {i} at {self.state.speed:.1f} m/s")
        self._front_prev = front

    # --- summaries ------------------------------------------------------------------------------------------

    def totals(self) -> dict[Kind, int]:
        out = {k: 0 for k in Kind}
        for inc in self.incidents:
            out[inc.kind] += 1
        return out


def _idm(v: float, v0: float, gap: float, lead_v: float, a: float = 1.5, b: float = 2.5, t_gap: float = 1.2,
         s0: float = 2.0) -> float:
    """Intelligent Driver Model speed update, one ``DT`` step: follow a leader ``gap`` metres ahead at ``lead_v``."""
    free = 1.0 - (v / max(v0, 0.1)) ** 4
    if math.isfinite(gap):
        s_star = s0 + max(0.0, v * t_gap + v * (v - lead_v) / (2.0 * math.sqrt(a * b)))
        free -= (s_star / max(gap, 0.1)) ** 2
    return max(0.0, v + max(-9.0, a * free) * DT)


class LoopRunner:
    """One planner driving the ego round the loop: plan, control, step, log. The loop shared by the CLI and the window."""

    def __init__(self, sim: LoopSim, planner: Planner, predictor: Predictor | None = None,
                 controller: Controller | None = None):
        self.sim, self.planner, self.predictor = sim, planner, predictor
        self.ctl = controller or Controller()
        self.plan: PlannerCommand | None = None
        self.plan_ms = 0.0
        self.speeds: list[float] = []

    def advance(self) -> None:
        sim = self.sim
        world = sim.world_model()
        preds: tuple[PredictedTrajectory, ...] = self.predictor(world) if self.predictor else ()
        route = sim.route()
        started = time.perf_counter()
        plan = self.planner.plan(world, route, preds)
        self.plan_ms = 1000.0 * (time.perf_counter() - started)
        cmd = self.ctl.step(world.stamp, world.ego, route, plan.target_speed, sim.dt)
        self.plan = plan
        self.speeds.append(sim.state.speed)
        sim.step(cmd, self.ctl.max_steer_angle)

    def run_laps(self, laps: int, max_seconds: float = 1800.0) -> None:
        while len(self.sim.laps) < laps and self.sim.time < max_seconds:
            self.advance()


def format_laps(laps: Sequence[LapRecord]) -> str:
    rows = [f"{'lap':>4} {'time':>7} {'mean v':>7} {'collisions':>11} {'red lights':>11} {'close calls':>12}"]
    for r in laps:
        rows.append(f"{r.lap:>4} {r.time:>6.1f}s {r.mean_speed:>6.1f} {r.collisions:>11} {r.red_lights:>11} {r.close_calls:>12}")
    return "\n".join(rows)


__all__ = ["LoopSim", "LoopRunner", "Incident", "Kind", "LapRecord", "format_laps", "Course", "Signal", "Walker"]
