"""The toy sim: ego on the bicycle model, a parked occluder, an optional hidden pedestrian.

Deterministic: the same parameters and controller give a bit-identical trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from av_core.control import BicycleState, Controller, accel_from_pedals, ego_center, step
from av_core.geometry import (
    SENSOR_RANGE,
    densify,
    distance_point_to_rect,
    is_visible,
    rect_corners,
    shadow_polygon,
)
from av_core.types import Agent, ControlCommand, EgoState, OccludedRegion, PredictedTrajectory, Route, WorldModel
from av_sim_toy import scenario as sc
from av_sim_toy.scenario import ScenarioParams

OCCLUDER_ID, PED_ID = 1, 2


def make_route() -> Route:
    return densify([[-10.0, 0.0], [250.0, 0.0]], sc.SPEED_LIMIT)


class ToySim:
    def __init__(self, params: ScenarioParams, dt: float = sc.DT):
        self.params, self.dt = params, dt
        self.route = make_route()
        self.state = BicycleState(0.0, 0.0, 0.0, params.initial_speed)
        self.step_count = 0
        self.ped_trigger_step: int | None = None
        self.outcome: str | None = None  # 'collision' | 'finished' | 'timeout'
        self._occluder = rect_corners(params.occluder_x, sc.PARKING_CENTER_Y, 0.0,
                                      params.occluder_length, sc.OCCLUDER_WIDTH)

    # --- geometry -------------------------------------------------------------------------

    @property
    def time(self) -> float:
        return self.step_count * self.dt

    @property
    def done(self) -> bool:
        return self.outcome is not None

    def ego_corners(self) -> np.ndarray:
        cx, cy = ego_center(self.state, sc.EGO_WHEELBASE)
        return rect_corners(cx, cy, self.state.yaw, sc.EGO_LENGTH, sc.EGO_WIDTH)

    def ego_front(self) -> np.ndarray:
        """Front-centre point: the sensor position and the reference for 'distance short of x'."""
        d = 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE)
        return np.array([self.state.x + d * math.cos(self.state.yaw), self.state.y + d * math.sin(self.state.yaw)])

    def ego_rear(self) -> np.ndarray:
        d = 0.5 * (sc.EGO_LENGTH - sc.EGO_WHEELBASE)
        return np.array([self.state.x - d * math.cos(self.state.yaw), self.state.y - d * math.sin(self.state.yaw)])

    def ped_xy(self) -> np.ndarray | None:
        p = self.params
        if not p.ped_present:
            return None
        walked = 0.0 if self.ped_trigger_step is None else (self.step_count - self.ped_trigger_step) * self.dt * p.ped_speed
        return np.array([p.ped_x, sc.PED_START_Y + walked])

    def ped_walking(self) -> bool:
        return self.ped_trigger_step is not None

    def ped_visible(self) -> bool:
        xy = self.ped_xy()
        return xy is not None and is_visible(self.ego_front(), xy, self._occluder, SENSOR_RANGE)

    # --- interface ------------------------------------------------------------------------

    def world_model(self) -> WorldModel:
        s, p = self.state, self.params
        ego = EgoState(s.x, s.y, s.yaw, s.speed, sc.EGO_LENGTH, sc.EGO_WIDTH, sc.EGO_WHEELBASE)
        agents = [Agent(OCCLUDER_ID, "vehicle", p.occluder_x, sc.PARKING_CENTER_Y, 0.0, 0.0, 0.0,
                        p.occluder_length, sc.OCCLUDER_WIDTH, True)]
        if self.ped_visible():
            xy = self.ped_xy()
            vy = p.ped_speed if self.ped_walking() else 0.0
            agents.append(Agent(PED_ID, "pedestrian", xy[0], xy[1], math.pi / 2, 0.0, vy,
                                2 * sc.PED_RADIUS, 2 * sc.PED_RADIUS, False))
        shadow = shadow_polygon(self.ego_front(), self._occluder, SENSOR_RANGE)
        occluded = () if shadow is None else (OccludedRegion(OCCLUDER_ID, shadow),)
        return WorldModel(self.time, ego, tuple(agents), occluded, "none", None)

    def step(self, cmd: ControlCommand, max_steer_angle: float) -> None:
        """Advance one dt under ``cmd`` and update the episode outcome."""
        if self.done:
            raise RuntimeError("episode is over")
        self.state = step(self.state, accel_from_pedals(cmd.throttle, cmd.brake),
                          cmd.steer * max_steer_angle, sc.EGO_WHEELBASE, self.dt)
        self.step_count += 1
        p = self.params
        if p.ped_present and self.ped_trigger_step is None:
            if p.ped_x - self.ego_front()[0] <= p.ped_trigger_distance:
                self.ped_trigger_step = self.step_count
        self.outcome = self._check_end()

    def collided(self) -> bool:
        xy = self.ped_xy()
        return xy is not None and distance_point_to_rect(self.ego_corners(), xy) <= sc.PED_RADIUS

    def _check_end(self) -> str | None:
        if self.collided():
            return "collision"
        if self.ego_rear()[0] > self._occluder[:, 0].max() + sc.END_MARGIN:
            return "finished"
        if self.time >= sc.MAX_TIME - 1e-9:
            return "timeout"
        return None


@dataclass
class Episode:
    """Per-step log of one run. Row k is the state at the start of step k, before the command."""

    params: ScenarioParams
    dt: float
    t: np.ndarray
    ego: np.ndarray  # (N, 4): rear-axle x, y, yaw, speed
    ped: np.ndarray  # (N, 2): pedestrian centre, NaN when absent
    ped_visible: np.ndarray  # (N,) bool
    target_speed: np.ndarray
    reason: list[str]
    throttle: np.ndarray
    brake: np.ndarray
    steer: np.ndarray
    outcome: str
    final_ego: np.ndarray = field(default_factory=lambda: np.zeros(4))  # state after the last step
    final_ped: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))


def run_episode(params: ScenarioParams, planner, controller: Controller | None = None,
                predictor=None, dt: float = sc.DT) -> Episode:
    """Run one episode. ``planner.plan(world, route, predictions)``; ``predictor(world)`` is optional."""
    sim, ctl = ToySim(params, dt), controller or Controller()
    rows: dict[str, list] = {k: [] for k in ("t", "ego", "ped", "vis", "tgt", "reason", "thr", "brk", "str")}
    while not sim.done:
        world = sim.world_model()
        preds: tuple[PredictedTrajectory, ...] = predictor(world) if predictor else ()
        plan = planner.plan(world, sim.route, preds)
        cmd = ctl.step(world.stamp, world.ego, sim.route, plan.target_speed, dt)
        ped = sim.ped_xy()
        rows["t"].append(sim.time)
        rows["ego"].append([sim.state.x, sim.state.y, sim.state.yaw, sim.state.speed])
        rows["ped"].append([np.nan, np.nan] if ped is None else ped)
        rows["vis"].append(sim.ped_visible())
        rows["tgt"].append(plan.target_speed)
        rows["reason"].append(plan.reason)
        rows["thr"].append(cmd.throttle)
        rows["brk"].append(cmd.brake)
        rows["str"].append(cmd.steer)
        sim.step(cmd, ctl.max_steer_angle)
    ped_end = sim.ped_xy()
    return Episode(
        params=params, dt=dt, t=np.array(rows["t"]), ego=np.array(rows["ego"]), ped=np.array(rows["ped"]),
        ped_visible=np.array(rows["vis"], dtype=bool), target_speed=np.array(rows["tgt"]),
        reason=rows["reason"], throttle=np.array(rows["thr"]), brake=np.array(rows["brk"]),
        steer=np.array(rows["str"]), outcome=sim.outcome,
        final_ego=np.array([sim.state.x, sim.state.y, sim.state.yaw, sim.state.speed]),
        final_ped=np.full(2, np.nan) if ped_end is None else ped_end,
    )
