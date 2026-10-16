"""The toy sim: ego on the bicycle model, parked vehicles beside the road, hidden pedestrians.

Deterministic: the same parameters and controller give a bit-identical trajectory.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from av_core.control import BicycleState, Controller, accel_from_pedals, ego_center, step
from av_core.geometry import (
    SENSOR_RANGE,
    convex_polygons_overlap,
    densify,
    distance_point_to_rect,
    is_visible,
    rect_corners,
    shadow_polygon,
)
from av_core.protocols import Planner, Predictor
from av_core.sweep.record import Outcome
from av_core.types import Agent, AgentClass, ControlCommand, EgoState, OccludedRegion, PredictedTrajectory, Route, WorldModel
from av_sim_toy import scenario as sc
from av_sim_toy.manual import Direction, ManualPedestrian, STOPPED_EGO_SPEED
from av_sim_toy.scenario import ScenarioParams

OCCLUDER_ID, PED_ID = 1, 2
EXTRA_ID_BASE = 10  # extra parked vehicles are agents 10, 11, ...
EXTRA_PED_BASE = 20  # pedestrians after the primary one are agents 21, 22, ...
MOVING_BASE = 30  # moving vehicles are agents 30, 31, ...
MANUAL_PED_ID = 40  # the keyboard-driven pedestrian


def make_route() -> Route:
    return densify([[-10.0, 0.0], [250.0, 0.0]], sc.SPEED_LIMIT)


class ToySim:
    def __init__(self, params: ScenarioParams, dt: float = sc.DT, manual_ped: ManualPedestrian | None = None):
        self.params, self.dt = params, dt
        self.route = make_route()
        self.state = BicycleState(0.0, 0.0, 0.0, params.initial_speed)
        self.step_count = 0
        self._peds = params.pedestrians  # the primary one (if present) first, then the extras
        self.ped_trigger_steps: list[int | None] = [None] * len(self._peds)
        self.outcome: Outcome | None = None
        # moving vehicles: current [x, speed] each; they accelerate nowhere, they only brake once
        self._moving = params.moving_vehicles
        self.moving_state = [[mv.x, mv.speed] for mv in self._moving]
        # (agent id, vehicle, rectangle) for the primary occluder and every extra vehicle
        self._vehicles = [(OCCLUDER_ID if i == 0 else EXTRA_ID_BASE + i - 1, v,
                           rect_corners(v.x, v.y, 0.0, v.length, v.width)) for i, v in enumerate(params.vehicles)]
        # an optional pedestrian a person drives; it comes after the scripted ones (index ``len(params.pedestrians)``)
        self.manual = manual_ped
        if manual_ped is not None and manual_ped.blocked(manual_ped.x, manual_ped.y, self._parked_rects()):
            raise ValueError(f"manual pedestrian starts inside a vehicle or outside the world: ({manual_ped.x}, {manual_ped.y})")

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

    # pedestrians: index i runs over ``params.pedestrians``; the unindexed versions are the primary pedestrian

    @property
    def n_peds(self) -> int:
        return len(self._peds) + (self.manual is not None)

    def _is_manual(self, i: int) -> bool:
        return self.manual is not None and i == len(self._peds)

    def _parked_rects(self) -> list[np.ndarray]:
        return [rect for _, _, rect in self._vehicles]

    @property
    def ped_trigger_step(self) -> int | None:
        return self.ped_trigger_steps[0] if self._peds else None

    @ped_trigger_step.setter
    def ped_trigger_step(self, value: int | None) -> None:
        self.ped_trigger_steps[0] = value

    def ped_id(self, i: int) -> int:
        if self._is_manual(i):
            return MANUAL_PED_ID
        return PED_ID if i == 0 and self.params.ped_present else EXTRA_PED_BASE + i

    def ped_xy_at(self, i: int) -> np.ndarray:
        if self.manual is not None and self._is_manual(i):
            return np.array([self.manual.x, self.manual.y])
        ped, trig = self._peds[i], self.ped_trigger_steps[i]
        walked = 0.0 if trig is None else (self.step_count - trig) * self.dt * ped.speed
        walked = min(walked, 2.0 * abs(sc.PED_START_Y))  # stops on the far sidewalk
        return np.array([ped.x, ped.y0 - ped.side * walked])

    def peds_xy(self) -> np.ndarray:
        """(P, 2) pedestrian centres; a single NaN row when there are none."""
        if self.n_peds == 0:
            return np.full((1, 2), np.nan)
        return np.array([self.ped_xy_at(i) for i in range(self.n_peds)])

    def ped_xy(self) -> np.ndarray | None:
        return self.ped_xy_at(0) if self.params.ped_present else None

    def ped_walking_at(self, i: int) -> bool:
        if self.manual is not None and self._is_manual(i):
            return self.manual.speed > 0.05
        return self.ped_trigger_steps[i] is not None

    def ped_walking(self) -> bool:
        return self.params.ped_present and self.ped_walking_at(0)

    def ped_moving_at(self, i: int) -> bool:
        """Triggered and not yet at the far sidewalk (a driven pedestrian: whether it is walking)."""
        if self.manual is not None and self._is_manual(i):
            return self.manual.speed > 0.05
        return self.ped_walking_at(i) and abs(self.ped_xy_at(i)[1] - self._peds[i].y0) < 2.0 * abs(sc.PED_START_Y) - 1e-9

    def ped_moving(self) -> bool:
        return self.params.ped_present and self.ped_moving_at(0)

    # moving vehicles

    @property
    def n_moving(self) -> int:
        return len(self._moving)

    def moving_pose(self, j: int) -> tuple[float, float, float]:
        """(x, y, yaw) of moving vehicle ``j``."""
        mv = self._moving[j]
        return self.moving_state[j][0], mv.y, mv.yaw

    def moving_rect(self, j: int) -> np.ndarray:
        mv = self._moving[j]
        x, y, yaw = self.moving_pose(j)
        return rect_corners(x, y, yaw, mv.length, mv.width)

    def moving_poses(self) -> np.ndarray:
        """(M, 3) x, y, yaw of every moving vehicle; shape (0, 3) when there are none."""
        return np.array([self.moving_pose(j) for j in range(self.n_moving)]).reshape(-1, 3)

    def occluder_rects(self) -> list[tuple[int, np.ndarray]]:
        """(agent id, rectangle) of everything that blocks sight right now: parked and moving vehicles."""
        return [(i, rect) for i, _, rect in self._vehicles] + [(MOVING_BASE + j, self.moving_rect(j))
                                                              for j in range(self.n_moving)]

    def ped_visible_at(self, i: int) -> bool:
        front, xy = self.ego_front(), self.ped_xy_at(i)
        return all(is_visible(front, xy, rect, SENSOR_RANGE) for _, rect in self.occluder_rects())

    def peds_visible(self) -> np.ndarray:
        """(P,) bool; one False when there are no pedestrians."""
        if self.n_peds == 0:
            return np.zeros(1, dtype=bool)
        return np.array([self.ped_visible_at(i) for i in range(self.n_peds)], dtype=bool)

    def ped_visible(self) -> bool:
        return self.params.ped_present and self.ped_visible_at(0)

    # --- interface ------------------------------------------------------------------------

    def world_model(self) -> WorldModel:
        s = self.state
        ego = EgoState(s.x, s.y, s.yaw, s.speed, sc.EGO_LENGTH, sc.EGO_WIDTH, sc.EGO_WHEELBASE)
        agents = [Agent(i, AgentClass.VEHICLE, v.x, v.y, 0.0, 0.0, 0.0, v.length, v.width, True)
                  for i, v, _ in self._vehicles]
        for i, ped in enumerate(self._peds):
            if self.ped_visible_at(i):
                xy = self.ped_xy_at(i)
                vy = -ped.side * ped.speed if self.ped_moving_at(i) else 0.0
                agents.append(Agent(self.ped_id(i), AgentClass.PEDESTRIAN, xy[0], xy[1], -ped.side * math.pi / 2, 0.0, vy,
                                    2 * sc.PED_RADIUS, 2 * sc.PED_RADIUS, False))
        if self.manual is not None and self.ped_visible_at(len(self._peds)):
            m = self.manual
            agents.append(Agent(MANUAL_PED_ID, AgentClass.PEDESTRIAN, m.x, m.y, m.heading, m.vx, m.vy,
                                2 * sc.PED_RADIUS, 2 * sc.PED_RADIUS, False))
        for j, mv in enumerate(self._moving):
            vx = self.moving_state[j][1]
            agents.append(Agent(MOVING_BASE + j, AgentClass.VEHICLE, self.moving_pose(j)[0], mv.y, mv.yaw,
                                vx, 0.0, mv.length, mv.width, False))
        front = self.ego_front()
        shadows = [(i, shadow_polygon(front, rect, SENSOR_RANGE)) for i, rect in self.occluder_rects()]
        occluded = tuple(OccludedRegion(i, sh) for i, sh in shadows if sh is not None)
        return WorldModel(self.time, ego, tuple(agents), occluded, ())

    def step(self, cmd: ControlCommand, max_steer_angle: float, ped_input: Direction = (0, 0)) -> None:
        """Advance one dt under ``cmd`` and update the episode outcome. ``ped_input`` drives the manual pedestrian."""
        if self.done:
            raise RuntimeError("episode is over")
        self.state = step(self.state, accel_from_pedals(cmd.throttle, cmd.brake),
                          cmd.steer * max_steer_angle, sc.EGO_WHEELBASE, self.dt)
        self.step_count += 1
        for j, mv in enumerate(self._moving):  # advance the moving vehicles
            x, v = self.moving_state[j]
            if mv.brake_time is not None and self.time - self.dt >= mv.brake_time - 1e-9:
                floor = abs(mv.brake_to)
                v = math.copysign(max(floor, abs(v) - mv.brake_decel * self.dt), v) if abs(v) > floor else v
            self.moving_state[j] = [x + 0.5 * (self.moving_state[j][1] + v) * self.dt, v]
        for i, ped in enumerate(self._peds):
            if self.ped_trigger_steps[i] is None and ped.x - self.ego_front()[0] <= ped.trigger_distance:
                self.ped_trigger_steps[i] = self.step_count
        if self.manual is not None:
            obstacles = self._parked_rects()
            if self.state.speed < STOPPED_EGO_SPEED:  # nobody walks into a car that is standing still
                obstacles.append(self.ego_corners())
            self.manual.step(ped_input, self.dt, obstacles)
        self.outcome = self._check_end()

    def collided(self) -> bool:
        corners = self.ego_corners()
        return any(distance_point_to_rect(corners, self.ped_xy_at(i)) <= sc.PED_RADIUS for i in range(self.n_peds))

    def hit_vehicle(self) -> bool:
        """Whether the ego rectangle overlaps any vehicle, parked or moving."""
        ego = self.ego_corners()
        return any(convex_polygons_overlap(ego, rect) for _, rect in self.occluder_rects())

    def _check_end(self) -> Outcome | None:
        if self.collided() or self.hit_vehicle():
            return "collision"
        if self.ego_rear()[0] > self.params.far_x + sc.END_MARGIN:
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
    peds: np.ndarray  # (N, P, 2): every pedestrian's centre; one NaN column when there are none
    peds_visible: np.ndarray  # (N, P) bool
    target_speed: np.ndarray
    reason: list[str]
    throttle: np.ndarray
    brake: np.ndarray
    steer: np.ndarray
    outcome: Outcome
    final_ego: np.ndarray = field(default_factory=lambda: np.zeros(4))  # state after the last step
    final_peds: np.ndarray = field(default_factory=lambda: np.full((1, 2), np.nan))  # (P, 2)
    moving: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))  # (N, M, 3): x, y, yaw of each moving vehicle
    final_moving: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (M, 3)
    hit_vehicle: bool = False  # the episode ended with the ego hitting a vehicle

    @property
    def ped(self) -> np.ndarray:
        """(N, 2): the primary pedestrian's centre, NaN when there is none (``peds[:, 0]``)."""
        return self.peds[:, 0] if self.params.ped_present else np.full((len(self.t), 2), np.nan)

    @property
    def ped_visible(self) -> np.ndarray:
        """(N,) bool: whether the primary pedestrian is in view."""
        return self.peds_visible[:, 0] & self.params.ped_present

    @property
    def final_ped(self) -> np.ndarray:
        return self.final_peds[0] if self.params.ped_present else np.full(2, np.nan)


class EpisodeRunner:
    """One episode, one step at a time: the loop shared by ``run_episode`` and the live window.

    ``advance`` runs a full cycle (world model, predictions, plan, control, sim step) and logs the row. Because both
    callers use it, a recorded live session replays to the identical trajectory.
    """

    def __init__(self, params: ScenarioParams, planner: Planner, controller: Controller | None = None,
                 predictor: Predictor | None = None, dt: float = sc.DT, manual_ped: ManualPedestrian | None = None):
        self.params, self.dt = params, dt
        self.sim, self.ctl = ToySim(params, dt, manual_ped), controller or Controller()
        self.planner, self.predictor = planner, predictor
        self.plan = None
        self.plan_ms = 0.0  # wall time of the last plan() call, for the live window
        self._rows: dict[str, list] = {k: [] for k in ("t", "ego", "peds", "vis", "moving", "tgt", "reason", "thr", "brk", "steer")}

    @property
    def done(self) -> bool:
        return self.sim.done

    def advance(self, ped_input: Direction = (0, 0)) -> None:
        sim, rows = self.sim, self._rows
        world = sim.world_model()
        preds: tuple[PredictedTrajectory, ...] = self.predictor(world) if self.predictor else ()
        started = time.perf_counter()
        plan = self.planner.plan(world, sim.route, preds)
        self.plan_ms = 1000.0 * (time.perf_counter() - started)
        cmd = self.ctl.step(world.stamp, world.ego, sim.route, plan.target_speed, self.dt)
        self.plan = plan
        rows["t"].append(sim.time)
        rows["ego"].append([sim.state.x, sim.state.y, sim.state.yaw, sim.state.speed])
        rows["peds"].append(sim.peds_xy())
        rows["vis"].append(sim.peds_visible())
        rows["moving"].append(sim.moving_poses())
        rows["tgt"].append(plan.target_speed)
        rows["reason"].append(plan.reason)
        rows["thr"].append(cmd.throttle)
        rows["brk"].append(cmd.brake)
        rows["steer"].append(cmd.steer)
        sim.step(cmd, self.ctl.max_steer_angle, ped_input)

    def episode(self) -> Episode:
        sim, rows = self.sim, self._rows
        assert sim.outcome is not None  # only meaningful once the episode is done
        return Episode(
            params=self.params, dt=self.dt, t=np.array(rows["t"]), ego=np.array(rows["ego"]), peds=np.array(rows["peds"]),
            peds_visible=np.array(rows["vis"], dtype=bool), target_speed=np.array(rows["tgt"]),
            reason=rows["reason"], throttle=np.array(rows["thr"]), brake=np.array(rows["brk"]),
            steer=np.array(rows["steer"]), outcome=sim.outcome,
            final_ego=np.array([sim.state.x, sim.state.y, sim.state.yaw, sim.state.speed]),
            final_peds=sim.peds_xy(), moving=np.array(rows["moving"]), final_moving=sim.moving_poses(),
            hit_vehicle=sim.hit_vehicle(),
        )


def run_episode(params: ScenarioParams, planner: Planner, controller: Controller | None = None,
                predictor: Predictor | None = None, dt: float = sc.DT, manual_ped: ManualPedestrian | None = None,
                ped_inputs: Sequence[Direction] = ()) -> Episode:
    """Run one episode. ``planner.plan(world, route, predictions)``; ``predictor(world)`` is optional.

    ``manual_ped`` adds a driven pedestrian, fed ``ped_inputs`` one per step (standing still once they run out).
    """
    runner = EpisodeRunner(params, planner, controller, predictor, dt, manual_ped)
    k = 0
    while not runner.done:
        runner.advance(ped_inputs[k] if k < len(ped_inputs) else (0, 0))
        k += 1
    return runner.episode()
