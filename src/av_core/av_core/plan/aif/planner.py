"""Active-inference planner for the occluded-pedestrian situation.

Each model step (``DT``) the planner

1. turns the WorldModel into discrete observations (``observe``),
2. updates its belief over the pedestrian with pymdp (``inference.update_posterior_states``), carrying the
   belief forward in time so a pedestrian seen once is remembered,
3. scores every 4-step policy of {keep, ease off, brake} by expected free energy
   (``efe.expected_free_energy``, a batched version of pymdp's): G = -(risk + ambiguity), where risk is how far the predicted
   outcomes sit from the preferred ones and ambiguity is the expected uncertainty about the pedestrian that
   remains. The ambiguity term is what makes slowing down to look worthwhile,
4. executes the first action of the most probable action sequence as a target speed.

The rule-based limits (route, lead vehicle, lights) still apply on top, so this planner only changes
behaviour around occluders. When there is no occluder it returns exactly planner v0's command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymdp import inference, maths

from av_core.geometry import RouteFrame
from av_core.plan.aif.efe import expected_free_energy
from av_core.plan.aif.model import (
    ACCEL,
    DT,
    ND,
    NE,
    NP,
    NV,
    SPEED_CENTRES,
    Action,
    AIFParams,
    GenerativeModel,
    PedObs,
    Safety,
    build_model,
    ego_index,
    split_distance,
    split_speed,
)
from av_core.plan.aif.observe import (
    Hazard,
    ego_distance,
    find_hazard,
    pedestrian_observation,
    visibility_profile,
)
from av_core.plan.rule_based import RuleBasedPlanner
from av_core.types import PlannerCommand, PlannerReason, PredictedTrajectory, Route, WorldModel

@dataclass(frozen=True)
class StepDiagnostics:
    """Everything worth plotting about one replanning step."""

    stamp: float
    ped_obs: PedObs
    ped_belief: NDArray[np.float64]  # posterior over ``Ped`` before the planner acts
    distance: float  # m, front bumper to crossing point
    speed: float
    policies: NDArray[np.int64]  # (P, policy_len) action indices (the ego control factor)
    q_pi: NDArray[np.float64]  # posterior over policies
    G: NDArray[np.float64]  # negative expected free energy per policy
    risk: NDArray[np.float64]  # per policy, summed over the horizon and modalities
    ambiguity: NDArray[np.float64]
    action: Action
    target_speed: float


@dataclass
class AIFPlanner:
    params: AIFParams = field(default_factory=AIFParams)
    base: RuleBasedPlanner = field(default_factory=RuleBasedPlanner)
    record: bool = False  # keep a StepDiagnostics per replan (off for sweeps)
    diagnostics: list[StepDiagnostics] = field(default_factory=list, init=False)
    _models: dict[tuple[float, ...], GenerativeModel] = field(default_factory=dict, init=False, repr=False)
    _belief: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _band_weights: NDArray[np.float64] | None = field(default=None, init=False, repr=False)  # ego band last step
    _hazard_id: int | None = field(default=None, init=False, repr=False)
    _last_replan: float | None = field(default=None, init=False, repr=False)
    _held_action: Action = field(default=Action.KEEP, init=False, repr=False)
    _held_target: float = field(default=math.inf, init=False, repr=False)

    # --- Planner protocol ---------------------------------------------------------------------

    def plan(self, world: WorldModel, route: Route,
             predictions: tuple[PredictedTrajectory, ...] = ()) -> PlannerCommand:
        frame = self.base.frame_for(route)
        limits = self.base.limits(world, route, frame, predictions)
        reason = min(limits, key=limits.__getitem__)
        cap = limits[reason]

        ego = world.ego
        s0, _ = frame.project(ego.x, ego.y)
        s_front = s0 + 0.5 * (ego.length + ego.wheelbase)
        hazard = find_hazard(world, frame, s_front)
        if hazard is None:
            self._reset()
            return PlannerCommand(world.stamp, cap, reason)

        if hazard.occluder.id != self._hazard_id:
            self._reset()
            self._hazard_id = hazard.occluder.id
        if self._last_replan is None or world.stamp - self._last_replan >= DT - 1e-9:
            self._replan(world, frame, hazard, s_front)

        if self._held_action is not Action.KEEP and self._held_target < cap:
            return PlannerCommand(world.stamp, self._held_target, PlannerReason.OCCLUSION)
        return PlannerCommand(world.stamp, cap, reason)

    # --- one model step -----------------------------------------------------------------------

    def _reset(self) -> None:
        self._belief, self._band_weights, self._hazard_id, self._last_replan = None, None, None, None
        self._held_action, self._held_target = Action.KEEP, math.inf

    def _model_for(self, hazard: Hazard, frame: RouteFrame) -> GenerativeModel:
        vis = visibility_profile(hazard, frame)
        key = tuple(vis)
        if key not in self._models:
            self._models[key] = build_model(vis, self.params)
        return self._models[key]

    def _replan(self, world: WorldModel, frame: RouteFrame, hazard: Hazard, s_front: float) -> None:
        model = self._model_for(hazard, frame)
        speed = world.ego.speed
        d = ego_distance(hazard, s_front)

        # prior over the pedestrian: the model's prior at first, else last belief advanced by the elapsed steps
        # under the transition for the distance bands the ego was in
        if self._belief is None or self._last_replan is None or self._band_weights is None:
            prior_ped = model.D_ped
        else:
            steps = max(1, round((world.stamp - self._last_replan) / DT))
            B_now = np.tensordot(self._band_weights, model.B_ped, axes=1)  # (NP, NP), averaged over bands
            prior_ped = self._belief
            for _ in range(steps):
                prior_ped = B_now @ prior_ped
        prior_ego = np.zeros(NE)
        for di, wd in split_distance(d):
            for vi, wv in split_speed(speed):
                prior_ego[ego_index(di, vi)] += wd * wv

        ped_obs = pedestrian_observation(world, frame, hazard)
        speed_obs = int(np.argmin(np.abs(SPEED_CENTRES - speed)))
        obs = [int(ped_obs), int(speed_obs), int(Safety.OK)]
        prior = np.empty(1, dtype=object)
        prior[0] = np.outer(prior_ego, prior_ped).ravel()  # ego known, so the joint is a product
        qs = inference.update_posterior_states(model.A, obs, prior=prior)
        posterior = np.asarray(qs[0], dtype=np.float64).reshape(NE, NP)
        ped_belief = posterior.sum(axis=0)
        ped_belief /= ped_belief.sum()

        policies, risk, ambiguity = expected_free_energy(
            np.asarray(qs[0], dtype=np.float64), model, model.params.policy_len)
        G = -(risk + ambiguity)  # equals pymdp's update_posterior_policies G (tests/test_aif.py), computed in batch
        q_pi = maths.softmax(model.params.gamma * G)
        marginal = np.bincount(policies[:, 0], weights=q_pi, minlength=len(Action))  # first-action marginal
        action = Action(int(np.argmax(marginal)))

        v_new = float(np.clip(speed + ACCEL[action] * DT, 0.0, np.inf))
        self._belief = ped_belief
        self._band_weights = prior_ego.reshape(ND, NV).sum(axis=1)
        self._last_replan = world.stamp
        self._held_action, self._held_target = action, v_new if action is not Action.KEEP else math.inf

        if self.record:
            self.diagnostics.append(StepDiagnostics(
                stamp=world.stamp, ped_obs=ped_obs, ped_belief=self._belief.copy(), distance=d, speed=speed,
                policies=policies, q_pi=np.asarray(q_pi, dtype=np.float64).ravel(),
                G=G, risk=risk, ambiguity=ambiguity, action=action,
                target_speed=self._held_target))


__all__ = ["AIFPlanner", "StepDiagnostics"]
