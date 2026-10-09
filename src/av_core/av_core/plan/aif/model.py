"""The generative model of the occluded-pedestrian situation, as pymdp arrays A, B, C, D.

Everything here is a starting guess for you to rewrite: the discretisation, the dynamics, the likelihoods and
above all the preferences C. pymdp softmaxes each ``C[m]`` into a distribution over that modality's outcomes,
so only the differences within a modality matter, and their scale sets how much a collision outweighs a
slow approach.

Hidden state (one factor, 8 x 8 x 4 = 256 states, indexed ``(distance band, speed band, pedestrian)``)
    ego         distance band to the crossing point and speed band; known to the ego, moved by the action
    pedestrian  none | waiting (not yet crossing, usually hidden) | crossing | cleared

The two are one factor because pymdp 0.0.7 cannot make one factor's transition depend on another, and the
pedestrian steps out as the ego closes in (``p_cross`` depends on the distance band), which is how the toy
sim triggers it. That is an assumption about the scenario; CARLA's triggers may differ.

Observation modalities
    0  pedestrian: not seen | seen waiting | seen crossing. A waiting pedestrian is seen only from distance
       bands where the geometry gives line of sight (``visible``): the part of the model that makes some
       actions informative.
    1  ego speed band (what the ego feels), carrying the preference for making progress
    2  safety: ok | near miss | collision, a function of where the ego is when the pedestrian is crossing

Actions (control factor 0): keep, ease off, brake. One model step is ``DT`` seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

DT = 0.5  # s per model step
# Distance bands, by centre, in metres *before* the crossing point (negative = past it). Each is 8 m wide, so
# the ego (<= 14 m/s, 7 m per step) cannot skip a band; the band at 0 is the zone where the ego is level with
# the pedestrian's path, and the one at 8 is the approach to it.
DIST_CENTRES: NDArray[np.float64] = np.array([48.0, 40.0, 32.0, 24.0, 16.0, 8.0, 0.0, -8.0])
SPEED_CENTRES: NDArray[np.float64] = np.arange(0.0, 15.0, 2.0)  # m/s
ND, NV = len(DIST_CENTRES), len(SPEED_CENTRES)
NE = ND * NV  # joint ego states
NP = 4  # pedestrian states, see ``Ped``
NS = NE * NP  # hidden states of the single world factor
ZONE = int(np.argmin(np.abs(DIST_CENTRES)))
APPROACH = ZONE - 1


class Action(IntEnum):
    KEEP = 0  # head for the route's speed limit
    EASE_OFF = 1
    BRAKE = 2


# Acceleration each action asks for, m/s^2. Also the speed change the planner commands per step.
ACCEL: dict[Action, float] = {Action.KEEP: 1.0, Action.EASE_OFF: -1.5, Action.BRAKE: -4.0}


class Ped(IntEnum):
    NONE = 0
    WAITING = 1
    CROSSING = 2
    CLEARED = 3


class PedObs(IntEnum):
    NOT_SEEN = 0
    SEEN_WAITING = 1
    SEEN_CROSSING = 2


class Safety(IntEnum):
    OK = 0
    NEAR = 1
    COLLISION = 2


def ego_index(d: int, v: int) -> int:
    return d * NV + v


def state_index(e: int, ped: int) -> int:
    return e * NP + ped


@dataclass(frozen=True)
class AIFParams:
    """Model parameters. Starting guesses, not results."""

    prior_present: float = 0.5  # P(a pedestrian is waiting behind the occluder); the sampler uses 0.5
    # per step, in each distance band (``DIST_CENTRES``): the chance a waiting pedestrian starts to cross. High
    # while the ego closes in from ~35 m to ~10 m (the toy's trigger window), none once it is on top of it.
    p_cross: tuple[float, ...] = (0.02, 0.02, 0.2, 0.2, 0.2, 0.05, 0.0, 0.0)
    p_clear: float = 0.2  # per step, a crossing pedestrian is out of the lane (mean 2.5 s, a walk across ~3.5 m)
    p_detect: float = 0.95  # a pedestrian in line of sight is reported
    # Preferences: relative log-probabilities of outcomes within each modality (softmaxed by pymdp)
    # From a small grid on tuning seeds 0-59 (see the PR): speed preferences much weaker than the safety ones, or
    # braking to a stop for the horizon costs more than hitting the pedestrian.
    c_safety: tuple[float, float, float] = (0.0, -8.0, -32.0)  # ok, near miss, collision
    c_speed: tuple[float, ...] = (-1.0, -0.75, -0.5, -0.3, -0.15, -0.05, 0.0, 0.0)  # per SPEED_CENTRES band
    gamma: float = 16.0  # policy precision
    policy_len: int = 6  # steps, i.e. 3 s of lookahead: 33 m at 11 m/s, enough to see the zone from outside it


@dataclass(frozen=True)
class GenerativeModel:
    """pymdp arrays: ``A[m][o, s]``, ``B[0][s', s, a]``, ``C[m][o]``; ``s`` indexes ``state_index(ego, ped)``."""

    A: np.ndarray  # object array, 3 modalities
    B: np.ndarray  # object array, 1 factor
    C: np.ndarray  # object array, 3 modalities
    D_ped: NDArray[np.float64]  # prior over the pedestrian
    B_ped: NDArray[np.float64]  # (ND, NP, NP): pedestrian transition given the distance band
    params: AIFParams
    visible: NDArray[np.float64]  # (ND,) line of sight to a waiting pedestrian from each distance band

    @property
    def num_states(self) -> list[int]:
        return [NS]

    @property
    def num_controls(self) -> list[int]:
        return [len(Action)]


def _split(value: float, centres: NDArray[np.float64]) -> list[tuple[int, float]]:
    """Linear split of ``value`` between its two neighbouring band centres (clamped at the ends)."""
    if value <= centres[0]:
        return [(0, 1.0)]
    if value >= centres[-1]:
        return [(len(centres) - 1, 1.0)]
    j = int(np.searchsorted(centres, value, side="right")) - 1
    w = (value - centres[j]) / (centres[j + 1] - centres[j])
    return [(j, 1.0 - w), (j + 1, w)]


def _split_desc(value: float, centres: NDArray[np.float64]) -> list[tuple[int, float]]:
    """``_split`` for centres in descending order (the distance bands)."""
    return [(len(centres) - 1 - j, w) for j, w in _split(value, centres[::-1])]


def split_distance(d: float) -> list[tuple[int, float]]:
    return _split_desc(d, DIST_CENTRES)


def split_speed(v: float) -> list[tuple[int, float]]:
    return _split(v, SPEED_CENTRES)


def build_B_ego() -> NDArray[np.float64]:
    """B[s', s, a]: speed moves by the action's acceleration, distance by the mean speed, both split between
    neighbouring bands so slow movement is not rounded away."""
    B = np.zeros((NE, NE, len(Action)))
    for a in Action:
        for d in range(ND):
            for v in range(NV):
                v_new = float(np.clip(SPEED_CENTRES[v] + ACCEL[a] * DT, 0.0, SPEED_CENTRES[-1]))
                d_new = DIST_CENTRES[d] - 0.5 * (SPEED_CENTRES[v] + v_new) * DT
                if d == ND - 1:  # past the crossing point: stay there
                    d_new = DIST_CENTRES[d]
                for dj, wd in split_distance(d_new):
                    for vj, wv in split_speed(v_new):
                        B[ego_index(dj, vj), ego_index(d, v), a] += wd * wv
    return B


def build_B_ped(p: AIFParams) -> NDArray[np.float64]:
    """B_ped[d, p', p]: the pedestrian's transition while the ego is in distance band ``d``."""
    B = np.zeros((ND, NP, NP))
    for d in range(ND):
        B[d, Ped.NONE, Ped.NONE] = 1.0
        B[d, Ped.WAITING, Ped.WAITING] = 1.0 - p.p_cross[d]
        B[d, Ped.CROSSING, Ped.WAITING] = p.p_cross[d]
        B[d, Ped.CROSSING, Ped.CROSSING] = 1.0 - p.p_clear
        B[d, Ped.CLEARED, Ped.CROSSING] = p.p_clear
        B[d, Ped.CLEARED, Ped.CLEARED] = 1.0
    return B


def build_B(B_ego: NDArray[np.float64], B_ped: NDArray[np.float64]) -> NDArray[np.float64]:
    """B[(e', p'), (e, p), a] = B_ego[e', e, a] * B_ped[band(e), p', p]."""
    band_of = np.arange(NE) // NV
    # (e', e, a) x (e, p', p) -> (e', p', e, p, a)
    B = np.einsum("fea,eqp->fqepa", B_ego, B_ped[band_of])
    return B.reshape(NS, NS, len(Action))


def build_A(p: AIFParams, visible: NDArray[np.float64]) -> np.ndarray:
    A_ped = np.zeros((len(PedObs), NE, NP))
    A_speed = np.zeros((NV, NE, NP))
    A_safety = np.zeros((len(Safety), NE, NP))
    for d in range(ND):
        for v in range(NV):
            e = ego_index(d, v)
            A_speed[v, e, :] = 1.0
            # pedestrian: only a crossing or a visible waiting pedestrian can be seen
            A_ped[PedObs.NOT_SEEN, e, Ped.NONE] = 1.0
            A_ped[PedObs.NOT_SEEN, e, Ped.CLEARED] = 1.0
            see = p.p_detect * visible[d]
            A_ped[PedObs.SEEN_WAITING, e, Ped.WAITING] = see
            A_ped[PedObs.NOT_SEEN, e, Ped.WAITING] = 1.0 - see
            A_ped[PedObs.SEEN_CROSSING, e, Ped.CROSSING] = p.p_detect
            A_ped[PedObs.NOT_SEEN, e, Ped.CROSSING] = 1.0 - p.p_detect
            # safety: matters only while the pedestrian is in the lane
            speed_frac = min(SPEED_CENTRES[v], 8.0) / 8.0
            hit = near = 0.0
            if d == ZONE:
                hit = 0.1 + 0.9 * speed_frac  # level with the path: even a slow ego is close
            elif d == APPROACH:
                near = 0.5 * speed_frac  # too fast to stop short of the path
            A_safety[:, e, :] = np.array([1.0, 0.0, 0.0])[:, None]
            A_safety[:, e, Ped.CROSSING] = [1.0 - hit - near, near, hit]
    return _obj([m.reshape(m.shape[0], NS) for m in (A_ped, A_speed, A_safety)])


def _obj(arrays: list[NDArray[np.float64]]) -> np.ndarray:
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def build_model(visible: NDArray[np.float64], params: AIFParams | None = None) -> GenerativeModel:
    """``visible[d]`` is 1 where a waiting pedestrian behind the occluder is in line of sight from distance
    band ``d`` and 0 where it is hidden (see ``observe.visibility_profile``)."""
    p = params or AIFParams()
    vis = np.asarray(visible, dtype=np.float64)
    C = _obj([np.zeros(len(PedObs)), np.asarray(p.c_speed, dtype=np.float64), np.asarray(p.c_safety, dtype=np.float64)])
    B_ped = build_B_ped(p)
    B = _obj([build_B(build_B_ego(), B_ped)])
    # a little mass everywhere, so a surprising sighting (a pedestrian already crossing) can still be explained
    D_ped = 0.99 * np.array([1.0 - p.prior_present, p.prior_present, 0.0, 0.0]) + 0.01 / NP
    return GenerativeModel(A=build_A(p, vis), B=B, C=C, D_ped=D_ped, B_ped=B_ped, params=p, visible=vis)
