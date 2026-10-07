import numpy as np
import pytest

from av_core.predict import ConstantVelocityPredictor
from av_core.types import Agent, EgoState, WorldModel

EGO = EgoState(0, 0, 0, 10, 4.7, 1.9, 2.9)


def world(*agents):
    return WorldModel(2.0, EGO, tuple(agents), (), "none", None)


def agent(id=1, x=0.0, y=0.0, vx=0.0, vy=0.0, static=False):
    return Agent(id, "pedestrian", x, y, 0.0, vx, vy, 0.6, 0.6, static)


def test_horizon_and_steps():
    (traj,) = ConstantVelocityPredictor()(world(agent()))
    assert len(traj.t) == 31
    assert traj.t[0] == 0.0 and traj.t[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(traj.t), 0.1)
    assert traj.prob == 1.0


def test_straight_line_cases_exact():
    p = ConstantVelocityPredictor()
    cases = [(5.0, -4.5, 0.0, 1.4), (-3.0, 2.0, 8.0, 0.0), (10.0, 10.0, -2.0, -1.0), (0.0, 0.0, 0.0, 0.0)]
    for x, y, vx, vy in cases:
        (traj,) = p(world(agent(x=x, y=y, vx=vx, vy=vy)))
        for t, (px, py) in zip(traj.t, traj.xy):
            assert px == pytest.approx(x + vx * t, abs=1e-12)
            assert py == pytest.approx(y + vy * t, abs=1e-12)


def test_one_per_agent_in_order_with_ids():
    trajs = ConstantVelocityPredictor()(world(agent(id=7), agent(id=3, x=5.0), agent(id=9, static=True)))
    assert [t.agent_id for t in trajs] == [7, 3, 9]


def test_static_agent_stays_put():
    (traj,) = ConstantVelocityPredictor()(world(agent(x=12.0, y=-2.75, static=True)))
    assert np.all(traj.xy == [12.0, -2.75])


def test_empty_world_and_custom_horizon():
    assert ConstantVelocityPredictor()(world()) == ()
    (traj,) = ConstantVelocityPredictor(horizon=1.0, dt=0.25)(world(agent(vx=4.0)))
    assert len(traj.t) == 5 and traj.xy[-1, 0] == pytest.approx(4.0)
