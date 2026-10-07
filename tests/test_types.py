import math

import numpy as np
import pytest

from av_core.types import (
    Agent,
    ControlCommand,
    EgoState,
    OccludedRegion,
    PlannerCommand,
    PredictedTrajectory,
    Route,
    WorldModel,
)

EGO = dict(x=0.0, y=0.0, yaw=0.0, speed=5.0, length=4.7, width=1.9, wheelbase=2.9)
AGENT = dict(id=1, cls="vehicle", x=10.0, y=-2.75, yaw=0.0, vx=0.0, vy=0.0,
             length=4.5, width=2.0, is_static=True)


def make_world(stop_line=None):
    return WorldModel(
        stamp=1.5,
        ego=EgoState(**EGO),
        agents=(Agent(**AGENT), Agent(**{**AGENT, "id": 2, "cls": "pedestrian", "is_static": False})),
        occluded=(OccludedRegion(1, [[0, 0], [50, 0], [50, -10]]),),
        light="red",
        stop_line=stop_line,
    )


ALL = [
    EgoState(**EGO),
    Agent(**AGENT),
    OccludedRegion(3, [[0, 0], [1, 0], [1, 1]]),
    make_world(),
    make_world(stop_line=[20.0, 0.0]),
    Route([[0, 0], [0.5, 0], [1, 0]], 13.9),
    PredictedTrajectory(4, [0.1, 0.2, 0.3], [[0, 0], [1, 0], [2, 0]], 0.5),
    PlannerCommand(1.0, 8.0, "lead"),
    ControlCommand(1.0, 0.2, 0.0, -0.3),
]


@pytest.mark.parametrize("obj", ALL, ids=lambda o: type(o).__name__)
def test_round_trip(obj):
    d = obj.to_dict()
    back = type(obj).from_dict(d)
    assert back == obj
    assert back.to_dict() == d


def test_arrays_are_float64_and_read_only():
    r = Route([[0, 0], [1, 0]], 10)
    assert r.points.dtype == np.float64
    with pytest.raises(ValueError):
        r.points[0, 0] = 1.0


def test_input_array_is_copied():
    src = np.array([[0.0, 0.0], [1.0, 0.0]])
    r = Route(src, 10)
    src[0, 0] = 99
    assert r.points[0, 0] == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        EgoState(**{**EGO, "x": bad})
    with pytest.raises(ValueError):
        Agent(**{**AGENT, "vy": bad})
    with pytest.raises(ValueError):
        PlannerCommand(bad, 1.0, "route")
    with pytest.raises(ValueError):
        ControlCommand(0.0, bad, 0.0, 0.0)
    with pytest.raises(ValueError):
        Route([[0, 0], [bad, 0]], 10)
    with pytest.raises(ValueError):
        OccludedRegion(1, [[0, 0], [1, 0], [bad, 1]])
    with pytest.raises(ValueError):
        PredictedTrajectory(1, [0.1, bad], [[0, 0], [1, 1]])


def test_rejects_bad_ranges():
    for kw in ({"speed": -0.1}, {"length": 0.0}, {"width": -1.0}, {"wheelbase": 0.0}):
        with pytest.raises(ValueError):
            EgoState(**{**EGO, **kw})
    with pytest.raises(ValueError):
        Agent(**{**AGENT, "cls": "truck"})
    with pytest.raises(ValueError):
        Agent(**{**AGENT, "id": 1.5})
    with pytest.raises(ValueError):
        PlannerCommand(0.0, -1.0, "route")
    with pytest.raises(ValueError):
        PlannerCommand(0.0, 1.0, "because")
    for kw in ({"throttle": 1.1}, {"brake": -0.1}, {"steer": 1.5}, {"steer": -1.5}):
        with pytest.raises(ValueError):
            ControlCommand(**{"stamp": 0.0, "throttle": 0.0, "brake": 0.0, "steer": 0.0, **kw})
    with pytest.raises(ValueError):
        Route([[0, 0]], 10)
    with pytest.raises(ValueError):
        Route([[0, 0], [1, 0]], 0.0)
    with pytest.raises(ValueError):
        Route([[0, 0, 0], [1, 0, 0]], 10)
    with pytest.raises(ValueError):
        OccludedRegion(1, [[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        PredictedTrajectory(1, [0.2, 0.1], [[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        PredictedTrajectory(1, [0.1, 0.2], [[0, 0]])
    with pytest.raises(ValueError):
        PredictedTrajectory(1, [0.1], [[0, 0]], prob=1.5)


def test_world_model_validation():
    ego = EgoState(**EGO)
    with pytest.raises(ValueError):
        WorldModel(0.0, ego, (), (), "blue", None)
    with pytest.raises(ValueError):
        WorldModel(0.0, ego, (Agent(**AGENT), Agent(**AGENT)), (), "none", None)
    with pytest.raises(ValueError):
        WorldModel(0.0, ego, (), (), "none", [1.0, 2.0, 3.0])


def test_frozen():
    with pytest.raises(AttributeError):
        EgoState(**EGO).x = 1.0  # type: ignore[misc]


def test_equality_distinguishes_values_and_types():
    assert PlannerCommand(0.0, 1.0, "route") != PlannerCommand(0.0, 2.0, "route")
    assert Route([[0, 0], [1, 0]], 5) != Route([[0, 0], [2, 0]], 5)
    assert PlannerCommand(0.0, 1.0, "route") != ControlCommand(0.0, 0.0, 0.0, 0.0)
