import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from av_core.geometry import SENSOR_RANGE, is_visible, point_in_convex, rect_corners, shadow_polygon

SENSOR = (0.0, 0.0)
BOX = rect_corners(50.0, 0.0, 0.0, 6.0, 2.0)  # x in [47, 53], y in [-1, 1]


def test_rect_corners_ccw_and_yaw():
    c = rect_corners(1, 2, math.pi / 2, 4.0, 2.0)
    assert sorted(map(tuple, np.round(c, 9))) == [(0.0, 0.0), (0.0, 4.0), (2.0, 0.0), (2.0, 4.0)]
    area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    assert area == pytest.approx(8.0)  # positive: counter-clockwise


def test_directly_behind_is_hidden():
    near = rect_corners(20.0, 0.0, 0.0, 6.0, 2.0)
    assert not is_visible(SENSOR, (30.0, 0.0), near)
    assert not is_visible(SENSOR, (40.0, 0.5), near)


def test_beside_is_visible():
    near = rect_corners(20.0, 0.0, 0.0, 6.0, 2.0)
    assert is_visible(SENSOR, (30.0, 5.0), near)
    assert is_visible(SENSOR, (30.0, -5.0), near)
    assert is_visible(SENSOR, (10.0, 0.0), near)  # in front of it


def test_along_corner_ray_is_visible():
    near = rect_corners(20.0, 0.0, 0.0, 6.0, 2.0)  # corners (17|23, +-1)
    # ray through the top-right corner (23, 1) and onward; (17,1) is the extreme bearing
    for x in (23.0, 30.0, 40.0):
        assert is_visible(SENSOR, (x, x / 17.0), near)  # along the ray through corner (17, 1)
    # a hair inside the ray is hidden
    assert not is_visible(SENSOR, (30.0, 30.0 / 17.0 - 1e-3), near)


def test_grazing_edge_is_visible():
    c = rect_corners(20.0, 3.0, 0.0, 6.0, 2.0)  # bottom edge on y = 2
    assert is_visible((0.0, 2.0), (40.0, 2.0), c)  # runs along the bottom edge
    assert is_visible((0.0, 0.0), (20.0, 2.0), c)  # ends on the edge


def test_range_limit():
    assert is_visible(SENSOR, (50.0, 0.0), rect_corners(0, 30, 0, 1, 1))
    assert not is_visible(SENSOR, (50.1, 0.0), rect_corners(0, 30, 0, 1, 1))


def test_point_inside_occluder_not_visible():
    assert not is_visible(SENSOR, (50.0, 0.0), BOX)


def test_shadow_polygon_shape():
    near = rect_corners(20.0, 0.0, 0.0, 6.0, 2.0)
    q = shadow_polygon(SENSOR, near)
    assert q.shape == (4, 2)
    # two corners are occluder corners at the extreme bearings: (17, 1) and (17, -1)
    assert {tuple(p) for p in np.round(q[[0, 3]], 9)} == {(17.0, 1.0), (17.0, -1.0)}
    # the far points sit on the corner rays, at least out to the 50 m range
    for far, near_c in zip(q[[1, 2]], q[[0, 3]]):
        assert far[0] * near_c[1] - far[1] * near_c[0] == pytest.approx(0.0, abs=1e-9)
        assert math.hypot(*far) >= 50.0
    assert point_in_convex(q, (30.0, 0.0))
    assert not point_in_convex(q, (30.0, 5.0))


def test_shadow_out_of_range_is_none():
    assert shadow_polygon(SENSOR, rect_corners(80.0, 0.0, 0.0, 4.0, 2.0)) is None


def test_sensor_inside_occluder_raises():
    with pytest.raises(ValueError):
        shadow_polygon((50.0, 0.0), BOX)


def _dist_to_polygon(poly, p):
    p = np.asarray(p)
    best = math.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        if not ab.any():
            continue
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


occluders = st.builds(
    rect_corners,
    cx=st.floats(-40, 40), cy=st.floats(-40, 40), yaw=st.floats(-math.pi, math.pi),
    length=st.floats(1.0, 12.0), width=st.floats(0.5, 4.0),
)
unit = st.floats(0.0, 1.0)


@settings(max_examples=300, deadline=None)
@given(corners=occluders, theta=st.floats(-math.pi, math.pi), r=st.floats(0.5, SENSOR_RANGE - 1e-6))  # stay off the range edge, like the 1e-6 margin below
def test_property_shadow_hidden_and_rest_visible(corners, theta, r):
    assume(not point_in_convex(corners, SENSOR))
    q = shadow_polygon(SENSOR, corners)
    assume(q is not None)
    p = np.array([r * math.cos(theta), r * math.sin(theta)])
    margin = 1e-6
    inside_occluder = point_in_convex(corners, p)
    d_shadow, d_occ = _dist_to_polygon(q, p), _dist_to_polygon(corners, p)
    if point_in_convex(q, p) and d_shadow > margin:
        assert not is_visible(SENSOR, p, corners)  # in the shadow and in range: never visible
    elif not point_in_convex(q, p) and d_shadow > margin and not inside_occluder and d_occ > margin:
        assert is_visible(SENSOR, p, corners)  # outside shadow and occluder: always visible


@settings(max_examples=200, deadline=None)
@given(corners=occluders, u=unit, v=unit)
def test_property_points_targeted_at_shadow(corners, u, v):
    """Sample inside the shadow wedge directly so the 'hidden' branch is well exercised."""
    assume(not point_in_convex(corners, SENSOR))
    q = shadow_polygon(SENSOR, corners)
    assume(q is not None)
    # a convex combination of the quad's vertices lies inside the quad
    p = (1 - v) * ((1 - u) * q[0] + u * q[3]) + v * ((1 - u) * q[1] + u * q[2])
    assume(math.hypot(*p) <= 50.0 and not point_in_convex(corners, p, strict=False))
    assume(_dist_to_polygon(q, p) > 1e-6 and _dist_to_polygon(corners, p) > 1e-6)
    assert not is_visible(SENSOR, p, corners)
