"""Matplotlib renderer: episode log -> MP4 (or a single frame).

Top-down, camera follows the ego. Shows the road, the parked occluder, the occluded (shadow)
region, ego, each pedestrian (solid when visible, dashed outline when hidden) and the planner's
target speed with its reason.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from av_core.geometry import SENSOR_RANGE, is_visible, rect_corners, shadow_polygon  # noqa: E402
from av_sim_toy import scenario as sc  # noqa: E402
from av_sim_toy.sim import Episode  # noqa: E402

VIEW_BEHIND, VIEW_AHEAD = 15.0, 75.0  # m of road shown behind / ahead of the ego front
Y_LIM = (-8.0, 8.0)
COLORS = dict(road="#d9d9d9", parking="#bdbdbd", walk="#eeeeee", ego="#1f77b4", occluder="#555555",
              shadow="#f2a900", ped="#d62728", edge="#222222", moving="#7a9e7e")


@dataclass(frozen=True)
class _Frames:
    t: np.ndarray
    ego: np.ndarray
    ped: np.ndarray  # (K, P, 2)
    moving: np.ndarray  # (K, M, 3): x, y, yaw
    target: np.ndarray
    reason: list[str]


def _frames(ep: Episode) -> _Frames:
    """Per-frame arrays: the logged rows plus the final state, so the last frame shows the outcome."""
    return _Frames(
        t=np.append(ep.t, ep.t[-1] + ep.dt),
        ego=np.vstack([ep.ego, ep.final_ego]),
        ped=np.concatenate([ep.peds, ep.final_peds[None]], axis=0),
        moving=np.concatenate([ep.moving, ep.final_moving[None]], axis=0) if ep.moving.size else
        np.zeros((len(ep.t) + 1, 0, 3)),
        target=np.append(ep.target_speed, ep.target_speed[-1]),
        reason=ep.reason + [ep.reason[-1]],
    )


def draw_background(ax: Axes, params: sc.ScenarioParams, y_lim: tuple[float, float] = Y_LIM) -> list[np.ndarray]:
    """Road, parking lanes, sidewalks, an oncoming lane if there is one, and the parked vehicles.

    Returns the parked vehicles' rectangles (the primary occluder first), for visibility and shadows.
    """
    ax.set_aspect("equal")
    ax.set_ylim(*y_lim)
    x0, x1 = -30.0, 300.0
    ax.add_patch(Rectangle((x0, -sc.LANE_WIDTH / 2), x1 - x0, sc.LANE_WIDTH, fc=COLORS["road"], lw=0))
    for sign in (-1.0, 1.0):  # parking lane and sidewalk on each side
        ax.add_patch(Rectangle((x0, sign * sc.LANE_WIDTH / 2), x1 - x0, sign * (abs(sc.SIDEWALK_Y) - sc.LANE_WIDTH / 2),
                               fc=COLORS["parking"], lw=0))
        ax.add_patch(Rectangle((x0, sign * abs(sc.SIDEWALK_Y)), x1 - x0, sign * (abs(y_lim[0]) - abs(sc.SIDEWALK_Y)),
                               fc=COLORS["walk"], lw=0))
    if any(mv.y > sc.LANE_WIDTH / 2 for mv in params.moving_vehicles):  # an oncoming lane: draw it as road
        ax.add_patch(Rectangle((x0, sc.LANE_WIDTH / 2), x1 - x0, sc.LANE_WIDTH, fc=COLORS["road"], lw=0, zorder=1.5))
        ax.axhline(sc.LANE_WIDTH / 2, color="#f2c400", lw=1.5, zorder=1.6)
    ax.axhline(0.0, color="white", ls=(0, (6, 6)), lw=1)
    rects = [rect_corners(v.x, v.y, 0.0, v.length, v.width) for v in params.vehicles]
    for rect in rects:
        ax.add_patch(Polygon(rect, fc=COLORS["occluder"], ec=COLORS["edge"], zorder=4))
    return rects


class _Scene:
    def __init__(self, ep: Episode, title: str = ""):
        self.ep, self.f = ep, _frames(ep)
        p = ep.params
        self.fig, self.ax = plt.subplots(figsize=(12, 4.2), dpi=100)
        ax = self.ax
        self.rects = draw_background(ax, p)
        self.moving = [Polygon(np.zeros((4, 2)), fc=COLORS["moving"], ec=COLORS["edge"], zorder=4) for _ in p.moving_vehicles]
        for patch in self.moving:
            ax.add_patch(patch)
        n_shadows = len(self.rects) + len(p.moving_vehicles)
        self.shadows = [Polygon(np.zeros((3, 2)), fc=COLORS["shadow"], alpha=0.3, ec="none", zorder=3) for _ in range(n_shadows)]
        self.ego = Polygon(np.zeros((4, 2)), fc=COLORS["ego"], ec=COLORS["edge"], zorder=5)
        self.peds = [Circle((0, 0), sc.PED_RADIUS, zorder=6) for _ in range(self.f.ped.shape[1])]
        for patch in (*self.shadows, self.ego, *self.peds):
            ax.add_patch(patch)
        self.text = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=10,
                            bbox=dict(fc="white", ec="none", alpha=0.8), zorder=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or (f"seed {p.seed}" if p.seed is not None else "episode"))

    def draw(self, k: int) -> None:
        f, ep = self.f, self.ep
        x, y, yaw, v = f.ego[k]
        cx, cy = x + 0.5 * sc.EGO_WHEELBASE * math.cos(yaw), y + 0.5 * sc.EGO_WHEELBASE * math.sin(yaw)
        self.ego.set_xy(rect_corners(cx, cy, yaw, sc.EGO_LENGTH, sc.EGO_WIDTH))
        front = np.array([x + 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE) * math.cos(yaw),
                          y + 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE) * math.sin(yaw)])
        p = ep.params
        moving_rects = [rect_corners(x_, y_, yaw_, mv.length, mv.width)
                        for (x_, y_, yaw_), mv in zip(f.moving[k], p.moving_vehicles)]
        for patch, rect in zip(self.moving, moving_rects):
            patch.set_xy(rect)
        all_rects = [*self.rects, *moving_rects]
        for patch, rect in zip(self.shadows, all_rects):
            shadow = shadow_polygon(front, rect, SENSOR_RANGE)
            patch.set_visible(shadow is not None)
            if shadow is not None:
                patch.set_xy(shadow)
        for patch, pxy in zip(self.peds, f.ped[k]):
            patch.set_visible(not np.isnan(pxy[0]))
            if not np.isnan(pxy[0]):
                seen = all(is_visible(front, pxy, rect, SENSOR_RANGE) for rect in all_rects)
                patch.center = (float(pxy[0]), float(pxy[1]))
                patch.set(fc=COLORS["ped"] if seen else "none", ec=COLORS["ped"], ls="-" if seen else "--", lw=1.5)
        self.ax.set_xlim(front[0] - VIEW_BEHIND, front[0] + VIEW_AHEAD)
        last = k == len(f.t) - 1
        label = f"t={f.t[k]:5.2f}s  v={v:5.2f} m/s  target={f.target[k]:5.2f} m/s ({f.reason[k]})"
        self.text.set_text(label + (f"\noutcome: {ep.outcome}" if last else ""))


def render_frame(ep: Episode, k: int = -1, title: str = ""):
    """Draw one frame and return the matplotlib figure (caller closes it)."""
    scene = _Scene(ep, title)
    scene.draw(k % len(scene.f.t))
    return scene.fig


def render_episode(ep: Episode, path: str | Path, fps: int | None = None, stride: int = 1, title: str = "") -> Path:
    """Write an MP4. ``stride`` keeps every n-th step (faster to render); real time at stride 1."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = _Scene(ep, title)
    idx = list(range(0, len(scene.f.t), stride))
    if idx[-1] != len(scene.f.t) - 1:
        idx.append(len(scene.f.t) - 1)  # always end on the outcome frame
    fps = fps or max(1, round(1.0 / (ep.dt * stride)))
    anim = FuncAnimation(scene.fig, lambda i: scene.draw(idx[i]), frames=len(idx), blit=False)
    anim.save(str(path), writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(scene.fig)
    return path
