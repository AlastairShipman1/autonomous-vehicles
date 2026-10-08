"""Matplotlib renderer: episode log -> MP4 (or a single frame).

Top-down, camera follows the ego. Shows the road, the parked occluder, the occluded (shadow)
region, ego, the pedestrian (solid when visible, dashed outline when hidden) and the planner's
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
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from av_core.geometry import SENSOR_RANGE, is_visible, rect_corners, shadow_polygon  # noqa: E402
from av_sim_toy import scenario as sc  # noqa: E402
from av_sim_toy.sim import Episode  # noqa: E402

VIEW_BEHIND, VIEW_AHEAD = 15.0, 75.0  # m of road shown behind / ahead of the ego front
Y_LIM = (-9.0, 5.0)
COLORS = dict(road="#d9d9d9", parking="#bdbdbd", walk="#eeeeee", ego="#1f77b4", occluder="#555555",
              shadow="#f2a900", ped="#d62728", edge="#222222")


@dataclass(frozen=True)
class _Frames:
    t: np.ndarray
    ego: np.ndarray
    ped: np.ndarray
    target: np.ndarray
    reason: list[str]


def _frames(ep: Episode) -> _Frames:
    """Per-frame arrays: the logged rows plus the final state, so the last frame shows the outcome."""
    return _Frames(
        t=np.append(ep.t, ep.t[-1] + ep.dt),
        ego=np.vstack([ep.ego, ep.final_ego]),
        ped=np.vstack([ep.ped, ep.final_ped]),
        target=np.append(ep.target_speed, ep.target_speed[-1]),
        reason=ep.reason + [ep.reason[-1]],
    )


class _Scene:
    def __init__(self, ep: Episode):
        self.ep, self.f = ep, _frames(ep)
        p = ep.params
        self.occluder = rect_corners(p.occluder_x, sc.PARKING_CENTER_Y, 0.0, p.occluder_length, sc.OCCLUDER_WIDTH)
        self.fig, self.ax = plt.subplots(figsize=(12, 4.2), dpi=100)
        ax = self.ax
        ax.set_aspect("equal")
        ax.set_ylim(*Y_LIM)
        x0, x1 = -30.0, 300.0
        ax.add_patch(Rectangle((x0, -sc.LANE_WIDTH / 2), x1 - x0, sc.LANE_WIDTH, fc=COLORS["road"], lw=0))
        ax.add_patch(Rectangle((x0, sc.SIDEWALK_Y), x1 - x0, -sc.LANE_WIDTH / 2 - sc.SIDEWALK_Y, fc=COLORS["parking"], lw=0))
        ax.add_patch(Rectangle((x0, Y_LIM[0]), x1 - x0, sc.SIDEWALK_Y - Y_LIM[0], fc=COLORS["walk"], lw=0))
        ax.axhline(0.0, color="white", ls=(0, (6, 6)), lw=1)
        ax.add_patch(Polygon(self.occluder, fc=COLORS["occluder"], ec=COLORS["edge"], zorder=4))
        self.shadow = Polygon(np.zeros((3, 2)), fc=COLORS["shadow"], alpha=0.35, ec="none", zorder=3)
        self.ego = Polygon(np.zeros((4, 2)), fc=COLORS["ego"], ec=COLORS["edge"], zorder=5)
        self.ped = Circle((0, 0), sc.PED_RADIUS, zorder=6)
        for patch in (self.shadow, self.ego, self.ped):
            ax.add_patch(patch)
        self.text = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=10,
                            bbox=dict(fc="white", ec="none", alpha=0.8), zorder=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"seed {p.seed}" if p.seed is not None else "episode")

    def draw(self, k: int) -> None:
        f, ep = self.f, self.ep
        x, y, yaw, v = f.ego[k]
        cx, cy = x + 0.5 * sc.EGO_WHEELBASE * math.cos(yaw), y + 0.5 * sc.EGO_WHEELBASE * math.sin(yaw)
        self.ego.set_xy(rect_corners(cx, cy, yaw, sc.EGO_LENGTH, sc.EGO_WIDTH))
        front = np.array([x + 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE) * math.cos(yaw),
                          y + 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE) * math.sin(yaw)])
        shadow = shadow_polygon(front, self.occluder, SENSOR_RANGE)
        self.shadow.set_visible(shadow is not None)
        if shadow is not None:
            self.shadow.set_xy(shadow)
        pxy = f.ped[k]
        self.ped.set_visible(not np.isnan(pxy[0]))
        if not np.isnan(pxy[0]):
            seen = is_visible(front, pxy, self.occluder, SENSOR_RANGE)
            self.ped.center = (float(pxy[0]), float(pxy[1]))
            self.ped.set(fc=COLORS["ped"] if seen else "none", ec=COLORS["ped"], ls="-" if seen else "--", lw=1.5)
        self.ax.set_xlim(front[0] - VIEW_BEHIND, front[0] + VIEW_AHEAD)
        last = k == len(f.t) - 1
        label = f"t={f.t[k]:5.2f}s  v={v:5.2f} m/s  target={f.target[k]:5.2f} m/s ({f.reason[k]})"
        self.text.set_text(label + (f"\noutcome: {ep.outcome}" if last else ""))


def render_frame(ep: Episode, k: int = -1):
    """Draw one frame and return the matplotlib figure (caller closes it)."""
    scene = _Scene(ep)
    scene.draw(k % len(scene.f.t))
    return scene.fig


def render_episode(ep: Episode, path: str | Path, fps: int | None = None, stride: int = 1) -> Path:
    """Write an MP4. ``stride`` keeps every n-th step (faster to render); real time at stride 1."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = _Scene(ep)
    idx = list(range(0, len(scene.f.t), stride))
    if idx[-1] != len(scene.f.t) - 1:
        idx.append(len(scene.f.t) - 1)  # always end on the outcome frame
    fps = fps or max(1, round(1.0 / (ep.dt * stride)))
    anim = FuncAnimation(scene.fig, lambda i: scene.draw(idx[i]), frames=len(idx), blit=False)
    anim.save(str(path), writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(scene.fig)
    return path
