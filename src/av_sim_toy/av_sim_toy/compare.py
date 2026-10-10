"""Overlay renderer: several planners' runs of the same scenario in one video.

One road and one set of parked and moving vehicles; each planner gets its own coloured ego, labelled, and its own
pedestrians in its colour (a pedestrian's trigger depends on where *its* ego is, so they differ between planners;
solid when that planner's ego can see it, dashed when it cannot). Underneath, every planner's speed (solid) and
commanded target (dashed) against time, with a cursor. Runs are aligned on time; a run that has ended (a collision,
or the finish) holds its last frame and says so.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from av_core.geometry import SENSOR_RANGE, is_visible, rect_corners, shadow_polygon  # noqa: E402
from av_sim_toy import scenario as sc  # noqa: E402
from av_sim_toy.render import COLORS, draw_background  # noqa: E402
from av_sim_toy.sim import Episode  # noqa: E402

VIEW_BEHIND = 12.0  # the window starts this far behind the rearmost ego front
VIEW_MIN, VIEW_MAX, VIEW_PAD = 55.0, 110.0, 40.0  # its width: the spread of the egos plus a margin, within these limits
Y_LIM = (-6.5, 6.5)
NEST = 0.14  # m: each planner's ego is this much smaller on every side than the previous one's
PALETTE = ["#d95f02", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
FIXED = {"v0": "#d95f02", "v1": "#1f77b4", "aif": "#2ca02c"}  # the same planner is the same colour in every video


@dataclass(frozen=True)
class Run:
    name: str
    ep: Episode


def colour_of(name: str, index: int) -> str:
    return FIXED.get(name, PALETTE[index % len(PALETTE)])


class _Track:
    """One run's per-frame arrays, padded by holding the last frame so every run spans the whole video."""

    def __init__(self, run: Run, n_frames: int, colour: str):
        ep = run.ep
        self.run, self.colour, self.ep = run, colour, ep
        self.n_real = len(ep.t) + 1  # logged rows plus the final state
        self.ego = np.vstack([ep.ego, ep.final_ego])
        self.ped = np.concatenate([ep.peds, ep.final_peds[None]], axis=0)
        self.moving = (np.concatenate([ep.moving, ep.final_moving[None]], axis=0) if ep.moving.size
                       else np.zeros((self.n_real, 0, 3)))
        self.target = np.append(ep.target_speed, ep.target_speed[-1])
        self.reason = ep.reason + [ep.reason[-1]]
        self.t = np.append(ep.t, ep.t[-1] + ep.dt)
        self.n_frames = n_frames

    def at(self, k: int) -> int:
        return min(k, self.n_real - 1)

    def ended(self, k: int) -> bool:
        return k >= self.n_real - 1

    def front(self, k: int) -> np.ndarray:
        x, y, yaw, _ = self.ego[self.at(k)]
        d = 0.5 * (sc.EGO_LENGTH + sc.EGO_WHEELBASE)
        return np.array([x + d * math.cos(yaw), y + d * math.sin(yaw)])


class _Overlay:
    def __init__(self, runs: Sequence[Run], title: str, shadows: str | None):
        if not runs:
            raise ValueError("need at least one run")
        dts = {round(r.ep.dt, 9) for r in runs}
        if len(dts) != 1:
            raise ValueError(f"runs must share a time step, got {sorted(dts)}")
        self.params = runs[0].ep.params
        if any(r.ep.params != self.params for r in runs):
            raise ValueError("runs must be of the same scenario")
        self.dt = runs[0].ep.dt
        self.n_frames = max(len(r.ep.t) + 1 for r in runs)
        self.tracks = [_Track(r, self.n_frames, colour_of(r.name, i)) for i, r in enumerate(runs)]
        self.shadow_track = next((t for t in self.tracks if t.run.name == shadows), None)
        if shadows is not None and self.shadow_track is None:
            raise ValueError(f"no run named {shadows!r} to draw shadows for")
        self.longest = max(self.tracks, key=lambda t: t.n_real)  # moving vehicles are the same in every run

        self.fig = plt.figure(figsize=(12, 7.4), dpi=100)
        grid = self.fig.add_gridspec(2, 1, height_ratios=[2.5, 1.6], hspace=0.32, top=0.86, bottom=0.08, left=0.07, right=0.98)
        self.ax = self.fig.add_subplot(grid[0])
        self.ax_v = self.fig.add_subplot(grid[1])
        self.rects = draw_background(self.ax, self.params, Y_LIM)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=13, fontweight="bold")

        self.moving = [Polygon(np.zeros((4, 2)), fc=COLORS["moving"], ec=COLORS["edge"], zorder=4)
                       for _ in self.params.moving_vehicles]
        n_shadow = len(self.rects) + len(self.params.moving_vehicles) if self.shadow_track else 0
        self.shadows = [Polygon(np.zeros((3, 2)), fc=COLORS["shadow"], alpha=0.25, ec="none", zorder=3) for _ in range(n_shadow)]
        # translucent with a thick outline of the planner's colour, each one slightly smaller than the last, so runs that
        # coincide (planners behaving alike) are still told apart
        self.egos = [Polygon(np.zeros((4, 2)), fc=t.colour, ec=t.colour, alpha=0.45, lw=2.6, zorder=5 + i)
                     for i, t in enumerate(self.tracks)]
        self.ego_outlines = [Polygon(np.zeros((4, 2)), fc="none", ec=t.colour, lw=2.6, zorder=5 + i, joinstyle="round")
                             for i, t in enumerate(self.tracks)]
        self.labels = [self.ax.text(0, 0, t.run.name, color=t.colour, fontsize=9, fontweight="bold", ha="center",
                                    va="bottom", zorder=12, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.0))
                       for t in self.tracks]
        self.peds = [[Circle((0, 0), sc.PED_RADIUS, zorder=8) for _ in range(t.ped.shape[1])] for t in self.tracks]
        for patch in (*self.moving, *self.shadows, *self.egos, *self.ego_outlines, *[c for row in self.peds for c in row]):
            self.ax.add_patch(patch)
        n = len(self.tracks)  # one status line per planner, side by side above the scene
        self.info = [self.fig.text(0.07 + 0.91 * i / n, 0.925, "", va="top", family="monospace", fontsize=9, color=t.colour,
                                   fontweight="bold") for i, t in enumerate(self.tracks)]

        # speed panel
        for t in self.tracks:
            self.ax_v.plot(t.t, t.ego[:, 3], color=t.colour, lw=1.8, label=t.run.name)
            self.ax_v.step(t.t, t.target, where="post", color=t.colour, lw=1.0, ls="--", alpha=0.55)
            if t.ep.outcome == "collision":
                self.ax_v.plot([t.t[-1]], [t.ego[-1, 3]], marker="X", color=t.colour, ms=10, mec="black", zorder=6)
        self.ax_v.set_xlim(0.0, self.n_frames * self.dt)
        self.ax_v.set_ylim(-0.5, max(15.0, max(float(t.ego[:, 3].max()) for t in self.tracks) + 1.0))
        self.ax_v.set_xlabel("time [s]")
        self.ax_v.set_ylabel("speed [m/s]  (dashed: target)")
        self.ax_v.grid(alpha=0.3)
        self.ax_v.legend(loc="upper right", ncols=len(self.tracks), fontsize=8)
        self.cursor = self.ax_v.axvline(0.0, color="black", lw=1.0)
        self.dots = [self.ax_v.plot([0.0], [0.0], "o", color=t.colour, mec="black", ms=6, zorder=7)[0] for t in self.tracks]

    def draw(self, k: int) -> None:
        longest = self.longest
        kk = longest.at(k)
        moving_rects = [rect_corners(x, y, yaw, mv.length, mv.width)
                        for (x, y, yaw), mv in zip(longest.moving[kk], self.params.moving_vehicles)]
        for patch, rect in zip(self.moving, moving_rects):
            patch.set_xy(rect)
        all_rects = [*self.rects, *moving_rects]
        if self.shadow_track is not None:
            front = self.shadow_track.front(k)
            for patch, rect in zip(self.shadows, all_rects):
                shadow = shadow_polygon(front, rect, SENSOR_RANGE)
                patch.set_visible(shadow is not None)
                if shadow is not None:
                    patch.set_xy(shadow)

        fronts = []
        for i, t in enumerate(self.tracks):
            j = t.at(k)
            x, y, yaw, v = t.ego[j]
            cx, cy = x + 0.5 * sc.EGO_WHEELBASE * math.cos(yaw), y + 0.5 * sc.EGO_WHEELBASE * math.sin(yaw)
            shrink = NEST * i
            body = rect_corners(cx, cy, yaw, sc.EGO_LENGTH - 2 * shrink, sc.EGO_WIDTH - 2 * shrink)
            self.egos[i].set_xy(body)
            self.ego_outlines[i].set_xy(body)
            front = t.front(k)
            fronts.append(front[0])
            self.labels[i].set_position((cx, cy + 1.2 + 1.15 * i))
            for patch, pxy in zip(self.peds[i], t.ped[j]):
                patch.set_visible(not np.isnan(pxy[0]))
                if not np.isnan(pxy[0]):
                    seen = all(is_visible(front, pxy, rect, SENSOR_RANGE) for rect in all_rects)
                    patch.center = (float(pxy[0]), float(pxy[1]))
                    patch.set(fc=t.colour if seen else "none", ec=t.colour, ls="-" if seen else "--", lw=2.2, alpha=0.9)
            end = f"  {t.ep.outcome.upper()}" if t.ended(k) else ""
            self.info[i].set_text(f"{t.run.name}  v={v:5.2f}  target={t.target[j]:5.2f}\n({t.reason[j]}){end}")
            self.dots[i].set_data([t.t[j]], [v])
        left = min(fronts) - VIEW_BEHIND
        width = float(np.clip(max(fronts) - min(fronts) + VIEW_PAD, VIEW_MIN, VIEW_MAX))
        self.ax.set_xlim(left, left + width)
        self.cursor.set_xdata([min(k, self.n_frames - 1) * self.dt])


def render_comparison_frame(runs: Sequence[Run], time: float | None = None, title: str = "",
                            shadows: str | None = None) -> Figure:
    """One frame of the overlay at ``time`` seconds (default: the end); the caller closes the figure."""
    ov = _Overlay(runs, title, shadows)
    k = ov.n_frames - 1 if time is None else min(max(int(round(time / ov.dt)), 0), ov.n_frames - 1)
    ov.draw(k)
    return ov.fig


def render_comparison(runs: Sequence[Run], path: str | Path, stride: int = 2, title: str = "",
                      shadows: str | None = None, fps: int | None = None) -> Path:
    """Write the overlay as an MP4, real time at ``stride`` 1; always ends on the final frame."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ov = _Overlay(runs, title, shadows)
    idx = list(range(0, ov.n_frames, stride))
    if idx[-1] != ov.n_frames - 1:
        idx.append(ov.n_frames - 1)
    fps = fps or max(1, round(1.0 / (ov.dt * stride)))
    anim = FuncAnimation(ov.fig, lambda i: ov.draw(idx[i]), frames=len(idx), blit=False)
    anim.save(str(path), writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(ov.fig)
    return path
