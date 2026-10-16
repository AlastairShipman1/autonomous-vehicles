"""The live window for the loop: a camera that follows the ego, a minimap of the whole track, and a heads-up display.

Controls: SPACE pause, N one step while paused, [ and ] speed (0.25x to 4x), V overview of the whole track,
R restart, TAB restart under the next planner, Q or Esc quit.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
import pygame

from av_core.geometry import SENSOR_RANGE, rect_corners, shadow_polygon
from av_core.types import TrafficLightState
from av_sim_toy import scenario as sc
from av_sim_toy.loop.course import PED_RADIUS
from av_sim_toy.loop.sim import Kind
from av_sim_toy.loop.session import LoopSession
from av_sim_toy.loop.track import PARKING_EDGE
from av_sim_toy.render import COLORS

WIDTH, HEIGHT = 1280, 760
FOLLOW_SCALE = 9.0  # pixels per metre
HELP = "SPACE pause  N step  [ ] speed  V overview  R restart  TAB next planner  Q quit"
LIGHT_COLOURS = {TrafficLightState.GREEN: (60, 190, 90), TrafficLightState.YELLOW: (240, 190, 40),
                 TrafficLightState.RED: (220, 60, 50), TrafficLightState.UNKNOWN: (150, 150, 150)}
BANDS = (  # (lateral from, lateral to, colour): right sidewalk, parking lane, lane 0, lane 1, inner sidewalk
    (-9.0, PARKING_EDGE, COLORS["walk"]), (PARKING_EDGE, -1.75, COLORS["parking"]), (-1.75, 1.75, COLORS["road"]),
    (1.75, 5.25, COLORS["road"]), (5.25, 9.0, COLORS["walk"]))


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=1)
def _font(size: int = 16) -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.SysFont("menlo,consolas,dejavusansmono,couriernew,monospace", size)


class _Camera:
    """World metres (x right, y up) to pixels, centred on a world point."""

    def __init__(self, cx: float, cy: float, scale: float, origin: tuple[float, float] = (WIDTH / 2, HEIGHT / 2)):
        self.cx, self.cy, self.scale, self.ox, self.oy = cx, cy, scale, origin[0], origin[1]

    def __call__(self, pts: np.ndarray) -> list[tuple[float, float]]:
        p = np.atleast_2d(pts)
        return [(self.ox + (x - self.cx) * self.scale, self.oy - (y - self.cy) * self.scale) for x, y in p]


def _draw_road(surf: pygame.Surface, session: LoopSession, cam: _Camera, s_from: float, s_to: float, step: float) -> None:
    track = session.track
    s = np.arange(s_from, s_to + step, step)
    edges = {lat: cam(track.xy_at(s, lat)) for lat in {b[0] for b in BANDS} | {b[1] for b in BANDS}}
    for lo, hi, colour in BANDS:
        a, b = edges[lo], edges[hi]
        for k in range(len(s) - 1):
            pygame.draw.polygon(surf, _rgb(colour), (a[k], a[k + 1], b[k + 1], b[k]))
    for lat, colour, dash in ((1.75, (255, 255, 255), True), (-1.75, (255, 255, 255), False)):  # lane line, parking line
        pts = cam(track.xy_at(s, lat))
        for k in range(0, len(pts) - 1, 1):
            if dash and (k // 3) % 2:
                continue
            pygame.draw.line(surf, colour, pts[k], pts[k + 1], 1)


def _draw_scene(surf: pygame.Surface, session: LoopSession, cam: _Camera, s_ego: float, reach: float, step: float,
                px_scale: float) -> None:
    sim, track = session.sim, session.track
    surf.fill(_rgb(COLORS["walk"]))
    _draw_road(surf, session, cam, s_ego - reach, s_ego + reach, step)
    for sig in sim.course.signals:  # stop lines, coloured by state
        line = cam(sim.stop_line(sig))
        pygame.draw.line(surf, LIGHT_COLOURS[sim.signal_state(sig)], line[0], line[1], max(2, round(0.5 * px_scale)))
    front = sim.ego_front()
    shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for _, rect in sim.occluder_rects():
        pygame.draw.polygon(surf, _rgb(COLORS["occluder"]), cam(rect))
    for _, rect in sim.blocking_rects(front):
        shadow = shadow_polygon(front, rect, SENSOR_RANGE)
        if shadow is not None and px_scale >= 4.0:
            pygame.draw.polygon(shade, (*_rgb(COLORS["shadow"]), 80), cam(shadow))
    for i in range(len(sim.cars)):
        pygame.draw.polygon(surf, _rgb(COLORS["moving"]), cam(sim.car_rect(i)))
    surf.blit(shade, (0, 0))
    pygame.draw.polygon(surf, _rgb(COLORS["ego"]), cam(sim.ego_corners()))
    pygame.draw.polygon(surf, _rgb(COLORS["edge"]), cam(sim.ego_corners()), 2)
    r = max(5, round(PED_RADIUS * px_scale))
    for i in range(len(sim.walkers)):
        (px, py) = cam(sim.walker_xy(i))[0]
        seen = sim.walker_visible(i, front)
        pygame.draw.circle(surf, _rgb(COLORS["ped"]), (px, py), r, 0 if seen else 2)


def _draw_minimap(surf: pygame.Surface, session: LoopSession, box: pygame.Rect) -> None:
    sim, track = session.sim, session.track
    pts = track.frame.points
    lo, hi = pts.min(axis=0) - 12.0, pts.max(axis=0) + 12.0
    scale = min(box.width / (hi[0] - lo[0]), box.height / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    cam = _Camera(float(mid[0]), float(mid[1]), scale, (box.centerx, box.centery))
    pygame.draw.rect(surf, (245, 245, 245), box, border_radius=6)
    pygame.draw.rect(surf, (120, 120, 120), box, 1, border_radius=6)
    for lat, width in ((0.0, 7), (track.lane_width, 7)):
        ring = cam(track.xy_at(np.arange(0, track.length, 4.0), lat))
        pygame.draw.lines(surf, _rgb(COLORS["road"]), True, ring, 4)
    for sig in sim.course.signals:
        p = cam(sim.stop_line(sig).mean(axis=0))[0]
        pygame.draw.circle(surf, LIGHT_COLOURS[sim.signal_state(sig)], p, 5)
    for _, rect in sim.occluder_rects()[: len(sim.course.parked)]:
        pygame.draw.circle(surf, _rgb(COLORS["occluder"]), cam(rect.mean(axis=0))[0], 2)
    for i in range(len(sim.cars)):
        pygame.draw.circle(surf, _rgb(COLORS["moving"]), cam(np.array(sim.car_pose(i)[:2]))[0], 3)
    for i in range(len(sim.walkers)):
        pygame.draw.circle(surf, _rgb(COLORS["ped"]), cam(sim.walker_xy(i))[0], 3)
    pygame.draw.circle(surf, _rgb(COLORS["ego"]), cam(np.array([sim.state.x, sim.state.y]))[0], 5)


def draw_frame(surf: pygame.Surface, session: LoopSession) -> None:
    """Draw the whole window onto ``surf`` (size ``(WIDTH, HEIGHT)``)."""
    sim, runner, track = session.sim, session.runner, session.track
    if session.overview:
        lo, hi = track.frame.points.min(axis=0) - 14.0, track.frame.points.max(axis=0) + 14.0
        scale = min(WIDTH / (hi[0] - lo[0]), (HEIGHT - 80) / (hi[1] - lo[1]))
        mid = (lo + hi) / 2
        cam = _Camera(float(mid[0]), float(mid[1]), scale, (WIDTH / 2, HEIGHT / 2 + 20))
        _draw_scene(surf, session, cam, track.length / 2, track.length / 2, 3.0, scale)
    else:
        cam = _Camera(float(sim.state.x), float(sim.state.y), FOLLOW_SCALE)
        _draw_scene(surf, session, cam, sim.ego_s(), 85.0, 1.5, FOLLOW_SCALE)
        _draw_minimap(surf, session, pygame.Rect(WIDTH - 236, 58, 224, 224))

    f = _font()
    plan, s = runner.plan, sim.state
    target = f"{plan.target_speed:5.2f} ({plan.reason})" if plan else "  -  "
    lag = "  LAGGING" if session.pacer.dropped else ""
    totals = sim.totals()
    last = f"{sim.laps[-1].time:.1f}s" if sim.laps else "-"
    lines = [
        f"{session.planner_name:<4} t={sim.time:6.1f}s  lap {sim.lap}  v={s.speed:5.2f} m/s  target={target}  "
        f"plan {runner.plan_ms:5.1f} ms  sim x{session.sim_speed:g}{lag}",
        f"collisions {totals[Kind.COLLISION]}   red lights {totals[Kind.RED_LIGHT]}   close calls "
        f"{totals[Kind.CLOSE_CALL]}   last lap {last}   course: {sim.course.name}",
    ]
    pygame.draw.rect(surf, (255, 255, 255), pygame.Rect(0, 0, WIDTH, 50))
    for k, text in enumerate(lines):
        surf.blit(f.render(text, True, (20, 20, 20)), (8, 6 + 22 * k))
    recent = sim.incidents[-1] if sim.incidents else None
    if recent is not None and sim.time - recent.time < 4.0:  # flash the latest incident for a few seconds
        banner = f"{recent.kind.value.upper()}: {recent.detail}"
        text = f.render(banner, True, (255, 255, 255))
        box = pygame.Rect(0, 0, text.get_width() + 24, text.get_height() + 12)
        box.midtop = (WIDTH // 2, 62)
        pygame.draw.rect(surf, (190, 40, 40), box, border_radius=6)
        surf.blit(text, (box.x + 12, box.y + 6))
    if session.paused:
        text = f.render("PAUSED   (SPACE resumes)", True, (255, 255, 255))
        box = pygame.Rect(0, 0, text.get_width() + 24, text.get_height() + 12)
        box.center = (WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(surf, (30, 30, 30), box, border_radius=6)
        surf.blit(text, (box.x + 12, box.y + 6))
    surf.blit(f.render(HELP, True, (70, 70, 70)), (8, HEIGHT - 24))


def run_app(session: LoopSession, fps: int = 60, max_frames: int | None = None) -> None:
    """Open the window and run until it is closed (or, for tests, until ``max_frames`` frames have been drawn)."""
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("av_sim_toy: the loop")
    running, frame, last = True, 0, time.perf_counter()
    while running:
        now = time.perf_counter()
        wall, last = min(now - last, 0.1), now
        step_once = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    session.toggle_pause()
                elif ev.key == pygame.K_n:
                    step_once = True
                elif ev.key == pygame.K_v:
                    session.toggle_view()
                elif ev.key == pygame.K_r:
                    session.reset()
                elif ev.key == pygame.K_TAB:
                    session.next_planner()
                elif ev.key == pygame.K_LEFTBRACKET:
                    session.slower()
                elif ev.key == pygame.K_RIGHTBRACKET:
                    session.faster()
        if step_once and session.paused:
            session.step_once()
        else:
            session.tick(wall)
        draw_frame(screen, session)
        pygame.display.flip()
        frame += 1
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))
        if max_frames is not None and frame >= max_frames:
            running = False
    pygame.display.quit()
