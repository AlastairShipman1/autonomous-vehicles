"""The live window: walk a pedestrian with the keyboard and watch a planner drive the ego.

Controls: arrows or WASD walk, SPACE pauses, N steps one tick while paused, [ and ] change the speed, R restarts,
TAB restarts under the next planner, Q or Esc quits. Every run is written to ``out/play`` as a recording.

The drawing here and the stepping in ``live.py`` are separate: ``draw_frame`` takes any surface, so it can be tested
(and screenshotted) without opening a window.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
import pygame

from av_core.geometry import SENSOR_RANGE, rect_corners, shadow_polygon
from av_sim_toy import scenario as sc
from av_sim_toy.live import LiveSession
from av_sim_toy.manual import direction
from av_sim_toy.render import COLORS, Y_LIM

VIEW_BEHIND, VIEW_AHEAD = 15.0, 75.0  # metres of road shown behind / ahead of the ego front
WIDTH = 1260
SCALE = WIDTH / (VIEW_BEHIND + VIEW_AHEAD)  # pixels per metre
TOP, BOTTOM = 62, 52  # heads-up lines above and below the scene
SCENE_H = int((Y_LIM[1] - Y_LIM[0]) * SCALE)
HEIGHT = TOP + SCENE_H + BOTTOM
HELP = "arrows/WASD walk   SPACE pause   N step   [ ] speed   R restart   TAB next planner   Q quit"


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=1)
def _font() -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.SysFont("menlo,consolas,dejavusansmono,couriernew,monospace", 16)


class _Camera:
    """World (x right, y up, metres) to screen pixels, following the ego front."""

    def __init__(self, front_x: float):
        self.x0 = front_x - VIEW_BEHIND

    def __call__(self, pts: np.ndarray) -> list[tuple[float, float]]:
        p = np.atleast_2d(pts)
        return [(float((x - self.x0) * SCALE), float(TOP + (Y_LIM[1] - y) * SCALE)) for x, y in p]


def _background(surf: pygame.Surface, cam: _Camera, oncoming: bool) -> None:
    def band(y_top: float, y_bottom: float, colour: str) -> None:
        a, b = cam(np.array([0.0, y_top]))[0][1], cam(np.array([0.0, y_bottom]))[0][1]
        pygame.draw.rect(surf, _rgb(colour), pygame.Rect(0, int(a), WIDTH, int(b - a) + 1))

    hw, walk = sc.LANE_WIDTH / 2, abs(sc.SIDEWALK_Y)
    band(Y_LIM[1], Y_LIM[0], COLORS["walk"])
    band(walk, -walk, COLORS["parking"])
    band(hw, -hw, COLORS["road"])
    if oncoming:
        band(hw + sc.LANE_WIDTH, hw, COLORS["road"])
        y = cam(np.array([0.0, hw]))[0][1]
        pygame.draw.line(surf, _rgb("#f2c400"), (0, y), (WIDTH, y), 2)
    y = cam(np.array([0.0, 0.0]))[0][1]
    first = int(math.floor(cam.x0 / 6.0)) * 6
    for x in range(first, int(cam.x0 + VIEW_BEHIND + VIEW_AHEAD) + 6, 6):  # dashed centre line
        sx = cam(np.array([float(x), 0.0]))[0][0]
        pygame.draw.line(surf, (255, 255, 255), (sx, y), (sx + 3 * SCALE, y), 1)
    for x in range(int(math.ceil(cam.x0 / 10.0)) * 10, int(cam.x0 + VIEW_BEHIND + VIEW_AHEAD), 10):  # distance ticks
        sx = cam(np.array([float(x), 0.0]))[0][0]
        pygame.draw.line(surf, (150, 150, 150), (sx, TOP + SCENE_H - 6), (sx, TOP + SCENE_H), 1)
        label = _font().render(f"{x}", True, (110, 110, 110))
        surf.blit(label, (sx - label.get_width() / 2, TOP + SCENE_H + 2))


def draw_frame(surf: pygame.Surface, session: LiveSession) -> None:
    """Draw the whole window onto ``surf`` (size ``(WIDTH, HEIGHT)``)."""
    sim, runner = session.sim, session.runner
    front = sim.ego_front()
    cam = _Camera(float(front[0]))
    surf.fill((250, 250, 250))
    _background(surf, cam, any(mv.y > sc.LANE_WIDTH / 2 for mv in session.params.moving_vehicles))
    surf.set_clip(pygame.Rect(0, TOP, WIDTH, SCENE_H))  # shadows run off the road; keep them off the heads-up lines
    for _, rect in sim.occluder_rects():
        pygame.draw.polygon(surf, _rgb(COLORS["occluder"]), cam(rect))
    for j in range(sim.n_moving):
        pygame.draw.polygon(surf, _rgb(COLORS["moving"]), cam(sim.moving_rect(j)))
    shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for _, rect in sim.occluder_rects():
        shadow = shadow_polygon(front, rect, SENSOR_RANGE)
        if shadow is not None:
            pygame.draw.polygon(shade, (*_rgb(COLORS["shadow"]), 85), cam(shadow))
    surf.blit(shade, (0, 0))
    pygame.draw.polygon(surf, _rgb(COLORS["ego"]), cam(sim.ego_corners()))
    pygame.draw.polygon(surf, _rgb(COLORS["edge"]), cam(sim.ego_corners()), 2)
    r = max(6, round(sc.PED_RADIUS * SCALE))  # drawn no smaller than 6 px so it can be found
    for i in range(sim.n_peds):
        (px, py), seen = cam(sim.ped_xy_at(i))[0], sim.ped_visible_at(i)
        pygame.draw.circle(surf, _rgb(COLORS["ped"]), (px, py), r, 0 if seen else 2)
        if sim.manual is not None and i == sim.n_peds - 1:  # the one you drive: a heading tick
            hx, hy = px + 1.8 * r * math.cos(sim.manual.heading), py - 1.8 * r * math.sin(sim.manual.heading)
            pygame.draw.line(surf, _rgb(COLORS["edge"]), (px, py), (hx, hy), 2)

    surf.set_clip(None)
    f = _font()
    plan, s = runner.plan, sim.state
    target = f"{plan.target_speed:5.2f} ({plan.reason})" if plan else "  -  "
    lag = "  LAGGING" if session.pacer.dropped else ""
    surf.blit(f.render(f"{session.planner_name:<4} t={sim.time:5.2f}s  v={s.speed:5.2f} m/s  target={target}  "
                       f"plan {runner.plan_ms:5.1f} ms  sim x{session.sim_speed:g}{lag}", True, (20, 20, 20)), (8, 8))
    if sim.manual is not None:
        seen = "seen by the AV" if sim.ped_visible_at(sim.n_peds - 1) else "hidden from the AV"
        surf.blit(f.render(f"you: x={sim.manual.x:6.1f} y={sim.manual.y:5.1f}  walking {sim.manual.speed:3.1f} m/s  {seen}",
                           True, _rgb(COLORS["ped"])), (8, 30))
    banner = (f"{sim.outcome.upper()}   (R restarts)" if sim.outcome else "PAUSED   (SPACE starts)" if session.paused else "")
    if banner:
        text = f.render(banner, True, (255, 255, 255))
        box = pygame.Rect(0, 0, text.get_width() + 24, text.get_height() + 12)
        box.center = (WIDTH // 2, TOP + SCENE_H // 2)
        pygame.draw.rect(surf, (30, 30, 30), box, border_radius=6)
        surf.blit(text, (box.x + 12, box.y + 6))
    saved = f"   saved: {session.saved_path}" if session.saved_path else ""
    surf.blit(f.render(HELP + saved, True, (90, 90, 90)), (8, HEIGHT - 24))


def run_app(session: LiveSession, fps: int = 60, max_frames: int | None = None) -> None:
    """Open the window and run until it is closed (or, for tests, until ``max_frames`` frames have been drawn)."""
    pygame.display.init()  # not pygame.init(): no audio is needed, and it fails noisily on machines without a sound device
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("av_sim_toy: walk the pedestrian")
    running, frame, last = True, 0, time.perf_counter()
    while running:
        now = time.perf_counter()  # (not pygame.time.Clock: it starts an SDL timer thread we have no use for)
        wall, last = min(now - last, 0.1), now  # a long stall (window drag) must not fast-forward the sim
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
                elif ev.key == pygame.K_r:
                    session.reset()
                elif ev.key == pygame.K_TAB:
                    session.next_planner()
                elif ev.key == pygame.K_LEFTBRACKET:
                    session.slower()
                elif ev.key == pygame.K_RIGHTBRACKET:
                    session.faster()
        keys = pygame.key.get_pressed()
        wanted = direction(keys[pygame.K_LEFT] or keys[pygame.K_a], keys[pygame.K_RIGHT] or keys[pygame.K_d],
                           keys[pygame.K_UP] or keys[pygame.K_w], keys[pygame.K_DOWN] or keys[pygame.K_s])
        if step_once and session.paused:
            session.step_once(wanted)
        else:
            session.tick(wall, wanted)
        draw_frame(screen, session)
        pygame.display.flip()
        frame += 1
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))
        if max_frames is not None and frame >= max_frames:
            running = False
    session.finish()
    pygame.display.quit()  # not pygame.quit(): that would invalidate the cached font for the next window
