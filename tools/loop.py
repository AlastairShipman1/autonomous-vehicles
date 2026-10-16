"""The loop: an endless circular track with a gauntlet of obstacles, for watching a planner drive round and round.

    uv run python -m tools.loop                              # window, planner v1, the first gauntlet
    uv run python -m tools.loop --planner aif --course gauntlet
    uv run python -m tools.loop --headless --laps 5 --planner all      # lap tables, no window
    uv run python -m tools.loop --shot out/loop.png --at 26            # a screenshot of t = 26 s

Window controls (also listed at the bottom of the window): SPACE pause, N one step, [ ] speed, V overview of the whole
track, R restart, TAB restart under the next planner, Q quit. The ego follows lane 0. Contacts are counted, not
physical: the ego drives on through whatever it hits. ``--headless`` runs each planner for ``--laps`` laps and prints
a row per lap (time, mean speed, collisions, red lights run, close calls with pedestrians) and the incident log.
"""

from __future__ import annotations

import os

# One BLAS thread: the AIF planner is matrix-heavy and gains nothing from more, and a live window needs the other cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

from av_core.protocols import Planner, Predictor
from av_sim_toy.loop import LoopRunner, LoopSim, Track, default_course, empty_course, format_laps
from av_sim_toy.loop.course import Course
from av_sim_toy.loop.session import LoopSession
from tools.planners import PLANNERS, expand, make_planner, make_predictor

COURSES = {"gauntlet": default_course, "empty": lambda track: empty_course()}


def make(name: str) -> tuple[Planner, Predictor | None]:
    return make_planner(name), make_predictor(name)


def run_headless(track: Track, course: Course, planners: list[str], laps: int, max_seconds: float) -> None:
    for name in planners:
        sim = LoopSim(track, course)
        runner = LoopRunner(sim, make_planner(name), make_predictor(name))
        runner.run_laps(laps, max_seconds)
        totals = sim.totals()
        print(f"\n{name}: {len(sim.laps)} laps in {sim.time:.0f} s of sim time on {course.name}")
        print(format_laps(sim.laps))
        print("totals: " + ", ".join(f"{k.value} {v}" for k, v in totals.items()))
        for inc in sim.incidents[:20]:
            print(f"  t={inc.time:6.1f}s lap {inc.lap}: {inc.kind.value}: {inc.detail}")
        if len(sim.incidents) > 20:
            print(f"  ... and {len(sim.incidents) - 20} more")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", nargs="+", default=["v1"], choices=[*sorted(PLANNERS), "all"],
                    help="live: the starting planner (TAB cycles); headless: the planners to run")
    ap.add_argument("--course", choices=sorted(COURSES), default="gauntlet")
    ap.add_argument("--radius", type=float, default=100.0, help="track radius in metres (lane 0's centreline)")
    ap.add_argument("--headless", action="store_true", help="no window: run --laps laps per planner and print the tables")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=1800.0, help="headless: stop a planner after this much sim time")
    ap.add_argument("--shot", type=Path, default=None, metavar="PNG", help="write a screenshot instead of opening a window")
    ap.add_argument("--at", type=float, default=20.0, help="with --shot, the sim time to photograph")
    ap.add_argument("--overview", action="store_true", help="with --shot, the whole-track view")
    args = ap.parse_args(argv)

    track = Track.circle(args.radius)
    course = COURSES[args.course](track)
    planners = expand(args.planner)

    if args.headless:
        run_headless(track, course, planners, args.laps, args.max_seconds)
        return

    session = LoopSession(track, course, make, list(PLANNERS), planners[0])
    if args.shot is not None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame

        from av_sim_toy.loop.view import HEIGHT, WIDTH, draw_frame

        while session.sim.time < args.at:
            session.step_once()
        session.overview = args.overview
        pygame.font.init()
        surf = pygame.Surface((WIDTH, HEIGHT))
        draw_frame(surf, session)
        args.shot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surf, str(args.shot))
        print(f"wrote {args.shot}")
        return

    from av_sim_toy.loop.view import run_app  # imported late: opening pygame is not needed for --headless

    run_app(session)


if __name__ == "__main__":
    main()
