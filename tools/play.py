"""Walk a pedestrian with the keyboard while a planner drives the ego; replay what you did against any planner.

    uv run python -m tools.play                              # single_car, planner v1
    uv run python -m tools.play row_of_cars --planner aif
    uv run python -m tools.play --seed 10042 --speed 8       # a sampled scenario, ego starting at 8 m/s
    uv run python -m tools.play --list

    uv run python -m tools.play --replay out/play/single_car__v1__20261016-101500.json --planner all
    uv run python -m tools.play --replay FILE --planner aif --video out/replay.mp4

The scripted pedestrians are removed (so the only one is yours); ``--keep-scripted`` leaves them in. The window starts
paused: press SPACE. Keys are listed at the bottom of the window. Each run is saved to ``--out`` (default ``out/play``)
as a small JSON recording; ``--no-record`` turns that off. ``--replay`` runs a recording headlessly against the chosen
planner(s) and prints the outcome next to the recorded one.
"""

from __future__ import annotations

import os

# One BLAS thread: the AIF planner is matrix-heavy and gains nothing from more, and a live window needs the other cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

from av_core.protocols import Planner, Predictor
from av_core.sweep import compute_metrics
from av_sim_toy import SCENARIOS, render_episode
from av_sim_toy.live import LiveSession, SessionConfig
from av_sim_toy.recording import Recording, replay
from av_sim_toy.sweep import to_record
from tools.planners import PLANNERS, expand, make_planner, make_predictor


def make(name: str) -> tuple[Planner, Predictor | None]:
    return make_planner(name), make_predictor(name)


def do_replay(path: Path, planners: list[str], video: Path | None, stride: int) -> None:
    rec = Recording.load(path)
    print(f"{path.name}: scenario {rec.scenario or 'seed ' + str(rec.seed)}, recorded under {rec.planner} -> {rec.outcome}, "
          f"{len(rec.inputs)} steps ({len(rec.inputs) * rec.dt:.1f} s)")
    for name in planners:
        ep = replay(rec, make_planner(name), make_predictor(name))
        m = compute_metrics(to_record(ep))
        dist = "-" if m.min_distance is None else f"{m.min_distance:.2f} m"
        print(f"  {name:<4} {ep.outcome:<10} min distance {dist:>8}   min v {ep.ego[:, 3].min():5.1f} m/s   "
              f"duration {m.duration:5.1f} s")
        if video is not None:
            target = video if len(planners) == 1 else video.with_name(f"{video.stem}__{name}{video.suffix}")
            print(f"       video: {render_episode(ep, target, stride=stride)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="single_car", help="scenario name (see --list)")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument("--seed", type=int, default=None, help="a sampled scenario instead of a named one")
    ap.add_argument("--planner", nargs="+", default=None, choices=[*sorted(PLANNERS), "all"],
                    help="live: the starting planner (TAB cycles through all); replay: the planners to run")
    ap.add_argument("--speed", type=float, default=None, help="the ego's starting speed in m/s")
    ap.add_argument("--keep-scripted", action="store_true", help="leave the scenario's own pedestrians in")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="where your pedestrian starts (default: right sidewalk, 1 m past the last parked vehicle)")
    ap.add_argument("--out", type=Path, default=Path("out/play"), help="where recordings go")
    ap.add_argument("--no-record", action="store_true", help="do not write recordings")
    ap.add_argument("--replay", type=Path, default=None, metavar="FILE", help="replay a recording headlessly and exit")
    ap.add_argument("--video", type=Path, default=None, help="with --replay, write an MP4")
    ap.add_argument("--stride", type=int, default=2, help="with --video, render every n-th step")
    args = ap.parse_args(argv)

    if args.list:
        for s in SCENARIOS.values():
            print(f"{s.name:<28}{s.description}")
        return

    if args.replay is not None:
        names = expand(args.planner) if args.planner else [Recording.load(args.replay).planner]
        do_replay(args.replay, names, args.video, args.stride)
        return

    planner = (expand(args.planner)[0] if args.planner else "v1")
    config = SessionConfig(scenario=None if args.seed is not None else args.scenario, seed=args.seed,
                           speed=args.speed, keep_scripted=args.keep_scripted,
                           start=tuple(args.start) if args.start else None,
                           out_dir=None if args.no_record else args.out)
    from av_sim_toy.play import run_app  # imported late: opening pygame is not needed for --list or --replay

    run_app(LiveSession(config, make, list(PLANNERS), planner))


if __name__ == "__main__":
    main()
