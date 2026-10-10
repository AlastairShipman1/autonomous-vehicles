"""Run the hand-built scenarios (several parked cars, both sides, trucks, fast pedestrians, ...) and render them.

    uv run python -m tools.render_scenario --list
    uv run python -m tools.render_scenario row_of_cars gap_between_cars --planner v1 aif
    uv run python -m tools.render_scenario --all --planner v0 v1 aif --out out/scenarios
    uv run python -m tools.render_scenario --all --no-video          # just the outcome table
    uv run python -m tools.render_scenario row_of_cars --planner all --overlay   # all planners in one video

Videos go to ``<out>/<scenario>__<planner>.mp4``; with ``--overlay`` one ``<scenario>__overlay.mp4`` holds every planner at
once (``--still 3.2`` writes a PNG of that moment instead, ``--shadows aif`` draws one planner's occluded regions). The table shows the outcome, the lowest speed, how close the
pedestrian came and the time penalty (negative = faster than constant speed).
"""

from __future__ import annotations

import os

# One BLAS thread: the AIF planner is matrix-heavy and gains nothing from more.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

from av_core.sweep import compute_metrics
from av_sim_toy import SCENARIOS, Run, render_comparison, render_comparison_frame, render_episode
from av_sim_toy.sweep import to_record
from tools.planners import PLANNERS, expand, run_planners


def row(name: str, run: Run) -> str:
    ep = run.ep
    m = compute_metrics(to_record(ep))
    dist = "-" if m.min_distance is None else f"{m.min_distance:5.2f}"
    penalty = "  n/a" if m.time_penalty != m.time_penalty else f"{m.time_penalty:5.1f}"
    return (f"{name:<28}{run.name:<5}{ep.outcome:<11}{m.duration:6.1f}  "
            f"min v {ep.ego[:, 3].min():5.1f}   min dist {dist:>5}   penalty {penalty}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (see --list)")
    ap.add_argument("--all", action="store_true", help="every scenario")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument("--planner", nargs="+", default=["v1"], choices=[*sorted(PLANNERS), "all"],
                    help="planners to run; 'all' for every one")
    ap.add_argument("--out", type=Path, default=Path("out/scenarios"))
    ap.add_argument("--stride", type=int, default=2, help="render every n-th step (faster, choppier)")
    ap.add_argument("--no-video", action="store_true", help="print the table only")
    ap.add_argument("--overlay", action="store_true",
                    help="one video per scenario with every planner overlaid, instead of one per planner")
    ap.add_argument("--still", type=float, default=None, metavar="SECONDS",
                    help="with --overlay, write a PNG of that moment (or -1 for the end) instead of a video")
    ap.add_argument("--shadows", default=None, metavar="PLANNER", help="with --overlay, draw this planner's occluded regions")
    args = ap.parse_args(argv)

    if args.list:
        for s in SCENARIOS.values():
            print(f"{s.name:<28}{s.description}")
        return
    names = list(SCENARIOS) if args.all or not args.names else args.names
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; try --list")
    planners = expand(args.planner)
    if args.shadows is not None and args.shadows not in planners:
        raise SystemExit(f"--shadows {args.shadows!r} is not one of the planners being run: {planners}")
    print(f"{'scenario':<28}{'plan':<5}{'outcome':<11}{'dur':>6}")
    for name in names:
        runs = run_planners(SCENARIOS[name].params, planners)
        for run in runs:
            print(row(name, run), flush=True)
        if args.no_video:
            continue
        if args.overlay:
            title = f"{name}  |  " + " vs ".join(planners)
            if args.still is not None:
                import matplotlib.pyplot as plt

                args.out.mkdir(parents=True, exist_ok=True)
                fig = render_comparison_frame(runs, None if args.still < 0 else args.still, title, args.shadows)
                fig.savefig(args.out / f"{name}__overlay.png", dpi=110)
                plt.close(fig)
            else:
                render_comparison(runs, args.out / f"{name}__overlay.mp4", stride=args.stride, title=title, shadows=args.shadows)
        else:
            for run in runs:
                render_episode(run.ep, args.out / f"{name}__{run.name}.mp4", stride=args.stride,
                               title=f"{name}  |  planner {run.name}")
    if not args.no_video:
        print(f"\nwrote to {args.out}/")


if __name__ == "__main__":
    main()
