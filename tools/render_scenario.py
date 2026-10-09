"""Run the hand-built scenarios (several parked cars, both sides, trucks, fast pedestrians, ...) and render them.

    uv run python -m tools.render_scenario --list
    uv run python -m tools.render_scenario row_of_cars gap_between_cars --planner v1 aif
    uv run python -m tools.render_scenario --all --planner v0 v1 aif --out out/scenarios
    uv run python -m tools.render_scenario --all --no-video          # just the outcome table

Videos go to ``<out>/<scenario>__<planner>.mp4``. The table shows the outcome, the lowest speed, how close the
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
from av_sim_toy import SCENARIOS, render_episode, run_episode
from av_sim_toy.sweep import to_record
from tools.planners import PLANNERS, make_planner, make_predictor


def run_one(name: str, planner_name: str, out: Path | None, stride: int) -> str:
    sc = SCENARIOS[name]
    ep = run_episode(sc.params, make_planner(planner_name), predictor=make_predictor(planner_name))
    m = compute_metrics(to_record(ep))
    if out is not None:
        render_episode(ep, out / f"{name}__{planner_name}.mp4", stride=stride, title=f"{name}  |  planner {planner_name}")
    dist = "-" if m.min_distance is None else f"{m.min_distance:5.2f}"
    penalty = "  n/a" if m.time_penalty != m.time_penalty else f"{m.time_penalty:5.1f}"
    return (f"{name:<22}{planner_name:<5}{ep.outcome:<11}{m.duration:6.1f}  "
            f"min v {ep.ego[:, 3].min():5.1f}   min dist {dist:>5}   penalty {penalty}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names (see --list)")
    ap.add_argument("--all", action="store_true", help="every scenario")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument("--planner", nargs="+", default=["v1"], choices=sorted(PLANNERS))
    ap.add_argument("--out", type=Path, default=Path("out/scenarios"))
    ap.add_argument("--stride", type=int, default=2, help="render every n-th step (faster, choppier)")
    ap.add_argument("--no-video", action="store_true", help="print the table only")
    args = ap.parse_args(argv)

    if args.list:
        for s in SCENARIOS.values():
            print(f"{s.name:<22}{s.description}")
        return
    names = list(SCENARIOS) if args.all or not args.names else args.names
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; try --list")
    print(f"{'scenario':<22}{'plan':<5}{'outcome':<11}{'dur':>6}")
    for name in names:
        for planner in args.planner:
            print(run_one(name, planner, None if args.no_video else args.out, args.stride), flush=True)
    if not args.no_video:
        print(f"\nvideos in {args.out}/")


if __name__ == "__main__":
    main()
