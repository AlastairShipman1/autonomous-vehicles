"""One command to render any seed to MP4.

    uv run python -m tools.render_seed 10000 --planner v0 --out out/seed_10000.mp4
    uv run python -m tools.render_seed 10000 --planner all --overlay          # every planner in one video
"""

from __future__ import annotations

import argparse
from pathlib import Path

from av_sim_toy import render_comparison, render_episode, sample_scenario
from tools.planners import PLANNERS, expand, run_planners


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", type=int)
    ap.add_argument("--planner", nargs="+", default=["v0"], choices=[*sorted(PLANNERS), "all"])
    ap.add_argument("--out", type=Path, default=None,
                    help="file (one planner, or --overlay); default out/seed_<seed>_<planner>.mp4")
    ap.add_argument("--stride", type=int, default=1, help="render every n-th step (faster, choppier)")
    ap.add_argument("--overlay", action="store_true", help="one video with every planner overlaid")
    ap.add_argument("--shadows", default=None, metavar="PLANNER", help="with --overlay, draw this planner's occluded regions")
    args = ap.parse_args(argv)
    planners = expand(args.planner)
    runs = run_planners(sample_scenario(args.seed), planners)
    if args.overlay:
        out = args.out or Path("out") / f"seed_{args.seed}_overlay.mp4"
        render_comparison(runs, out, stride=args.stride, title=f"seed {args.seed}  |  " + " vs ".join(planners),
                          shadows=args.shadows)
        print(f"{out}  " + "  ".join(f"{r.name}={r.ep.outcome}" for r in runs))
        return
    if len(runs) > 1 and args.out is not None:
        raise SystemExit("--out names one file; with several planners drop it or use --overlay")
    for run in runs:
        out = args.out or Path("out") / f"seed_{args.seed}_{run.name}.mp4"
        render_episode(run.ep, out, stride=args.stride)
        print(f"{out}  outcome={run.ep.outcome}  duration={run.ep.t[-1] + run.ep.dt:.2f}s")


if __name__ == "__main__":
    main()
