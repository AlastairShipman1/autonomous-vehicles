"""One command to render any seed to MP4.

    uv run python -m tools.render_seed 10000 --planner v0 --out out/seed_10000.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from av_sim_toy import render_episode, run_episode, sample_scenario
from tools.planners import make_planner


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", type=int)
    ap.add_argument("--planner", default="v0")
    ap.add_argument("--out", type=Path, default=None, help="default: out/seed_<seed>_<planner>.mp4")
    ap.add_argument("--stride", type=int, default=1, help="render every n-th step (faster, choppier)")
    args = ap.parse_args(argv)
    out = args.out or Path("out") / f"seed_{args.seed}_{args.planner}.mp4"
    ep = run_episode(sample_scenario(args.seed), make_planner(args.planner))
    render_episode(ep, out, stride=args.stride)
    print(f"{out}  outcome={ep.outcome}  duration={ep.t[-1] + ep.dt:.2f}s")


if __name__ == "__main__":
    main()
