"""Run a planner over a set of seeds and write ``episodes.csv`` and ``summary.txt``.

    uv run python -m tools.run_sweep --planner v1 --seeds reporting --out out/sweep_v1
    uv run python -m tools.run_sweep --planner v0 --seeds 0:100 --workers 4

``--seeds`` is ``tuning`` (0-999), ``reporting`` (10000-10999), ``A:B`` (half-open), or ``--n`` takes the
first n of the chosen set.
"""

from __future__ import annotations

import os

# One BLAS thread per process: with --workers N the default (a thread per core in every worker) oversubscribes
# the CPU and makes the matrix-heavy AIF planner an order of magnitude slower. Must precede the numpy import.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from pathlib import Path

from av_core.sweep import format_summary, run_sweep, summarize, write_csv
from av_sim_toy import REPORTING_SEEDS, TUNING_SEEDS
from av_sim_toy.sweep import toy_scenario_factory
from tools.planners import PLANNERS, PREDICTORS


def parse_seeds(spec: str, n: int | None) -> range:
    if spec == "tuning":
        seeds = TUNING_SEEDS
    elif spec == "reporting":
        seeds = REPORTING_SEEDS
    else:
        a, b = spec.split(":")
        seeds = range(int(a), int(b))
    return seeds[:n] if n else seeds


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", default="v0", choices=sorted(PLANNERS))
    ap.add_argument("--seeds", default="reporting")
    ap.add_argument("--n", type=int, default=None, help="use only the first n seeds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="default: out/sweep_<planner>")
    args = ap.parse_args(argv)
    out = args.out or Path("out") / f"sweep_{args.planner}"
    seeds = parse_seeds(args.seeds, args.n)
    t0 = time.time()
    rows = run_sweep(seeds, toy_scenario_factory, PLANNERS[args.planner], PREDICTORS.get(args.planner),
                     workers=args.workers)
    write_csv(rows, out / "episodes.csv")
    text = f"planner {args.planner}, seeds {args.seeds} ({len(rows)} episodes, {time.time() - t0:.0f} s)\n\n"
    text += format_summary(summarize(rows))
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote {out / 'episodes.csv'} and {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
