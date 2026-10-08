"""Sweep harness: run a planner over many seeds, write one CSV row per episode, summarise with intervals.

``run_sweep`` takes a scenario factory and a planner factory, so the same code drives the toy sim and,
later, CARLA. With ``workers > 1`` the factories must be picklable (module-level callables).
"""

from __future__ import annotations

import csv
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from av_core.protocols import Planner, Predictor
from av_core.sweep.metrics import EpisodeMetrics, compute_metrics
from av_core.sweep.record import EpisodeRecord, ParamValue, Scenario
from av_core.sweep.stats import bootstrap_mean, wilson_interval

ScenarioFactory = Callable[[int], Scenario]
PlannerFactory = Callable[[], Planner]
PredictorFactory = Callable[[], Predictor]
Row = tuple[EpisodeMetrics, dict[str, ParamValue]]
Interval = tuple[float, float, float]  # (estimate, low, high)
SummaryEntry = dict[str, int | Interval]  # counts (n, n_*) and intervals


def run_episode_for_seed(seed: int, scenario_factory: ScenarioFactory, planner_factory: PlannerFactory,
                         predictor_factory: PredictorFactory | None = None) -> EpisodeRecord:
    predictor = predictor_factory() if predictor_factory else None
    return scenario_factory(seed).run(planner_factory(), predictor)


def _metrics_row(args: tuple[int, ScenarioFactory, PlannerFactory, PredictorFactory | None]) -> Row:
    rec = run_episode_for_seed(*args)
    return compute_metrics(rec), rec.params


def run_sweep(seeds: Iterable[int], scenario_factory: ScenarioFactory, planner_factory: PlannerFactory,
              predictor_factory: PredictorFactory | None = None, workers: int = 1
              ) -> list[Row]:
    """One (metrics, scenario params) pair per seed, in seed order."""
    jobs = [(s, scenario_factory, planner_factory, predictor_factory) for s in seeds]
    if workers <= 1:
        return [_metrics_row(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_metrics_row, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


def write_csv(rows: list[Row], path: str | Path) -> Path:
    """One row per episode: metrics columns, then ``param_*`` columns, so later plots need no rerun."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_cols = [f.name for f in fields(EpisodeMetrics)]
    param_cols = sorted({k for _, p in rows for k in p})
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(metric_cols + [f"param_{k}" for k in param_cols])
        for m, params in rows:
            d = asdict(m)
            w.writerow([_cell(d[c]) for c in metric_cols] + [_cell(params.get(k)) for k in param_cols])
    return path


def _cell(v: ParamValue | np.generic) -> str | int:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))  # round-trips exactly; nan/inf spelled out
    if isinstance(v, np.integer):
        return int(v)
    return v if isinstance(v, (int, str)) else str(v)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows: list[Row]) -> dict[str, SummaryEntry]:
    """Summary per split ('pedestrian present' / 'pedestrian absent'); every entry is (estimate, low, high).

    Rates use Wilson 95% intervals and continuous metrics a 95% bootstrap interval of the mean over the
    episodes where they are defined (``n_*`` gives that count). Minimum TTC is infinite when there is no
    conflict, so it is reported as a rate of finite TTCs plus the mean over those.
    """
    ms = [m for m, _ in rows]
    out: dict[str, SummaryEntry] = {}
    for name, group in (("pedestrian present", [m for m in ms if m.ped_present]),
                        ("pedestrian absent", [m for m in ms if not m.ped_present])):
        s: SummaryEntry = {"n": len(group)}
        if name.endswith("present"):
            s["collision_rate"] = wilson_interval(sum(bool(m.collision) for m in group), len(group))
            s["min_distance_m"] = bootstrap_mean([m.min_distance for m in group])
            finite = [m.min_ttc for m in group if m.min_ttc is not None and math.isfinite(m.min_ttc)]
            s["finite_ttc_rate"] = wilson_interval(len(finite), len(group))
            s["min_ttc_s"] = bootstrap_mean(finite)
            s["n_finite_ttc"] = len(finite)
        else:
            s["needless_stop_rate"] = wilson_interval(sum(bool(m.needless_stop) for m in group), len(group))
        onset = [m.braking_onset for m in group]
        s["braking_onset_m"] = bootstrap_mean(onset)
        s["n_braking_onset"] = int(np.isfinite(onset).sum())
        pen = [m.time_penalty for m in group]
        s["time_penalty_s"] = bootstrap_mean(pen)
        s["n_time_penalty"] = int(np.isfinite(pen).sum())
        out[name] = s
    return out


_DEFINED_IN = {"min_ttc_s": "n_finite_ttc", "braking_onset_m": "n_braking_onset", "time_penalty_s": "n_time_penalty"}


def format_summary(summary: dict[str, SummaryEntry]) -> str:
    lines: list[str] = []
    for name, s in summary.items():
        lines.append(f"{name} (n = {s['n']})")
        for k, v in s.items():
            if isinstance(v, int):  # the counts: n and n_*
                continue
            est, lo, hi = v
            fmt = "{:.1%}" if k.endswith("rate") else "{:.2f}"
            extra = f"   [defined in {s[_DEFINED_IN[k]]}]" if k in _DEFINED_IN else ""
            lines.append(f"  {k:<20} {fmt.format(est)}  (95% CI {fmt.format(lo)} to {fmt.format(hi)}){extra}")
        lines.append("")
    return "\n".join(lines)


__all__ = ["run_sweep", "run_episode_for_seed", "write_csv", "read_csv", "summarize", "format_summary"]
