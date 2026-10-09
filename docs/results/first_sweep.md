# First sweep: planners v0 and v1, 200 reporting seeds

Seeds 10000-10199 (the first 200 reporting seeds), toy sim, planners as merged in PRs 1.4 and 2.6 with their
starting-guess constants (**not tuned**). Reproduce with:

```sh
uv run python -m tools.run_sweep --planner v0 --seeds reporting --n 200 --out docs/results/sweep_v0
uv run python -m tools.run_sweep --planner v1 --seeds reporting --n 200 --out docs/results/sweep_v1
```

Per-episode rows are in `sweep_v0/episodes.csv` and `sweep_v1/episodes.csv`; full summaries in each `summary.txt`.
Rates are Wilson 95% intervals; the other metrics are bootstrap 95% intervals of the mean over the episodes where the
metric is defined (count in brackets). Metric definitions: see PR 2.7.

**Scenario geometry.** The pedestrian steps out just past the occluder's far end (0.5 to 1.5 m beyond it), so its walk no
longer crosses the parked car. An earlier version of this table, with the pedestrian's x sampled over the car's span as
the spec words it, had the pedestrian walking through the car; it gave v1 a 33% collision rate and is superseded.

## Pedestrian present (96 episodes)

| Metric | v0 | v1 |
| --- | --- | --- |
| Collision rate | 24.0% (16.5 to 33.4) | 0.0% (0.0 to 3.8) |
| Min distance, ped centre to ego rectangle (m) | 0.94 (0.81 to 1.08) | 2.96 (2.80 to 3.10) |
| Episodes with a finite min TTC | 25.0% (17.4 to 34.5) | 100.0% (96.2 to 100.0) |
| Min TTC where finite (s) | 0.08 (0.00 to 0.23) [24] | 1.33 (1.29 to 1.38) [96] |
| Braking onset, ego front to occluder near end (m) | -3.48 (-5.34 to -1.63) [17] | 20.29 (19.85 to 20.74) [96] |
| Time penalty (s) | -1.97 (-2.18 to -1.76) [73] | 4.13 (3.84 to 4.42) [96] |

## Pedestrian absent (104 episodes)

| Metric | v0 | v1 |
| --- | --- | --- |
| Needless stop rate (speed < 1 m/s) | 0.0% (0.0 to 3.6) | 0.0% (0.0 to 3.6) |
| Braking onset (m) | undefined (never brakes above 1 m/s²) | 19.92 (19.58 to 20.26) [104] |
| Time penalty (s) | -2.03 (-2.23 to -1.84) [104] | 1.18 (0.98 to 1.36) [104] |

*Update:* the pedestrian now stops on the far sidewalk instead of walking on indefinitely (#16), so a pedestrian the ego
never meets ends 3.55 m from the lane edge rather than ever farther away; v1's minimum distance (earlier 3.15 m) is
re-measured above. Nothing else in this table changed.

## Reading the numbers

- **v0 is blind to the hidden pedestrian and v1 is not.** v0 collides in 23 of 96 pedestrian episodes (24%); v1 in none,
  at a cost in time. v1 starts braking about 20 m before the occluder in every episode, pedestrian or not (its occlusion
  cap), and takes 1.2 s longer than constant speed without a pedestrian and 4.1 s longer with one.
- **That cost is what an AIF planner has to beat.** The two baselines bracket the trade-off: v0 is fast and unsafe, v1
  safe and slow. The package-gate test (fewer collisions at equal or lower over-caution, or equal collisions with clearly
  less over-caution) is against v1's 0 collisions, so the target is about the time penalty.
- **v1's time penalty is comparable across the two splits only loosely**: the with-pedestrian figure includes episodes
  where v1 stops for the pedestrian and waits. v0's negative penalty is it accelerating toward 13.9 m/s from slower
  starts. v0's penalty excludes its 23 collisions (the finish is never reached); v1 has none to exclude.
- **There was no needless stop.** Neither planner dropped below 1 m/s without a pedestrian (0 of 104, upper bound 3.6%),
  so there is no needless-stop clip. The second clip is the closest thing: the pedestrian-absent episode with the largest
  time penalty.
- Only a quarter of v0's pedestrian episodes have a finite min TTC, since constant-velocity extrapolation rarely
  predicts an overlap until the pedestrian is in the lane.

## Clips (planner v1, `docs/results/clips/`)

| Clip | Seed | What it shows |
| --- | --- | --- |
| `near_miss_seed10097_v1.mp4` | 10097 | Pedestrian present, no collision, smallest centre distance of any non-collision episode: 0.67 m (0.37 m clearance from the 0.3 m disc). The ego comes to a full stop and the pedestrian passes beside it, so this is a close pass rather than a braking failure. |
| `slowdown_no_ped_seed10145_v1.mp4` | 10145 | No pedestrian; the largest time penalty (2.98 s). Ego slows from 13.0 m/s to 3.8 m/s along the 10 m occluder (the 4 m/s floor), then resumes. Not a stop, but the nearest needless-caution case. |

Re-render either with `uv run python -m tools.render_seed <seed> --planner v1`.
