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

## Pedestrian present (96 episodes)

| Metric | v0 | v1 |
| --- | --- | --- |
| Collision rate | 24.0% (16.5 to 33.4) | 33.3% (24.7 to 43.2) |
| Min distance, ped centre to ego rectangle (m) | 0.95 (0.82 to 1.09) | 1.89 (1.61 to 2.18) |
| Episodes with a finite min TTC | 26.0% (18.3 to 35.6) | 97.9% (92.7 to 99.4) |
| Min TTC where finite (s) | 0.18 (0.00 to 0.46) [25] | 0.71 (0.58 to 0.83) [94] |
| Braking onset, ego front to occluder near end (m) | 0.91 (-1.39 to 3.28) [17] | 21.93 (21.14 to 22.75) [96] |
| Time penalty (s) | -1.97 (-2.18 to -1.76) [73] | 3.90 (3.57 to 4.22) [64] |

## Pedestrian absent (104 episodes)

| Metric | v0 | v1 |
| --- | --- | --- |
| Needless stop rate (speed < 1 m/s) | 0.0% (0.0 to 3.6) | 0.0% (0.0 to 3.6) |
| Braking onset (m) | undefined (never brakes above 1 m/s²) | 19.92 (19.58 to 20.26) [104] |
| Time penalty (s) | -2.03 (-2.23 to -1.84) [104] | 1.18 (0.98 to 1.36) [104] |

## Reading the numbers

- **v1 collides more often than v0 (33% against 24%)**, though the intervals overlap. It brakes about 20 m before the
  occluder and slows by about 1 to 4 s overall, but that does not buy safety here. This is a finding about the untuned
  baseline, not a bug I could find. In the 8 v1 collisions I looked at, the ego was slow at impact (0.4 to 4 m/s: at
  the 4 m/s occlusion floor, or after braking for a pedestrian already in the lane). One plausible mechanism is that
  the pedestrian's walk is triggered by the ego's distance, so a slower ego gives it more time to reach the lane,
  while the fast v0 ego often passes first. I did not test that, nor check where the ego was relative to the car at
  impact.
- **Time penalty excludes collisions** (the finish is never reached), so v1's 64 defined episodes are the ones that
  avoided a collision; compare with the collision rate before reading it as a cost.
- **There was no needless stop.** Neither planner dropped below 1 m/s without a pedestrian (0 of 104, upper bound 3.6%),
  so there is no needless-stop clip. The clip below is the closest thing: the pedestrian-absent episode with the largest
  time penalty.
- Only about a quarter of v0's pedestrian episodes have a finite min TTC, since constant-velocity extrapolation rarely
  predicts an overlap until the pedestrian is in the lane. The v1 figure is high because it spends so long near the
  pedestrian's path.

## Caveat that affects every number above: where the pedestrian appears

The spec samples the pedestrian's x within the occluder span +/- 1 m, and the pedestrian walks straight in +y from
y = -4.5. For most seeds that path crosses the parked car's footprint, so the pedestrian emerges from the car's top
edge at y = -1.75, only about 0.8 m (about 0.6 s at 1.4 m/s) from the ego's body. Most collisions are therefore
essentially unavoidable by sight for any planner that does not slow to a crawl well beforehand. The rates above are
only meaningful relative to one another, and if the sampler changes (PR 2.1 and 2.3 raised this) they will move a lot.

## Clips (planner v1, `docs/results/clips/`)

| Clip | Seed | What it shows |
| --- | --- | --- |
| `near_miss_seed10195_v1.mp4` | 10195 | Pedestrian present, no collision, smallest centre distance of any non-collision episode: 0.34 m (0.04 m clearance from the 0.3 m disc). Min TTC is infinite: constant velocity never predicts the overlap. |
| `slowdown_no_ped_seed10145_v1.mp4` | 10145 | No pedestrian; the largest time penalty (2.98 s). Ego slows from 13.0 m/s to 3.8 m/s along the 10 m occluder (the 4 m/s floor), then resumes. Not a stop, but the nearest needless-caution case. |

Re-render either with `uv run python -m tools.render_seed <seed> --planner v1`.
