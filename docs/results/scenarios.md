# Hand-built scenarios

Sixteen fixed scenarios for watching a planner on harder streets than the seeded single-car one. Each has one video per
planner in `docs/results/scenarios/<scenario>__<planner>.mp4` (v0, v1, aif). Regenerate with:

```sh
uv run python -m tools.render_scenario --list
uv run python -m tools.render_scenario --all --planner v0 v1 aif --out docs/results/scenarios
uv run python -m tools.render_scenario row_of_cars --planner v1 aif     # just some
uv run python -m tools.render_scenario --all --planner v0 v1 aif --no-video   # table only, 20 s
```

These are single deterministic runs for looking at, **not statistics**: one run per cell says what happened here, not how
often. The sweeps (`tools.run_sweep`) are for rates.

| Scenario | What it tests | v0 | v1 | aif |
| --- | --- | --- | --- | --- |
| `single_car` | The baseline: one 6 m car, pedestrian steps out past it | finished, grazes the pedestrian's 0.3 m disc (0.33) | finished, stops, +4.3 s | finished, +4.8 s |
| `row_of_cars` | Four cars nose to tail; pedestrian steps out past the last | **collision** | finished, +4.8 s | finished, +4.1 s |
| `gap_between_cars` | Two cars, 2.5 m gap; pedestrian steps out of the gap | **collision** | finished, +5.6 s | finished, +5.7 s |
| `long_truck` | 12 m truck hides the pedestrian until almost alongside | finished (0.33) | finished, +4.2 s | finished, +6.6 s |
| `cars_both_sides` | A parked row on the left too; pedestrian on the right | finished (0.33) | finished, +5.4 s | finished, +5.2 s |
| `ped_from_left` | Mirror image of `single_car` | finished (0.33) | finished, +4.3 s | finished, +4.8 s |
| `left_cars_ped_right` | Cars both sides; pedestrian from the right-hand gap, crossing a gap in the left row | **collision** | finished, +7.0 s | finished, +5.2 s |
| `dart_out` | 2.0 m/s pedestrian, 12 m/s ego, in the lane just as the ego arrives | **collision** | finished, +3.8 s | finished, +4.6 s |
| `slow_ped` | 0.8 m/s pedestrian triggered 35 m out; should be easy | finished (1.31) | finished, +6.5 s | finished, +6.8 s |
| `two_peds_one_gap` | Two pedestrians out of the same gap, the second a moment later | **collision** | finished, +6.6 s | finished, +6.6 s |
| `second_pedestrian_follows` | One crosses early and is gone; a second follows from the same spot later | **collision** | finished, +4.1 s | finished, +4.8 s |
| `group_of_children` | Three children (1.1 to 1.6 m/s) stepping out one after another | **collision** | finished, +5.3 s | finished, +6.8 s |
| `peds_both_sides` | One pedestrian from behind a right-hand car, another from behind a left-hand one | **collision** | finished, +5.8 s | finished, +4.9 s |
| `row_two_gaps` | Three cars, a pedestrian in each of the two gaps | **collision** | finished, +6.9 s | finished, +6.7 s |
| `row_no_pedestrian` | The four-car row, nobody there: any slowing is over-caution | finished, −1.7 s | finished, +4.8 s | finished, +0.5 s |
| `truck_no_pedestrian` | The truck, nobody there | finished, −1.5 s | finished, +2.0 s | finished, +1.8 s |

Times are the time penalty (seconds slower than constant initial speed; v0 is negative because it accelerates to
13.9 m/s). Numbers in brackets are the closest the pedestrian's centre came to the ego rectangle: 0.3 m is contact, so
0.33 is a graze.

## What to look at

- **v0 on `row_of_cars`, `gap_between_cars`, `dart_out`.** It does not slow at all and hits the pedestrian as they step
  into the lane. The four v0 collisions are the ones where the pedestrian is in the lane when the ego arrives; in the
  others it gets by, narrowly.
- **v1 stops dead for the pedestrian in most scenarios** (minimum speed 0): it brakes for the occluded region, then for the
  pedestrian when they appear, and waits. The AIF planner does the same here. That is why the pedestrian-absent scenarios
  are the interesting comparison.
- **`row_no_pedestrian`: v1 +4.8 s, aif +0.5 s.** With nobody there the AIF planner keeps most of its speed (lowest
  8.6 m/s) while v1 slows to 3.8 m/s. That is the over-caution gap from the sweep numbers, visible in one scenario. I have
  not looked into why the AIF planner slows less here (`tools.aif_trace` would show its belief and G over the run).
- **The multi-pedestrian scenarios are lethal for v0 and handled by the others, mostly by waiting.** v1 comes to a full
  stop in all five and the AIF planner in four (it slows to 0.3 m/s in `second_pedestrian_follows`), so the interesting
  question is how long they wait. The AIF planner's model has one pedestrian hypothesis behind the nearest hazard; I
  have not looked at how it copes with two, or for cases where that breaks.
- **The sim is still simple.** No moving vehicles, a straight road. A truck or van in the ego's own lane (a lead vehicle
  that also occludes) is not a scenario yet.

## Not covered

Moving traffic or an oncoming lane; a curved road (the planners handle one, the sim does not draw it);
occlusion from the opposite side hiding a pedestrian on the right.
