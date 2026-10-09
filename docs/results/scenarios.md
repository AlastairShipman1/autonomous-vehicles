# Hand-built scenarios

Twenty-two fixed scenarios for watching a planner on harder streets than the seeded single-car one. Each has one video per
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
| `slow_lead_car` | A car ahead in the lane doing 7 m/s; a hazard further on | finished, +6.1 s | finished, +6.6 s | finished, +8.2 s |
| `lead_brakes_hard` | The lead car brakes at 5 m/s² from 11 to 3 m/s; nobody else | finished, +14.1 s | finished, +14.1 s | finished, +14.1 s |
| `follow_the_leader` | Following a 10 m/s car past the gap; the leader holds the ego back | finished, +4.0 s | finished, +6.2 s | finished, +6.5 s |
| `truck_ahead_hides_view` | A 10 m truck 12 m ahead blocks the view of the roadside | finished, +4.9 s | finished, +5.8 s | finished, +5.7 s |
| `oncoming_traffic` | A car and a van in the opposite lane, gone before the pedestrian steps out | finished (0.33), −1.5 s | finished, +4.3 s | finished, +5.2 s |
| `oncoming_no_pedestrian` | The same oncoming traffic, nobody hidden | finished, −1.5 s | finished, +1.6 s | finished, +0.2 s |
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
- **Moving vehicles.** In the lane ahead the three planners behave alike: they follow. With the lead braking to 3 m/s
  (`lead_brakes_hard`) all three finish with identical times (+14.1 s, behind the lead throughout) and nobody hits it.
  v0's stopping rule ignores the lead's speed, and on `slow_lead_car` and `follow_the_leader` its minimum speed is 0, so it
  is stopping short of the lead rather than matching its speed. Oncoming vehicles do not change the lowest speed of v0 or
  v1 by more than 0.3 m/s against the same scenario without them (tested; I did not test the AIF planner).
- **The pedestrian-absent oncoming scenario separates the planners the same way the no-traffic one does** (v1 slows
  to 3.8 m/s for the parked car; the AIF planner stays above 8.9 m/s).
- **The moving vehicles do not react and are not hit by pedestrians.** Constant speed, one optional braking event; a
  pedestrian can in principle walk through one (the oncoming scenarios are timed so that does not happen, and a test checks
  it for all three planners). The AIF planner treats any vehicle beside the route as a parked occluder.

## Not covered

Vehicles that react to the ego or to pedestrians, overtaking, cut-ins and vehicles pulling out; a curved road (the planners handle one, the sim does not draw it);
occlusion from the opposite side hiding a pedestrian on the right.
