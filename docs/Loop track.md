# The loop track

An endless circular track with a gauntlet of obstacles, for watching a planner drive round and round and for counting
what goes wrong. It is the straight-road toy sim's successor for open-ended testing: same ego (bicycle model, pure
pursuit, speed PID), same planner interface, same WorldModel, but nothing ends the run.

```sh
uv run python -m tools.loop                                  # window; planner v1; the first gauntlet
uv run python -m tools.loop --planner aif                    # start under another planner (TAB cycles in the window)
uv run python -m tools.loop --headless --laps 5 --planner all   # lap tables and the incident log, no window
uv run python -m tools.loop --shot out/loop.png --at 26      # a screenshot of t = 26 s (--overview for the whole track)
```

Window keys: SPACE pause, N one step while paused, `[` `]` speed (0.25x to 4x), V follow camera or whole track,
R restart, TAB restart under the next planner, Q quit. Heads-up lines show the planner, lap, speed, target speed and
reason, planning time, and running counts of collisions, red lights run and close calls; the latest incident flashes
as a red banner. The minimap (top right) shows the whole loop.

## How it works

- **Track** (`loop/track.py`): a closed centreline in arc length `s` (wraps at the lap length) and lateral offset
  (left positive, so on this counter-clockwise circle left is towards the middle). Lane 0 is the right lane (the
  ego's), lane 1 the one to its left; the parking lane and sidewalk are to the right of lane 0, an inner sidewalk to
  the left of lane 1. Only a circle (radius 100 m, a 45 s lap at the speed limit) exists, but everything goes through
  `Track.pose` and `Track.project`, so another closed shape is a different set of points.
- **Endless route:** planners and the controller still get an ordinary open `Route`. The sim hands them a 250 m window
  of lane 0 starting a little behind the ego and makes a new one every 45 m or so, so the loop never runs out.
- **Course** (`loop/course.py`): plain data. `ParkedCar`, `Walker`, `Signal`, `TrafficCar`, collected in a `Course`
  (validated: a walker's path may not run through a parked car, traffic may not start on top of each other).
  `default_course` lays out the first gauntlet as fractions of a lap:
  a traffic light; a row of three parked cars with a pedestrian who runs out from behind the last one (hidden until
  the ego is almost level); an open crosswalk where a pedestrian crosses in plain sight; four cars circulating in the
  left lane.
- **Walkers** wait on the sidewalk, cross when the ego is `trigger_distance` short of them, finish on the far
  sidewalk, and re-arm once the ego has gone by, so every lap meets them fresh.
- **Traffic cars** follow the Intelligent Driver Model: they keep a gap to whatever is ahead in their lane (the ego,
  another car, a pedestrian in the road, a parked car) and stop for red and yellow lights. They never change lane.
- **Incidents** are logged once per contact and the ego drives on: a collision (the ego's body overlaps a vehicle or
  pedestrian), a red light run (the ego's front crosses a stop line while it is red) and a close call (within 1 m of a
  pedestrian while moving). There is no physical response to contact. Each lap becomes a row: time, mean speed, and the
  counts.
- `LoopRunner` is the plan-control-step loop; `LoopSession` holds the window's controls (testable without a display).

## What it shows today

Three laps of the first gauntlet, from `--headless --laps 3 --planner all` on this branch:

| planner | lap time | collisions | red lights | close calls |
|---|---|---|---|---|
| v0 | 45 to 52 s | 1 per lap (the hidden runner) | 0 | 2 per lap |
| v1 | 64 to 71 s | 0 | 0 | 0 |
| AIF | 62 to 71 s | 1 in 3 laps (the open crosswalk) | 0 | 1 per lap |

One lap is one sample of each obstacle and the timings depend on the light phase, so read these as "the gauntlet
discriminates", not as a ranking. AIF handles occlusion but not a pedestrian crossing in plain sight; v1's conflict
check does. That is the kind of thing the loop is for.

## A planner bug this found

Planner v0 released a red light as soon as the ego "could no longer stop at b", checked every step. With a speed PID
that lags its target, the check tripped just as the stopping profile started to bind, and the ego drove through the red
at full speed. `RuleBasedPlanner._light_limit` now releases only a *yellow* (the dilemma-zone case the rule was written
for); a red keeps its stopping profile and the ego brakes as hard as it must. Unit test added; the straight-road
sweeps have no lights, so their numbers are unchanged.

## Adding to the gauntlet

1. A frozen dataclass in `course.py` for the new thing, a tuple of them on `Course`, and a check in `Course.validate`.
2. Its state and motion in `LoopSim` (`__post_init__`, `_advance_*`), what the ego can see of it in `world_model`,
   and any contact rule in `_log_contacts`.
3. Draw it in `view.py`, and put it somewhere in `default_course` (or make a new named course in `tools/loop.py`).
4. A test in `tests/test_loop.py` that the planners can be told apart by it.

On the list, not built: lane changes and overtaking (the planners only change speed along a fixed lane today, and the
ego never leaves lane 0), a stalled car in the ego's lane, a lane closure, an emergency vehicle, other tracks (oval,
figure of eight with a crossing), traffic that reacts badly, and a human-driven car or pedestrian in the window.

## Limits

- The ego follows lane 0 for ever; traffic stays in its lane. No overtaking exists yet, so a slow car in lane 0 would
  hold the ego behind it permanently.
- Contacts are counted, not simulated. A collision costs nothing but a tally.
- Traffic ignores walkers that have not stepped out, lights are ground truth, and perception is perfect except for
  line of sight, as in the straight-road sim.
- AIF's model was built for one occluder beside the route. By its rules a traffic car in the left lane is also an
  occluder beside the route; I have not checked how much that changes its behaviour, so read its numbers here with care.
- Not run in a real window (the container has no display): drawing was checked on offscreen screenshots.
