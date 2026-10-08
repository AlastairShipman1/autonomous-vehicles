# Laptop work: M1 and M2

Oct 7, 2026 · @Alastair

Everything from Doc that runs on the MacBook with no ROS and no CARLA, in the order to do it.

## Scope

Thirteen PRs, about 38 hours: M1's four pure-Python PRs (13 h) and all of M2 (25 h). Twelve can be done start to finish on the laptop.

The exception is PR 2.8 (MCAP replay, 2 h). It needs one bag recorded on the desktop in PR 1.9, so leave it until that exists; everything else in M2 runs without it.

Not here: PRs 1.5 to 1.9 (messages, harness, ROS nodes, Foxglove bridge, the CARLA run). They stay in the build spec and need the desktop.

## Setup

Python 3.12 to match the desktop, so `av_core` behaves identically on both; no ROS on the Mac.

- Python 3.12 venv (uv or venv) at the repo root.
- `pip install -e src/av_core src/av_sim_toy` plus `numpy scipy matplotlib pytest hypothesis mcap mcap-ros2-support`.
- `brew install ffmpeg` for exporting toy-sim videos.
- Foxglove desktop app, for viewing bags copied from the desktop (only needed for PR 2.8).

Repo folders you'll touch from the laptop:

```text
src/av_core/        pure Python, no ROS: types, control/, plan/, predict/, sweep/, geometry/
src/av_sim_toy/     2D occlusion sim
tools/              mcap_io, run scripts
tests/              pytest, runs here and in CI
```

`src/av_interfaces/` and `src/av_ros/` also live in the repo but are desktop-only (colcon); leave them alone from the laptop.

## Shared contracts

These types are PR 1.2 and everything else builds on them, so review them line by line. The desktop's ROS messages will mirror them field for field, which is why they freeze at the end of M1.

```python
# av_core/types.py  (frozen dataclasses; arrays are numpy, float64)
EgoState(x, y, yaw, speed, length, width, wheelbase)
Agent(id: int, cls: 'vehicle' | 'pedestrian' | 'cyclist',
      x, y, yaw, vx, vy, length, width, is_static: bool)
OccludedRegion(occluder_id: int, polygon: (K, 2) array)       # filled by the toy sim in M2
TrafficLight(id: int, state: 'red' | 'yellow' | 'green' | 'unknown',
             stop_line: (2, 2) array)                          # the stop line's two endpoints
WorldModel(stamp, ego: EgoState, agents: tuple[Agent, ...],
           occluded: tuple[OccludedRegion, ...],
           traffic_lights: tuple[TrafficLight, ...])           # every known light; consumers decide which apply
Route(points: (N, 2) array, speed_limit: float)                # densified to 0.5 m spacing
PredictedTrajectory(agent_id, t: (H,) array, xy: (H, 2) array, prob: float = 1.0)
PlannerCommand(stamp, target_speed, reason: str)               # reason: 'route' | 'lead' | 'light' | 'conflict' | 'occlusion'
ControlCommand(stamp, throttle, brake, steer)                  # CARLA ranges: [0,1], [0,1], [-1,1]
```

- **Frame.** One `map` frame, right-handed, x and y in metres, yaw in radians counter-clockwise from +x, speeds in m/s, time in seconds. CARLA's left-handed frame is converted on the desktop only; nothing on the laptop sees it.
- **Rate.** 20 Hz (dt = 0.05 s), matching the Leaderboard, so parameters tuned in the toy carry over.
- **Stamps.** Every output carries the stamp of the WorldModel it was computed from.

## Tasks

Do them in this order: 1.1 to 1.4 first, because the toy sim reuses the bicycle model and planner v0. Each row is one PR.

| # | PR | Who | Hours | Done when | Status |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Repo skeleton, `pyproject`s, pytest, GitHub Actions running `tests/` on Ubuntu and macOS | Claude | 2 | CI green on an empty test | Not started |
| 1.2 | `av_core/types.py` as in Shared contracts, with validation (finite values, valid ranges) | Claude writes, you review | 2 | Types construct, reject NaN, round-trip to dict | Not started |
| 1.3 | Kinematic bicycle model and controller (pure pursuit + PID) in `av_core/control` | You write the control law, Claude the tests | 5 | Controller tests below pass | Not started |
| 1.4 | Planner v0 in `av_core/plan/rule_based.py`, tested on synthetic WorldModels | You write the rules | 4 | Empty road gives route speed; lead car and red light give the stopping profile | Not started |
| 2.1 | `av_sim_toy` core: road, ego on the bicycle model, parked occluder, pedestrian, collision, episode end | Claude writes, you review the geometry | 5 | Same seed gives a bit-identical trajectory; collision cases pass | Not started |
| 2.2 | Visibility and occluded-region polygon in `av_core/geometry` | You write, Claude tests | 3 | Geometry tests below pass | Not started |
| 2.3 | Scenario sampler: tuning seeds 0–999, reporting seeds 10000–10999 | Claude | 1 | Parameters reproducible from the seed alone | Not started |
| 2.4 | Renderer: matplotlib animation to MP4 with occluded region, pedestrian (visible or hidden) and target speed with reason | Claude | 2 | One-command video of any seed | Not started |
| 2.5 | Constant-velocity predictor in `av_core/predict` (3 s horizon, 0.1 s steps) | You | 1 | Straight-line cases exact | Not started |
| 2.6 | Planner v1: v0 plus predicted-conflict stop and occlusion speed cap | You | 4 | Planner v1 tests below pass | Not started |
| 2.7 | Sweep harness in `av_core/sweep`: episodes, metrics, CSV, summary with intervals | You define the metrics, Claude builds it | 5 | 200 episodes in under 5 minutes | Not started |
| 2.9 | First sweep: 200 reporting-seed episodes, results table, clips of a near miss and a needless stop | You | 2 | Table and two clips saved | Not started |
| 2.8 | `tools/mcap_io`: read `/world_model` and `/planner_cmd` from a desktop bag, replay into the planner. Waits for a bag from PR 1.9 | Claude | 2 | Replay reproduces the logged planner commands | Not started |

2.9 sits before 2.8 here on purpose: the first sweep doesn't need the bag.

## M1 specs

The constants are starting guesses to tune, not results. The tests check behaviour, not the constants.

### Controller (PR 1.3)

Pure pursuit picks a target point L\_d ahead along the route; α is its bearing in the ego frame and L the wheelbase.

```latex
L_d = \operatorname{clip}(0.5\,v + 3,\ 4,\ 12)\ \text{m}, \qquad \delta = \arctan\frac{2L\sin\alpha}{L_d}, \qquad \text{steer} = \operatorname{clip}(\delta/\delta_{\max},\ -1,\ 1)
```

δ\_max is a parameter here; on the desktop it comes from the vehicle's physics control. Speed is a PID on e = v\* − v with a clamped integral: positive output becomes throttle (capped at 0.75), negative output becomes brake. Start at K\_p = 0.5, K\_i = 0.05, K\_d = 0.

Tests, on the bicycle model with acceleration = 4·throttle − 8·brake m/s² (a stand-in for CARLA's dynamics):

- A straight route and a 30 m radius circle at 8 m/s keep lateral error under 0.3 m after the first 2 s.
- A speed step from 0 to 10 m/s settles within 0.5 m/s in 6 s, with under 1 m/s overshoot.

The gains get retuned in CARLA later; these tests only prove the control law is right.

### Planner v0 (PR 1.4)

The target speed is the minimum of three limits, and `reason` names the one that bound.

```latex
v^\ast = \min\left(v_{\text{route}},\ v_{\text{lead}},\ v_{\text{light}}\right)
```

```latex
v_{\text{route}} = \min\left(v_{\text{limit}},\ \min_{s \in [s_0,\, s_0 + 30]} \sqrt{a_{\text{lat}} / |\kappa(s)|}\right), \qquad a_{\text{lat}} = 2\ \text{m/s}^2
```

```latex
v_{\text{lead}} = \sqrt{2\, b\, \max(0,\ s_{\text{lead}} - d_0)}, \qquad b = 3\ \text{m/s}^2,\ d_0 = 6\ \text{m}
```

- κ(s) is the route curvature from the resampled points; s₀ is ego's arc-length position (closest point).
- s\_lead is the arc length to the nearest agent ahead whose lateral offset from the route is below half the ego width plus half its width plus 0.3 m.
- v\_light uses the v\_lead formula with the stop line as the obstacle and d₀ = 2 m, for each light that is not green (red, yellow or unknown) whose stop-line midpoint is within 2 m of the route and that ego can still stop at b. The nearest binds; with none it is infinite.

Test it with hand-built WorldModels and Routes; no simulator is needed.

## M2 specs

The toy is one scenario on purpose: a pedestrian hidden behind a parked vehicle, which is what the AIF planner targets in M3.

### Toy scenario (PRs 2.1, 2.3)

| Element | Spec | Sampled per seed |
| --- | --- | --- |
| Road | Straight along +x; ego lane centre y = 0, width 3.5 m; parking lane centre y = −2.75 m; sidewalk below y = −3.75 m | none |
| Ego | Bicycle model from PR 1.3; 4.7 × 1.9 m, wheelbase 2.9 m; starts at x = 0, y = 0; speed limit 13.9 m/s | Initial speed U(8, 13) m/s |
| Occluder | Parked vehicle, width 2.0 m, centred at y = −2.75 m; in the WorldModel as a static agent | Centre x U(40, 60) m; length one of 4.5, 6.0, 10.0 m |
| Pedestrian | Disc of radius 0.3 m at y = −4.5 m, walks straight across in +y once triggered; in the WorldModel only while visible | Present with probability 0.5; x within the occluder span +1 m; speed U(0.8, 2.0) m/s; trigger when ego front is U(10, 35) m short of it |
| Sensor | Ego front-centre point, 50 m range, no noise | none |
| End | Collision, ego rear 30 m past the occluder, or 30 s | none |

Episodes without a pedestrian matter as much as those with one: they are how over-caution gets measured.

### Visibility and occluded region (PR 2.2)

- A point is visible if it is within range and the segment from the sensor to it doesn't pass through the occluder rectangle's interior. Grazing an edge counts as visible.
- The occluded region is the shadow quad: the two occluder corners at the extreme bearings from the sensor, each extended along its bearing ray out to the 50 m range.
- Tests: hand-built cases (directly behind, beside, along the corner ray), plus a hypothesis property test: a random point inside the shadow and range is never visible, and a random point outside the shadow and occluder is always visible.

### Planner v1 (PR 2.6)

Planner v1 adds two limits to v0's minimum. Both are starting points for you to rewrite. The AIF planner gets compared against this, so it should be sensible, not a strawman.

- **Predicted conflict.** For each prediction, find the first time t\_c its position enters ego's corridor (half ego width + 0.5 m) at arc length s\_c. With ego arrival time t\_e = (s\_c − s₀)/max(v, 0.1), if |t\_e − t\_c| < 2 s, treat s\_c as a lead obstacle in the v\_lead formula.
- **Occlusion cap.** With s\_occ the arc length of the nearest route point ahead within 2.5 m of any occluded region:

```latex
v_{\text{occ}} = \max\left(v_{\text{floor}},\ \sqrt{2\, b_{\text{occ}} \max(0,\ s_{\text{occ}} - s_0 - d_0)}\right), \qquad v_{\text{floor}} = 4\ \text{m/s},\ b_{\text{occ}} = 4\ \text{m/s}^2
```

Tune v\_floor and b\_occ on tuning seeds only. Unit tests: no occluder leaves v1 equal to v0; a pedestrian on a crossing course 20 m ahead produces a stop with `reason = 'conflict'`.

### Sweep metrics (PR 2.7)

| Metric | Definition | Episodes |
| --- | --- | --- |
| Collision | Ego rectangle and pedestrian disc overlap at any step | Pedestrian present |
| Minimum distance | Smallest distance from pedestrian centre to the ego rectangle | Pedestrian present |
| Minimum TTC | Smallest time until overlap if both held constant velocity, searched to 5 s; infinite if never | Pedestrian present |
| Braking onset | Distance from ego front to the occluder's near end when deceleration first exceeds 1 m/s²; negative if already past | All |
| Needless stop | Ego speed drops below 1 m/s | Pedestrian absent |
| Time penalty | Time to reach 30 m past the occluder, minus the time at constant initial speed | All |

Report rates with Wilson 95% intervals and continuous metrics with bootstrap 95% intervals (2,000 resamples), split by pedestrian present and absent. Write one CSV row per episode so later plots don't need a rerun. The harness takes a scenario factory and a planner factory, so M4 can point it at CARLA unchanged.

### MCAP replay (PR 2.8, needs a desktop bag)

Read `/world_model` and `/planner_cmd` with the `mcap` and `mcap-ros2-support` libraries, which decode the message definitions stored in the bag, so no ROS install is needed. Convert to `av_core` types, run planner v0 over the recorded WorldModels and check its output matches the logged commands. That match is the regression test that the laptop and desktop code paths agree.
