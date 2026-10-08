# M1 and M2 build spec

Oct 7, 2026 · @Alastair

PR-sized tasks, acceptance criteria and specs for M1 (ground-truth loop) and M2 (laptop loop) from Doc.

## Where each piece runs

M2 is almost all laptop work. M1 is not: anything that touches CARLA, the Leaderboard harness or ROS needs the desktop. About 13 of M1's 30 hours (types, controller, planner rules) are pure Python and can be done on the laptop first, then dropped into ROS on the desktop.

|  | Laptop (M1 MacBook, no ROS) | Desktop (Ubuntu 24.04, Jazzy, CARLA 0.9.16) |
| --- | --- | --- |
| M1 | Repo skeleton, `av_core` types, controller with bicycle-model tests, planner v0 rules | Messages, harness agent, ROS nodes, Foxglove, end-to-end run on route 24206 |
| M2 | Toy sim, visibility geometry, predictor, planner v1, sweep harness, MCAP replay, first sweep | Nothing new; only bags recorded in M1 |

One change from the plan: `av_core` starts as pure Python in M1's first PR, so M2's restructure task disappears and its hours go to the sweep harness.

## Setup

Both machines use Python 3.12 so `av_core` behaves identically; the laptop never installs ROS.

**Laptop**

- Python 3.12 venv (uv or venv) at the repo root.
- `pip install -e av_core av_sim_toy` plus `numpy scipy matplotlib pytest hypothesis mcap mcap-ros2-support`.
- `brew install ffmpeg` for exporting toy-sim videos.
- Foxglove desktop app for viewing MCAP bags copied from the desktop.

**Desktop**

- Same venv recipe, plus CARLA's `cp312` wheel and the Leaderboard requirements from the Bench2Drive setup notes in the 2027 Plan.
- `colcon` workspace containing `av_interfaces` and `av_ros`; source Jazzy, then the venv, so `rclpy` and `carla` import in one process.
- `sudo apt install ros-jazzy-foxglove-bridge`.
- Bags live outside the repo (for example `~/av_bags/`) and are copied to the laptop by `rsync` or a shared drive.

**Repo**

```text
av_core/        pure Python, no ROS: types, control/, plan/, predict/, sweep/, geometry/
av_sim_toy/     2D occlusion sim (M2)
av_interfaces/  ROS 2 messages (colcon, desktop only)
av_ros/         thin nodes and the harness agent (colcon, desktop only)
tools/          mcap_io, Foxglove layouts, run scripts
tests/          pytest for av_core and av_sim_toy, runs on both machines and in CI
```

## Shared contracts

These types are the one thing every later milestone depends on, so review them line by line and freeze them at the end of M1. The ROS messages in `av_interfaces` mirror them field for field, and one module in `av_ros` converts between the two.

```python
# av_core/types.py  (frozen dataclasses; arrays are numpy, float64)
EgoState(x, y, yaw, speed, length, width, wheelbase)
Agent(id: int, cls: 'vehicle' | 'pedestrian' | 'cyclist',
      x, y, yaw, vx, vy, length, width, is_static: bool)
OccludedRegion(occluder_id: int, polygon: (K, 2) array)       # empty in M1, filled by the toy sim in M2
WorldModel(stamp, ego: EgoState, agents: tuple[Agent, ...],
           occluded: tuple[OccludedRegion, ...],
           light: 'none' | 'red' | 'yellow' | 'green',
           stop_line: (2,) array | None)                       # stop line of the light affecting ego
Route(points: (N, 2) array, speed_limit: float)                # densified to 0.5 m spacing
PredictedTrajectory(agent_id, t: (H,) array, xy: (H, 2) array, prob: float = 1.0)
PlannerCommand(stamp, target_speed, reason: str)               # reason: 'route' | 'lead' | 'light' | 'conflict' | 'occlusion'
ControlCommand(stamp, throttle, brake, steer)                  # CARLA ranges: [0,1], [0,1], [-1,1]
```

Conventions:

- **Frame.** One `map` frame, right-handed, x and y in metres, yaw in radians counter-clockwise from +x, speeds in m/s, time in seconds.
- **CARLA is left-handed.** Convert only at the harness boundary: y = −y\_carla, vy = −vy\_carla, yaw = −radians(yaw\_carla). Nothing else in the codebase knows about CARLA's frame.
- **Rate.** 20 Hz (dt = 0.05 s), the Leaderboard's fixed step. The toy sim uses the same dt so tuned parameters carry over.
- **Stamps.** Every message carries the sim time of the WorldModel it was computed from. A `ControlCommand` is valid only for the frame whose stamp it carries.
- **Topics.** `/world_model`, `/route` (latched), `/predictions`, `/planner_cmd`, `/control_cmd`, `/clock`, plus `/viz/*` markers.

## M1 PRs (30 h)

Nine PRs; the four laptop ones can all land before the first desktop session. M1 is done when your own nodes drive route 24206 on ground truth with zero control misses, and the run is recorded and viewable in Foxglove.

| # | PR | Where | Who | Hours | Done when |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Repo skeleton, `pyproject`s, pytest, GitHub Actions running `tests/` on Ubuntu and macOS | Laptop | Claude | 2 | CI green on an empty test |
| 1.2 | `av_core/types.py` as in Shared contracts, with validation (finite values, unit-length checks) | Laptop | Claude writes, you review | 2 | Types construct, reject NaN, round-trip to dict |
| 1.3 | Kinematic bicycle model and controller (pure pursuit + PID) in `av_core/control`, tested on the bicycle model | Laptop | You write the control law, Claude the tests | 5 | Tests in M1 specs pass |
| 1.4 | Planner v0 in `av_core/plan/rule_based.py` | Laptop | You write the rules | 4 | Unit tests: empty road gives route speed; lead car and red light give the stopping profile |
| 1.5 | `av_interfaces` messages and `av_ros/convert.py` | Desktop | Claude | 3 | `colcon build` clean; dataclass → msg → dataclass round-trip is exact |
| 1.6 | `av_harness` Leaderboard agent: sensors, ground-truth WorldModel, dense route, `/clock`, frame-matched control | Desktop | Claude, you review the frame conversion | 6 | Agent runs route 24206 with a dummy constant-brake controller and logs zero misses |
| 1.7 | Planner and control nodes wrapping `av_core`, one launch file | Desktop | Claude | 3 | Launch file brings up harness, planner, control |
| 1.8 | `foxglove_bridge`, markers (route, agent boxes, target speed and reason), saved layout, `tools/record.sh` (MCAP) | Desktop | Claude | 2 | Layout in the repo opens a recorded bag correctly |
| 1.9 | End-to-end run on route 24206: retune gains in CARLA, record the clip, check whether Leaderboard 2.1 submissions are open | Desktop | You | 3 | Route completed, no collisions or red-light infractions, clip saved |

The harness uses CARLA's privileged actor list for ground truth. That's fine for local runs; a SENSORS-track submission forbids it, which is what M5's perception replaces.

## M1 specs

Starting values below are guesses to tune, not results; the tests check behaviour, not the constants.

### Harness (PR 1.6)

- Subclass the Leaderboard `AutonomousAgent`. `sensors()` declares one front RGB camera and one LiDAR (published for Foxglove now, used in M5).
- `set_global_plan` gives a sparse route. If any spacing exceeds 2 m, densify it with CARLA's `GlobalRoutePlanner` at 1 m, then resample to 0.5 m. Publish once on `/route`.
- Each `run_step`: build the WorldModel from the actor list (vehicles and walkers within 60 m, `is_static` for parked vehicles), the light affecting ego and its stop line. Convert frames, then publish `/clock`, sensors and `/world_model`.
- Block until a `/control_cmd` with the matching stamp arrives, up to 0.5 s wall time. On a miss, reapply the previous command, increment a counter and log the stamp. Never step the sim with a stale command silently.
- Run `rclpy` in a background executor thread inside the agent process; the Leaderboard owns the main thread.

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
- v\_light uses the same formula as v\_lead, with the stop line in place of the lead and d₀ = 2 m, when the light is red or yellow and ego can still stop at b. Otherwise it is infinite.

### Controller (PR 1.3)

Pure pursuit picks a target point L\_d ahead along the route; α is its bearing in the ego frame and L the wheelbase.

```latex
L_d = \operatorname{clip}(0.5\,v + 3,\ 4,\ 12)\ \text{m}, \qquad \delta = \arctan\frac{2L\sin\alpha}{L_d}, \qquad \text{steer} = \operatorname{clip}(\delta/\delta_{\max},\ -1,\ 1)
```

δ\_max comes from the vehicle's physics control in CARLA and is a parameter in the toy. Speed is a PID on e = v\* − v with clamped integral: a positive output becomes throttle (capped at 0.75), a negative one brake. Start at K\_p = 0.5, K\_i = 0.05, K\_d = 0.

Laptop tests on the bicycle model (acceleration = 4·throttle − 8·brake m/s², a stand-in for CARLA's dynamics):

- A straight route and a 30 m radius circle at 8 m/s keep lateral error under 0.3 m after the first 2 s.
- A speed step from 0 to 10 m/s settles within 0.5 m/s in 6 s with under 1 m/s overshoot.

Expect to retune gains in CARLA in PR 1.9; the bicycle model only proves the control law is right.

## M2 PRs (25 h, all laptop)

M2 is done when a 200-episode seeded sweep of the rule-based planner produces a results table and a video, entirely on the laptop. Only PR 2.8 needs anything from the desktop: one bag recorded in PR 1.9.

| # | PR | Who | Hours | Done when |
| --- | --- | --- | --- | --- |
| 2.1 | `av_sim_toy` core: straight road, ego on the M1 bicycle model, parked occluder, pedestrian, collision check, episode end | Claude writes, you review the geometry | 5 | Same seed gives a bit-identical trajectory; collision test cases pass |
| 2.2 | Visibility and the occluded-region polygon in `av_core/geometry` | You write, Claude tests | 3 | Tests in M2 specs pass |
| 2.3 | Scenario sampler with tuning seeds 0–999 and reporting seeds 10000–10999 | Claude | 1 | Parameters reproducible from the seed alone |
| 2.4 | Renderer: matplotlib animation to MP4, showing the occluded region, visible and hidden pedestrian, and the target speed with its reason | Claude | 2 | One-command video of any seed |
| 2.5 | Constant-velocity predictor in `av_core/predict` (3 s horizon, 0.1 s steps) | You | 1 | Straight-line test cases exact |
| 2.6 | Planner v1: v0 plus predicted-conflict stop and occlusion speed cap | You | 4 | Unit tests in M2 specs pass |
| 2.7 | Sweep harness in `av_core/sweep`: runs episodes, computes metrics, writes CSV and a summary with confidence intervals | You define the metrics, Claude builds it | 5 | Runs 200 episodes in under 5 minutes on the laptop |
| 2.8 | `tools/mcap_io`: read `/world_model` and `/planner_cmd` from a desktop bag on the Mac, replay into planner v1 | Claude | 2 | Replay reproduces the logged planner commands with v0 settings |
| 2.9 | First sweep: 200 reporting-seed episodes, results table and a video of a near miss and a needless stop | You | 2 | Table and two clips saved |

The sweep harness takes a scenario factory and a planner factory, so M4 can point it at CARLA without changing the metrics.

## M2 specs

The toy is deliberately one scenario: a pedestrian hidden behind a parked vehicle, which is exactly what the AIF planner targets in M3.

### Toy scenario (PRs 2.1, 2.3)

| Element | Spec | Sampled per seed |
| --- | --- | --- |
| Road | Straight along +x; ego lane centre y = 0, width 3.5 m; parking lane centre y = −2.75 m; sidewalk below y = −3.75 m | none |
| Ego | Bicycle model from M1; 4.7 × 1.9 m, wheelbase 2.9 m; starts at x = 0, y = 0; speed limit 13.9 m/s | Initial speed U(8, 13) m/s |
| Occluder | Parked vehicle, width 2.0 m, centred at y = −2.75 m; in the WorldModel as a static agent | Centre x U(40, 60) m; length one of 4.5, 6.0, 10.0 m |
| Pedestrian | Disc of radius 0.3 m at y = −4.5 m, walks straight across in +y once triggered; in the WorldModel only while visible | Present with probability 0.5; x within the occluder span +1 m; speed U(0.8, 2.0) m/s; trigger when ego front is U(10, 35) m short of it |
| Sensor | Ego front-centre point, 50 m range, no noise | none |
| End | Collision, ego rear 30 m past the occluder, or 30 s | none |

Episodes without a pedestrian matter as much as those with one: they are how over-caution gets measured.

### Visibility and occluded region (PR 2.2)

- A point is visible if it is within range and the segment from the sensor to it does not pass through the occluder rectangle's interior. Grazing an edge counts as visible.
- The occluded region is the shadow quad: the two occluder corners at the extreme bearings from the sensor, c\_a and c\_b, plus each extended along its bearing ray out to the 50 m range.
- Tests: hand-built cases (directly behind, beside, along the corner ray), plus a hypothesis property test: a random point inside the shadow and range is never visible, and a random point outside the shadow and occluder is always visible.

### Planner v1 (PR 2.6)

Planner v1 adds two limits to v0's minimum. Both are starting points for you to rewrite; the AIF planner is compared against this, so it should be sensible, not a strawman.

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

Report rates with Wilson 95% intervals and continuous metrics with bootstrap 95% intervals (2,000 resamples), split by pedestrian present and absent. Write one CSV row per episode so any later plot can be redone without rerunning.

## Work order

&#91;embedded content: work order · laptop and desktop lanes\]

Start on the laptop with 1.1 to 1.4 and carry straight on into M2; batch the five desktop PRs into one or two RDP sessions once the types are frozen. Only the MCAP replay waits on a desktop recording.
