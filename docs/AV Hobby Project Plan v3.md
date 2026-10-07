# AV Hobby Project Plan v3

Oct 7, 2026 · @Alastair

## Summary

Build a full CARLA + ROS 2 driving stack over about 300 hours (Oct 2026 to Oct 2027), with active inference (AIF) as one component: the ego behaviour planner for occluded-pedestrian situations. Ground truth comes first and perception is swapped in later behind the same interface. The end evaluation is CARLA Leaderboard 2.1, with Bench2Drive as the fallback. A standalone AIF package is optional and decided at a gate after AIF runs in CARLA.

Principles:

- **A visible result every milestone**: a Foxglove clip or video you could show in an application. The first lands in about 5 weeks.
- **Logic in plain Python, ROS as a thin shell**: every module (planner, predictor, tracker) is a pure-Python library with a small ROS 2 node around it, so most of it runs on the laptop with no CARLA.
- **One interface, three sources**: CARLA ground truth, the laptop toy sim and, later, your perception all publish the same world-model message.
- **Claude writes the plumbing, you write and check the maths**: see Who writes what.
- **Overruns apply the fallback**: a milestone that runs over its hours takes its stated fallback rather than borrowing from the next.
- **One small C++ milestone**: most AV software roles expect C++, and you have none yet. It is scoped to a port of code you already wrote.

## Where AIF fits

AIF goes in the planner. Prediction gets a Bayesian intent filter that feeds it. The instinct that AIF is mainly for prediction is the part of the old plans to drop.

**Why planning.** AIF is a theory of how an agent chooses actions. It scores each candidate plan by expected free energy, roughly risk (how far predicted outcomes sit from preferred ones) plus ambiguity (how uncertain the agent expects to stay about hidden state). The ambiguity term makes an AIF agent pick actions that reveal information. At an occlusion, that produces slow down, look, then resume without a hand-written rule. Engström et al. (2024), a Waymo-affiliated paper, shows exactly this behaviour, which makes it the AIF work closest to the job spec you saw.

**Why not prediction.** Predicting another road user's intent uses only the inference half of AIF: updating a belief over a hidden intent from observations. That half is ordinary Bayesian filtering over a discrete state, which an HMM does equally well. Calling it AIF is fair, but it is not the distinctive part, and an interviewer who knows the field will notice. Distinctively AIF prediction would model each road user as an agent that itself minimises expected free energy (inverse planning). That is research-grade and outside this budget.

**What you still get in prediction.** The predictor infers pedestrian and vehicle intent with pymdp's state inference (or plain Bayes), then rolls out one kinematic trajectory per intent. Those predictions become the AIF planner's transition model for visible agents. A prior over hidden pedestrians covers what can't be seen.

**Why one scenario first.** Three reasons:

- The information-gathering term only does anything when there is hidden state the ego can reveal by moving. On an open road with everything visible, an AIF planner behaves like any cost-based planner, so a comparison shows nothing.
- The Leaderboard driving score is dominated by basics: route completion, red lights and collisions. A planner difference on one situation is invisible in the aggregate unless you also report per-scenario results.
- Each scenario needs its own generative model (states, observations, preferences). One model done properly beats three half-built ones.

The first scenario is the **occluded pedestrian**: Leaderboard's ParkingCrossingPedestrian and DynamicObjectCrossing. You already scored route 24206 (ParkingCrossingPedestrian) with the autopilot. An unprotected left turn with occluded oncoming traffic is the second, only after the package gate.

## Architecture

Three interchangeable sources produce one `WorldModel`; everything downstream is pure Python that doesn't know which source fed it.

&#91;embedded content: stack architecture · 4 sources, 1 interface\]

The predictor feeds the planner's transition model for visible agents; the planner also reads occluded regions straight from the WorldModel.

**Desktop (Ubuntu 24.04, Jazzy, CARLA 0.9.16, RTX 3090 Ti).** CARLA, the harness agent, the bridge for teleop and free-roam, perception, and every CARLA sweep and Leaderboard run. You reach it over RDP; Claude Code and Dispatch run PRs and overnight sweeps there.

**Laptop (M1 MacBook, no ROS).** The toy sim, the AIF model, the predictor, the tracker maths and their unit tests, all against the same `av_core` Python packages. CARLA recordings come over as MCAP files, read with the `mcap` Python libraries and viewed in the Foxglove app. Sync is plain git.

Repo layout:

```text
av_core/        pure Python, no ROS: world_model, predict/, plan/ (rule_based, aif/), track/, fuse/, control/, sweep/
av_sim_toy/     2D occlusion sim for the laptop
av_ros/         thin nodes: harness, perception, predictor, planner, control
av_interfaces/  ROS messages, frozen in M1
tools/          MCAP readers, Foxglove layouts, eval scripts
```

## Milestones

Seven milestones and a 20-hour buffer make 300 hours, finishing around October 2027 at 6 hours a week. Dates are estimates from the hour budgets; bursty weeks move them, the order doesn't change.

&#91;embedded content: milestone timeline · 7 milestones, 1 gate\]

The package gate after M4 is also the natural moment to resubmit to Waymo with the CARLA comparison in hand.

### M1: Ground-truth loop (30 h, about mid-Nov 2026)

Your own nodes drive a Leaderboard route on ground truth, recorded and replayable.

- [ ] Freeze `av_interfaces`: `WorldModel` (agents with id, class, pose, velocity, size; occluded regions), `PredictedTrajectoryArray`, `PlannerCommand`.
- [ ] `av_harness`: a Leaderboard agent that republishes sensors, `/clock` and a ground-truth `WorldModel` to ROS, and waits for the control matching the current frame (timeout plus logged miss).
- [ ] `av_control`: pure pursuit plus PID speed control.
- [ ] Rule-based planner v0: follow the route, stop for lead objects, obey lights (ground truth for now).
- [ ] Bag recording to MCAP and a saved Foxglove layout in the repo.
- [ ] Check whether Leaderboard 2.1 submissions are open.

**Visible result:** route 24206 driven by your stack, as a Foxglove clip. **Fallback:** if harness sync is still flaky at 25 h, drive the route through the bridge with scenario\_runner and fix the harness in M4.

### M2: Laptop loop (25 h, about mid-Dec 2026)

The same planner runs on the MacBook in a toy sim, and a seeded sweep produces numbers.

- [ ] Restructure as pure-Python `av_core` packages with thin ROS nodes.
- [ ] Toy 2D sim: straight road, parked-car occluder, pedestrian emerging at seeded times, ray-cast visibility, outputs the same `WorldModel` as Python objects.
- [ ] Read CARLA MCAP bags on the Mac with the `mcap` Python libraries.
- [ ] Constant-velocity predictor.
- [ ] Sweep harness shared by toy and CARLA: collision rate, minimum distance, minimum time-to-collision, braking onset, needless stops, time to clear the hazard.

**Visible result:** toy animation plus a first sweep table for the rule-based planner.

### M3: AIF planner in the toy (50 h, about mid-Feb 2027)

The toy reproduces slow, look, resume.

- [ ] Reading (about 20 h): Engström et al. (2024), Smith et al. (2022) tutorial, Parr, Pezzulo and Friston chapters 1 to 4, pymdp T-maze tutorial.
- [ ] Generative model by hand: states (distance band to occluder, ego speed band, pedestrian hidden or not, visibility), observations, actions {keep, ease off, brake}, preferences.
- [ ] Log risk, ambiguity and G per policy each step; plot over an episode.
- [ ] Toy sweep: AIF vs rule-based on the same seeds.

**Visible result:** toy video with the G breakdown plot. **Hard cap:** 30 h on the model itself. If slow, look, resume hasn't appeared, keep what you have and carry the rule-based planner forward.

### M4: AIF in CARLA (40 h, about early Apr 2027)

The AIF planner drives the occluded-pedestrian scenarios in CARLA, compared with the rule-based planner.

- [ ] Intent predictor: Bayesian intent filter plus one kinematic rollout per intent.
- [ ] Map toy quantities to CARLA ground truth, including the occluded region behind parked vehicles.
- [ ] AIF planner node behind the same interface as the rule-based one.
- [ ] Seeded CARLA sweep on ParkingCrossingPedestrian and DynamicObjectCrossing variants, about 100 episodes per planner, separate tuning and reporting seeds.
- [ ] Package gate (see below).

**Visible result:** side-by-side clips plus sweep numbers. A natural point to resubmit to Waymo.

### M5: Perception (70 h, about early Jul 2027)

Perception replaces ground truth behind the same `WorldModel`.

- [ ] Pretrained YOLO on one front camera: vehicles, pedestrians, cyclists, traffic lights.
- [ ] LiDAR ground removal and clustering.
- [ ] LiDAR-to-camera fusion using CARLA's exact calibration.
- [ ] SORT-style tracker: Kalman filter per object plus Hungarian assignment.
- [ ] Occluded regions from perceived parked vehicles; traffic-light state from YOLO crops.
- [ ] Re-run the M4 sweep on perception and report the gap to ground truth.

**Visible result:** tracked boxes and occluded regions in Foxglove, and the ground-truth vs perception table. **Fallback at 55 h:** per-frame fused detections with velocity by differencing.

### M6: C++ port (25 h, about early Aug 2027)

The Kalman tracker runs as an `rclcpp` node on the same topics, matching the Python version numerically on recorded bags.

**Fallback:** port the controller instead, which is smaller.

### M7: Leaderboard evaluation (40 h, about late Sep 2027)

- [ ] Package the agent for the SENSORS track (no privileged information), in Docker.
- [ ] Local runs on routes containing the target scenarios: driving score, infraction breakdown, per-scenario results, AIF vs rule-based.
- [ ] Submit if submissions are open; otherwise run Bench2Drive Dev10.

The writeup comes after M7, outside the 300 hours.

## Package decision gate

The AIF code lives as an ordinary module (`av_core/aif/`) until the end of M4. At that point, extract it as a pip-installable `aif_driving` package only if either test passes. Extraction is budgeted at about 10 hours from the buffer.

| Test | Pass means |
| --- | --- |
| Beats a baseline in CARLA | On the seeded occluded-pedestrian sweep, the AIF planner has fewer collisions than the rule-based planner at equal or lower over-caution (needless stops, time to clear the hazard), or equal collisions with clearly less over-caution. |
| Usable by another AV project | Depends only on numpy and pymdp, takes a plain world-model input (agents with pose, velocity, class, plus an occluded region) and returns a speed command or action, with one example that runs without your stack. |

If neither passes, the module stays in the repo and is shown through the M4 video and per-scenario numbers. Nothing else in the plan depends on the package existing.

The Oct 5 implementation spec (a domain-free AIF library with JAX backends and SPM cross-checks) is retired. It duplicates pymdp, and the AV work never needed it.

## Who writes what

Claude Code writes the plumbing through PRs; you hand-write or line-by-line check anything with maths in it, because that is what you will be asked about in an interview.

| Area | You write | You review closely | Claude writes |
| --- | --- | --- | --- |
| AIF planner | Generative model (A, B, C, D), state and action discretisation, preference tuning | pymdp calls, EFE breakdown logging | ROS node, parameter files, plots |
| Intent predictor | Intent likelihoods and transition priors | Per-intent kinematic rollout | Message conversion, markers |
| Tracking | Kalman filter (predict, update, gating) | Hungarian assignment wrapper | Track bookkeeping, node, tests scaffold |
| Fusion | LiDAR-to-camera projection | Box-to-cluster association | Calibration loading from CARLA |
| Control |  | Pure pursuit and PID gains | Controller code, node |
| Rule-based planner | Rules for the occlusion case | Everything else | Route following, lights, lead-vehicle logic |
| Infra |  | Message definitions (frozen early) | Harness agent, launch files, toy sim, MCAP tooling, CI, Foxglove layouts, eval scripts |
| C++ port | The port itself, with Claude as tutor |  | CMake and package setup |

Rule of thumb: if a PR touches a probability, a matrix or a frame transform, you read every line before merging.

## Risks

| Risk | Mitigation |
| --- | --- |
| Leaderboard 2.1 submissions may not be open. The site says the 2025 challenge did not run and encourages local use ([leaderboard.carla.org](https://leaderboard.carla.org/)). | Check in M1. Run the Leaderboard evaluator locally either way; Bench2Drive Dev10 is the fallback external number. |
| Leaderboard 2.x targets CARLA 0.9.15; you run 0.9.16 | Bench2Drive already ran cleanly on 0.9.16. State the version in any reported numbers. |
| The AIF toy doesn't reproduce slow, look, resume | 30-hour cap in M3, then ship the toy video and evaluate the rule-based planner only. |
| Harness-to-ROS sync drops or delays controls | Frame-matched control with a timeout and logged misses, built in M1. |
| Perception eats the year | Fallback in M5: per-frame fused detections, velocity by differencing. Ground truth keeps the AIF comparison valid. |
| ROS 2 on the M1 MacBook is painful | Don't install it. Read bags with the `mcap` Python libraries and view them in the Foxglove app. |
| Claude-written code you can't explain in an interview | The review rule in Who writes what. |

## Decisions log

All made on 2026-10-07.

| Decision | Reason |
| --- | --- |
| AIF is the ego planner; prediction is a Bayesian intent filter feeding it | Planning is where AIF's information-gathering term is distinctive |
| First scenario: occluded pedestrian | It exercises that term, and route 24206 already runs |
| Ground truth first, perception swapped in at M5 | Early visible results; AIF comparison doesn't wait on CV |
| Perception kept (YOLO, LiDAR, fusion, SORT) | Wanted for the CV learning, beyond ground truth |
| Leaderboard 2.1 primary, Bench2Drive fallback | Your preference |
| pymdp as a dependency | No need to rebuild the engine |
| Laptop work via a toy sim and MCAP replay, no ROS on the Mac | Work away from the desktop |
| One C++ milestone | Full-stack AV roles expect it |
| Writeup only after the work is done | Your preference |

## Retired from earlier plans

- The Oct 5 AIF implementation spec (domain-free library, JAX, SPM cross-checks, sophisticated inference).
- The 2027 Plan's Bench2Drive-first evaluation and its writeup deliverable.
- The Year Plan's AIF-as-predictor framing.
- Hardware, including the zero-hour stretch.
- Ubuntu 26; you're on 24.04 with Jazzy and a working bridge. RDP stays for remote work.
