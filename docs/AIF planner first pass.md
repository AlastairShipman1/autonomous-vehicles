# AIF planner, first pass

Where it lives: `src/av_core/av_core/plan/aif/` (`model.py`, `observe.py`, `efe.py`, `planner.py`). Run it with
`--planner aif` in `tools.run_sweep` / `tools.render_seed`; plot one episode with `tools.aif_trace`.
This is a working starting point for the hand-written generative model, not a result. Every constant is in
`AIFParams` or at the top of `model.py` / `observe.py`.

## The model (`model.py`)

One hidden-state factor of 256 states: ego distance band (8, each 8 m wide, measured from the ego's front bumper to
the crossing point) × ego speed band (8, 0 to 14 m/s in 2 m/s steps) × pedestrian (none, waiting, crossing, cleared).
Ego and pedestrian are one factor because pymdp 0.0.7 cannot make one factor's transition depend on another, and the
pedestrian steps out as the ego closes in. That is how the toy triggers it; CARLA's scenario triggers may differ.

| Piece | Content |
| --- | --- |
| Actions | keep (toward the speed limit, +1 m/s²), ease off (−1.5), brake (−4). One model step is 0.5 s. |
| **B** | Ego: speed moves by the action, distance by the mean speed, mass split linearly between neighbouring bands. Pedestrian: waiting → crossing with a per-band probability (`p_cross`: 0.2 in the 12 to 36 m bands, small elsewhere, 0 once level with the crossing), crossing → cleared at 0.2/step. |
| **A** | Pedestrian modality: a waiting pedestrian is seen only from bands where line of sight exists (computed from the occluder geometry in `observe.visibility_profile`); a crossing pedestrian is seen from anywhere. Speed modality: the ego's own speed band, exactly. Safety modality: collision only when level with the path and the pedestrian is crossing, more likely the faster the ego; near miss in the approach band when fast. |
| **C** | Per modality, relative log preferences: safety (0, −8, −32), speed (−1 at standstill rising to 0 above 10 m/s), pedestrian modality indifferent. pymdp softmaxes each, so only differences within a modality matter. |
| **D** | Pedestrian prior 50% waiting (what the sampler uses), with 1% smoothing. The ego is known. |

Policies are all 3⁶ = 729 action sequences (3 s). The first action of the most probable sequence becomes a target speed
(current speed plus the action's acceleration times 0.5 s); keep means the rule-based cap. The rule-based limits
(route, lead, lights) still apply on top, and with no occluder in the WorldModel the command is exactly planner v0's.

## What the planner reads from the WorldModel (`observe.py`)

- The nearest occluder with an `OccludedRegion`, assumed parallel to the route. A hidden pedestrian is assumed to wait
  1 m past its far end and 0.75 m beyond its outer side. That is the crossing point the distance bands are measured to.
- Pedestrians near the crossing point, classified as waiting or crossing (in the lane, or moving toward the route).

## Where pymdp is used, and where it is not

pymdp is used for state inference (`inference.update_posterior_states`) and its softmax. Expected free energy is
computed in `efe.py`, batched over all policies, instead of by `control.update_posterior_policies`: that loops in
Python, taking about 0.7 s per decision for 81 policies and far too long for 729. The two agree to 5e-7
(`tests/test_aif.py::test_batched_efe_equals_pymdp`), including pymdp taking the information gain over the joint
outcome of all modalities. `G = -(risk + ambiguity)` with risk the KL from predicted to preferred outcomes and
ambiguity the expected entropy of the likelihood; both are logged per policy and plotted by `tools.aif_trace`.

## First numbers (tuning seeds only; reporting seeds not touched)

Preference grid on tuning seeds 0 to 59 (34 with a pedestrian), `c_collision` against a scale on the speed
preferences. Time penalties are in seconds:

| Collision pref | Speed pref scale | Collisions | Time penalty (ped) | Time penalty (no ped) | Needless stops |
| --- | --- | --- | --- | --- | --- |
| −16 | 0.5 | 32.4% | 1.21 | −1.13 | 0% |
| −16 | 0.25 | 11.8% | 4.06 | −0.01 | 0% |
| −16 | 0.1 | 5.9% | 7.84 | 4.81 | 0% |
| −32 | 0.5 | 29.4% | 2.43 | −0.91 | 0% |
| **−32** | **0.25** | **5.9%** | **4.89** | **0.22** | **0%** |
| −32 | 0.1 | 0.0% | 9.12 | 5.33 | 19.2% |

The defaults are the bold row. On tuning seeds 0 to 99 with those defaults, next to planner v1:

| | AIF | v1 |
| --- | --- | --- |
| Collision rate (54 with pedestrian) | 3.7% (1.0 to 12.5) | 0.0% (0.0 to 6.6) |
| Time penalty, pedestrian present (s) | 5.11 | 4.58 |
| Time penalty, pedestrian absent (s) | 0.39 | 1.30 |
| Needless stops | 0 of 46 | 0 of 46 |

So the first pass does not beat v1 yet: it is less over-cautious without a pedestrian, slightly slower and less safe
with one. The model is tuned on 34 pedestrian episodes, so the table above is a rough shape, not an optimum.

## Known weaknesses to look at first

- **Late pedestrians.** The model believes anyone who hasn't stepped out during the 12 to 36 m bands probably isn't
  there. Seed 39 (trigger at 14 m) loses its belief (0.5 → 0.14) and collides. The belief dynamics are the thing to
  refine; the pedestrian modality could also carry more information.
- **Coarse bands.** 8 m bands, 3 s horizon: speed and distance are rounded in ways that matter near the crossing.
- **Chatter.** At low speed the first action alternates between keep and ease off, so the target speed oscillates
  (visible in `tools.aif_trace 2`). A held action or a switching cost in the policy prior would help.
- **No information-seeking beyond slowing.** The only visible bands are level with and past the crossing point, so the
  epistemic value is concentrated at the zone; there is no lateral movement to look around the car.
- **Predictions unused.** `predictions` is ignored; in M4 the predictor feeds the transition model for visible agents.
