# autonomous-vehicles

Laptop-side stack (no ROS, no CARLA). See `docs/Laptop work M1 and M2.md` for the plan.

```text
src/av_core/        pure Python: types, control/, plan/, predict/, sweep/, geometry/
src/av_sim_toy/     2D occlusion sim
tools/              mcap_io, run scripts
tests/              pytest, runs locally and in CI
docs/               plan and results
```

## Setup

```sh
uv sync --python 3.12 --all-packages
uv run pytest
```

`ffmpeg` is needed to export toy-sim videos (`brew install ffmpeg`).

## Commands

```sh
uv run python -m tools.render_seed 10000 --planner v1      # MP4 of one seed -> out/
uv run python -m tools.run_sweep --planner v1 --seeds reporting --n 200 --workers 4
uv run python -m tools.mcap_io bag.mcap --planner v0       # replay a desktop bag through planner v0
uv run python -m tools.run_sweep --planner aif --seeds tuning --n 100 --workers 4   # the AIF planner
uv run pytest                                              # add -m slow for the 200-episode timing test
```

First sweep results: `docs/results/first_sweep.md`.
