# autonomous-vehicles

Laptop-side stack (no ROS, no CARLA). See `docs/Laptop work M1 and M2.md` for the plan.

```text
av_core/        pure Python: types, control/, plan/, predict/, sweep/, geometry/
av_sim_toy/     2D occlusion sim
tools/          mcap_io, run scripts
tests/          pytest, runs locally and in CI
```

## Setup

```sh
uv sync --python 3.12 --all-packages
uv run pytest
```

`ffmpeg` is needed to export toy-sim videos (`brew install ffmpeg`).
