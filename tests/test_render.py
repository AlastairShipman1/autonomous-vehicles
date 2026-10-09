import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from av_core.plan import RuleBasedPlanner
from av_sim_toy import ScenarioParams, render_episode, render_frame, run_episode, sample_scenario
from tools.render_seed import main

PLANNER = RuleBasedPlanner()


def episode_with_hidden_ped():
    p = ScenarioParams(initial_speed=12.0, occluder_length=10.0, ped_x=56.0, ped_trigger_distance=20.0)
    return run_episode(p, PLANNER)


def test_render_frame_shows_hidden_pedestrian_dashed_and_label():
    ep = episode_with_hidden_ped()
    k = int(np.argmax(~ep.ped_visible))  # first hidden step
    fig = render_frame(ep, k)
    try:
        ped = next(p for p in fig.axes[0].patches if p.__class__.__name__ == "Circle")
        assert ped.get_linestyle() == "--" and ped.get_facecolor()[3] == 0.0
        assert "target=" in fig.axes[0].texts[0].get_text()
    finally:
        plt.close(fig)


def test_render_frame_visible_pedestrian_solid_and_outcome_on_last_frame():
    ep = episode_with_hidden_ped()
    fig = render_frame(ep, int(np.flatnonzero(ep.ped_visible)[-1]))  # revealed as the ego draws level
    try:
        ped = next(p for p in fig.axes[0].patches if p.__class__.__name__ == "Circle")
        assert ped.get_linestyle() == "-" and ped.get_facecolor()[3] == 1.0
    finally:
        plt.close(fig)
    fig = render_frame(ep, -1)
    try:
        assert f"outcome: {ep.outcome}" in fig.axes[0].texts[0].get_text()
    finally:
        plt.close(fig)


def test_render_frame_without_pedestrian():
    fig = render_frame(run_episode(ScenarioParams(ped_present=False), PLANNER), 5)
    plt.close(fig)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_episode_writes_mp4(tmp_path):
    out = render_episode(episode_with_hidden_ped(), tmp_path / "sub" / "ep.mp4", stride=10)
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_one_command_renders_a_seed(tmp_path, capsys):
    out = tmp_path / "s.mp4"
    main(["10000", "--out", str(out), "--stride", "10"])
    assert out.exists()
    assert "outcome=" in capsys.readouterr().out
    assert sample_scenario(10000).seed == 10000
