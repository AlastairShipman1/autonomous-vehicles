import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from av_core.plan import RuleBasedPlanner
from av_sim_toy import SCENARIOS, Run, render_comparison, render_comparison_frame, run_episode, sample_scenario
from av_sim_toy.compare import FIXED, NEST, PALETTE, colour_of
from tools import render_scenario, render_seed
from tools.planners import expand, run_planners

NAMES = ["v0", "v1", "aif"]


@pytest.fixture(scope="module")
def runs():
    return run_planners(SCENARIOS["two_peds_one_gap"].params, NAMES)  # v0 collides early; the others run on


def is_outline(p):
    return p.get_facecolor()[3] == 0.0  # the unfilled ring drawn on top of each translucent ego


def polygons(fig):
    return [p for p in fig.axes[0].patches if p.__class__.__name__ == "Polygon"]


def circles(fig):
    return [p for p in fig.axes[0].patches if p.__class__.__name__ == "Circle"]


def test_one_figure_with_a_scene_and_a_speed_panel(runs):
    fig = render_comparison_frame(runs, 1.0, title="t")
    try:
        assert len(fig.axes) == 2
        assert len(fig.axes[1].lines) >= 2 * len(runs)  # a speed line and a cursor, dots and markers besides
        assert [t.get_text() for t in fig.axes[1].get_legend().get_texts()] == NAMES
    finally:
        plt.close(fig)


def test_each_planner_keeps_its_colour_and_unknown_ones_get_the_palette():
    assert [colour_of(n, i) for i, n in enumerate(NAMES)] == [FIXED[n] for n in NAMES]
    assert colour_of("mine", 0) == PALETTE[0] and colour_of("other", 7) == PALETTE[7 % len(PALETTE)]
    assert len({FIXED[n] for n in NAMES}) == 3


def test_each_ego_is_where_its_own_run_says_it_is(runs):
    for time in (0.0, 1.5, 3.0):
        fig = render_comparison_frame(runs, time)
        try:
            k = int(round(time / runs[0].ep.dt))
            outlines = [p for p in polygons(fig) if is_outline(p)]
            assert len(outlines) == len(runs)
            for run, poly in zip(runs, outlines):
                x, y, yaw, _ = run.ep.ego[k]
                cx = x + 0.5 * 2.9 * np.cos(yaw)  # the body centre, half a wheelbase ahead of the rear axle
                assert poly.get_xy()[:4, 0].mean() == pytest.approx(cx, abs=1e-6), (run.name, time)
        finally:
            plt.close(fig)


def test_egos_are_nested_so_coinciding_runs_stay_distinguishable(runs):
    fig = render_comparison_frame(runs, 0.0)  # every planner starts at the same place
    try:
        outlines = [p for p in polygons(fig) if is_outline(p)]
        widths = [np.ptp(p.get_xy()[:4, 1]) for p in outlines]
        assert widths == pytest.approx([1.9 - 2 * NEST * i for i in range(3)], abs=1e-6)
    finally:
        plt.close(fig)


def test_a_finished_run_holds_its_last_frame_and_says_so(runs):
    v0 = runs[0]
    assert v0.ep.outcome == "collision" and len(v0.ep.t) < len(runs[1].ep.t)
    late = render_comparison_frame(runs, 10.0)  # long after v0 hit the pedestrian
    try:
        text = " ".join(t.get_text() for t in late.texts)
        assert "COLLISION" in text and "v0" in text
        outline = next(p for p in polygons(late) if is_outline(p))
        fx = v0.ep.final_ego[0] + 0.5 * 2.9 * np.cos(v0.ep.final_ego[2])
        assert outline.get_xy()[:4, 0].mean() == pytest.approx(fx, abs=1e-6)
    finally:
        plt.close(late)
    end = render_comparison_frame(runs, None)  # default: the end of the longest run
    plt.close(end)


def test_each_planners_pedestrians_are_drawn_in_its_colour_and_styled_by_what_it_sees(runs):
    fig = render_comparison_frame(runs, 3.0)
    try:
        cs = circles(fig)
        assert len(cs) == 2 * len(runs)  # two pedestrians per run
        by_colour = {}
        for c in cs:
            by_colour.setdefault(matplotlib_hex(c.get_edgecolor()), []).append(c)
        assert set(by_colour) == {FIXED[n] for n in NAMES} and all(len(v) == 2 for v in by_colour.values())
        assert {c.get_linestyle() for c in cs} <= {"-", "--"}
    finally:
        plt.close(fig)


def matplotlib_hex(rgba):
    from matplotlib.colors import to_hex

    return to_hex(rgba)


def test_shadows_are_drawn_only_for_the_chosen_planner(runs):
    plain, shadowed = render_comparison_frame(runs, 2.0), render_comparison_frame(runs, 2.0, shadows="v1")
    try:
        assert len(polygons(shadowed)) > len(polygons(plain))
    finally:
        plt.close(plain)
        plt.close(shadowed)
    with pytest.raises(ValueError, match="no run named"):
        render_comparison_frame(runs, 2.0, shadows="nope")


def test_moving_vehicles_are_drawn_once_from_the_longest_run():
    moving = run_planners(SCENARIOS["oncoming_traffic"].params, ["v0", "v1"])
    fig = render_comparison_frame(moving, 3.0)
    try:
        green = [p for p in polygons(fig) if np.allclose(p.get_facecolor()[:3], (0x7a / 255, 0x9e / 255, 0x7e / 255))]
        assert len(green) == 2  # the car and the van, however many planners
    finally:
        plt.close(fig)


def test_mismatched_runs_are_rejected(runs):
    other = Run("v1", run_episode(SCENARIOS["slow_ped"].params, RuleBasedPlanner()))
    with pytest.raises(ValueError, match="same scenario"):
        render_comparison_frame([runs[0], other], 1.0)
    with pytest.raises(ValueError, match="at least one"):
        render_comparison_frame([], 1.0)
    coarse = Run("c", run_episode(runs[0].ep.params, RuleBasedPlanner(), dt=0.1))
    with pytest.raises(ValueError, match="time step"):
        render_comparison_frame([runs[0], coarse], 1.0)


def test_one_run_alone_still_works():
    one = run_planners(sample_scenario(3), ["v1"])
    fig = render_comparison_frame(one, 2.0)
    plt.close(fig)


def test_expand_all():
    assert expand(["all"]) == ["v0", "v1", "aif"] and expand(["v1", "aif"]) == ["v1", "aif"]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@needs_ffmpeg
def test_overlay_video_is_written_and_ends_on_the_last_frame(tmp_path, runs):
    out = render_comparison(runs, tmp_path / "x" / "o.mp4", stride=10, title="t")
    assert out.exists() and out.stat().st_size > 1000


def test_scenario_tool_still_and_table(tmp_path, capsys):
    render_scenario.main(["slow_ped", "--planner", "all", "--overlay", "--still", "3", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("slow_ped") == 3 and (tmp_path / "slow_ped__overlay.png").stat().st_size > 10_000
    render_scenario.main(["slow_ped", "--planner", "v0", "v1", "--overlay", "--still", "-1", "--out", str(tmp_path / "end")])
    assert (tmp_path / "end" / "slow_ped__overlay.png").exists()


@needs_ffmpeg
def test_scenario_tool_overlay_video_and_separate_videos(tmp_path):
    render_scenario.main(["slow_ped", "--planner", "v0", "v1", "--overlay", "--out", str(tmp_path / "o"), "--stride", "10"])
    assert [p.name for p in (tmp_path / "o").iterdir()] == ["slow_ped__overlay.mp4"]
    render_scenario.main(["slow_ped", "--planner", "v0", "v1", "--out", str(tmp_path / "s"), "--stride", "10"])
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["slow_ped__v0.mp4", "slow_ped__v1.mp4"]


def test_scenario_tool_rejects_shadows_for_a_planner_not_run(tmp_path):
    with pytest.raises(SystemExit, match="shadows"):
        render_scenario.main(["slow_ped", "--planner", "v0", "--overlay", "--shadows", "aif", "--out", str(tmp_path)])


@needs_ffmpeg
def test_seed_tool_overlay_and_multiple_planners(tmp_path, capsys):
    out = tmp_path / "o.mp4"
    render_seed.main(["10000", "--planner", "v0", "v1", "--overlay", "--out", str(out), "--stride", "10"])
    assert out.exists() and "v0=" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="one file"):
        render_seed.main(["10000", "--planner", "v0", "v1", "--out", str(tmp_path / "x.mp4")])


@needs_ffmpeg
def test_seed_tool_with_several_planners_and_no_overlay_writes_one_video_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    render_seed.main(["10000", "--planner", "v0", "v1", "--stride", "10"])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["seed_10000_v0.mp4", "seed_10000_v1.mp4"]
