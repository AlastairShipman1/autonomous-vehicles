"""Run the AIF planner on one seed and plot what it believed and why it acted.

    uv run python -m tools.aif_trace 10097 --out out/aif_trace_10097.png

Panels: speed against the commanded target; belief over the pedestrian; the expected free energy of the chosen
first action's best policy split into risk and ambiguity; and the action taken at each replan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from av_core.plan import AIFPlanner  # noqa: E402
from av_core.plan.aif import Action, StepDiagnostics  # noqa: E402
from av_core.plan.aif.model import Ped  # noqa: E402
from av_sim_toy import Episode, run_episode, sample_scenario  # noqa: E402


def best_policy_index(d: StepDiagnostics) -> int:
    """The highest-G policy that starts with the action actually taken."""
    mask = d.policies[:, 0] == int(d.action)
    return int(np.flatnonzero(mask)[np.argmax(d.G[mask])])


def plot_trace(ep: Episode, diagnostics: list[StepDiagnostics], title: str):  # noqa: ANN201
    t = np.array([d.stamp for d in diagnostics])
    best = [best_policy_index(d) for d in diagnostics]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 1.5, 0.8]})
    ax[0].plot(ep.t, ep.ego[:, 3], label="speed")
    ax[0].step(ep.t, ep.target_speed, where="post", alpha=0.6, label="target")
    ax[0].set_ylabel("m/s")
    ax[0].legend(loc="upper right")
    ax[0].set_title(title)
    for state, label in ((Ped.WAITING, "waiting"), (Ped.CROSSING, "crossing"), (Ped.NONE, "none")):
        ax[1].plot(t, [d.ped_belief[state] for d in diagnostics], label=label)
    ax[1].set_ylabel("belief")
    ax[1].legend(loc="upper right")
    ax[2].stackplot(t, [d.risk[i] for d, i in zip(diagnostics, best)],
                    [d.ambiguity[i] for d, i in zip(diagnostics, best)], labels=["risk", "ambiguity"], alpha=0.8)
    ax[2].set_ylabel("-G of chosen policy")
    ax[2].legend(loc="upper right")
    ax[3].step(t, [int(d.action) for d in diagnostics], where="post")
    ax[3].set_yticks(list(map(int, Action)), [a.name.lower() for a in Action])
    ax[3].set_xlabel("time [s]")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", type=Path, default=None, help="default: out/aif_trace_<seed>.png")
    args = ap.parse_args(argv)
    out = args.out or Path("out") / f"aif_trace_{args.seed}.png"
    planner = AIFPlanner(record=True)
    params = sample_scenario(args.seed)
    ep = run_episode(params, planner)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_trace(ep, planner.diagnostics, f"seed {args.seed}: {ep.outcome}, pedestrian {'present' if params.ped_present else 'absent'}")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"{out}  outcome={ep.outcome}  replans={len(planner.diagnostics)}")


if __name__ == "__main__":
    main()
