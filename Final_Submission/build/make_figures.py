"""Figures for the final report: pipeline diagram, win rates, training loss.

Usage: python make_figures.py <output_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NAVY, AMBER, TEAL, RUST, SLATE, MIST = "#1F3A5F", "#C8862A", "#2E7D6B", "#B5443A", "#5A6472", "#EEF2F7"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# Results from training/stage*.log and data/evaluation_history.json
ITERS = [3, 4, 5, 6, 7, 8, 9, 10]
WIN = [30, 70, 45, 50, 55, 52.5, 72.5, 50]
PROMOTED = {4, 7, 9}
LOSS = [4.83, 4.25, 3.66, 3.36, 3.39, 3.34, 3.27, 3.29]
POLICY = [4.78, 4.19, 3.62, 3.33, 3.35, 3.31, 3.24, 3.26]


def pipeline():
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 3.4); ax.axis("off")
    steps = [("Self-play", "MCTS + best network\n25-game batches"),
             ("Replay data", "(s, π, z) examples\n2-iteration window"),
             ("Training", "CE(π, p) + MSE(z, v)\nSGD on the GPU"),
             ("Evaluation", "candidate vs best\n≥ 55% to promote")]
    xs = [0.15, 2.65, 5.15, 7.65]
    for (head, body), x in zip(steps, xs):
        ax.add_patch(FancyBboxPatch((x, 1.35), 2.2, 1.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=MIST, ec=NAVY, lw=1.2))
        ax.text(x + 1.1, 2.75, head, ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)
        ax.text(x + 1.1, 1.95, body, ha="center", va="center", fontsize=9.5, color="#1B1F24", linespacing=1.4)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 2.48, 2.23), xytext=(x + 2.22, 2.23),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4))
    # Feedback path runs underneath the boxes: evaluation -> self-play
    ax.plot([8.75, 8.75], [1.33, 0.75], color=TEAL, lw=1.6)
    ax.plot([8.75, 1.25], [0.75, 0.75], color=TEAL, lw=1.6)
    ax.annotate("", xy=(1.25, 1.33), xytext=(1.25, 0.75), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6))
    ax.text(5.0, 0.45, "if promoted, the candidate becomes the new best network", ha="center", va="top",
            fontsize=10, color=TEAL, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", dpi=200)
    plt.close(fig)


def win_rates():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [TEAL if i in PROMOTED else RUST for i in ITERS]
    bars = ax.bar([str(i) for i in ITERS], WIN, color=colors, width=0.62)
    for b, v in zip(bars, WIN):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}%", ha="center", fontsize=9.5,
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axhline(55, color=NAVY, ls="--", lw=1.2, label="55% promotion threshold")
    ax.bar([0], [0], color=TEAL, label="Promoted"); ax.bar([0], [0], color=RUST, label="Rejected")
    ax.set_ylim(0, 90); ax.set_ylabel("Candidate win rate (%)"); ax.set_xlabel("Iteration")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=3)
    ax.grid(axis="y", color="#DDE3EA", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_winrate.png", dpi=200)
    plt.close(fig)


def losses():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(ITERS, LOSS, marker="o", color=NAVY, lw=2, label="Total loss")
    ax.plot(ITERS, POLICY, marker="s", color=AMBER, lw=1.6, ls="--", label="Policy loss")
    ax.axhline(4.98, color=SLATE, ls=":", lw=1.2)
    ax.text(10, 4.9, "uniform guess over 145 moves (≈4.98)", ha="right", va="top", fontsize=9, color=SLATE)
    for x, y in zip(ITERS, LOSS):
        ax.text(x, y + 0.07, f"{y:.2f}", ha="center", fontsize=9)
    ax.set_xticks(ITERS); ax.set_xlabel("Iteration"); ax.set_ylabel("Final-epoch loss")
    ax.set_ylim(3.0, 5.2)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.85), fontsize=9.5)
    ax.grid(axis="y", color="#DDE3EA", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); win_rates(); losses()
    print("figures written to", OUT.resolve())
