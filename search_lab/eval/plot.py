"""Render the eval results as an MRR-per-config bar chart.

Takes the same list[EvalResult] that run_eval() returns and writes a PNG.
The chart is generated straight from the scored data, so it always matches the table.
Re-running the eval regenerates it.

The baseline (first config) is drawn as a horizontal reference line so the
story reads at a glance: bars above the line beat the dense baseline,
bars below it (sparse) lose to it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

from search_lab.eval.runner import EvalResult


def plot_mrr(results: list[EvalResult], out_path: str | Path) -> Path:
    """Bar chart of MRR per config, baseline drawn as a reference line."""
    if not results:
        raise ValueError("no results to plot")

    baseline_mrr = results[0].mrr
    names = [r.name for r in results]
    mrrs = [r.mrr for r in results]

    # Colour by relation to baseline: above = better, below = worse,
    # the baseline bar itself a neutral tone.
    colors = []
    for i, r in enumerate(results):
        if i == 0:
            colors.append("#6c757d")  # baseline: grey
        elif r.mrr >= baseline_mrr:
            colors.append("#2a9d8f")  # beats baseline: teal
        else:
            colors.append("#e76f51")  # below baseline: red

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, mrrs, color=colors)

    # Baseline reference line + label (left side, clear of the bars).
    ax.axhline(baseline_mrr, color="#6c757d", linestyle="--", linewidth=1)
    ax.text(
        -0.4,
        baseline_mrr + 0.008,
        f"baseline {baseline_mrr:.3f}",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#6c757d",
    )

    # Value labels on top of each bar.
    for bar, mrr in zip(bars, mrrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mrr + 0.005,
            f"{mrr:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("MRR (higher is better)")
    ax.set_title("Retrieval quality by configuration (20-question DPDP eval)")
    ax.set_ylim(0, max(mrrs) * 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Standalone smoke test with dummy data, so the module can be run alone.
    from search_lab.eval.runner import EvalResult as _ER

    demo = [
        _ER("fixed/dense/none", 0.637, {1: 0.5, 3: 0.75, 5: 0.85}),
        _ER("fixed/sparse/none", 0.443, {1: 0.3, 3: 0.45, 5: 0.65}),
        _ER("fixed/hybrid/local", 0.746, {1: 0.65, 3: 0.85, 5: 0.9}),
        _ER("fixed/hybrid/cohere", 0.817, {1: 0.75, 3: 0.9, 5: 0.9}),
    ]
    p = plot_mrr(demo, "eval_results_demo.png")
    print(f"wrote {p}")
