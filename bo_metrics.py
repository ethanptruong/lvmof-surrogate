"""
bo_metrics.py — Simulation metrics and visualization for BO benchmarking.

Classes:
  SimulationMetrics — AF, EF, Top%, cumulative best tracking

Functions:
  plot_convergence()       — cumulative best vs iteration
  plot_topk_curves()       — Top% vs evaluations
  plot_af_ef_comparison()  — AF/EF bar charts across ablation conditions
  plot_batch_comparison()  — constant_liar vs kriging_believer
  save_simulation_results() — CSV export of full history
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# SimulationMetrics
# ─────────────────────────────────────────────────────────────
class SimulationMetrics:
    """Compute Liang et al. metrics for BO simulation evaluation.

    Metrics:
      AF   — Acceleration Factor: how much faster BO finds top-k vs random
      EF   — Enhancement Factor: quality of BO selections vs random
      Top% — fraction of top-p% discovered after N evaluations
      Cumulative best — best observed score vs iteration
    """

    def __init__(self, y_all, top_fraction=0.05):
        """
        Parameters
        ----------
        y_all : array — all raw 0-9 pxrd_scores in the dataset
        top_fraction : float — define "top" as this fraction (default 5%)
        """
        self.y_all = np.asarray(y_all, dtype=float)
        self.top_fraction = top_fraction
        self.n_total = len(self.y_all)

        # Identify top-k indices
        k = max(1, int(np.ceil(self.n_total * top_fraction)))
        sorted_idx = np.argsort(self.y_all)[::-1]
        self.top_k_indices = set(sorted_idx[:k].tolist())
        self.top_k_threshold = self.y_all[sorted_idx[k - 1]]
        self.k = k

    def acceleration_factor(self, selected_indices, n_init):
        """AF = (fraction of top-k found) / (fraction of pool evaluated).

        AF > 1 means BO is finding top experiments faster than random.
        """
        n_evaluated = len(selected_indices)
        if n_evaluated == 0:
            return 0.0
        n_pool = self.n_total - n_init
        frac_evaluated = n_evaluated / max(n_pool, 1)

        found = sum(1 for idx in selected_indices if idx in self.top_k_indices)
        frac_found = found / self.k

        return frac_found / frac_evaluated if frac_evaluated > 0 else 0.0

    def enhancement_factor(self, selected_indices):
        """EF = mean(y_selected) / mean(y_all).

        EF > 1 means BO is selecting better-than-average experiments.
        """
        if len(selected_indices) == 0:
            return 0.0
        y_selected = self.y_all[selected_indices]
        mean_all = self.y_all.mean()
        if mean_all == 0:
            return 0.0
        return y_selected.mean() / mean_all

    def top_percent_curve(self, history):
        """Compute fraction of top-k discovered vs iteration.

        Returns arrays (iterations, frac_discovered).
        """
        found = set()
        fracs = []
        init_indices = set(history.get("init_indices", []))

        # Check init set
        for idx in init_indices:
            if idx in self.top_k_indices:
                found.add(idx)

        for i, idx in enumerate(history["selected_indices"]):
            if idx in self.top_k_indices:
                found.add(idx)
            fracs.append(len(found) / self.k)

        iterations = list(range(1, len(fracs) + 1))
        return np.array(iterations), np.array(fracs)

    def cumulative_best_curve(self, history):
        """Cumulative best observed score vs iteration."""
        best_so_far = history["best_so_far"]
        return np.arange(1, len(best_so_far) + 1), np.array(best_so_far)

    def summary(self, history):
        """Return dict with all metrics for a single run."""
        sel = history["selected_indices"]
        n_init = len(history.get("init_indices", []))
        af = self.acceleration_factor(sel, n_init)
        ef = self.enhancement_factor(sel)
        _, top_curve = self.top_percent_curve(history)
        final_top = top_curve[-1] if len(top_curve) > 0 else 0.0
        final_best = history["best_so_far"][-1] if history["best_so_far"] else 0.0

        return {
            "AF": af,
            "EF": ef,
            "Top_percent_final": final_top,
            "best_score_final": final_best,
            "n_evaluated": len(sel),
            "n_init": n_init,
        }


# ─────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────
def plot_convergence(histories, labels, y_all, save_path="docs/bo_convergence.png"):
    """Plot cumulative best vs iteration for multiple methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = SimulationMetrics(y_all)

    for hist, label in zip(histories, labels):
        iters, bests = metrics.cumulative_best_curve(hist)
        ax.plot(iters, bests, label=label, linewidth=2)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Best Observed pxrd_score")
    ax.set_title("BO Convergence: Cumulative Best")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved convergence plot → {save_path}")


def plot_topk_curves(histories, labels, y_all, save_path="docs/bo_topk.png"):
    """Plot Top% discovery rate vs evaluations for multiple methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = SimulationMetrics(y_all)

    for hist, label in zip(histories, labels):
        iters, fracs = metrics.top_percent_curve(hist)
        ax.plot(iters, fracs * 100, label=label, linewidth=2)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel(f"Top-{metrics.top_fraction*100:.0f}% Discovered (%)")
    ax.set_title("Top-k Discovery Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved top-k plot → {save_path}")


def plot_af_ef_comparison(summaries, save_path="docs/bo_af_ef.png"):
    """AF/EF bar charts across ablation conditions.

    Parameters
    ----------
    summaries : list of (label, summary_dict) — from SimulationMetrics.summary()
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [s[0] for s in summaries]
    afs = [s[1]["AF"] for s in summaries]
    efs = [s[1]["EF"] for s in summaries]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x, afs, width, color="steelblue")
    ax1.set_ylabel("Acceleration Factor (AF)")
    ax1.set_title("AF Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, efs, width, color="darkorange")
    ax2.set_ylabel("Enhancement Factor (EF)")
    ax2.set_title("EF Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved AF/EF comparison → {save_path}")


def plot_batch_comparison(
    hist_cl, hist_kb, y_all,
    save_path="docs/bo_batch_comparison.png"
):
    """Compare constant_liar vs kriging_believer convergence."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    metrics = SimulationMetrics(y_all)

    # Convergence
    for hist, label in [(hist_cl, "Constant Liar"), (hist_kb, "Kriging Believer")]:
        iters, bests = metrics.cumulative_best_curve(hist)
        ax1.plot(iters, bests, label=label, linewidth=2)
    ax1.set_xlabel("Evaluation")
    ax1.set_ylabel("Best Observed")
    ax1.set_title("Batch BO Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-k
    for hist, label in [(hist_cl, "Constant Liar"), (hist_kb, "Kriging Believer")]:
        iters, fracs = metrics.top_percent_curve(hist)
        ax2.plot(iters, fracs * 100, label=label, linewidth=2)
    ax2.set_xlabel("Evaluation")
    ax2.set_ylabel(f"Top-{metrics.top_fraction*100:.0f}% Discovered (%)")
    ax2.set_title("Batch BO Top-k Discovery")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved batch comparison → {save_path}")


def save_simulation_results(histories, labels, y_all, save_path="docs/bo_results.csv"):
    """Export full simulation results to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = SimulationMetrics(y_all)
    rows = []
    for hist, label in zip(histories, labels):
        s = metrics.summary(hist)
        s["method"] = label
        rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[bo_metrics] Saved results → {save_path}")
    return df


def save_full_history(history, label, save_path=None):
    """Export per-iteration history to CSV."""
    if save_path is None:
        save_path = f"docs/bo_history_{label.replace(' ', '_').replace('|', '')}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "iteration": history["iterations"],
        "selected_index": history["selected_indices"],
        "y_selected": history["y_selected"],
        "best_so_far": history["best_so_far"],
    })
    df.to_csv(save_path, index=False)
    print(f"[bo_metrics] Saved history → {save_path}")
