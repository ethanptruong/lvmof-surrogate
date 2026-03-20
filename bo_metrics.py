"""
bo_metrics.py — Simulation metrics and visualization for BO benchmarking.

Classes:
  SimulationMetrics — AF, EF, Top%, cumulative best, simple regret tracking

Functions:
  plot_convergence()              — cumulative best vs iteration
  plot_average_score()            — rolling mean of selected scores vs iteration
  plot_topk_curves()              — Top% vs evaluations
  plot_simple_regret()            — simple regret (y_best_possible - y_best_found) vs iteration
  plot_af_ef_comparison()         — AF/EF bar charts across ablation conditions
  plot_seed_aggregated_comparison() — mean ± std AF/EF error bar chart across seeds
  plot_sensitive_heatmap()        — acquisition × surrogate heatmap of mean AF
  plot_seed_averaged_convergence() — convergence bands (mean ± std) collapsed over seeds
  plot_batch_comparison()         — constant_liar vs kriging_believer
  compute_surrogate_calibration() — z-score calibration check for surrogate sigma
  plot_calibration()              — z-score histogram + coverage bar chart
  save_simulation_results()       — CSV export of full history
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm


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

    def simple_regret_curve(self, history):
        """Simple regret = y_best_possible - best_found_by_BO_so_far per iteration.

        Regret is computed against BO's own selections only (init set excluded).
        This avoids a flat-zero curve when the init set already contains the
        global optimum — a common occurrence with large init fractions.

        A well-performing BO should drive regret toward zero faster than random.
        """
        y_best_possible = self.y_all.max()
        y_selected = np.array(history["y_selected"], dtype=float)
        bo_best_so_far = np.maximum.accumulate(y_selected)
        regret = y_best_possible - bo_best_so_far
        return np.arange(1, len(regret) + 1), regret

    def summary(self, history):
        """Return dict with all metrics for a single run."""
        sel = history["selected_indices"]
        n_init = len(history.get("init_indices", []))
        af = self.acceleration_factor(sel, n_init)
        ef = self.enhancement_factor(sel)
        _, top_curve = self.top_percent_curve(history)
        final_top = top_curve[-1] if len(top_curve) > 0 else 0.0
        final_best = history["best_so_far"][-1] if history["best_so_far"] else 0.0

        _, regret_curve = self.simple_regret_curve(history)
        final_regret = float(regret_curve[-1]) if len(regret_curve) > 0 else float(self.y_all.max())

        return {
            "AF": af,
            "EF": ef,
            "Top_percent_final": final_top,
            "best_score_final": final_best,
            "simple_regret_final": final_regret,
            "n_evaluated": len(sel),
            "n_init": n_init,
        }


    def per_cluster_summary(self, history, groups):
        """Compute AF, EF, Top-5% broken down by chemistry cluster.

        For each cluster, metrics are computed *within* that cluster's pool —
        i.e. top-5% is relative to the cluster, not the global dataset.  This
        answers "did the BO find the best experiments *within* cluster X?"

        Parameters
        ----------
        history : dict from run_simulation
        groups  : int array (n,) — cluster labels for the full dataset

        Returns
        -------
        dict : {cluster_id: {"AF": float, "EF": float, "Top%": float,
                              "n_pool": int, "n_selected": int}}
        """
        groups = np.asarray(groups, dtype=int)
        init_set = set(history.get("init_indices", []))
        selected = history["selected_indices"]
        results = {}

        for cid in sorted(np.unique(groups)):
            cid_mask = (groups == cid)
            cid_indices = set(np.where(cid_mask)[0].tolist())

            pool_in_cluster = cid_indices - init_set
            selected_in_cluster = [s for s in selected if s in pool_in_cluster]
            n_pool_c = len(pool_in_cluster)
            n_sel_c  = len(selected_in_cluster)

            if n_pool_c == 0:
                results[cid] = {"AF": 0.0, "EF": 0.0, "Top%": 0.0,
                                "n_pool": 0, "n_selected": 0}
                continue

            # Cluster-local top-5%
            y_pool_c = self.y_all[sorted(pool_in_cluster)]
            k_c = max(1, int(np.ceil(n_pool_c * self.top_fraction)))
            threshold_c = np.sort(y_pool_c)[::-1][min(k_c - 1, len(y_pool_c) - 1)]
            pool_list = sorted(pool_in_cluster)
            top_k_local = set()
            for idx in pool_list:
                if self.y_all[idx] >= threshold_c:
                    top_k_local.add(idx)
                    if len(top_k_local) >= k_c:
                        break

            found = sum(1 for s in selected_in_cluster if s in top_k_local)
            frac_found = found / max(k_c, 1)
            frac_eval  = n_sel_c / max(n_pool_c, 1)
            af = frac_found / frac_eval if frac_eval > 0 else 0.0

            y_sel = self.y_all[selected_in_cluster] if n_sel_c > 0 else np.array([0.0])
            y_pool_mean = y_pool_c.mean() if len(y_pool_c) > 0 else 1e-9
            ef = float(y_sel.mean() / max(y_pool_mean, 1e-9))

            # Hit rate: fraction of BO selections with score >= 7
            hit_rate = float((y_sel >= 7).mean()) if n_sel_c > 0 else 0.0
            baseline_hit = float((y_pool_c >= 7).mean()) if len(y_pool_c) > 0 else 0.0

            results[cid] = {
                "AF": af, "EF": ef, "Top%": frac_found,
                "hit_rate": hit_rate, "baseline_hit": baseline_hit,
                "n_pool": n_pool_c, "n_selected": n_sel_c,
            }

        return results


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


def plot_average_score(histories, labels, window=10, save_path="docs/bo_avg_score.png"):
    """Plot rolling mean of selected scores vs iteration.

    Unlike cumulative best, this shows whether BO is consistently selecting
    better experiments over time — useful when the global best is found early
    (flat cumulative-best curve) or to compare exploitation quality across methods.

    Parameters
    ----------
    histories : list of history dicts (must contain 'y_selected')
    labels    : list of str
    window    : int — rolling window size (default 10)
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for hist, label in zip(histories, labels):
        y_sel = np.array(hist["y_selected"], dtype=float)
        iters = np.arange(1, len(y_sel) + 1)

        # cumulative mean
        cum_mean = np.cumsum(y_sel) / iters
        ax.plot(iters, cum_mean, lw=2, label=label)

        # rolling mean (shown as shaded band behind line)
        if len(y_sel) >= window:
            rolling = np.convolve(y_sel, np.ones(window) / window, mode="valid")
            ax.plot(iters[window - 1:], rolling, lw=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Score")
    ax.set_title(
        f"BO Selection Quality\n"
        f"Solid = cumulative mean  |  Dashed = rolling mean (window={window})"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved average score plot → {save_path}")


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


def plot_simple_regret(histories, labels, y_all, save_path="docs/bo_simple_regret.png"):
    """Plot simple regret (y_best_possible - y_best_found) vs iteration.

    Lower is better; a well-calibrated BO should reach zero faster than random.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = SimulationMetrics(y_all)

    for hist, label in zip(histories, labels):
        iters, regret = metrics.simple_regret_curve(hist)
        ax.plot(iters, regret, label=label, linewidth=2)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Simple Regret (y* − best BO selection)")
    ax.set_title("BO Simple Regret (lower = better)\n"
                 "Regret relative to BO selections only — init set excluded")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved simple regret plot → {save_path}")


def compute_surrogate_calibration(surrogate, X_test, y_test):
    """Evaluate how well the surrogate's uncertainty estimates (sigma) are calibrated.

    A perfectly calibrated surrogate satisfies:
      z = (y_true - mu) / sigma  ~  N(0, 1)

    This means:
      - z-score mean  ≈ 0   (no systematic bias)
      - z-score std   ≈ 1   (sigma matches actual error magnitude)
      - ~68% of points fall within 1σ, ~95% within 2σ

    Parameters
    ----------
    surrogate : RegressionSurrogate — already fitted surrogate
    X_test    : array (n_test, n_features)
    y_test    : array (n_test,) — true pxrd_scores

    Returns
    -------
    dict with keys:
      z_scores, mu, sigma, y_true,
      mean_z, std_z,
      fraction_within_1sigma, fraction_within_2sigma, fraction_within_3sigma,
      n_valid, n_zero_sigma, calibration_error
    """
    y_test = np.asarray(y_test, dtype=float)
    mu, sigma = surrogate.predict(X_test)

    valid = sigma > 1e-10
    n_valid = int(valid.sum())
    n_zero_sigma = int((~valid).sum())

    if n_valid < 5:
        return {
            "error": f"Too few non-zero sigma values ({n_valid}). "
                     "Surrogate may not support uncertainty estimation.",
            "n_valid": n_valid,
            "n_zero_sigma": n_zero_sigma,
        }

    mu_v = mu[valid]
    sigma_v = sigma[valid]
    y_v = y_test[valid]

    z_scores = (y_v - mu_v) / sigma_v

    frac_1 = float((np.abs(z_scores) <= 1.0).mean())
    frac_2 = float((np.abs(z_scores) <= 2.0).mean())
    frac_3 = float((np.abs(z_scores) <= 3.0).mean())

    # Calibration error: mean absolute difference between observed and expected coverage
    # across a grid of confidence levels
    confidence_levels = np.linspace(0.05, 0.95, 19)
    z_quantiles = scipy_norm.ppf((1 + confidence_levels) / 2)
    observed_coverage = np.array(
        [(np.abs(z_scores) <= zq).mean() for zq in z_quantiles]
    )
    calibration_error = float(np.mean(np.abs(observed_coverage - confidence_levels)))

    return {
        "z_scores": z_scores,
        "mu": mu_v,
        "sigma": sigma_v,
        "y_true": y_v,
        "mean_z": float(z_scores.mean()),
        "std_z": float(z_scores.std()),
        "fraction_within_1sigma": frac_1,
        "fraction_within_2sigma": frac_2,
        "fraction_within_3sigma": frac_3,
        "calibration_error": calibration_error,
        "confidence_levels": confidence_levels,
        "observed_coverage": observed_coverage,
        "n_valid": n_valid,
        "n_zero_sigma": n_zero_sigma,
    }


def plot_calibration(
    cal_dict,
    surrogate_name="surrogate",
    save_path="docs/bo_calibration.png",
):
    """Plot surrogate calibration diagnostics: z-score histogram + coverage curve.

    Left panel  — z-score histogram with N(0,1) reference.
                  A well-calibrated surrogate should look approximately normal.

    Right panel — reliability diagram: observed coverage vs expected coverage at
                  each confidence level.  Perfect calibration = diagonal line.
                  Points above the diagonal → overconfident (sigma too small).
                  Points below → underconfident (sigma too large).

    Parameters
    ----------
    cal_dict      : dict returned by compute_surrogate_calibration()
    surrogate_name: str, used in plot title
    save_path     : output path
    """
    if "error" in cal_dict:
        print(f"[bo_metrics] Calibration plot skipped: {cal_dict['error']}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    z = cal_dict["z_scores"]
    conf = cal_dict["confidence_levels"]
    obs_cov = cal_dict["observed_coverage"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: z-score histogram ───────────────────────────────────────────────
    z_clip = np.clip(z, -5, 5)
    ax1.hist(z_clip, bins=30, density=True, alpha=0.7, color="steelblue",
             label=f"Observed z-scores (n={cal_dict['n_valid']})")
    x_ref = np.linspace(-5, 5, 300)
    ax1.plot(x_ref, scipy_norm.pdf(x_ref), "r-", linewidth=2, label="N(0,1) ideal")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)

    stats_text = (
        f"mean={cal_dict['mean_z']:.2f}  std={cal_dict['std_z']:.2f}\n"
        f"within 1σ: {cal_dict['fraction_within_1sigma']*100:.1f}%  "
        f"(ideal 68.3%)\n"
        f"within 2σ: {cal_dict['fraction_within_2sigma']*100:.1f}%  "
        f"(ideal 95.4%)"
    )
    ax1.text(
        0.03, 0.97, stats_text,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )
    ax1.set_xlabel("Standardised residual  (y_true − μ) / σ")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Surrogate Calibration — {surrogate_name}\nZ-score Distribution")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Right: reliability diagram ────────────────────────────────────────────
    ax2.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration")
    ax2.plot(conf, obs_cov, "o-", color="steelblue", linewidth=2,
             markersize=5, label="Observed coverage")
    ax2.fill_between(conf, conf, obs_cov,
                     where=(obs_cov > conf), alpha=0.15, color="orange",
                     label="Underconfident (σ too large)")
    ax2.fill_between(conf, conf, obs_cov,
                     where=(obs_cov < conf), alpha=0.15, color="red",
                     label="Overconfident (σ too small)")

    ax2.text(
        0.03, 0.97,
        f"Mean calibration error: {cal_dict['calibration_error']:.3f}",
        transform=ax2.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )
    ax2.set_xlabel("Expected coverage (confidence level)")
    ax2.set_ylabel("Observed coverage")
    ax2.set_title(f"Reliability Diagram — {surrogate_name}")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved calibration plot → {save_path}")


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


def plot_seed_aggregated_comparison(
    summaries,
    save_path="docs/bo_ablation_seed_aggregated.png",
):
    """Mean ± std AF and EF per method, aggregated across seeds.

    Labels are expected in the format  'acq|seed=N'  or  'acq|surrogate|seed=N'.
    Seeds are collapsed by stripping the '|seed=N' suffix, then mean and std
    are computed across the resulting groups.

    Parameters
    ----------
    summaries : list of (label, summary_dict)
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Group by method (label without seed suffix)
    groups: dict = {}
    for label, s in summaries:
        # Strip seed component: everything after the last |seed=
        parts = label.split("|")
        method = "|".join(p for p in parts if not p.startswith("seed="))
        groups.setdefault(method, {"AF": [], "EF": [], "Top%": []})
        groups[method]["AF"].append(s["AF"])
        groups[method]["EF"].append(s["EF"])
        groups[method]["Top%"].append(s["Top_percent_final"] * 100)

    methods = list(groups.keys())
    af_means  = np.array([np.mean(groups[m]["AF"])  for m in methods])
    af_stds   = np.array([np.std(groups[m]["AF"])   for m in methods])
    ef_means  = np.array([np.mean(groups[m]["EF"])  for m in methods])
    ef_stds   = np.array([np.std(groups[m]["EF"])   for m in methods])
    top_means = np.array([np.mean(groups[m]["Top%"]) for m in methods])
    top_stds  = np.array([np.std(groups[m]["Top%"])  for m in methods])

    # Sort by mean AF descending
    order = np.argsort(af_means)[::-1]
    methods   = [methods[i]   for i in order]
    af_means  = af_means[order];  af_stds  = af_stds[order]
    ef_means  = ef_means[order];  ef_stds  = ef_stds[order]
    top_means = top_means[order]; top_stds = top_stds[order]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, means, stds, ylabel, title, baseline in [
        (axes[0], af_means,  af_stds,  "AF",         "Acceleration Factor (mean ± std)", 1.0),
        (axes[1], ef_means,  ef_stds,  "EF",         "Enhancement Factor (mean ± std)",  1.0),
        (axes[2], top_means, top_stds, "Top-5% (%)", "Top-5% Discovered (mean ± std)",   None),
    ]:
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8,
               error_kw={"linewidth": 1.5})
        if baseline is not None:
            ax.axhline(baseline, color="red", linestyle="--", alpha=0.6,
                       label="Random baseline")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved seed-aggregated comparison → {save_path}")


def plot_sensitive_heatmap(
    summaries,
    sensitive_acquisitions=("ei", "lcb", "thompson"),
    metric="AF",
    save_path="docs/bo_ablation_heatmap.png",
):
    """Heatmap of mean AF (or EF) across acquisition × surrogate for surrogate-sensitive methods.

    Cells show the mean metric value across seeds.  Annotated with the value.
    Directly answers: for each acquisition, which surrogate works best?

    Parameters
    ----------
    summaries             : list of (label, summary_dict)
    sensitive_acquisitions: acquisitions to include (must use surrogate)
    metric                : 'AF' or 'EF'
    save_path             : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Collect values: cell[acq][surrogate] = list of metric values across seeds
    cell: dict = {}
    for label, s in summaries:
        parts = label.split("|")
        if len(parts) < 3:
            continue  # agnostic labels have no surrogate field
        acq, surr = parts[0], parts[1]
        if acq not in sensitive_acquisitions:
            continue
        if parts[2].startswith("seed="):
            cell.setdefault(acq, {}).setdefault(surr, []).append(s[metric])

    if not cell:
        print(f"[bo_metrics] Heatmap skipped: no surrogate-sensitive data found.")
        return

    acqs  = sorted(cell.keys())
    surrs = sorted({s for acq_data in cell.values() for s in acq_data})

    data = np.full((len(acqs), len(surrs)), np.nan)
    for i, acq in enumerate(acqs):
        for j, surr in enumerate(surrs):
            vals = cell.get(acq, {}).get(surr, [])
            if vals:
                data[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(surrs) * 1.4), max(4, len(acqs) * 1.2)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label=f"Mean {metric}")

    ax.set_xticks(range(len(surrs)))
    ax.set_xticklabels(surrs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(acqs)))
    ax.set_yticklabels(acqs, fontsize=9)
    ax.set_title(f"Mean {metric} — Acquisition × Surrogate\n(mean across seeds)")

    # Annotate cells
    for i in range(len(acqs)):
        for j in range(len(surrs)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved heatmap → {save_path}")


def plot_seed_averaged_convergence(
    histories, labels, y_all,
    metric="avg_score",
    window=10,
    save_path="docs/bo_ablation_seed_avg_convergence.png",
):
    """Convergence curves collapsed over seeds: one shaded band per unique method.

    Mean line + ±1 std shaded region across seeds, so variance from random
    initialisation doesn't clutter the chart.

    Parameters
    ----------
    histories : list of history dicts
    labels    : list of str in format 'acq|...|seed=N'
    y_all     : array — full dataset scores (for simple regret)
    metric    : 'avg_score' (cumulative mean of y_selected) or 'simple_regret'
    window    : rolling window for avg_score (ignored for simple_regret)
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sim = SimulationMetrics(y_all)

    # Group histories by method (strip seed)
    groups: dict = {}
    for hist, label in zip(histories, labels):
        parts = label.split("|")
        method = "|".join(p for p in parts if not p.startswith("seed="))
        groups.setdefault(method, []).append(hist)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab20")

    for idx, (method, hists) in enumerate(groups.items()):
        curves = []
        for hist in hists:
            if metric == "simple_regret":
                _, vals = sim.simple_regret_curve(hist)
            else:
                y_sel = np.array(hist["y_selected"], dtype=float)
                iters = np.arange(1, len(y_sel) + 1)
                vals = np.cumsum(y_sel) / iters
            curves.append(vals)

        # Align to shortest run
        min_len = min(len(c) for c in curves)
        mat = np.array([c[:min_len] for c in curves])
        iters = np.arange(1, min_len + 1)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        color = cmap(idx % 20)
        ax.plot(iters, mean, color=color, lw=2, label=method)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)

    if metric == "simple_regret":
        ax.set_ylabel("Simple Regret (y* − best BO selection)")
        ax.set_title("BO Simple Regret — Mean ± Std across Seeds")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("Cumulative Mean Score")
        ax.set_title("BO Selection Quality — Mean ± Std across Seeds")

    ax.set_xlabel("BO Iteration")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved seed-averaged convergence → {save_path}")


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


# ─────────────────────────────────────────────────────────────
# Multi-seed per-cluster and LOCO plots
# ─────────────────────────────────────────────────────────────

def plot_per_cluster_bar(cluster_stats_by_seed, metric="AF",
                         save_path=None):
    """Grouped bar chart: per-cluster metric with error bars across seeds.

    Parameters
    ----------
    cluster_stats_by_seed : list[dict]
        Each element is the output of SimulationMetrics.per_cluster_summary(),
        one per seed.
    metric : str — "AF", "EF", or "Top%"
    save_path : str or None
    """
    if save_path is None:
        save_path = f"docs/bo_per_cluster_{metric.replace('%', 'pct')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(cluster_stats_by_seed[0].keys())
    means, stds = [], []
    for cid in cluster_ids:
        vals = [seed_stats[cid][metric] for seed_stats in cluster_stats_by_seed
                if cid in seed_stats]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if vals else 0.0)

    # Add aggregate bar
    agg_vals = []
    for seed_stats in cluster_stats_by_seed:
        all_vals = [seed_stats[cid][metric] for cid in cluster_ids if cid in seed_stats]
        agg_vals.append(np.mean(all_vals) if all_vals else 0.0)
    means.append(np.mean(agg_vals))
    stds.append(np.std(agg_vals))

    labels = [f"C{cid}" for cid in cluster_ids] + ["Agg"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue"] * len(cluster_ids) + ["firebrick"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-Cluster {metric} (mean ± std across seeds)")
    if metric == "AF":
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Random (AF=1)")
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved per-cluster {metric} → {save_path}")


def plot_evaluate_hit_rate(cluster_stats_by_seed,
                           save_path="docs/bo_per_cluster_hit_rate.png"):
    """Grouped bar chart: BO hit rate vs random baseline, mean ± std across seeds.

    Hit rate  = fraction of BO selections with score >= 7.
    Baseline  = fraction of pool with score >= 7 (random guessing).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(cluster_stats_by_seed[0].keys())

    hit_means, hit_stds = [], []
    base_means = []
    for cid in cluster_ids:
        hits = [s[cid]["hit_rate"] for s in cluster_stats_by_seed if cid in s]
        hit_means.append(np.mean(hits) * 100 if hits else 0.0)
        hit_stds.append(np.std(hits) * 100 if hits else 0.0)
        bases = [s[cid]["baseline_hit"] for s in cluster_stats_by_seed if cid in s]
        base_means.append(np.mean(bases) * 100 if bases else 0.0)

    # Aggregate across clusters (weighted by seeds)
    agg_hits = []
    agg_bases = []
    for s in cluster_stats_by_seed:
        h = [s[c]["hit_rate"] for c in cluster_ids if c in s]
        b = [s[c]["baseline_hit"] for c in cluster_ids if c in s]
        agg_hits.append(np.mean(h) * 100 if h else 0.0)
        agg_bases.append(np.mean(b) * 100 if b else 0.0)
    hit_means.append(np.mean(agg_hits))
    hit_stds.append(np.std(agg_hits))
    base_means.append(np.mean(agg_bases))

    labels = [f"C{cid}" for cid in cluster_ids] + ["Agg"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bo = ["steelblue"] * len(cluster_ids) + ["firebrick"]
    colors_base = ["lightcoral"] * len(cluster_ids) + ["salmon"]

    ax.bar(x - width / 2, hit_means, width, yerr=hit_stds, capsize=4,
           color=colors_bo, edgecolor="black", linewidth=0.5, label="BO hit rate")
    ax.bar(x + width / 2, base_means, width,
           color=colors_base, edgecolor="black", linewidth=0.5, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit Rate (%):  score ≥ 7")
    ax.set_title("Per-Cluster BO Hit Rate vs Random Baseline (mean ± std across seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(hit_means, default=0), max(base_means, default=0)) + 10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved per-cluster hit rate → {save_path}")


def plot_loco_bar(loco_results, metric="AF", save_path=None):
    """Bar chart of leave-one-cluster-out results.

    Parameters
    ----------
    loco_results : dict {cluster_id: {"AF": ..., "EF": ..., "Top%": ..., "n_pool": ...}}
    metric : str
    save_path : str or None
    """
    if save_path is None:
        save_path = f"docs/bo_loco_{metric.replace('%', 'pct')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(loco_results.keys())
    vals = [loco_results[cid][metric] for cid in cluster_ids]
    pool_sizes = [loco_results[cid].get("n_pool", 0) for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"C{c}" for c in cluster_ids], vals,
                  color="steelblue", edgecolor="black", linewidth=0.5)

    # Annotate pool size on each bar
    for bar, ps in zip(bars, pool_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={ps}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(metric)
    ax.set_title(f"Leave-One-Cluster-Out: {metric} per held-out cluster")
    if metric == "AF":
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Random (AF=1)")
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved LOCO {metric} → {save_path}")


def plot_loco_hit_rate(loco_results, save_path="docs/bo_loco_hit_rate.png"):
    """Grouped bar chart: BO hit rate vs random baseline per held-out cluster.

    Hit rate = fraction of BO selections with score >= 7.
    Baseline = fraction of the cluster with score >= 7 (i.e. random guessing).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(loco_results.keys())
    hit_rates = [loco_results[c]["hit_rate"] * 100 for c in cluster_ids]
    baselines = [loco_results[c]["baseline_hit"] * 100 for c in cluster_ids]
    pool_sizes = [loco_results[c].get("n_pool", 0) for c in cluster_ids]
    labels = [f"C{c}" for c in cluster_ids]

    x = np.arange(len(cluster_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_bo = ax.bar(x - width / 2, hit_rates, width, label="BO hit rate",
                     color="steelblue", edgecolor="black", linewidth=0.5)
    bars_rand = ax.bar(x + width / 2, baselines, width, label="Random baseline",
                       color="lightcoral", edgecolor="black", linewidth=0.5)

    # Annotate pool size above the taller bar in each pair
    for i, ps in enumerate(pool_sizes):
        y_top = max(hit_rates[i], baselines[i])
        ax.text(x[i], y_top + 1.5, f"n={ps}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit Rate (%):  score ≥ 7")
    ax.set_title("LOCO: BO Hit Rate vs Random Baseline per Cluster")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(hit_rates, default=0), max(baselines, default=0)) + 10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved LOCO hit rate → {save_path}")


def plot_learning_curve(lc_results, save_path="docs/bo_learning_curve.png"):
    """Plot AF, EF, and Hit Rate vs number of initial experiments.

    Parameters
    ----------
    lc_results : list[dict]
        Each dict has keys: 'init_frac', 'n_init_mean', 'AF_mean', 'AF_std',
        'EF_mean', 'EF_std', 'hit_mean', 'hit_std', 'baseline_hit'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_init = [r["n_init_mean"] for r in lc_results]
    fracs  = [r["init_frac"]   for r in lc_results]
    baseline_hit = lc_results[0].get("baseline_hit", 0.0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, ylabel in zip(
        axes,
        ["AF", "EF", "hit"],
        ["Acceleration Factor", "Enhancement Factor", "Hit Rate (score ≥ 7)"],
    ):
        means = [r[f"{metric}_mean"] for r in lc_results]
        stds  = [r[f"{metric}_std"]  for r in lc_results]
        scale = 100.0 if metric == "hit" else 1.0

        ax.errorbar(n_init, [m * scale for m in means],
                    yerr=[s * scale for s in stds],
                    marker="o", capsize=4, linewidth=2, markersize=6,
                    color="steelblue", label="BO")
        ax.set_xlabel("Number of initial (random) experiments")
        ax.set_ylabel(f"{ylabel}" + (" (%)" if metric == "hit" else ""))
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)

        if metric == "AF":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                       label="Random (AF=1)")
            ax.axhline(2.0, color="green", linestyle=":", linewidth=0.8,
                       label="Useful (AF=2)")
            ax.legend(fontsize=8)
        elif metric == "hit":
            ax.axhline(baseline_hit, color="lightcoral", linestyle="--",
                       linewidth=2, label=f"Random baseline ({baseline_hit:.1f}%)")
            ax.legend(fontsize=8)

        # Secondary x-axis showing init fraction
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        frac_ticks = [r["n_init_mean"] for r in lc_results]
        ax2.set_xticks(frac_ticks)
        ax2.set_xticklabels([f"{f:.0%}" for f in fracs], fontsize=7)
        ax2.set_xlabel("Init fraction", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved learning curve → {save_path}")
