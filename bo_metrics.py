"""
bo_metrics.py - Simulation metrics and visualization for BO benchmarking.

All success metrics use a single threshold-based criterion: a "hit" is a row
with pxrd_score >= config.BO_HIT_THRESHOLD (default 7.0, crystalline product).
This unifies the acquisition target, the gamma sweep, and the headline AF/EF
numbers around the same scientific definition of success.

Classes:
  SimulationMetrics - AF, EF, hit-discovery rate, simple regret tracking

Functions:
  plot_convergence()              - cumulative best vs iteration
  plot_average_score()            - rolling mean of selected scores vs iteration
  plot_topk_curves()              - hit-discovery rate vs evaluations
  plot_simple_regret()            - simple regret (y_best_possible - y_best_found) vs iteration
  plot_af_ef_comparison()         - AF/EF bar charts across ablation conditions
  plot_seed_aggregated_comparison() - mean ± std AF/EF/Hit% error bar chart across seeds
  plot_sensitive_heatmap()        - acquisition × surrogate heatmap of mean AF
  plot_seed_averaged_convergence() - convergence bands (mean ± std) collapsed over seeds
  plot_batch_comparison()         - constant_liar vs kriging_believer
  compute_surrogate_calibration() - z-score calibration check for surrogate sigma
  plot_calibration()              - z-score histogram + coverage bar chart
  save_simulation_results()       - CSV export of full history
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

from config import BO_HIT_THRESHOLD


# ---
# SimulationMetrics
# ---
class SimulationMetrics:
    """Threshold-based BO simulation metrics.

    All metrics share a single success criterion: y >= hit_threshold.

    Metrics:
      AF       - Acceleration Factor: (fraction of pool-hits found) / (fraction
                 of pool evaluated). AF=1 is random; >1 means BO finds hits
                 faster than uniform sampling. NaN when the pool contains no
                 hits (the metric is undefined, not zero).
      EF       - Enhancement Factor (graded quality ratio):
                 mean(y_selected) / mean(y_pool). EF=1 is random;
                 threshold-agnostic - captures graded selection quality that
                 the binary AF can miss (e.g. selections at y=6 vs y=3 both
                 count as misses for AF but differ in EF).
      Hit%     - Fraction of BO selections with y >= threshold.
      HitDisc% - Fraction of pool hits discovered after N evaluations.
      Cumulative best, simple regret - convergence diagnostics.
    """

    def __init__(self, y_all, hit_threshold=None):
        """
        Parameters
        ---
        y_all         : array - all raw 0-9 pxrd_scores in the dataset
        hit_threshold : float or None - y >= threshold counts as a hit.
                        Defaults to config.BO_HIT_THRESHOLD.
        """
        self.y_all = np.asarray(y_all, dtype=float)
        self.hit_threshold = (BO_HIT_THRESHOLD if hit_threshold is None
                              else float(hit_threshold))
        self.n_total = len(self.y_all)

        # Identify hits (y >= threshold)
        hit_mask = self.y_all >= self.hit_threshold
        self.hit_indices = set(np.where(hit_mask)[0].tolist())
        self.n_hits = len(self.hit_indices)
        self.baseline_hit_rate = self.n_hits / max(self.n_total, 1)

    def acceleration_factor(self, selected_indices, n_init, init_indices=None):
        """AF = (fraction of pool-hits found) / (fraction of pool evaluated).

        Hits are restricted to candidates actually in the pool (excluding init),
        so random selection has an expected AF of 1.0. Returns NaN when the
        pool contains no hits - the metric is undefined, not zero, and lumping
        them as zeros biases aggregate means downward.
        """
        n_evaluated = len(selected_indices)
        if n_evaluated == 0:
            return float("nan")

        if init_indices is not None:
            init_set = set(init_indices)
            pool_hits = self.hit_indices - init_set
        else:
            pool_hits = self.hit_indices
        n_pool_hits = len(pool_hits)
        if n_pool_hits == 0:
            return float("nan")

        n_pool = self.n_total - n_init
        frac_evaluated = n_evaluated / max(n_pool, 1)

        found = sum(1 for idx in selected_indices if idx in pool_hits)
        frac_found = found / n_pool_hits

        return frac_found / frac_evaluated if frac_evaluated > 0 else float("nan")

    def enhancement_factor(self, selected_indices, init_indices=None):
        """EF = mean(y_selected) / mean(y_pool) - graded quality ratio.

        Complements AF: AF measures *speed* of discovery at the binary
        hit_threshold; EF measures graded *quality* of selections vs the
        pool's average score (threshold-agnostic by design - a hit-rate-based
        EF is algebraically equivalent to AF and would carry no new info).

        Pool-aware: excludes the init set from the baseline so a high-quality
        init split doesn't depress EF.
        """
        if len(selected_indices) == 0:
            return float("nan")
        if init_indices is not None and len(init_indices) > 0:
            pool_mask = np.ones(self.n_total, dtype=bool)
            pool_mask[list(init_indices)] = False
            mean_pool = float(self.y_all[pool_mask].mean()) if pool_mask.any() else 0.0
        else:
            mean_pool = float(self.y_all.mean())
        if mean_pool <= 0:
            return float("nan")
        return float(self.y_all[selected_indices].mean() / mean_pool)

    def hit_discovery_curve(self, history):
        """Cumulative fraction of pool-hits discovered vs iteration.

        Only counts hits that were in the pool (not init).
        Returns arrays (iterations, frac_discovered).
        """
        init_set = set(history.get("init_indices", []))
        pool_hits = self.hit_indices - init_set
        n_pool_hits = len(pool_hits)
        if n_pool_hits == 0:
            iters = list(range(1, len(history["selected_indices"]) + 1))
            return np.array(iters), np.zeros(len(iters))

        found = set()
        fracs = []

        for idx in history["selected_indices"]:
            if idx in pool_hits:
                found.add(idx)
            fracs.append(len(found) / n_pool_hits)

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
        global optimum - a common occurrence with large init fractions.

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
        init_indices = history.get("init_indices", [])
        n_init = len(init_indices)
        af = self.acceleration_factor(sel, n_init, init_indices=init_indices)
        ef = self.enhancement_factor(sel, init_indices=init_indices)
        _, hit_curve = self.hit_discovery_curve(history)
        hit_discovery_rate = float(hit_curve[-1]) if len(hit_curve) > 0 else 0.0
        final_best = history["best_so_far"][-1] if history["best_so_far"] else 0.0

        _, regret_curve = self.simple_regret_curve(history)
        final_regret = float(regret_curve[-1]) if len(regret_curve) > 0 else float(self.y_all.max())

        if len(sel) > 0:
            hit_rate = float((self.y_all[sel] >= self.hit_threshold).mean())
        else:
            hit_rate = 0.0
        if init_indices:
            pool_mask = np.ones(self.n_total, dtype=bool)
            pool_mask[list(init_indices)] = False
            baseline_hit_rate = float(
                (self.y_all[pool_mask] >= self.hit_threshold).mean()
            ) if pool_mask.any() else 0.0
        else:
            baseline_hit_rate = self.baseline_hit_rate

        return {
            "AF": af,
            "EF": ef,
            "hit_rate": hit_rate,
            "baseline_hit_rate": baseline_hit_rate,
            "hit_discovery_rate": hit_discovery_rate,
            "best_score_final": final_best,
            "simple_regret_final": final_regret,
            "n_evaluated": len(sel),
            "n_init": n_init,
            "hit_threshold": self.hit_threshold,
        }


    def per_cluster_summary(self, history, groups):
        """Compute threshold-based AF, EF, Hit% per chemistry cluster.

        For each cluster, metrics are computed *within* that cluster's pool -
        AF asks "did BO find this cluster's hits faster than uniform sampling?"
        EF asks "are BO's selections in this cluster richer in hits than the
        cluster's base rate?" Both share the global `hit_threshold`.

        Returns NaN for AF/EF in clusters with no hits in the pool - the
        metric is undefined there, and lumping those as zeros biases the
        aggregate downward.

        Parameters
        ---
        history : dict from run_simulation
        groups  : int array (n,) - cluster labels for the full dataset

        Returns
        ---
        dict : {cluster_id: {"AF": float, "EF": float,
                              "hit_rate": float, "baseline_hit": float,
                              "hit_discovery_rate": float,
                              "n_pool": int, "n_selected": int,
                              "n_pool_hits": int}}
        """
        groups = np.asarray(groups, dtype=int)
        init_set = set(history.get("init_indices", []))
        selected = history["selected_indices"]
        thr = self.hit_threshold
        results = {}

        for cid in sorted(np.unique(groups)):
            cid_mask = (groups == cid)
            cid_indices = set(np.where(cid_mask)[0].tolist())

            pool_in_cluster = cid_indices - init_set
            selected_in_cluster = [s for s in selected if s in pool_in_cluster]
            n_pool_c = len(pool_in_cluster)
            n_sel_c  = len(selected_in_cluster)

            if n_pool_c == 0:
                results[cid] = {
                    "AF": float("nan"), "EF": float("nan"),
                    "hit_rate": 0.0, "baseline_hit": 0.0,
                    "hit_discovery_rate": 0.0,
                    "n_pool": 0, "n_selected": 0, "n_pool_hits": 0,
                }
                continue

            y_pool_c = self.y_all[sorted(pool_in_cluster)]
            pool_hits_c = {idx for idx in pool_in_cluster
                           if self.y_all[idx] >= thr}
            n_pool_hits_c = len(pool_hits_c)

            y_sel = self.y_all[selected_in_cluster] if n_sel_c > 0 else np.array([])
            hit_rate = float((y_sel >= thr).mean()) if n_sel_c > 0 else 0.0
            baseline_hit = float((y_pool_c >= thr).mean())

            if n_pool_hits_c == 0 or n_sel_c == 0:
                af = float("nan")
                hit_discovery_rate = 0.0
            else:
                found = sum(1 for s in selected_in_cluster if s in pool_hits_c)
                hit_discovery_rate = found / n_pool_hits_c
                frac_found = hit_discovery_rate
                frac_eval  = n_sel_c / n_pool_c
                af = frac_found / frac_eval if frac_eval > 0 else float("nan")

            # EF = graded quality ratio (mirror of global EF). Distinct from AF:
            # rewards selections that score high even when they miss the binary
            # hit threshold.
            mean_pool_c = float(y_pool_c.mean()) if len(y_pool_c) > 0 else 0.0
            if n_sel_c == 0 or mean_pool_c <= 0:
                ef = float("nan")
            else:
                ef = float(y_sel.mean() / mean_pool_c)

            results[cid] = {
                "AF": af, "EF": ef,
                "hit_rate": hit_rate, "baseline_hit": baseline_hit,
                "hit_discovery_rate": hit_discovery_rate,
                "n_pool": n_pool_c, "n_selected": n_sel_c,
                "n_pool_hits": n_pool_hits_c,
            }

        return results


# ---
# Plotting functions
# ---
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
    better experiments over time - useful when the global best is found early
    (flat cumulative-best curve) or to compare exploitation quality across methods.

    Parameters
    ---
    histories : list of history dicts (must contain 'y_selected')
    labels    : list of str
    window    : int - rolling window size (default 10)
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
    """Plot hit-discovery rate vs evaluations for multiple methods.

    Hit = y >= BO_HIT_THRESHOLD. The y-axis is the cumulative fraction of pool
    hits found by BO, so a perfect run reaches 100% once every hit is queried.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = SimulationMetrics(y_all)
    thr = metrics.hit_threshold

    for hist, label in zip(histories, labels):
        iters, fracs = metrics.hit_discovery_curve(hist)
        ax.plot(iters, fracs * 100, label=label, linewidth=2)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel(f"Pool Hits Discovered (y >= {thr:.0f}) [%]")
    ax.set_title("Hit Discovery Rate")
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
                 "Regret relative to BO selections only - init set excluded")
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
    ---
    surrogate : RegressionSurrogate - already fitted surrogate
    X_test    : array (n_test, n_features)
    y_test    : array (n_test,) - true pxrd_scores

    Returns
    ---
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

    Left panel  - z-score histogram with N(0,1) reference.
                  A well-calibrated surrogate should look approximately normal.

    Right panel - reliability diagram: observed coverage vs expected coverage at
                  each confidence level.  Perfect calibration = diagonal line.
                  Points above the diagonal → overconfident (sigma too small).
                  Points below → underconfident (sigma too large).

    Parameters
    ---
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

    # -- Left: z-score histogram ---
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
    ax1.set_title(f"Surrogate Calibration - {surrogate_name}\nZ-score Distribution")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -- Right: reliability diagram ---
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
    ax2.set_title(f"Reliability Diagram - {surrogate_name}")
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
    ---
    summaries : list of (label, summary_dict) - from SimulationMetrics.summary()
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
    ---
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
        groups.setdefault(method, {"AF": [], "EF": [], "HitDisc%": []})
        groups[method]["AF"].append(s["AF"])
        groups[method]["EF"].append(s["EF"])
        groups[method]["HitDisc%"].append(s["hit_discovery_rate"] * 100)

    methods = list(groups.keys())
    af_means  = np.array([np.nanmean(groups[m]["AF"])  for m in methods])
    af_stds   = np.array([np.nanstd(groups[m]["AF"])   for m in methods])
    ef_means  = np.array([np.nanmean(groups[m]["EF"])  for m in methods])
    ef_stds   = np.array([np.nanstd(groups[m]["EF"])   for m in methods])
    top_means = np.array([np.mean(groups[m]["HitDisc%"]) for m in methods])
    top_stds  = np.array([np.std(groups[m]["HitDisc%"])  for m in methods])

    # Sort by mean AF descending (NaNs to the bottom)
    order = np.argsort(np.where(np.isnan(af_means), -np.inf, af_means))[::-1]
    methods   = [methods[i]   for i in order]
    af_means  = af_means[order];  af_stds  = af_stds[order]
    ef_means  = ef_means[order];  ef_stds  = ef_stds[order]
    top_means = top_means[order]; top_stds = top_stds[order]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, means, stds, ylabel, title, baseline in [
        (axes[0], af_means,  af_stds,  "AF",          "Acceleration Factor (mean ± std)", 1.0),
        (axes[1], ef_means,  ef_stds,  "EF",          "Enrichment Factor (mean ± std)",   1.0),
        (axes[2], top_means, top_stds, "Hit Disc (%)","Pool Hits Discovered (mean ± std)", None),
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
    ---
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
    ax.set_title(f"Mean {metric} - Acquisition × Surrogate\n(mean across seeds)")

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
    ---
    histories : list of history dicts
    labels    : list of str in format 'acq|...|seed=N'
    y_all     : array - full dataset scores (for simple regret)
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
        ax.set_title("BO Simple Regret - Mean ± Std across Seeds")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("Cumulative Mean Score")
        ax.set_title("BO Selection Quality - Mean ± Std across Seeds")

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

    # Hit discovery
    for hist, label in [(hist_cl, "Constant Liar"), (hist_kb, "Kriging Believer")]:
        iters, fracs = metrics.hit_discovery_curve(hist)
        ax2.plot(iters, fracs * 100, label=label, linewidth=2)
    ax2.set_xlabel("Evaluation")
    ax2.set_ylabel(f"Pool Hits Discovered (y >= {metrics.hit_threshold:.0f}) [%]")
    ax2.set_title("Batch BO Hit Discovery")
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


# ---
# Multi-seed per-cluster and LOCO plots
# ---

def plot_per_cluster_bar(cluster_stats_by_seed, metric="AF",
                         save_path=None):
    """Grouped bar chart: per-cluster metric with error bars across seeds.

    NaN values (clusters with no pool hits) are excluded from mean/std and the
    bar is left blank with an "n/a" annotation, instead of being silently
    counted as zero.

    Parameters
    ---
    cluster_stats_by_seed : list[dict]
        Each element is the output of SimulationMetrics.per_cluster_summary(),
        one per seed.
    metric : str - "AF", "EF", or "hit_discovery_rate"
    save_path : str or None
    """
    if save_path is None:
        save_path = f"docs/bo_per_cluster_{metric.replace('%', 'pct')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(cluster_stats_by_seed[0].keys())
    means, stds, na_flags = [], [], []
    for cid in cluster_ids:
        vals = [seed_stats[cid][metric] for seed_stats in cluster_stats_by_seed
                if cid in seed_stats]
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        if valid:
            means.append(float(np.mean(valid)))
            stds.append(float(np.std(valid)))
            na_flags.append(False)
        else:
            means.append(0.0)
            stds.append(0.0)
            na_flags.append(True)

    # Add aggregate bar (NaNs excluded)
    agg_vals = []
    for seed_stats in cluster_stats_by_seed:
        all_vals = [seed_stats[cid][metric] for cid in cluster_ids
                    if cid in seed_stats]
        valid = [v for v in all_vals if v is not None and not np.isnan(v)]
        if valid:
            agg_vals.append(float(np.mean(valid)))
    means.append(float(np.mean(agg_vals)) if agg_vals else 0.0)
    stds.append(float(np.std(agg_vals)) if agg_vals else 0.0)
    na_flags.append(not bool(agg_vals))

    labels = [f"C{cid}" for cid in cluster_ids] + ["Agg"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue"] * len(cluster_ids) + ["firebrick"]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
                  linewidth=0.5)
    for bar, na in zip(bars, na_flags):
        if na:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "n/a", ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    title_metric = "AF" if metric == "AF" else "EF" if metric == "EF" else metric
    ax.set_title(f"Per-Cluster {title_metric} (mean ± std across seeds)\n"
                 f"NaN clusters (no pool hits) shown as n/a")
    if metric in ("AF", "EF"):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Random ({metric}=1)")
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved per-cluster {metric} → {save_path}")


def plot_evaluate_hit_rate(cluster_stats_by_seed,
                           save_path="docs/bo_per_cluster_hit_rate.png"):
    """Grouped bar chart: BO hit rate vs random baseline, mean ± std across seeds.

    Hit rate  = fraction of BO selections with score >= BO_HIT_THRESHOLD.
    Baseline  = fraction of pool with score >= BO_HIT_THRESHOLD (random guessing).
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
    ax.set_ylabel(f"Hit Rate (%):  score >= {BO_HIT_THRESHOLD:.0f}")
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

    NaN values (clusters with no pool hits) render as a blank bar with an
    "n/a" annotation rather than a misleading zero.

    Parameters
    ---
    loco_results : dict {cluster_id: {"AF": ..., "EF": ..., "n_pool": ...}}
    metric : str
    save_path : str or None
    """
    if save_path is None:
        save_path = f"docs/bo_loco_{metric.replace('%', 'pct')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cluster_ids = sorted(loco_results.keys())
    raw_vals = [loco_results[cid].get(metric, float("nan")) for cid in cluster_ids]
    na_flags = [v is None or (isinstance(v, float) and np.isnan(v)) for v in raw_vals]
    plot_vals = [0.0 if na else v for v, na in zip(raw_vals, na_flags)]
    pool_sizes = [loco_results[cid].get("n_pool", 0) for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"C{c}" for c in cluster_ids], plot_vals,
                  color="steelblue", edgecolor="black", linewidth=0.5)

    for bar, ps, na in zip(bars, pool_sizes, na_flags):
        if na:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    f"n/a\nn={ps}", ha="center", va="bottom", fontsize=8,
                    color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={ps}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(metric)
    ax.set_title(f"Leave-One-Cluster-Out: {metric} per held-out cluster")
    if metric in ("AF", "EF"):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Random ({metric}=1)")
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
    ax.set_ylabel(f"Hit Rate (%):  score >= {BO_HIT_THRESHOLD:.0f}")
    ax.set_title("LOCO: BO Hit Rate vs Random Baseline per Cluster")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(hit_rates, default=0), max(baselines, default=0)) + 10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved LOCO hit rate → {save_path}")


def plot_loco_bar_multiseed(loco_results_by_seed, metric="AF", save_path=None):
    """LOCO bar chart with mean ± std error bars across seeds.

    Same shape as plot_loco_bar but operates on a list of LOCO result dicts
    (one per seed) and draws error bars for each held-out cluster. NaN values
    (clusters with no pool hits) are excluded from the mean/std and rendered
    as a blank bar with an "n/a" annotation.

    Parameters
    ---
    loco_results_by_seed : list[dict]
        Each element is a {cluster_id: {metric: ..., n_pool: ...}} dict
        produced by one LOCO run (one seed).
    metric : str - "AF", "EF", or "hit_discovery_rate"
    save_path : str or None
    """
    if not loco_results_by_seed:
        print(f"[bo_metrics] plot_loco_bar_multiseed: empty input, skipping.")
        return
    if save_path is None:
        save_path = f"docs/bo_loco_{metric.replace('%', 'pct')}_multiseed.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Union of cluster ids across seeds (a cluster could be skipped in some
    # seeds if it ever falls below the min-pool threshold; in practice the
    # threshold is fixed, so this is just defensive).
    all_cids = sorted({cid for r in loco_results_by_seed for cid in r.keys()})

    means, stds, na_flags, pool_sizes = [], [], [], []
    for cid in all_cids:
        vals = [r[cid].get(metric, float("nan"))
                for r in loco_results_by_seed if cid in r]
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        if valid:
            means.append(float(np.mean(valid)))
            stds.append(float(np.std(valid)))
            na_flags.append(False)
        else:
            means.append(0.0)
            stds.append(0.0)
            na_flags.append(True)
        # Pool size is invariant across seeds; take the first one we see.
        ps = next((r[cid].get("n_pool", 0)
                   for r in loco_results_by_seed if cid in r), 0)
        pool_sizes.append(ps)

    # Pool-size weighted aggregate per seed, then mean ± std across seeds.
    agg_per_seed = []
    for r in loco_results_by_seed:
        pairs = [(r[c][metric], r[c].get("n_pool", 0)) for c in r
                 if metric in r[c]
                 and not (isinstance(r[c][metric], float)
                          and np.isnan(r[c][metric]))]
        if not pairs:
            continue
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            continue
        agg_per_seed.append(sum(v * w for v, w in pairs) / total_w)
    if agg_per_seed:
        means.append(float(np.mean(agg_per_seed)))
        stds.append(float(np.std(agg_per_seed)))
        na_flags.append(False)
    else:
        means.append(0.0)
        stds.append(0.0)
        na_flags.append(True)
    pool_sizes.append(sum(pool_sizes))

    labels = [f"C{cid}" for cid in all_cids] + ["W.Mean"]
    x = np.arange(len(labels))
    colors = ["steelblue"] * len(all_cids) + ["firebrick"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, ps, na, m, s in zip(bars, pool_sizes, na_flags, means, stds):
        if na:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    f"n/a\nn={ps}", ha="center", va="bottom", fontsize=8,
                    color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                    f"n={ps}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    n_seeds = len(loco_results_by_seed)
    ax.set_title(f"LOCO {metric} per held-out cluster "
                 f"(mean ± std across {n_seeds} seeds)")
    if metric in ("AF", "EF"):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Random ({metric}=1)")
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved LOCO {metric} (multi-seed) → {save_path}")


def plot_loco_hit_rate_multiseed(loco_results_by_seed,
                                 save_path="docs/bo_loco_hit_rate_multiseed.png"):
    """LOCO BO hit-rate vs random baseline with error bars across seeds.

    Shows a grouped bar chart: BO hit rate (mean ± std across seeds) vs the
    cluster's random baseline (deterministic per cluster, so no error bar).
    """
    if not loco_results_by_seed:
        print(f"[bo_metrics] plot_loco_hit_rate_multiseed: empty input, skipping.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_cids = sorted({cid for r in loco_results_by_seed for cid in r.keys()})

    hit_means, hit_stds, base_vals, pool_sizes = [], [], [], []
    for cid in all_cids:
        hits = [r[cid]["hit_rate"] for r in loco_results_by_seed if cid in r]
        bases = [r[cid]["baseline_hit"] for r in loco_results_by_seed if cid in r]
        hit_means.append(np.mean(hits) * 100 if hits else 0.0)
        hit_stds.append(np.std(hits) * 100 if hits else 0.0)
        # Baseline is deterministic per cluster; same across seeds.
        base_vals.append(np.mean(bases) * 100 if bases else 0.0)
        ps = next((r[cid].get("n_pool", 0)
                   for r in loco_results_by_seed if cid in r), 0)
        pool_sizes.append(ps)

    # Pool-size weighted aggregate over clusters, per seed → mean ± std.
    agg_hits, agg_bases = [], []
    for r in loco_results_by_seed:
        pairs = [(r[c]["hit_rate"], r[c].get("n_pool", 0)) for c in r]
        bpairs = [(r[c]["baseline_hit"], r[c].get("n_pool", 0)) for c in r]
        total = sum(w for _, w in pairs)
        if total <= 0:
            continue
        agg_hits.append(sum(v * w for v, w in pairs) / total * 100)
        agg_bases.append(sum(v * w for v, w in bpairs) / total * 100)
    if agg_hits:
        hit_means.append(float(np.mean(agg_hits)))
        hit_stds.append(float(np.std(agg_hits)))
        base_vals.append(float(np.mean(agg_bases)))
    else:
        hit_means.append(0.0)
        hit_stds.append(0.0)
        base_vals.append(0.0)
    pool_sizes.append(sum(pool_sizes))

    labels = [f"C{cid}" for cid in all_cids] + ["W.Mean"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bo = ["steelblue"] * len(all_cids) + ["firebrick"]
    colors_base = ["lightcoral"] * len(all_cids) + ["salmon"]

    ax.bar(x - width / 2, hit_means, width, yerr=hit_stds, capsize=4,
           color=colors_bo, edgecolor="black", linewidth=0.5,
           label="BO hit rate (mean ± std)")
    ax.bar(x + width / 2, base_vals, width,
           color=colors_base, edgecolor="black", linewidth=0.5,
           label="Random baseline")

    for i, ps in enumerate(pool_sizes):
        y_top = max(hit_means[i] + hit_stds[i], base_vals[i])
        ax.text(x[i], y_top + 1.5, f"n={ps}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Hit Rate (%):  score >= {BO_HIT_THRESHOLD:.0f}")
    n_seeds = len(loco_results_by_seed)
    ax.set_title(f"LOCO BO Hit Rate vs Random Baseline "
                 f"(mean ± std across {n_seeds} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(h + s for h, s in zip(hit_means, hit_stds)),
                       max(base_vals, default=0)) + 10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[bo_metrics] Saved LOCO hit rate (multi-seed) → {save_path}")


def plot_learning_curve(lc_results, save_path="docs/bo_learning_curve.png"):
    """Plot AF, EF, and Hit Rate vs number of initial experiments.

    Parameters
    ---
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
        ["Acceleration Factor", "Enrichment Factor",
         f"Hit Rate (score >= {BO_HIT_THRESHOLD:.0f})"],
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
