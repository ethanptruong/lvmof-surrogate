# -*- coding: utf-8 -*-
"""
bo_gamma_sweep.py -- Retrospective sweep over BO_LFBO_GAMMA.

For each gamma, trains the LFBO classifier on stratified K-fold splits and
evaluates how well the predicted hit-probabilities rank the high-pxrd region
on held-out rows. Unlike alpha (which only touches the regressor), gamma is
the knob that actually controls LFBO's decision boundary, so this sweep
should produce a real signal.

Metrics (per fold, averaged):
  - hit@k      : top-k probability-ranked test rows that have y >= hit_threshold
  - ndcg@k     : graded ranking quality with raw y as relevance
  - roc_auc    : overall classifier separability for y >= hit_threshold
  - pr_auc     : average precision (top-region focused, robust to imbalance)

The classifier is trained with adaptive_gamma=False so we test each gamma
value directly (the production loop anneals from gamma toward 0.10).

Usage:
    python bo_gamma_sweep.py
    python bo_gamma_sweep.py --gammas 0.05,0.10,0.15,0.20,0.25,0.33,0.50
    python bo_gamma_sweep.py --top-k 20 --hit-threshold 7
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from bo_core import LFBOAcquisition
from config import BO_LFBO_GAMMA, RANDOM_STATE


# ---- data ---

def load_features(data_path=None):
    from main import _load_bo_data
    X_cv, y_raw, _, _, _ = _load_bo_data(data_path)
    return X_cv, y_raw


# ---- metrics ---

def ndcg_at_k(y_true, y_score, k):
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = y_true[ideal_order]
    idcg = float((ideal_gains * discounts).sum())
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_gamma(X, y, gamma, k_folds, top_k, hit_threshold, random_state):
    """Run K-fold CV with the given LFBO gamma. Returns dict of per-fold arrays."""
    strat = (y >= hit_threshold).astype(int)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                          random_state=random_state)

    metrics = {"hit@k": [], "ndcg@k": [], "roc_auc": [], "pr_auc": [],
               "tau_train": [], "n_pos_train": []}

    for tr_idx, te_idx in skf.split(X, strat):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Fixed-gamma LFBO (no adaptive schedule for this retrospective test)
        lfbo = LFBOAcquisition(gamma=gamma, adaptive_gamma=False,
                               random_state=random_state)
        scores = np.asarray(lfbo.score(X_tr, y_tr, X_te))

        # Diagnostic: what tau did the classifier actually use?
        tau = float(np.quantile(y_tr, 1.0 - gamma))
        metrics["tau_train"].append(tau)
        metrics["n_pos_train"].append(int((y_tr >= tau).sum()))

        # hit@k (BO-aligned: y >= 7 regardless of gamma)
        k_eff = min(top_k, len(y_te))
        order = np.argsort(-scores)[:k_eff]
        metrics["hit@k"].append(float((y_te[order] >= hit_threshold).mean()))

        # NDCG@k with raw y as relevance
        metrics["ndcg@k"].append(ndcg_at_k(y_te, scores, k_eff))

        # AUC metrics on the binary y >= hit_threshold task
        binary_te = (y_te >= hit_threshold).astype(int)
        if 0 < binary_te.sum() < len(binary_te):
            metrics["roc_auc"].append(float(roc_auc_score(binary_te, scores)))
            metrics["pr_auc"].append(float(average_precision_score(binary_te, scores)))
        else:
            metrics["roc_auc"].append(np.nan)
            metrics["pr_auc"].append(np.nan)

    return {k: np.asarray(v, dtype=float) for k, v in metrics.items()}


# ---- plotting ---

def plot_results(results, gammas, save_path, hit_threshold, top_k,
                 n_total, n_tail, default_gamma):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.ravel()

    panels = [
        ("hit@k",   f"Top-{top_k} hit rate (y >= {hit_threshold})",
         "fraction", True),
        ("ndcg@k",  f"NDCG@{top_k}",
         "score", True),
        ("roc_auc", f"ROC AUC (y >= {hit_threshold})",
         "AUC", True),
        ("pr_auc",  f"PR AUC / Avg Precision (y >= {hit_threshold})",
         "AP", True),
    ]
    gammas_arr = np.asarray(gammas, dtype=float)

    for ax, (key, title, ylabel, higher_better) in zip(axes, panels):
        means = np.array([np.nanmean(results[g][key]) for g in gammas])
        stds  = np.array([np.nanstd(results[g][key])  for g in gammas])

        ax.errorbar(gammas_arr, means, yerr=stds, fmt="o-", capsize=4,
                    color="#2b6cb0", lw=1.7, ms=6,
                    label="K-fold mean +/- 1 SD")

        valid = ~np.isnan(means)
        if valid.any():
            if higher_better:
                best_idx = int(np.nanargmax(np.where(valid, means, -np.inf)))
            else:
                best_idx = int(np.nanargmin(np.where(valid, means, np.inf)))
            ax.scatter([gammas_arr[best_idx]], [means[best_idx]], s=160,
                       facecolor="none", edgecolor="#c53030", lw=2.2,
                       zorder=5, label=f"best gamma = {gammas_arr[best_idx]}")

        ax.axvline(default_gamma, color="gray", ls="--", lw=0.9, alpha=0.6,
                   label=f"config default ({default_gamma})")
        ax.axvline(0.10, color="purple", ls=":", lw=0.9, alpha=0.5,
                   label="adaptive floor (0.10)")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("gamma (BO_LFBO_GAMMA, top-quantile threshold)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"LFBO gamma sweep | n={n_total}, tail (y>={hit_threshold})={n_tail}",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  saved figure -> {save_path}")


# ---- driver ---

def main():
    parser = argparse.ArgumentParser(
        description="Retrospective LFBO gamma sweep "
                    "(top-quantile threshold for the LFBO classifier).")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--gammas", type=str,
                        default="0.05,0.10,0.15,0.20,0.25,0.33,0.50",
                        help="Comma-separated gammas to sweep.")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hit-threshold", type=float, default=7.0,
                        help="y >= this counts as a hit (BO-aligned).")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", type=str, default="docs/bo_gamma_sweep.png")
    parser.add_argument("--csv", type=str, default="docs/bo_gamma_sweep.csv")
    args = parser.parse_args()

    gammas = [float(g) for g in args.gammas.split(",")]

    print("=" * 70)
    print("  LFBO GAMMA SWEEP")
    print("=" * 70)
    print(f"  gammas        : {gammas}")
    print(f"  k_folds       : {args.k_folds}")
    print(f"  top_k         : {args.top_k}")
    print(f"  hit_threshold : {args.hit_threshold}")

    print("\nLoading featurized data...")
    X, y = load_features(args.data)
    n_tail = int((y >= args.hit_threshold).sum())
    print(f"  n_total = {len(y)}, tail (y >= {args.hit_threshold}) = {n_tail}")

    results = {}
    for gamma in gammas:
        print(f"\ngamma = {gamma}")
        m = evaluate_gamma(X, y, gamma, args.k_folds, args.top_k,
                           args.hit_threshold, args.random_state)
        results[gamma] = m
        print(f"  tau_train (median across folds): {np.median(m['tau_train']):.2f}")
        print(f"  n_pos in train fold (median):   {int(np.median(m['n_pos_train']))}")
        print(f"  hit@{args.top_k}      = "
              f"{np.nanmean(m['hit@k']):.3f} +/- {np.nanstd(m['hit@k']):.3f}")
        print(f"  ndcg@{args.top_k}     = "
              f"{np.nanmean(m['ndcg@k']):.3f} +/- {np.nanstd(m['ndcg@k']):.3f}")
        print(f"  roc_auc       = "
              f"{np.nanmean(m['roc_auc']):.3f} +/- {np.nanstd(m['roc_auc']):.3f}")
        print(f"  pr_auc        = "
              f"{np.nanmean(m['pr_auc']):.3f} +/- {np.nanstd(m['pr_auc']):.3f}")

    # Summary CSV
    rows = []
    for g in gammas:
        m = results[g]
        rows.append({
            "gamma": g,
            "tau_train_median":           float(np.median(m["tau_train"])),
            "n_pos_train_median":         int(np.median(m["n_pos_train"])),
            f"hit_at_{args.top_k}_mean":  float(np.nanmean(m["hit@k"])),
            f"hit_at_{args.top_k}_std":   float(np.nanstd(m["hit@k"])),
            f"ndcg_at_{args.top_k}_mean": float(np.nanmean(m["ndcg@k"])),
            f"ndcg_at_{args.top_k}_std":  float(np.nanstd(m["ndcg@k"])),
            "roc_auc_mean":               float(np.nanmean(m["roc_auc"])),
            "roc_auc_std":                float(np.nanstd(m["roc_auc"])),
            "pr_auc_mean":                float(np.nanmean(m["pr_auc"])),
            "pr_auc_std":                 float(np.nanstd(m["pr_auc"])),
        })
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df_out.to_csv(args.csv, index=False)
    print(f"\n  saved CSV    -> {args.csv}")

    plot_results(results, gammas, args.out,
                 args.hit_threshold, args.top_k, len(y), n_tail,
                 default_gamma=BO_LFBO_GAMMA)

    means_hit = [float(np.nanmean(results[g]["hit@k"])) for g in gammas]
    best_gamma = gammas[int(np.argmax(means_hit))]
    print(f"\n  Recommended gamma (by hit@{args.top_k}): {best_gamma}")
    print(f"  (current default in config.py is {BO_LFBO_GAMMA})")


if __name__ == "__main__":
    main()
