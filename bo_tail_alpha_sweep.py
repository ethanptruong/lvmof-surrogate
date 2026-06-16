# -*- coding: utf-8 -*-
"""
bo_tail_alpha_sweep.py -- Retrospective sweep over BO_TAIL_WEIGHT_ALPHA.

For each alpha, fits the surrogate on stratified K-fold splits of the training
data and evaluates how well the held-out predictions concentrate on the
high-pxrd region BO actually cares about. Cheap (no BO loop), useful for
narrowing the alpha range before running a full --bo-mode simulate sweep.

Metrics (per fold, averaged):
  - hit@k        : fraction of top-k surrogate predictions with y >= hit_threshold
  - spearman     : rank correlation across all test rows (sanity check)
  - ndcg@k       : graded ranking quality with raw y as relevance
  - tail_rmse    : RMSE on test rows with y >= hit_threshold

Usage:
    python bo_tail_alpha_sweep.py
    python bo_tail_alpha_sweep.py --surrogate rf_mi --alphas 0,0.25,0.5,0.75,1.0,1.5
    python bo_tail_alpha_sweep.py --k-folds 5 --top-k 10 --hit-threshold 7
"""

import argparse
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from bo_core import _compute_tail_weights
from config import BO_TAIL_WEIGHT_THRESHOLD


# ---- data + pipeline construction ---

def load_features(data_path=None):
    """Use the same loader BO simulate uses."""
    from main import _load_bo_data
    X_cv, y_raw, _, _, _ = _load_bo_data(data_path)
    return X_cv, y_raw


def build_pipeline(surrogate_name):
    """Fresh pipeline + the name of its terminal regressor step."""
    from main import _resolve_surrogate
    PARAMS_CKPT = os.path.join("checkpoints", "best_params.pkl")
    try:
        params = joblib.load(PARAMS_CKPT)
    except Exception:
        params = {}
    surr = _resolve_surrogate(surrogate_name, params, ranking_target=False)
    return surr.pipeline, surr._regressor_step_name()


# ---- metrics ---

def ndcg_at_k(y_true, y_pred, k):
    """Standard NDCG@k with raw y as relevance."""
    k = min(k, len(y_true))
    order = np.argsort(-y_pred)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())

    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = y_true[ideal_order]
    idcg = float((ideal_gains * discounts).sum())
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_alpha(X, y, alpha, surrogate_name, k_folds, top_k,
                   hit_threshold, tail_threshold, random_state):
    """Run K-fold CV with the given alpha. Returns dict of per-fold arrays."""
    # Stratify on binary tail label so every fold has hits in both splits
    strat = (y >= hit_threshold).astype(int)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                          random_state=random_state)

    metrics = {"hit@k": [], "spearman": [], "ndcg@k": [], "tail_rmse": []}

    for tr_idx, te_idx in skf.split(X, strat):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe, step_name = build_pipeline(surrogate_name)

        if alpha > 0:
            sw = _compute_tail_weights(y_tr, threshold=tail_threshold,
                                       alpha=alpha)
            fit_kwargs = {f"{step_name}__sample_weight": sw}
        else:
            fit_kwargs = {}

        pipe.fit(X_tr, y_tr, **fit_kwargs)
        y_pred = np.asarray(pipe.predict(X_te))

        # hit@k
        k_eff = min(top_k, len(y_te))
        order = np.argsort(-y_pred)[:k_eff]
        metrics["hit@k"].append(float((y_te[order] >= hit_threshold).mean()))

        # Spearman overall
        if np.std(y_te) > 0 and np.std(y_pred) > 0:
            rho, _ = spearmanr(y_te, y_pred)
            metrics["spearman"].append(rho if np.isfinite(rho) else np.nan)
        else:
            metrics["spearman"].append(np.nan)

        # NDCG@k
        metrics["ndcg@k"].append(ndcg_at_k(y_te, y_pred, k_eff))

        # Tail RMSE
        tail_mask = y_te >= hit_threshold
        if tail_mask.sum() > 0:
            err = y_pred[tail_mask] - y_te[tail_mask]
            metrics["tail_rmse"].append(float(np.sqrt((err ** 2).mean())))
        else:
            metrics["tail_rmse"].append(np.nan)

    return {k: np.asarray(v, dtype=float) for k, v in metrics.items()}


# ---- plotting ---

def plot_results(results, alphas, save_path, surrogate_name,
                 hit_threshold, top_k, n_total, n_tail):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.ravel()

    panels = [
        ("hit@k",     f"Top-{top_k} hit rate (y ≥ {hit_threshold})",
         "fraction", True),
        ("spearman",  "Spearman ρ (overall ranking)",
         "correlation", True),
        ("ndcg@k",    f"NDCG@{top_k}",
         "score", True),
        ("tail_rmse", f"Tail RMSE (test rows with y ≥ {hit_threshold})",
         "RMSE", False),
    ]

    alphas_arr = np.asarray(alphas, dtype=float)

    for ax, (key, title, ylabel, higher_better) in zip(axes, panels):
        means = np.array([np.nanmean(results[a][key]) for a in alphas])
        stds  = np.array([np.nanstd(results[a][key])  for a in alphas])

        ax.errorbar(alphas_arr, means, yerr=stds, fmt="o-", capsize=4,
                    color="#2b6cb0", lw=1.7, ms=6, label="K-fold mean ±1 SD")

        # Highlight the best alpha
        valid = ~np.isnan(means)
        if valid.any():
            if higher_better:
                best_idx = int(np.nanargmax(np.where(valid, means, -np.inf)))
            else:
                best_idx = int(np.nanargmin(np.where(valid, means, np.inf)))
            ax.scatter([alphas_arr[best_idx]], [means[best_idx]], s=160,
                       facecolor="none", edgecolor="#c53030", lw=2.2,
                       zorder=5, label=f"best α = {alphas_arr[best_idx]}")

        # Mark current default
        ax.axvline(0.5, color="gray", ls="--", lw=0.9, alpha=0.6,
                   label="config default (0.5)")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("alpha (BO_TAIL_WEIGHT_ALPHA)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Tail-weight α sweep - surrogate={surrogate_name}, "
        f"threshold={BO_TAIL_WEIGHT_THRESHOLD} | "
        f"n={n_total}, tail (y≥{hit_threshold})={n_tail}",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  saved figure -> {save_path}")


# ---- driver ---

def main():
    parser = argparse.ArgumentParser(
        description="Retrospective tail-weight alpha sweep "
                    "(Option A: held-out surrogate ranking quality).")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to experiment Excel; uses cache otherwise.")
    parser.add_argument("--surrogate", type=str, default="rf_mi",
                        choices=["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                                 "rf_cl_only", "xgb_cl_only"])
    parser.add_argument("--alphas", type=str,
                        default="0,0.25,0.5,0.75,1.0,1.5",
                        help="Comma-separated alphas to sweep.")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10,
                        help="k for hit@k and NDCG@k.")
    parser.add_argument("--hit-threshold", type=float, default=7.0,
                        help="y >= this counts as a hit (matches BO eval).")
    parser.add_argument("--tail-threshold", type=float,
                        default=BO_TAIL_WEIGHT_THRESHOLD,
                        help="Threshold passed to _compute_tail_weights.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="docs/bo_tail_alpha_sweep.png")
    parser.add_argument("--csv", type=str,
                        default="docs/bo_tail_alpha_sweep.csv")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 70)
    print("  BO TAIL-WEIGHT ALPHA SWEEP")
    print("=" * 70)
    print(f"  surrogate       : {args.surrogate}")
    print(f"  alphas          : {alphas}")
    print(f"  k_folds         : {args.k_folds}")
    print(f"  top_k           : {args.top_k}")
    print(f"  hit_threshold   : {args.hit_threshold}")
    print(f"  tail_threshold  : {args.tail_threshold}")

    print("\nLoading featurized data...")
    X, y = load_features(args.data)
    n_tail = int((y >= args.hit_threshold).sum())
    print(f"  n_total = {len(y)},  tail (y >= {args.hit_threshold}) = {n_tail}")
    if n_tail < args.k_folds:
        print(f"  WARNING: only {n_tail} tail rows for {args.k_folds} folds - "
              f"some test folds may have zero tail rows.")

    results = {}
    for alpha in alphas:
        print(f"\nalpha = {alpha}")
        m = evaluate_alpha(X, y, alpha,
                           args.surrogate, args.k_folds, args.top_k,
                           args.hit_threshold, args.tail_threshold,
                           args.random_state)
        results[alpha] = m
        print(f"  hit@{args.top_k}      = "
              f"{np.nanmean(m['hit@k']):.3f} +/- {np.nanstd(m['hit@k']):.3f}")
        print(f"  spearman      = "
              f"{np.nanmean(m['spearman']):.3f} +/- {np.nanstd(m['spearman']):.3f}")
        print(f"  ndcg@{args.top_k}     = "
              f"{np.nanmean(m['ndcg@k']):.3f} +/- {np.nanstd(m['ndcg@k']):.3f}")
        print(f"  tail_rmse     = "
              f"{np.nanmean(m['tail_rmse']):.3f} +/- {np.nanstd(m['tail_rmse']):.3f}")

    # Summary CSV
    rows = []
    for a in alphas:
        m = results[a]
        rows.append({
            "alpha": a,
            f"hit_at_{args.top_k}_mean":  float(np.nanmean(m["hit@k"])),
            f"hit_at_{args.top_k}_std":   float(np.nanstd(m["hit@k"])),
            "spearman_mean":              float(np.nanmean(m["spearman"])),
            "spearman_std":               float(np.nanstd(m["spearman"])),
            f"ndcg_at_{args.top_k}_mean": float(np.nanmean(m["ndcg@k"])),
            f"ndcg_at_{args.top_k}_std":  float(np.nanstd(m["ndcg@k"])),
            "tail_rmse_mean":             float(np.nanmean(m["tail_rmse"])),
            "tail_rmse_std":              float(np.nanstd(m["tail_rmse"])),
        })
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df_out.to_csv(args.csv, index=False)
    print(f"\n  saved CSV    -> {args.csv}")

    plot_results(results, alphas, args.out, args.surrogate,
                 args.hit_threshold, args.top_k, len(y), n_tail)

    # Pick a single recommended alpha (max of hit@k, BO-aligned)
    means_hit = [float(np.nanmean(results[a]["hit@k"])) for a in alphas]
    best_alpha = alphas[int(np.argmax(means_hit))]
    print(f"\n  Recommended alpha (by hit@{args.top_k}): {best_alpha}")
    print("  (current default in config.py is 0.5)")


if __name__ == "__main__":
    main()
