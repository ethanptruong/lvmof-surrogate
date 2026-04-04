# -*- coding: utf-8 -*-
"""
bo_cluster_check.py -- SSL cluster-assumption diagnostics for BO observed data.

Checks whether the SSL cluster assumption holds for the current BO observations:
  1. Pairwise feature distance vs. outcome difference (smoothness scatter)
  2. Within-solvent vs. between-solvent outcome variance
  3. k-NN label consistency

Usage:
    python bo_cluster_check.py
    python bo_cluster_check.py --k 5 --gamma 0.25 --out docs/cluster_check.png
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
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

FEATURES_CKPT = os.path.join("checkpoints", "features.pkl")
BO_BORE_GAMMA = 0.25


# ---- helpers -----------------------------------------------------------------

def load_data():
    cached = joblib.load(FEATURES_CKPT)
    X_cv      = cached["X_cv"]
    y_raw     = cached["y_raw"]
    df_merged = cached["df_merged"]
    mask      = cached["mask"]
    return X_cv, y_raw, df_merged[mask].reset_index(drop=True)


def get_solvent_label(df):
    """Primary solvent string per experiment."""
    s1 = df["solvent_1"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    if "solvent_2" in df.columns:
        s2 = df["solvent_2"].fillna("").astype(str).str.strip().str.upper()
        if "phi_1" in df.columns:
            pure = df["phi_1"].fillna(1.0) >= 1.0
            return s1.where(pure, s1 + "+" + s2)
        return s1.where(s2.isin(["", "NA", "NONE"]), s1 + "+" + s2)
    return s1


# ---- check 1: pairwise distance vs. outcome difference -----------------------

def check_pairwise(X, y, ax, n_sample=2000, rng=42):
    n = len(y)
    D     = pairwise_distances(X)
    Ydiff = np.abs(y[:, None] - y[None, :])

    idx = np.triu_indices(n, k=1)
    d_vals = D[idx]
    y_vals = Ydiff[idx]

    rng_obj = np.random.RandomState(rng)
    if len(d_vals) > n_sample:
        sel = rng_obj.choice(len(d_vals), n_sample, replace=False)
        d_vals, y_vals = d_vals[sel], y_vals[sel]

    rho, pval = spearmanr(d_vals, y_vals)

    ax.scatter(d_vals, y_vals, alpha=0.35, s=18, color="steelblue", edgecolors="none")
    ax.set_xlabel("Feature-space distance  ||x_i - x_j||")
    ax.set_ylabel("|y_i - y_j|  (PXRD score difference)")

    verdict = ("PASS: positive correlation -> smoothness holds"
               if rho > 0.15 else
               "FAIL: weak/no correlation -> landscape not smooth")
    ax.set_title(
        f"Check 1: Distance vs. outcome difference\n"
        f"Spearman r = {rho:.3f}  (p = {pval:.3f})\n{verdict}"
    )

    # Bin means overlay
    bins = np.percentile(d_vals, np.linspace(0, 100, 11))
    bin_idx = np.digitize(d_vals, bins)
    bx, by, be = [], [], []
    for b in range(1, len(bins)):
        sel = bin_idx == b
        if sel.sum() >= 3:
            bx.append(np.median(d_vals[sel]))
            by.append(np.mean(y_vals[sel]))
            be.append(np.std(y_vals[sel]) / np.sqrt(sel.sum()))
    ax.errorbar(bx, by, yerr=be, fmt="o-", color="tomato",
                linewidth=1.5, markersize=5, label="bin mean +/- SE", zorder=5)
    ax.legend(fontsize=8)

    return rho, pval


# ---- check 2: within vs. between solvent variance ----------------------------

def check_solvent_variance(y, solvent_labels, ax):
    df = pd.DataFrame({"y": y, "solvent": solvent_labels})
    counts   = df["solvent"].value_counts()
    multi    = counts[counts >= 2].index
    df_multi = df[df["solvent"].isin(multi)]

    if len(df_multi) < 4:
        ax.text(0.5, 0.5,
                "Not enough repeat-solvent observations\n(need >= 2 per solvent)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Check 2: Solvent variance (insufficient data)")
        return None, None

    within_var  = df_multi.groupby("solvent")["y"].var().mean()
    group_means = df_multi.groupby("solvent")["y"].mean()
    between_var = group_means.var()
    ratio       = between_var / (within_var + 1e-9)

    solvents_ordered = group_means.sort_values(ascending=False).index.tolist()
    data_per = [df_multi.loc[df_multi["solvent"] == s, "y"].values
                for s in solvents_ordered]
    ax.boxplot(data_per, tick_labels=solvents_ordered, vert=True)
    ax.set_xticklabels(solvents_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("PXRD score (raw)")

    verdict = ("PASS: between >> within -> solvents form clusters"
               if ratio > 2 else
               "FAIL: within ~ between -> process params dominate")
    ax.set_title(
        f"Check 2: Solvent variance\n"
        f"Within = {within_var:.2f},  Between = {between_var:.2f},  "
        f"Ratio = {ratio:.2f}\n{verdict}"
    )

    return within_var, between_var


# ---- check 3: k-NN label consistency ----------------------------------------

def check_knn_consistency(X, y, gamma, k, ax):
    tau    = np.quantile(y, 1 - gamma)
    labels = (y >= tau).astype(int)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nn.kneighbors(X)

    consistencies = np.array([
        np.mean(labels[indices[i, 1:]] == labels[i])
        for i in range(len(labels))
    ])
    mean_consistency = consistencies.mean()

    colors = ["tomato" if c < 0.5 else "steelblue" for c in consistencies]
    ax.bar(np.arange(len(consistencies)), consistencies, color=colors)
    ax.axhline(mean_consistency, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_consistency:.2f}")
    ax.axhline(0.7, color="green", linestyle=":", linewidth=1.2,
               label="Threshold = 0.70")
    ax.set_xlabel("Observation index")
    ax.set_ylabel(f"Fraction of {k} neighbors with same label")

    verdict = ("PASS: >= 0.70 -> cluster assumption holds"
               if mean_consistency >= 0.70 else
               "FAIL: < 0.70 -> SSL pseudo-labels likely unreliable")
    ax.set_title(
        f"Check 3: k-NN label consistency  (k={k}, gamma={gamma}, tau={tau:.1f})\n"
        f"Mean consistency = {mean_consistency:.2f}\n{verdict}"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    return mean_consistency, tau


# ---- check 4: semivariogram --------------------------------------------------

def check_semivariogram(X, y, ax, n_bins=10):
    """Empirical semivariogram: semivariance vs. distance bin.

    A rising-then-plateauing curve (classic 'sill') indicates spatial
    correlation -- the cluster assumption holds.  A flat curve from the
    start means no correlation structure.
    """
    n = len(y)
    D     = pairwise_distances(X)
    Ydiff = np.abs(y[:, None] - y[None, :])

    idx    = np.triu_indices(n, k=1)
    d_vals = D[idx]
    y_vals = Ydiff[idx]

    bins    = np.percentile(d_vals, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(d_vals, bins)

    bin_centers, semivar, counts = [], [], []
    for b in range(1, len(bins)):
        sel = bin_idx == b
        if sel.sum() >= 3:
            bin_centers.append(np.median(d_vals[sel]))
            semivar.append(np.mean(y_vals[sel] ** 2) / 2.0)
            counts.append(sel.sum())

    bin_centers = np.array(bin_centers)
    semivar     = np.array(semivar)

    # Detect a sill: semivariance rises in first half and flattens in second half
    mid = len(semivar) // 2
    rising  = np.mean(np.diff(semivar[:mid + 1]) > 0) if mid > 0 else 0.0
    flat    = np.std(semivar[mid:]) / (np.mean(semivar[mid:]) + 1e-9)
    has_sill = rising >= 0.5 and flat < 0.25

    sc = ax.scatter(bin_centers, semivar, c=counts, cmap="viridis",
                    s=60, zorder=5)
    ax.plot(bin_centers, semivar, color="steelblue", linewidth=1.5)
    plt.colorbar(sc, ax=ax, label="pair count")
    ax.set_xlabel("Distance bin (median)")
    ax.set_ylabel("Semivariance  (mean |dy|^2 / 2)")

    verdict = ("PASS: rising + sill -> spatial correlation present"
               if has_sill else
               "FAIL: flat/noisy -> no correlation structure")
    ax.set_title(
        f"Check 4: Semivariogram\n"
        f"Rising fraction = {rising:.2f},  Plateau CV = {flat:.2f}\n{verdict}"
    )

    return has_sill, rising, flat


# ---- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSL cluster-assumption diagnostics")
    parser.add_argument("--k",     type=int,   default=3,
                        help="Number of nearest neighbors for consistency check (default 3)")
    parser.add_argument("--gamma", type=float, default=BO_BORE_GAMMA,
                        help="BORE gamma threshold (default 0.25)")
    parser.add_argument("--out",   type=str,   default="docs/cluster_check.png",
                        help="Output figure path")
    args = parser.parse_args()

    print(f"Loading features from {FEATURES_CKPT} ...")
    X, y, df = load_data()
    solvent_labels = get_solvent_label(df)

    print(f"  {len(y)} observations, {X.shape[1]} features")
    print(f"  Unique solvents: {solvent_labels.nunique()}")
    print(f"  Score range: {int(y.min())} - {int(y.max())}")
    print()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"SSL Cluster-Assumption Diagnostics  (n={len(y)}, k={args.k}, gamma={args.gamma})",
        fontsize=13, fontweight="bold"
    )

    rho, pval           = check_pairwise(X, y, axes[0])
    within_var, between_var = check_solvent_variance(y, solvent_labels, axes[1])
    mean_consistency, tau   = check_knn_consistency(X, y, args.gamma, args.k, axes[2])
    has_sill, rising, flat  = check_semivariogram(X, y, axes[3])

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {args.out}")

    print("\n-- Summary ----------------------------------------------------------")
    print(f"  Check 1  Pairwise smoothness:  Spearman r = {rho:.3f}  (p = {pval:.3f})")
    if within_var is not None:
        ratio = between_var / (within_var + 1e-9)
        print(f"  Check 2  Solvent variance:     within = {within_var:.2f},  "
              f"between = {between_var:.2f},  ratio = {ratio:.2f}")
    print(f"  Check 3  k-NN consistency:     mean = {mean_consistency:.2f}  "
          f"(tau = {tau:.1f}, k = {args.k})")
    print(f"  Check 4  Semivariogram:        sill = {has_sill}  "
          f"(rising = {rising:.2f}, plateau CV = {flat:.2f})")

    ssl_ok = (
        (rho > 0.15) and
        (within_var is None or (between_var / (within_var + 1e-9)) > 2) and
        (mean_consistency >= 0.70) and
        has_sill
    )
    print()
    if ssl_ok:
        print("  PASS: All checks pass -- LFBO-SSL is justified for this dataset.")
    else:
        print("  FAIL: One or more checks failed -- prefer LFBO or Thompson over LFBO-SSL.")
    print("-" * 68)


if __name__ == "__main__":
    main()
