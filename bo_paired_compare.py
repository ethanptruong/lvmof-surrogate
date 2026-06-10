"""Paired-comparison reporting for BO ablation results.

Reads the per-seed metric table written by ``run_bo_ablation`` and runs a
paired statistical comparison between two methods that share the same set of
seeds (and, for surrogate-sensitive methods, the same surrogate).

Defensibility
-------------
The comparison is **paired** by seed: each seed produces a (method_A, method_B)
pair on identical init splits, so seed-to-seed variance is differenced out.
This is much more powerful than comparing aggregate means.  We use a
one-sided Wilcoxon signed-rank test (``alternative="greater"``) to avoid
Gaussianity assumptions on small n; report median delta + IQR + p-value.

Decision rule (defensible default)
----------------------------------
A method "wins" only if **all** of the following hold:
    1. n_seeds >= 10  (statistical power; warn otherwise)
    2. median delta > 0
    3. one-sided Wilcoxon p < 0.05
    4. win-rate (fraction of seeds where A > B) >= 0.6

Anything weaker is reported as a tie.  Tie => prefer the simpler method.

Usage
-----
    python bo_paired_compare.py \
        --csv docs/bo_ablation_results.csv \
        --method-a lfbo_ssl --method-b ei \
        --surrogate rf_cl_mi

If --surrogate is omitted, surrogate-agnostic methods (lfbo, random) can be
compared without it.  For sensitive methods, --surrogate is required and the
two methods are matched against the same surrogate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("[paired-compare] scipy is required: pip install scipy", file=sys.stderr)
    raise


METRICS = ["AF", "EF", "hit_discovery_rate", "hit_rate", "best_score_final",
           "simple_regret_final"]
METRIC_HIGHER_IS_BETTER = {
    "AF": True, "EF": True,
    "hit_discovery_rate": True, "hit_rate": True,
    "best_score_final": True,
    "simple_regret_final": False,   # lower regret is better
}


def parse_label(label: str) -> dict:
    """Parse a method label produced by run_bo_ablation.

    Formats:
      - surrogate-agnostic:  "{acq}|seed={seed}"           e.g. "lfbo|seed=42"
      - surrogate-sensitive: "{acq}|{surr}|seed={seed}"     e.g. "ei|rf_cl_mi|seed=42"
      - batch:               "batch|{acq}|{strategy}"       e.g. "batch|lfbo|kriging_believer"

    Returns dict with keys: acq, surrogate (or None), seed (or None), is_batch.
    """
    parts = label.split("|")
    out = {"acq": None, "surrogate": None, "seed": None, "is_batch": False}
    if parts[0] == "batch":
        out["is_batch"] = True
        out["acq"] = parts[1] if len(parts) > 1 else None
        return out

    seed_match = None
    non_seed_parts = []
    for p in parts:
        m = re.match(r"^seed=(\d+)$", p)
        if m:
            seed_match = int(m.group(1))
        else:
            non_seed_parts.append(p)

    out["seed"] = seed_match
    out["acq"] = non_seed_parts[0] if non_seed_parts else None
    if len(non_seed_parts) >= 2:
        out["surrogate"] = non_seed_parts[1]
    return out


def select_runs(df: pd.DataFrame, acq: str,
                surrogate: Optional[str]) -> pd.DataFrame:
    """Return rows of `df` matching (acq, surrogate), indexed by seed."""
    parsed = df["method"].apply(parse_label).apply(pd.Series)
    sub = df.assign(**parsed)
    sub = sub[~sub["is_batch"]]
    sub = sub[sub["acq"] == acq]
    if surrogate is not None:
        sub = sub[sub["surrogate"] == surrogate]
    sub = sub.dropna(subset=["seed"])
    sub["seed"] = sub["seed"].astype(int)
    indexed = sub.set_index("seed")

    # If multiple rows share a seed, the caller's filter is ambiguous —
    # typically means a surrogate-sensitive method was queried with
    # surrogate=None, so rows from every surrogate were returned.
    if indexed.index.has_duplicates:
        dup_seeds = sorted(indexed.index[indexed.index.duplicated()].unique().tolist())
        surrs = sorted(sub["surrogate"].dropna().unique().tolist())
        raise ValueError(
            f"Multiple rows per seed for acq='{acq}' "
            f"(surrogate filter = {surrogate!r}). Duplicated seeds: {dup_seeds}. "
            f"Surrogates present: {surrs}. "
            f"Pass --surrogate <name> (or --surrogate-a/--surrogate-b) "
            f"to disambiguate which surrogate to use for this method."
        )
    return indexed


def paired_compare(df: pd.DataFrame, acq_a: str, acq_b: str,
                   surrogate: Optional[str] = None,
                   surrogate_a: Optional[str] = None,
                   surrogate_b: Optional[str] = None) -> dict:
    """Compute paired deltas and Wilcoxon p-values for all metrics.

    `surrogate` sets both sides; `surrogate_a` / `surrogate_b` override per-side
    (use ``None`` for surrogate-agnostic methods like lfbo or random).
    """
    surr_a = surrogate_a if surrogate_a is not None else surrogate
    surr_b = surrogate_b if surrogate_b is not None else surrogate
    sub_a = select_runs(df, acq_a, surr_a)
    sub_b = select_runs(df, acq_b, surr_b)

    common_seeds = sorted(set(sub_a.index) & set(sub_b.index))
    if not common_seeds:
        a_str = f"{acq_a}" + (f"|{surr_a}" if surr_a else "")
        b_str = f"{acq_b}" + (f"|{surr_b}" if surr_b else "")
        raise ValueError(
            f"No overlapping seeds between '{a_str}' and '{b_str}'.  "
            f"A had seeds {sorted(sub_a.index.tolist())}; "
            f"B had seeds {sorted(sub_b.index.tolist())}."
        )

    n = len(common_seeds)
    results = {"n_seeds": n, "common_seeds": common_seeds, "metrics": {}}

    for metric in METRICS:
        if metric not in df.columns:
            continue
        a_vals = sub_a.loc[common_seeds, metric].astype(float).to_numpy()
        b_vals = sub_b.loc[common_seeds, metric].astype(float).to_numpy()

        # NaN-aware: drop any seed where either method returned NaN
        valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
        a_v = a_vals[valid]
        b_v = b_vals[valid]
        n_valid = len(a_v)
        if n_valid < 3:
            results["metrics"][metric] = {
                "n_valid": n_valid,
                "median_delta": float("nan"),
                "iqr_delta": (float("nan"), float("nan")),
                "p_value": float("nan"),
                "win_rate": float("nan"),
                "decision": "insufficient_data",
                "mean_a": float("nan"), "mean_b": float("nan"),
            }
            continue

        # Sign convention: delta = A - B for higher-is-better; B - A otherwise
        if METRIC_HIGHER_IS_BETTER[metric]:
            delta = a_v - b_v
        else:
            delta = b_v - a_v
        # Now positive delta == "A is better"

        median_delta = float(np.median(delta))
        q1, q3 = np.percentile(delta, [25, 75])

        # Wilcoxon: H0 median(delta)=0; H1 median(delta)>0 ("A is better")
        # zero_method='zsplit' handles ties, mode='approx' for n<25 fallback
        try:
            stat, p = wilcoxon(delta, alternative="greater",
                               zero_method="zsplit")
            p_value = float(p)
        except ValueError:
            # All deltas zero
            p_value = 1.0

        wins  = int(np.sum(delta > 0))
        win_rate = wins / n_valid

        # Defensible decision rule
        if n_valid < 10:
            decision = "underpowered"
        elif median_delta > 0 and p_value < 0.05 and win_rate >= 0.6:
            decision = "A_wins"
        elif median_delta < 0 and (1 - p_value) < 0.05 and win_rate <= 0.4:
            # symmetric check for B winning would need a separate test; we
            # report this only as suggestive.
            decision = "B_likely_better"
        else:
            decision = "tie"

        results["metrics"][metric] = {
            "n_valid": n_valid,
            "median_delta": median_delta,
            "iqr_delta": (float(q1), float(q3)),
            "p_value": p_value,
            "win_rate": win_rate,
            "decision": decision,
            "mean_a": float(a_v.mean()),
            "mean_b": float(b_v.mean()),
        }

    return results


def print_report(results: dict, acq_a: str, acq_b: str,
                 surrogate: Optional[str]) -> None:
    surr_str = f" | surrogate={surrogate}" if surrogate else ""
    n = results["n_seeds"]
    print(f"\n{'=' * 78}")
    print(f"Paired comparison:  A = {acq_a}    vs    B = {acq_b}{surr_str}")
    print(f"Common seeds: n = {n}  ({results['common_seeds']})")
    print("=" * 78)

    if n < 10:
        print(f"[WARN] n_seeds = {n} < 10.  Wilcoxon test is underpowered; "
              f"all 'wins' should be treated as suggestive only.\n"
              f"       Re-run with --bo-ablation-n-seeds 10 (or more) for a "
              f"defensible test.")

    header = (f"{'metric':<22} {'n':>3} {'mean_A':>8} {'mean_B':>8} "
              f"{'med delta':>8} {'IQR delta':>17} {'p (A>B)':>9} {'win%':>6}  decision")
    print(header)
    print("-" * len(header))
    for metric, m in results["metrics"].items():
        if m["decision"] == "insufficient_data":
            print(f"{metric:<22} {m['n_valid']:>3}  insufficient data (need >=3 valid pairs)")
            continue
        iqr_str = f"[{m['iqr_delta'][0]:+.3f},{m['iqr_delta'][1]:+.3f}]"
        print(f"{metric:<22} {m['n_valid']:>3} "
              f"{m['mean_a']:>8.3f} {m['mean_b']:>8.3f} "
              f"{m['median_delta']:>+8.3f} {iqr_str:>17} "
              f"{m['p_value']:>9.4f} {m['win_rate']*100:>5.0f}%  {m['decision']}")
    print("=" * 78)
    print("Sign convention: positive delta means method A is better on that metric.")
    print("Decision rule: A_wins requires n>=10 AND median delta>0 AND p<0.05 AND win-rate>=60%.")


_UNSET = object()


def plot_paired_deltas(results: dict, df: pd.DataFrame,
                       acq_a: str, acq_b: str, surrogate: Optional[str],
                       save_path: str,
                       surrogate_a=_UNSET,
                       surrogate_b=_UNSET) -> None:
    """Per-seed paired-line plot for the headline metrics."""
    import matplotlib.pyplot as plt

    surr_a = surrogate_a if surrogate_a is not _UNSET else surrogate
    surr_b = surrogate_b if surrogate_b is not _UNSET else surrogate
    headline_metrics = ["AF", "EF", "hit_discovery_rate"]
    sub_a = select_runs(df, acq_a, surr_a)
    sub_b = select_runs(df, acq_b, surr_b)
    seeds = results["common_seeds"]

    n_metrics = len(headline_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    if surr_a == surr_b:
        surr_str = f" ({surr_a})" if surr_a else ""
    else:
        surr_str = f" (A={surr_a or 'none'} vs B={surr_b or 'none'})"
    for ax, metric in zip(axes, headline_metrics):
        if metric not in df.columns:
            ax.set_title(f"{metric} — column missing")
            continue
        a_vals = sub_a.loc[seeds, metric].astype(float).to_numpy()
        b_vals = sub_b.loc[seeds, metric].astype(float).to_numpy()

        for a, b in zip(a_vals, b_vals):
            if np.isnan(a) or np.isnan(b):
                continue
            color = "tab:blue" if (
                (a > b) if METRIC_HIGHER_IS_BETTER[metric] else (a < b)
            ) else "tab:red"
            ax.plot([0, 1], [a, b], color=color, alpha=0.5, marker="o", markersize=4)

        # Means as bold horizontal markers
        with np.errstate(invalid="ignore"):
            ax.plot([0], [np.nanmean(a_vals)], marker="_", markersize=30,
                    color="black", markeredgewidth=3)
            ax.plot([1], [np.nanmean(b_vals)], marker="_", markersize=30,
                    color="black", markeredgewidth=3)

        m = results["metrics"].get(metric, {})
        title_extra = ""
        if m and not np.isnan(m.get("p_value", float("nan"))):
            title_extra = f"\nmedian delta={m['median_delta']:+.3f}, p={m['p_value']:.3f}, win={m['win_rate']*100:.0f}%"
        ax.set_xticks([0, 1])
        ax.set_xticklabels([acq_a, acq_b])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}{surr_str}{title_extra}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Paired per-seed comparison: {acq_a} vs {acq_b}  (n={results['n_seeds']} seeds)",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[paired-compare] Saved paired-delta plot -> {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="docs/bo_ablation_results.csv",
                   help="Path to the ablation results CSV (default: docs/bo_ablation_results.csv)")
    p.add_argument("--method-a", required=True,
                   help="Acquisition name for method A (e.g. lfbo_ssl)")
    p.add_argument("--method-b", required=True,
                   help="Acquisition name for method B (e.g. ei)")
    p.add_argument("--surrogate", default=None,
                   help="Surrogate name to hold fixed for BOTH methods "
                        "(required when both are surrogate-sensitive). Omit "
                        "for surrogate-agnostic methods like lfbo or random.")
    p.add_argument("--surrogate-a", default=None,
                   help="Override the surrogate for method A only. Use this "
                        "when the two methods differ in surrogate-sensitivity "
                        "(e.g. A=lfbo_ssl|<surr> vs B=lfbo).")
    p.add_argument("--surrogate-b", default=None,
                   help="Override the surrogate for method B only.")
    p.add_argument("--plot", default=None,
                   help="Save a per-seed paired-delta plot to this path. "
                        "Default: docs/bo_paired_{A}_vs_{B}[_{surr}].png")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[paired-compare] ERROR: results CSV not found at {args.csv}\n"
              f"                 Run `python main.py --bo-ablation` first.",
              file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    # Per-side overrides take precedence; otherwise fall back to --surrogate.
    # Do NOT mirror surrogate_a -> surrogate_b (or vice versa): passing only
    # --surrogate-a is the documented way to express asymmetric sensitivity
    # (e.g. A=lfbo_ssl|rf_cl_mi vs B=lfbo, where B has no surrogate in its
    # label). Use --surrogate to set both sides at once.
    surr_a = args.surrogate_a if args.surrogate_a is not None else args.surrogate
    surr_b = args.surrogate_b if args.surrogate_b is not None else args.surrogate

    results = paired_compare(
        df, args.method_a, args.method_b,
        surrogate_a=surr_a, surrogate_b=surr_b,
    )
    # For the report header, show the per-side surrogates if they differ.
    label_surr = surr_a if surr_a == surr_b else f"A={surr_a or 'none'}, B={surr_b or 'none'}"
    print_report(results, args.method_a, args.method_b, label_surr)

    plot_path = args.plot
    if plot_path is None:
        suffix_parts = []
        if surr_a: suffix_parts.append(f"A-{surr_a}")
        if surr_b: suffix_parts.append(f"B-{surr_b}")
        suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
        plot_path = f"docs/bo_paired_{args.method_a}_vs_{args.method_b}{suffix}.png"
    plot_paired_deltas(results, df, args.method_a, args.method_b,
                       label_surr, plot_path,
                       surrogate_a=surr_a, surrogate_b=surr_b)


if __name__ == "__main__":
    main()
