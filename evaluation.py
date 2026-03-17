"""
evaluation.py
Evaluation and visualization: ROC/PRC, learning curves, confusion matrices, SHAP.
All plot logic copied EXACTLY from the source notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
from sklearn.base import clone
from sklearn.model_selection import (cross_val_predict, learning_curve,
                                     StratifiedGroupKFold)
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from models import scoring_ordinal


def _partition_cv(cv):
    """Return a single-repeat StratifiedGroupKFold for use with cross_val_predict.
    cross_val_predict requires partitions; repeated CV violates this."""
    n_splits = getattr(cv, "n_splits", 5)
    random_state = getattr(cv, "random_state", 42)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)


# ─────────────────────────────────────────────────────────────
# plot_roc_prc
# ─────────────────────────────────────────────────────────────
def plot_roc_prc(pipelines, X, y, cv, groups,
                 positive_classes=None) -> None:
    """
    Generate ROC and precision-recall curves for each pipeline.

    Parameters
    ----------
    pipelines        : list of (name, pipe, n_jobs)
    positive_classes : list of int class ids (default [2])
    """
    if positive_classes is None:
        positive_classes = [2]

    POSITIVE_CLASSES = positive_classes
    CLASS_LABELS = {0: "Amorphous", 1: "Partial", 2: "Crystalline"}

    oof_pred_proba = {}
    summary_rows = []

    print("\n─── Generating out-of-fold probabilities ─────────────────────────")
    for name, pipe, n_jobs_cvpred in pipelines:
        print(f"Running: {name}")
        proba = cross_val_predict(
            pipe,
            X, y,
            cv=_partition_cv(cv),
            groups=groups,
            method="predict_proba",
            n_jobs=n_jobs_cvpred,
            verbose=0,
        )
        oof_pred_proba[name] = proba

    n_rows = len(POSITIVE_CLASSES)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14, 5 * n_rows),
        squeeze=False
    )

    for r, pos_class in enumerate(POSITIVE_CLASSES):
        y_bin = (y == pos_class).astype(int)
        prevalence = y_bin.mean()

        ax_roc = axes[r, 0]
        ax_pr  = axes[r, 1]

        for name, _, _ in pipelines:
            proba = oof_pred_proba[name][:, pos_class]

            # ROC
            fpr, tpr, _ = roc_curve(y_bin, proba)
            auroc = roc_auc_score(y_bin, proba)

            # PRC
            precision, recall, _ = precision_recall_curve(y_bin, proba)
            ap = average_precision_score(y_bin, proba)

            ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auroc:.3f})")
            ax_pr.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")

            summary_rows.append({
                "positive_class": pos_class,
                "class_name": CLASS_LABELS.get(pos_class, str(pos_class)),
                "pipeline": name,
                "roc_auc": auroc,
                "average_precision": ap,
                "prevalence": prevalence,
            })

        # ROC cosmetics
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        ax_roc.set_title(f"ROC — {CLASS_LABELS.get(pos_class, pos_class)} vs Rest")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(fontsize=9)
        ax_roc.grid(alpha=0.25)

        # PR cosmetics
        ax_pr.axhline(prevalence, linestyle="--", color="gray", lw=1,
                      label=f"No-skill baseline = {prevalence:.3f}")
        ax_pr.set_title(f"Precision-Recall — {CLASS_LABELS.get(pos_class, pos_class)} vs Rest")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=9)
        ax_pr.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("roc_prc_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: roc_prc_comparison.png")

    results_auc = (
        pd.DataFrame(summary_rows)
          .sort_values(["positive_class", "roc_auc", "average_precision"],
                       ascending=[True, False, False])
          .reset_index(drop=True)
    )

    print("\n─── AUC / AP summary ─────────────────────────────────────────────")
    for pos_class in POSITIVE_CLASSES:
        cls_name = CLASS_LABELS.get(pos_class, str(pos_class))
        sub = results_auc[results_auc["positive_class"] == pos_class].copy()

        print(f"\n{cls_name} vs Rest")
        print(sub[["pipeline", "roc_auc", "average_precision", "prevalence"]]
                .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_auc.to_csv("roc_prc_auc_summary.csv", index=False)
    print("\nSaved: roc_prc_auc_summary.csv")


# ─────────────────────────────────────────────────────────────
# plot_learning_curves
# ─────────────────────────────────────────────────────────────
def plot_learning_curves(pipelines, X, y, cv, groups, scoring,
                         scoring_key="qwk", score_name="QWK") -> None:
    """
    Compute and plot learning curves for each pipeline.

    Parameters
    ----------
    pipelines    : list of (name, pipe, n_jobs)
    scoring_key  : key in scoring dict (default "qwk")
    score_name   : display label (default "QWK")
    """
    TRAIN_SIZES_FRAC = np.linspace(0.2, 1.0, 5)
    LC_SCORING_KEY = scoring_key
    LC_SCORE_NAME  = score_name

    def compute_learning_curve(name, pipe, n_jobs_lc):
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=pipe,
            X=X,
            y=y,
            groups=groups,
            cv=cv,
            scoring=scoring[LC_SCORING_KEY],
            train_sizes=TRAIN_SIZES_FRAC,
            n_jobs=n_jobs_lc,
            shuffle=True,
            random_state=42,
            error_score="raise",
        )

        out = {
            "name": name,
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        }
        return out

    learning_curve_results = []
    print("\n─── Computing learning curves ─────────────────────────────")
    for name, pipe, n_jobs_lc in pipelines:
        print(f"Running: {name}")
        res = compute_learning_curve(name, pipe, n_jobs_lc)
        learning_curve_results.append(res)

    n_pipes = len(learning_curve_results)
    n_cols = 2
    n_rows = (n_pipes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.ravel()

    for ax, res in zip(axes, learning_curve_results):
        ts = res["train_sizes"]
        tr_m, tr_s = res["train_mean"], res["train_std"]
        va_m, va_s = res["val_mean"], res["val_std"]

        ax.plot(ts, tr_m, marker="o", lw=2, label=f"Train {LC_SCORE_NAME}", color="tab:blue")
        ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.18, color="tab:blue")

        ax.plot(ts, va_m, marker="s", lw=2, label=f"Validation {LC_SCORE_NAME}", color="tab:orange")
        ax.fill_between(ts, va_m - va_s, va_m + va_s, alpha=0.18, color="tab:orange")

        ax.set_title(res["name"])
        ax.set_xlabel("Training samples")
        ax.set_ylabel(LC_SCORE_NAME)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    for ax in axes[n_pipes:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("learning_curves_qwk.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: learning_curves_qwk.png")

    summary_rows = []
    for res in learning_curve_results:
        for i, n_train in enumerate(res["train_sizes"]):
            summary_rows.append({
                "pipeline": res["name"],
                "n_train": int(n_train),
                "train_mean": float(res["train_mean"][i]),
                "train_std": float(res["train_std"][i]),
                "val_mean": float(res["val_mean"][i]),
                "val_std": float(res["val_std"][i]),
                "generalization_gap": float(res["train_mean"][i] - res["val_mean"][i]),
            })

    lc_df = pd.DataFrame(summary_rows)
    lc_df.to_csv("learning_curve_summary_qwk.csv", index=False)
    print("Saved: learning_curve_summary_qwk.csv")

    print("\n─── Learning curve summary (largest train size) ─────────────────")
    final_rows = (
        lc_df.sort_values(["pipeline", "n_train"])
             .groupby("pipeline", as_index=False)
             .tail(1)
             .sort_values("val_mean", ascending=False)
    )

    for _, row in final_rows.iterrows():
        print(
            f"{row['pipeline']}\n"
            f"  n_train={int(row['n_train'])}\n"
            f"  Train {LC_SCORE_NAME}: {row['train_mean']:.4f} ± {row['train_std']:.4f}\n"
            f"  Valid {LC_SCORE_NAME}: {row['val_mean']:.4f} ± {row['val_std']:.4f}\n"
            f"  Gap: {row['generalization_gap']:.4f}\n"
        )

    print("\n─── Quick interpretation ───────────────────────────────────────")
    for _, row in final_rows.iterrows():
        gap = row["generalization_gap"]
        if gap > 0.10:
            flag = "possible overfitting"
        elif gap > 0.04:
            flag = "moderate gap"
        else:
            flag = "fairly tight train/validation fit"
        print(f"{row['pipeline']}: {flag}")


# ─────────────────────────────────────────────────────────────
# plot_confusion_matrices
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(pipelines, X, y, cv, groups) -> None:
    """
    Generate raw-count and row-normalized confusion matrices for each pipeline.

    Parameters
    ----------
    pipelines : list of (name, pipe, n_jobs)
    """
    CLASS_LABELS = ["Amorphous", "Partial", "Crystalline"]
    CLASS_IDS = [0, 1, 2]

    oof_preds = {}
    print("\n─── Generating out-of-fold predictions for confusion matrices ───")
    for name, pipe, n_jobs_cvpred in pipelines:
        print(f"Running: {name}")
        y_pred = cross_val_predict(
            pipe,
            X, y,
            cv=_partition_cv(cv),
            groups=groups,
            method="predict",
            n_jobs=n_jobs_cvpred,
            verbose=0,
        )
        oof_preds[name] = y_pred

    n_pipes = len(pipelines)
    n_cols = 2
    n_rows = (n_pipes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    axes = axes.ravel()

    for ax, (name, _, _) in zip(axes, pipelines):
        cm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\nCounts")

    for ax in axes[n_pipes:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("confusion_matrices_counts.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrices_counts.png")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    axes = axes.ravel()

    for ax, (name, _, _) in zip(axes, pipelines):
        cm_norm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"{name}\nNormalized by true class")

    for ax in axes[n_pipes:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("confusion_matrices_normalized.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrices_normalized.png")

    count_tables = []
    norm_tables = []

    print("\n─── Confusion matrices (counts) ─────────────────────────────────")
    for name, _, _ in pipelines:
        cm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS)
        df_cm = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_LABELS],
                                columns=[f"pred_{c}" for c in CLASS_LABELS])
        print(f"\n{name}")
        print(df_cm.to_string())

        df_long = df_cm.reset_index().rename(columns={"index": "true_label"})
        df_long.insert(0, "pipeline", name)
        count_tables.append(df_long)

    print("\n─── Confusion matrices (row-normalized) ─────────────────────────")
    for name, _, _ in pipelines:
        cm_norm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS, normalize="true")
        df_cm_norm = pd.DataFrame(cm_norm, index=[f"true_{c}" for c in CLASS_LABELS],
                                     columns=[f"pred_{c}" for c in CLASS_LABELS])
        print(f"\n{name}")
        print(df_cm_norm.to_string(float_format=lambda x: f"{x:.3f}"))

        df_long = df_cm_norm.reset_index().rename(columns={"index": "true_label"})
        df_long.insert(0, "pipeline", name)
        norm_tables.append(df_long)

    pd.concat(count_tables, ignore_index=True).to_csv("confusion_matrix_counts.csv", index=False)
    pd.concat(norm_tables, ignore_index=True).to_csv("confusion_matrix_normalized.csv", index=False)

    print("\nSaved: confusion_matrix_counts.csv")
    print("Saved: confusion_matrix_normalized.csv")


# ─────────────────────────────────────────────────────────────
# run_shap_analysis
# ─────────────────────────────────────────────────────────────
def run_shap_analysis(pipes, X, y) -> None:
    """
    Run SHAP analysis for tree-based pipelines (XGBoost or RandomForest).

    Parameters
    ----------
    pipes : list of (label, pipe)
    X     : np.ndarray
    y     : np.ndarray
    """

    def fit_full_pipeline(pipe, X, y):
        fitted = clone(pipe)
        fitted.fit(X, y)
        return fitted

    def transform_until_before_smote(pipe_fitted, X):
        """
        Preferred for SHAP:
        transform through all steps up to but NOT including SMOTE and final model.
        This gives the actual feature space seen by the model at inference time.
        """
        Xt = X
        used_steps = []
        for name, step in pipe_fitted.steps:
            if name in ("smote", "ordinal_xgb", "ordinal_rf"):
                break
            Xt = step.transform(Xt)
            used_steps.append(name)
        return Xt, used_steps

    def get_feature_names_generic(n_features, prefix="f"):
        return [f"{prefix}_{i:04d}" for i in range(n_features)]

    def get_top_features(mean_abs_shap, feature_names, top_n=25):
        df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        return df.head(top_n)

    def save_barplot(df_top, title, filename, color="tab:blue"):
        plt.figure(figsize=(8, max(6, 0.3 * len(df_top))))
        df_plot = df_top.iloc[::-1]
        plt.barh(df_plot["feature"], df_plot["mean_abs_shap"], color=color)
        plt.xlabel("Mean |SHAP value|")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=180, bbox_inches="tight")
        plt.show()
        print(f"Saved: {filename}")

    all_summary_rows = []

    print("\n─── SHAP analysis for tree pipelines ────────────────────────────")

    _BAR_COLORS = {
        "XGB | MI only":          "tab:blue",
        "XGB | CL + MI":          "tab:orange",
        "XGB | CL only (triplet)": "tab:purple",
        "RF  | MI only":          "tab:cyan",
        "RF  | CL + MI":          "tab:olive",
        "RF  | CL only (triplet)": "tab:pink",
    }
    _AGG_COLORS = {
        "XGB | MI only":          "tab:green",
        "XGB | CL + MI":          "tab:red",
        "XGB | CL only (triplet)": "tab:brown",
        "RF  | MI only":          "tab:cyan",
        "RF  | CL + MI":          "tab:olive",
        "RF  | CL only (triplet)": "tab:pink",
    }

    for pipe_label, pipe in pipes:
        print(f"\nFitting full pipeline for SHAP: {pipe_label}")
        pipe_slug = pipe_label.replace(" ", "_").replace("|", "-").replace("/", "-")
        fitted_pipe = fit_full_pipeline(pipe, X, y)

        # Transform through impute -> vt -> [cl] -> mi
        X_model, used_steps = transform_until_before_smote(fitted_pipe, X)
        feature_names = get_feature_names_generic(X_model.shape[1],
                                                   prefix=f"{pipe_slug.lower()}_feat")

        print(f"  Feature space entering model: {X_model.shape}")
        print(f"  Steps used before model: {used_steps}")

        ordinal_key = "ordinal_xgb" if "ordinal_xgb" in fitted_pipe.named_steps else "ordinal_rf"
        ordinal_step = fitted_pipe.named_steps[ordinal_key]
        classifiers = ordinal_step.classifiers_   # dict: threshold -> fitted binary estimator

        threshold_importances = []

        for threshold, base_model in classifiers.items():
            print(f"  Explaining Frank-Hall binary model for threshold k={threshold}")

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_model)

            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[..., -1]

            mean_abs = np.abs(shap_values).mean(axis=0)

            df_imp = pd.DataFrame({
                "pipeline": pipe_label,
                "threshold_k": threshold,
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }).sort_values("mean_abs_shap", ascending=False)

            threshold_importances.append(df_imp)
            all_summary_rows.append(df_imp)

            csv_name = f"shap_importance_{pipe_slug}_threshold_{threshold}.csv"
            df_imp.to_csv(csv_name, index=False)
            print(f"Saved: {csv_name}")

            top25 = df_imp.head(25)
            plot_name = f"shap_bar_{pipe_slug}_threshold_{threshold}.png"
            save_barplot(
                top25,
                title=f"{pipe_label} — Mean |SHAP| (threshold k={threshold})",
                filename=plot_name,
                color=_BAR_COLORS.get(pipe_label, "tab:gray"),
            )

            try:
                shap.summary_plot(
                    shap_values,
                    X_model,
                    feature_names=feature_names,
                    max_display=20,
                    show=False
                )
                plt.title(f"{pipe_label} — SHAP beeswarm (threshold k={threshold})")
                beeswarm_name = f"shap_beeswarm_{pipe_slug}_threshold_{threshold}.png"
                plt.tight_layout()
                plt.savefig(beeswarm_name, dpi=180, bbox_inches="tight")
                plt.show()
                print(f"Saved: {beeswarm_name}")
            except Exception as e:
                print(f"  Beeswarm skipped for {pipe_label}, threshold {threshold}: {e}")

        df_all = pd.concat(threshold_importances, ignore_index=True)
        df_agg = (
            df_all.groupby("feature", as_index=False)["mean_abs_shap"]
                  .mean()
                  .sort_values("mean_abs_shap", ascending=False)
                  .rename(columns={"mean_abs_shap": "mean_abs_shap_avg_over_thresholds"})
        )

        agg_csv = f"shap_importance_{pipe_slug}_aggregated.csv"
        df_agg.to_csv(agg_csv, index=False)
        print(f"Saved: {agg_csv}")

        top25_agg = df_agg.head(25)
        agg_plot = f"shap_bar_{pipe_slug}_aggregated.png"
        save_barplot(
            top25_agg.rename(columns={"mean_abs_shap_avg_over_thresholds": "mean_abs_shap"}),
            title=f"{pipe_label} — Aggregated Mean |SHAP| Across Frank-Hall Thresholds",
            filename=agg_plot,
            color=_AGG_COLORS.get(pipe_label, "tab:gray"),
        )

    print("\n─── Combined SHAP comparison ────────────────────────────────────")

    df_combined = pd.concat(all_summary_rows, ignore_index=True)
    df_compare = (
        df_combined.groupby(["pipeline", "feature"], as_index=False)["mean_abs_shap"]
                   .mean()
                   .rename(columns={"mean_abs_shap": "mean_abs_shap_avg_over_thresholds"})
    )

    top_features_union = (
        df_compare.groupby("feature")["mean_abs_shap_avg_over_thresholds"]
                  .max()
                  .sort_values(ascending=False)
                  .head(20)
                  .index
    )

    df_plot = df_compare[df_compare["feature"].isin(top_features_union)].copy()
    pivot_df = df_plot.pivot(index="feature", columns="pipeline",
                             values="mean_abs_shap_avg_over_thresholds").fillna(0.0)

    pivot_df = pivot_df.loc[pivot_df.max(axis=1).sort_values().index]

    cols = list(pivot_df.columns)
    n_cols = len(cols)
    total_width = 0.8
    bar_h = total_width / n_cols
    offsets = np.linspace(-(total_width - bar_h) / 2,
                          (total_width - bar_h) / 2, n_cols)
    _palette = list(_BAR_COLORS.values()) + [
        "tab:gray", "tab:olive", "tab:cyan", "tab:pink"
    ]

    plt.figure(figsize=(10, max(6, 0.35 * len(pivot_df))))
    x = np.arange(len(pivot_df))
    for i, col in enumerate(cols):
        color = _BAR_COLORS.get(col, _palette[i % len(_palette)])
        plt.barh(x + offsets[i], pivot_df[col], height=bar_h, label=col, color=color)

    plt.yticks(x, pivot_df.index)
    plt.xlabel("Mean |SHAP| averaged over ordinal thresholds")
    plt.title("Top SHAP Features — Pipeline Comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("shap_pipeline_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: shap_pipeline_comparison.png")

    pivot_df.reset_index().to_csv("shap_pipeline_comparison.csv", index=False)
    print("Saved: shap_pipeline_comparison.csv")


# -----------------------------------------------------------------------------
# SECTION B  --  Propagate names through pipeline VT + [CL] + MI
# -----------------------------------------------------------------------------

def transform_with_names(fitted_pipe, X_in, names_in, groups_in, label=""):
    # Run X through impute -> vt -> [cl] -> mi, tracking which features survive
    # each selection step.  Stops before 'smote' and the final ordinal model.
    # Returns (X_model, feat_names_arr, feat_groups_arr).
    Xt    = X_in.copy()
    names = np.array(names_in,  dtype=object)
    grps  = np.array(groups_in, dtype=object)

    for step_name, step in fitted_pipe.steps:
        if step_name in ("smote", "ordinal_xgb", "ordinal_rf"):
            break

        if step_name == "impute":
            Xt = step.transform(Xt)
            # SimpleImputer keeps all columns; names unchanged

        elif step_name == "vt":
            Xt    = step.transform(Xt)
            mask  = step.get_support()
            names = names[mask]
            grps  = grps[mask]

        elif step_name == "cl":
            Xt    = step.transform(Xt)
            n_out = Xt.shape[1]
            emb   = getattr(step, "embedding_dim", n_out)
            # If concat_original=True the output is [CL_emb | original_features]
            cl_n  = [f"cl_emb_{i:03d}" for i in range(min(emb, n_out))]
            cl_g  = ["CL Embedding"] * len(cl_n)
            if n_out > emb:
                n_orig = n_out - emb
                orig_n = (list(names[:n_orig]) if len(names) >= n_orig
                          else [f"orig_{i}" for i in range(n_orig)])
                orig_g = (list(grps[:n_orig])  if len(grps)  >= n_orig
                          else ["CL Original"] * n_orig)
                names = np.array(cl_n + orig_n, dtype=object)
                grps  = np.array(cl_g + orig_g, dtype=object)
            else:
                names = np.array(cl_n, dtype=object)
                grps  = np.array(cl_g, dtype=object)

        elif step_name == "mi":
            Xt    = step.transform(Xt)
            mask  = step.get_support()
            names = names[mask]
            grps  = grps[mask]

        else:
            try:
                Xt_new = step.transform(Xt)
                if Xt_new.shape[1] == Xt.shape[1]:
                    Xt = Xt_new
                else:
                    print(f"  [warn] '{step_name}' changed shape "
                          f"{Xt.shape[1]}->{Xt_new.shape[1]}; names reset.")
                    Xt    = Xt_new
                    names = np.array([f"{step_name}_{i}"
                                      for i in range(Xt.shape[1])], dtype=object)
                    grps  = np.array(["Unknown"] * Xt.shape[1], dtype=object)
            except Exception as err:
                print(f"  [warn] Could not transform '{step_name}': {err}")

    if label:
        print(f"  [{label}] features entering XGB: {Xt.shape[1]} "
              f" (names tracked: {len(names)})")
    return Xt, names, grps


# -----------------------------------------------------------------------------
# SECTION C  --  Colour palette (one colour per feature category)
# -----------------------------------------------------------------------------

_GRP_PAL = {
    "Process Variables":          "#e63946",
    "Process Interactions":       "#f4722b",
    "Metal Center (mendeleev)":   "#2a9d8f",
    "Metal Precursor Complex":    "#48cae4",
    "Linker ChemBERT":            "#6a4c93",
    "Mod ChemBERT":               "#9b5de5",
    "Linker Physchem/FP":         "#b5a0d8",
    "Mod Physchem/FP":            "#c8b6e2",
    "Linker Morgan FP":           "#457b9d",
    "Modulator Morgan FP":        "#1d3557",
    "Modulator Equiv.":           "#4e9af1",
    "Precursor Ligand FP":        "#235789",
    "Precursor Ligand RAC":       "#52b788",
    "Modulator RAC":              "#74c69d",
    "Ligand TEP (Electronic)":    "#f4a261",
    "Ligand Sterics":             "#e76f51",
    "Reaction FP (DRFP)":         "#264653",
    "3D SOAP Descriptor":         "#2b9348",
    "3D SOAP (Precursor)":        "#2b9348",
    "3D SOAP (Linker)":           "#55a630",
    "G14 Hub Topology":           "#e9c46a",
    "G14 Hub SMARTS":             "#f4d58d",
    "Linker TTP":                 "#f9c74f",
    "Linker EState":              "#90e0ef",
    "Linker Topological":         "#00b4d8",
    "Linker Torsion FP":          "#0077b6",
    "Linker Atom-Pair FP":        "#023e8a",
    "Halide Features":            "#d4a5a5",
    "Inventory Numeric":          "#a8dadc",
    "Process Variables (raw)":    "#f1faee",
    "KMeans Cluster OHE":         "#adb5bd",
    "CL Embedding":               "#ff006e",
    "Other Structural":           "#888888",
    "Unknown":                    "#cccccc",
}

def _pal(group):
    return _GRP_PAL.get(group, "#888888")


# -----------------------------------------------------------------------------
# SECTION D  --  Main SHAP analysis + three-plot output per pipeline
# -----------------------------------------------------------------------------

def run_shap_featurized(pipe_label, pipe, X, y, X_names, X_groups, top_n=15,
                        fitted_pipe=None):
    print(f"\n{'='*70}")
    print(f"  SHAP  >>  {pipe_label}")
    print(f"{'='*70}")

    # Sanitize label for filenames (remove chars invalid on Windows)
    import re
    safe_label = re.sub(r'[|<>:"/\\?*]', '_', pipe_label).strip()

    if fitted_pipe is not None:
        fitted = fitted_pipe
    else:
        fitted = clone(pipe)
        fitted.fit(X, y)

    X_model, feat_names, feat_grps = transform_with_names(
        fitted, X, X_names, X_groups, label=pipe_label)

    # Support both XGB and RF ordinal pipelines
    for step_name in ("ordinal_xgb", "ordinal_rf"):
        if step_name in fitted.named_steps:
            ordinal_step = fitted.named_steps[step_name]
            break
    else:
        raise KeyError("No ordinal step found (expected 'ordinal_xgb' or 'ordinal_rf')")
    classifiers  = ordinal_step.classifiers_

    thresh_abs = {}
    shap_stack = []
    for thresh, tree_model in classifiers.items():
        explainer = shap.TreeExplainer(tree_model)
        sv = np.asarray(explainer.shap_values(X_model))
        if sv.ndim == 3:
            sv = sv[..., -1]
        thresh_abs[thresh] = np.abs(sv).mean(axis=0)
        shap_stack.append(sv)

    mean_abs_shap = np.stack(list(thresh_abs.values())).mean(axis=0)
    shap_avg      = np.mean(shap_stack, axis=0)

    df_imp = pd.DataFrame({
        "feature":       feat_names,
        "group":         feat_grps,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df_imp.to_csv(f"shap_named_{safe_label}.csv", index=False)
    print(f"  Saved: shap_named_{safe_label}.csv")

    # ── Plot 1 -- Feature-group bar chart (SUM) ───────────────────────────
    # Total group importance -- larger groups will naturally score higher.
    # Useful for understanding raw predictive weight of each block.
    df_grp     = (df_imp.groupby("group")["mean_abs_shap"]
                        .sum()
                        .sort_values(ascending=True))
    df_grp_cnt = df_imp.groupby("group")["mean_abs_shap"].count()

    fig1, ax1 = plt.subplots(figsize=(11, max(5, 0.42 * len(df_grp))))
    cols1 = [_pal(g) for g in df_grp.index]
    bars1 = ax1.barh(df_grp.index, df_grp.values,
                     color=cols1, edgecolor="white", linewidth=0.6)
    xmax = df_grp.max()
    for bar, val, grp in zip(bars1, df_grp.values, df_grp.index):
        n_feat = df_grp_cnt[grp]
        ax1.text(val + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}  (n={n_feat})", va="center", ha="left", fontsize=7.5)
    ax1.set_xlabel(
        "Sum of Mean |SHAP value|  (summed across Frank-Hall thresholds)",
        fontsize=11)
    ax1.set_title(
        f"{pipe_label}\nSHAP Feature-Category Importance (total)  --  LVMOF Crystallinity",
        fontsize=12, fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig1.savefig(f"shap_group_{safe_label}.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"  Saved: shap_group_{safe_label}.png")

    df_grp.sort_values(ascending=False).reset_index().rename(
        columns={"mean_abs_shap": "sum_mean_abs_shap"}).assign(
        n_features=lambda d: d["group"].map(df_grp_cnt)).to_csv(
        f"shap_group_{safe_label}.csv", index=False)
    print(f"  Saved: shap_group_{safe_label}.csv")

    # ── Plot 1b -- Feature-group bar chart (MEAN, size-normalised) ────────
    # Divides each group's total SHAP by the number of features in that group.
    # Answers: "which feature TYPE carries the most signal per individual
    # variable?" -- corrects for large blocks (SOAP, fingerprints) inflating
    # their apparent importance purely through feature count.
    df_grp_avg = (df_imp.groupby("group")["mean_abs_shap"]
                        .mean()
                        .sort_values(ascending=True))

    fig1b, ax1b = plt.subplots(figsize=(11, max(5, 0.42 * len(df_grp_avg))))
    cols1b = [_pal(g) for g in df_grp_avg.index]
    bars1b = ax1b.barh(df_grp_avg.index, df_grp_avg.values,
                       color=cols1b, edgecolor="white", linewidth=0.6)
    xmax1b = df_grp_avg.max()
    for bar, val, grp in zip(bars1b, df_grp_avg.values, df_grp_avg.index):
        n_feat = df_grp_cnt[grp]
        ax1b.text(val + xmax1b * 0.01, bar.get_y() + bar.get_height() / 2,
                  f"{val:.5f}  (n={n_feat})", va="center", ha="left", fontsize=7.5)
    ax1b.set_xlabel(
        "Mean |SHAP value| per feature  (averaged across Frank-Hall thresholds)",
        fontsize=11)
    ax1b.set_title(
        f"{pipe_label}\nSHAP Feature-Category Importance (size-normalised)"
        "  --  LVMOF Crystallinity",
        fontsize=12, fontweight="bold")
    ax1b.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig1b.savefig(f"shap_group_avg_{safe_label}.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"  Saved: shap_group_avg_{safe_label}.png")

    df_grp_avg.sort_values(ascending=False).reset_index().rename(
        columns={"mean_abs_shap": "mean_per_feature_shap"}).assign(
        n_features=lambda d: d["group"].map(df_grp_cnt)).to_csv(
        f"shap_group_avg_{safe_label}.csv", index=False)
    print(f"  Saved: shap_group_avg_{safe_label}.csv")

    # ── Plot 2 -- Top-N individual features (bar, colour = category) ──────
    top_df  = df_imp.head(top_n).iloc[::-1]
    cols2   = [_pal(g) for g in top_df["group"]]

    fig2, ax2 = plt.subplots(figsize=(11, max(6, 0.38 * top_n)))
    ax2.barh(top_df["feature"], top_df["mean_abs_shap"],
             color=cols2, edgecolor="white", linewidth=0.4)
    legend_h = [Patch(facecolor=_pal(g), label=g)
                for g in pd.unique(top_df["group"])]
    ax2.legend(handles=legend_h, loc="lower right", fontsize=7.5,
               framealpha=0.8, title="Feature Category", title_fontsize=8)
    ax2.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax2.set_title(
        f"{pipe_label}  --  Top {top_n} Individual Features\n"
        "(colour = feature category)",
        fontsize=12, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig2.savefig(f"shap_top{top_n}_{safe_label}.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"  Saved: shap_top{top_n}_{safe_label}.png")

    # ── Plot 3 -- SHAP beeswarm (signed, top-N, avg over thresholds) ──────
    top_feat_set = set(df_imp["feature"].iloc[:top_n])
    top_cols     = [i for i, nm in enumerate(feat_names)
                    if nm in top_feat_set][:top_n]
    top_feat_ord = [feat_names[i] for i in top_cols]

    try:
        shap.summary_plot(
            shap_avg[:, top_cols],
            X_model[:, top_cols],
            feature_names=top_feat_ord,
            max_display=top_n,
            show=False,
            plot_type="dot",
            color_bar_label="Feature value (normalised)",
        )
        plt.title(
            f"{pipe_label}  --  SHAP Beeswarm  (avg over ordinal thresholds)\n"
            f"Top {top_n} features  |  "
            "positive SHAP => pushes toward Crystalline",
            fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"shap_beeswarm_{safe_label}.png",
                    dpi=180, bbox_inches="tight")
        plt.show()
        print(f"  Saved: shap_beeswarm_{safe_label}.png")
    except Exception as _e:
        plt.close("all")
        print(f"  Beeswarm skipped: {_e}")

    # Console summary
    print(f"\n  Top 15 features -- {pipe_label}")
    print(f"  {'Feature':<45} {'Category':<30} {'Mean|SHAP|':>10}")
    print(f"  {'-'*88}")
    for _, row in df_imp.head(15).iterrows():
        print(f"  {str(row['feature']):<45} {str(row['group']):<30} "
              f"{row['mean_abs_shap']:>10.5f}")

    return df_imp
