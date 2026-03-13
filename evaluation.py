"""
evaluation.py
Evaluation and visualization: ROC/PRC, learning curves, confusion matrices, SHAP.
All plot logic copied EXACTLY from the source notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.base import clone
from sklearn.model_selection import (cross_val_predict, learning_curve)
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from models import scoring_ordinal


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
            cv=cv,
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
            cv=cv,
            groups=groups,
            method="predict",
            n_jobs=n_jobs_cvpred,
            verbose=0,
        )
        oof_preds[name] = y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for ax, (name, _, _) in zip(axes, pipelines):
        cm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\nCounts")

    plt.tight_layout()
    plt.savefig("confusion_matrices_counts.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrices_counts.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for ax, (name, _, _) in zip(axes, pipelines):
        cm_norm = confusion_matrix(y, oof_preds[name], labels=CLASS_IDS, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"{name}\nNormalized by true class")

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
def run_shap_analysis(xgb_pipes, X, y) -> None:
    """
    Run SHAP analysis for XGBoost pipelines.

    Parameters
    ----------
    xgb_pipes : list of (label, pipe)  — only XGB pipelines
    X         : np.ndarray
    y         : np.ndarray
    """

    def fit_full_pipeline(pipe, X, y):
        fitted = clone(pipe)
        fitted.fit(X, y)
        return fitted

    def transform_until_before_smote(pipe_fitted, X):
        """
        Preferred for SHAP:
        transform through all steps up to but NOT including SMOTE and final model.
        This gives the actual feature space seen by the XGB model at inference time.
        """
        Xt = X
        used_steps = []
        for name, step in pipe_fitted.steps:
            if name in ("smote", "ordinal_xgb"):
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

    print("\n─── SHAP analysis for XGBoost pipelines ─────────────────────────")

    for pipe_label, pipe in xgb_pipes:
        print(f"\nFitting full pipeline for SHAP: {pipe_label}")
        fitted_pipe = fit_full_pipeline(pipe, X, y)

        # Transform through impute -> vt -> [cl] -> mi
        X_model, used_steps = transform_until_before_smote(fitted_pipe, X)
        feature_names = get_feature_names_generic(X_model.shape[1],
                                                   prefix=f"{pipe_label.lower()}_feat")

        print(f"  Feature space entering XGB: {X_model.shape}")
        print(f"  Steps used before model: {used_steps}")

        ordinal_step = fitted_pipe.named_steps["ordinal_xgb"]
        classifiers = ordinal_step.classifiers_   # dict: threshold -> fitted binary estimator

        threshold_importances = []

        for threshold, xgb_model in classifiers.items():
            print(f"  Explaining Frank-Hall binary model for threshold k={threshold}")

            explainer = shap.TreeExplainer(xgb_model)
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

            csv_name = f"shap_importance_{pipe_label}_threshold_{threshold}.csv"
            df_imp.to_csv(csv_name, index=False)
            print(f"Saved: {csv_name}")

            top25 = df_imp.head(25)
            plot_name = f"shap_bar_{pipe_label}_threshold_{threshold}.png"
            save_barplot(
                top25,
                title=f"{pipe_label} — Mean |SHAP| (threshold k={threshold})",
                filename=plot_name,
                color="tab:blue" if pipe_label == "XGB_MI_only" else "tab:orange",
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
                beeswarm_name = f"shap_beeswarm_{pipe_label}_threshold_{threshold}.png"
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

        agg_csv = f"shap_importance_{pipe_label}_aggregated.csv"
        df_agg.to_csv(agg_csv, index=False)
        print(f"Saved: {agg_csv}")

        top25_agg = df_agg.head(25)
        agg_plot = f"shap_bar_{pipe_label}_aggregated.png"
        save_barplot(
            top25_agg.rename(columns={"mean_abs_shap_avg_over_thresholds": "mean_abs_shap"}),
            title=f"{pipe_label} — Aggregated Mean |SHAP| Across Frank-Hall Thresholds",
            filename=agg_plot,
            color="tab:green" if pipe_label == "XGB_MI_only" else "tab:red",
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

    plt.figure(figsize=(10, max(6, 0.35 * len(pivot_df))))
    x = np.arange(len(pivot_df))
    width = 0.38

    cols = list(pivot_df.columns)
    plt.barh(x - width/2, pivot_df[cols[0]], height=width, label=cols[0], color="tab:blue")
    if len(cols) > 1:
        plt.barh(x + width/2, pivot_df[cols[1]], height=width, label=cols[1], color="tab:orange")

    plt.yticks(x, pivot_df.index)
    plt.xlabel("Mean |SHAP| averaged over ordinal thresholds")
    plt.title("Top SHAP Features: XGB MI vs XGB CL+MI")
    plt.legend()
    plt.tight_layout()
    plt.savefig("shap_xgb_pipeline_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved: shap_xgb_pipeline_comparison.png")

    pivot_df.reset_index().to_csv("shap_xgb_pipeline_comparison.csv", index=False)
    print("Saved: shap_xgb_pipeline_comparison.csv")
