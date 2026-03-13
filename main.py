"""
LVMOF-Surrogate: MOF Synthesis Prediction Pipeline
Entry point — orchestrates data loading, featurization, dimensionality reduction,
model training, and evaluation.

Usage:
    python main.py

    Or with custom data file:
    python main.py --data path/to/data.xlsx
"""

import argparse
import numpy as np

from config import COLMAP, N_CLUSTERS, RANDOM_STATE
from data_processing import load_data, build_inventory, merge_data, run_process_variable_audit
from feature_assembly import assemble_features
from dimensionality import (prepare_labels, remap_score, apply_variance_threshold,
                             build_umap_embedding, select_kmeans_groups,
                             run_mi_diagnostic, build_process_interactions,
                             assemble_cv_matrix)
from models import (scoring_ordinal, FrankHallOrdinalClassifier,
                    make_rf_pipe, make_xgb_pipe,
                    make_rf_pipe_cl_only, make_xgb_pipe_cl_only)
from pipeline import (objective_xgb, objective_rf, progress_callback,
                      objective_xgb_cl_only, objective_rf_cl_only,
                      eval_pipe)
from evaluation import (plot_roc_prc, plot_learning_curves,
                         plot_confusion_matrices, run_shap_analysis)

def main(data_path=None):
    import optuna

    # 1. Load data
    df = load_data(data_path or "data/Experiments_with_Calculated_Properties_no_linker.xlsx")
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    run_process_variable_audit(df_merged)

    # 2. Assemble features
    (X_final, df_merged, fp_cols, num_descriptors, calc,
     linker_col, mod_col, process_cols_present, X_process,
     X_linker, X_modulator, mod_eq, X_precursor_perlig, Xinventorynumeric) = assemble_features(df_merged, df_inventory)

    # 3. Prepare labels
    X_raw, y, mask = prepare_labels(df_merged, X_final)
    y = np.array([remap_score(s) for s in y])

    # 4. Dimensionality reduction
    X_vt, vt_pre = apply_variance_threshold(X_raw)
    mi_pre = run_mi_diagnostic(X_vt, y)
    Xprocnorm, interactions, _ = build_process_interactions(df_merged, mask, process_cols_present)
    X_for_umap = assemble_cv_matrix(mi_pre.transform(X_vt), Xprocnorm, interactions)
    X_2d = build_umap_embedding(X_for_umap)
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)
    groups, best_k, cv = select_kmeans_groups(X_2d, y)

    # 5. Optuna tuning — XGBoost
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda t: objective_xgb(t, X_cv, y, cv, groups),
                       n_trials=100, callbacks=[progress_callback])

    # 6. Optuna tuning — Random Forest
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(lambda t: objective_rf(t, X_cv, y, cv, groups),
                      n_trials=50, callbacks=[progress_callback])

    # 7. Build final pipelines
    from config import XGB_TUNED_KEYS
    best_xgb_params = {k: v for k, v in study_xgb.best_params.items() if k in XGB_TUNED_KEYS}
    best_rf_params  = study_rf.best_params

    pipe_rf_mi     = make_rf_pipe(best_rf_params,  with_cl=False)
    pipe_rf_cl_mi  = make_rf_pipe(best_rf_params,  with_cl=True)
    pipe_xgb_mi    = make_xgb_pipe(best_xgb_params, with_cl=False)
    pipe_xgb_cl_mi = make_xgb_pipe(best_xgb_params, with_cl=True)

    pipelines = [
        ("RF  | MI only",  pipe_rf_mi,     -1),
        ("RF  | CL + MI",  pipe_rf_cl_mi,   1),
        ("XGB | MI only",  pipe_xgb_mi,    -1),
        ("XGB | CL + MI",  pipe_xgb_cl_mi,  1),
    ]

    # 8. Evaluate
    print("\n─── FINAL COMPARISON ──────────────────────────────────")
    for name, pipe, n_jobs in pipelines:
        eval_pipe(name, pipe, X_cv, y, cv, groups, scoring_ordinal, n_jobs=n_jobs)

    # 9. Plots
    plot_roc_prc(pipelines, X_cv, y, cv, groups)
    plot_learning_curves(pipelines, X_cv, y, cv, groups, scoring_ordinal)
    plot_confusion_matrices(pipelines, X_cv, y, cv, groups)
    run_shap_analysis(
        [("XGB_MI_only", pipe_xgb_mi), ("XGB_CL_plus_MI", pipe_xgb_cl_mi)],
        X_cv, y
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    main(data_path=args.data)
