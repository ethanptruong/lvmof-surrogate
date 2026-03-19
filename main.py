"""
LVMOF-Surrogate: MOF Synthesis Prediction Pipeline
Entry point — orchestrates data loading, featurization, dimensionality reduction,
model training, evaluation, and Bayesian Optimization.

Usage:
    python main.py                          # classification pipeline (with Optuna tuning)
    python main.py --skip-tuning            # re-evaluate with existing hyperparams, no Optuna
    python main.py --bo --bo-mode simulate  # BO simulation
    python main.py --bo --bo-mode recommend --bo-precursor SMILES --bo-linker SMILES
"""

import argparse
import os
import joblib
import numpy as np
import optuna

import pandas as pd

from config import (COLMAP, N_CLUSTERS, RANDOM_STATE, XGB_TUNED_KEYS,
                    BO_N_ITERATIONS, BO_BATCH_SIZE, BO_INIT_FRACTION,
                    BO_BORE_GAMMA, BO_CHECKPOINT_DIR, BO_DEFAULT_SURROGATE,
                    BO_DEFAULT_ACQUISITION, BO_CONTROLLABLE_PARAMS,
                    TARGET_METALS, METAL_BLOCK_DIM, COLIGAND_BLOCK_DIM,
                    COMPLEX_BLOCK_DIM, TOTAL_VOLUME_ML,
                    BO_BORE_ADAPTIVE_GAMMA, BO_SSL_ALPHA, BO_SSL_N_PSEUDO)
from data_processing import load_data, build_inventory, merge_data, run_process_variable_audit, fix_missingness
from feature_assembly import (assemble_features, build_feature_catalog,
                               build_discrete_mask,
                               build_chemberta_block, build_g14_features,
                               build_ttp_features, build_linker_extra_features,
                               build_halide_block, build_drfp_block,
                               build_soap_block, build_mordred_rac_features,
                               build_precursor_full_block,
                               build_physicochem_features, build_tep_features,
                               build_steric_features)
from featurization import get_metal_descriptors, lookup_metal_descriptors
from dimensionality import (prepare_labels, remap_score, apply_variance_threshold,
                             build_umap_embedding, select_kmeans_groups,
                             run_mi_diagnostic, plot_mi_cliff,
                             build_process_interactions,
                             assemble_cv_matrix, remove_correlated_features,
                             RepeatedStratifiedGroupKFold)
from models import (scoring_ordinal, FrankHallOrdinalClassifier,
                    make_rf_pipe, make_xgb_pipe,
                    make_rf_pipe_cl_only, make_xgb_pipe_cl_only)
from pipeline import (objective_xgb, objective_rf, progress_callback,
                      objective_xgb_cl_mi, objective_rf_cl_mi,
                      objective_xgb_cl_only, objective_rf_cl_only,
                      eval_pipe)
from evaluation import (plot_roc_prc, plot_learning_curves,
                         plot_confusion_matrices, run_shap_featurized)
import random
import torch

# ── Global reproducibility seed ───────────────────────────────────────────────
SEED = RANDOM_STATE
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
DATA_CKPT      = os.path.join(CHECKPOINT_DIR, "data.pkl")
PARAMS_CKPT    = os.path.join(CHECKPOINT_DIR, "best_params.pkl")
OPTUNA_DB      = f"sqlite:///{CHECKPOINT_DIR}/optuna.db"


def _load(path):
    if os.path.exists(path):
        print(f"[checkpoint] loading {path}")
        return joblib.load(path)
    return None


def _featurize_data(data_path=None):
    """Run the full featurization pipeline from the Excel file.

    Returns (X_cv, y, groups, cv_tune, cv_eval, X_names, X_groups).
    Always re-featurizes from the source file (no data.pkl cache).
    """
    df = load_data(data_path or "data/Experiments_with_Calculated_Properties_no_linker.xlsx")
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    df_merged = fix_missingness(df_merged)
    run_process_variable_audit(df_merged)

    (X_final, df_merged, fp_cols, num_descriptors, calc,
     linker_col, mod_col, process_cols_present, X_process,
     X_linker, X_modulator, mod_eq, X_precursor_perlig,
     Xinventorynumeric) = assemble_features(df_merged, df_inventory)

    X_raw, y, mask = prepare_labels(df_merged, X_final)
    y = np.array([remap_score(s) for s in y])

    X_vt, vt_pre = apply_variance_threshold(X_raw)
    # Diagnostic MI is re-run below with the corrected discrete mask;
    # run a quick placeholder here so UMAP grouping can proceed.
    mi_pre = run_mi_diagnostic(X_vt, y)
    Xprocnorm, interactions, _ = build_process_interactions(df_merged, mask, process_cols_present)
    X_for_umap = assemble_cv_matrix(mi_pre.transform(X_vt), Xprocnorm, interactions)
    X_2d = build_umap_embedding(X_for_umap)
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)
    groups, best_k, cv_tune, cv_eval = select_kmeans_groups(X_2d, y)

    # ── Build feature name catalog (mirrors run_shap.py) ──────────────────────
    print("── Building feature name catalog ──")
    X_modulator_rac_aug, _, X_precursor_perlig_rac = \
        build_mordred_rac_features(df_merged, fp_cols, num_descriptors, calc)

    metal_ohe_df = pd.get_dummies(
        df_merged['metal_atom'].fillna('Unknown'), prefix='metal_is'
    )
    for sym in TARGET_METALS:
        col = f'metal_is_{sym}'
        if col not in metal_ohe_df.columns:
            metal_ohe_df[col] = 0
    ohe_cols = sorted([c for c in metal_ohe_df.columns if c.startswith('metal_is_')])
    metal_ohe_df = metal_ohe_df[ohe_cols]

    metal_descriptor_cache = {sym: get_metal_descriptors(sym) for sym in TARGET_METALS}
    zero_descriptor = get_metal_descriptors("XYZ")
    metal_descriptor_names = list(next(iter(metal_descriptor_cache.values())).keys())
    metal_descriptor_rows = df_merged['metal_atom'].apply(
        lambda s: lookup_metal_descriptors(s, metal_descriptor_cache, zero_descriptor)
    )
    X_metal = np.array([
        [row[name] for name in metal_descriptor_names]
        for row in metal_descriptor_rows
    ], dtype=float)
    bad = ~np.isfinite(X_metal)
    if bad.any():
        X_metal[bad] = 0.0
    X_metal_block = np.hstack([X_metal, metal_ohe_df.values.astype(float)])

    Xprecursor_full = build_precursor_full_block(df_merged)
    X_linker_phys10, X_modulator_phys10 = build_physicochem_features(
        df_merged, linker_col, mod_col
    )
    X_modulator_tep, X_linker_tep, X_precursor_perlig_tep = \
        build_tep_features(df_merged, linker_col, mod_col, fp_cols)
    df_merged, X_precursor_perlig_steric = build_steric_features(df_merged, fp_cols)
    X_chemberta_block, chemberta_names = build_chemberta_block(
        df_merged, linker_col, mod_col
    )
    X_g14_block, g14_names = build_g14_features(df_merged, linker_col, mod_col)
    Xlinker_ttp, _ = build_ttp_features(df_merged, linker_col)
    X_linker_extra = build_linker_extra_features(df_merged, linker_col)
    Xhalide_full = build_halide_block(df_merged, df_inventory)
    X_drfp, _ = build_drfp_block(df_merged)
    X_soap_precursor, X_soap_linker, soap_names = build_soap_block(
        df_merged, linker_col
    )

    vt_mask = vt_pre.get_support()
    X_names, X_groups = build_feature_catalog(
        X_final=X_final,
        X_linker=X_linker,
        X_modulator=X_modulator,
        mod_eq=mod_eq,
        X_precursor_perlig=X_precursor_perlig,
        Xinventorynumeric=Xinventorynumeric,
        X_process=X_process,
        fp_cols=fp_cols,
        num_descriptors=num_descriptors,
        ohe_cols=ohe_cols,
        process_cols_present=process_cols_present,
        n_clusters=N_CLUSTERS,
        X_modulator_rac_aug=X_modulator_rac_aug,
        X_metal_block=X_metal_block,
        Xprecursor_full=Xprecursor_full,
        X_precursor_perlig_rac=X_precursor_perlig_rac,
        X_linker_phys10=X_linker_phys10,
        X_modulator_phys10=X_modulator_phys10,
        X_modulator_tep=X_modulator_tep,
        X_linker_tep=X_linker_tep,
        X_precursor_perlig_tep=X_precursor_perlig_tep,
        X_precursor_perlig_steric=X_precursor_perlig_steric,
        X_chemberta_block=X_chemberta_block,
        chemberta_names=chemberta_names,
        X_g14_block=X_g14_block,
        g14_names=g14_names,
        Xlinker_ttp=Xlinker_ttp,
        X_linker_extra=X_linker_extra,
        Xhalide_full=Xhalide_full,
        X_drfp=X_drfp,
        X_soap_precursor=X_soap_precursor,
        X_soap_linker=X_soap_linker,
        soap_names=soap_names,
        vt_mask=vt_mask,
    )

    # ── Build discrete/continuous feature mask (Fix 1) ─────────────────────
    discrete_mask, vt_discrete_mask = build_discrete_mask(
        X_linker=X_linker,
        X_modulator=X_modulator,
        mod_eq=mod_eq,
        X_precursor_perlig=X_precursor_perlig,
        Xinventorynumeric=Xinventorynumeric,
        X_process=X_process,
        fp_cols=fp_cols,
        num_descriptors=num_descriptors,
        ohe_cols=ohe_cols,
        process_cols_present=process_cols_present,
        n_clusters=N_CLUSTERS,
        X_modulator_rac_aug=X_modulator_rac_aug,
        X_metal_block=X_metal_block,
        Xprecursor_full=Xprecursor_full,
        X_precursor_perlig_rac=X_precursor_perlig_rac,
        X_linker_phys10=X_linker_phys10,
        X_modulator_phys10=X_modulator_phys10,
        X_modulator_tep=X_modulator_tep,
        X_linker_tep=X_linker_tep,
        X_precursor_perlig_tep=X_precursor_perlig_tep,
        X_precursor_perlig_steric=X_precursor_perlig_steric,
        X_chemberta_block=X_chemberta_block,
        chemberta_names=chemberta_names,
        X_g14_block=X_g14_block,
        g14_names=g14_names,
        Xlinker_ttp=Xlinker_ttp,
        X_linker_extra=X_linker_extra,
        Xhalide_full=Xhalide_full,
        X_drfp=X_drfp,
        X_soap_precursor=X_soap_precursor,
        X_soap_linker=X_soap_linker,
        soap_names=soap_names,
        vt_mask=vt_mask,
    )

    # Pad/trim discrete mask to match X_cv width (same logic as names)
    if len(discrete_mask) < X_cv.shape[1]:
        extra = X_cv.shape[1] - len(discrete_mask)
        discrete_mask = np.concatenate([discrete_mask, np.zeros(extra, dtype=bool)])
    else:
        discrete_mask = discrete_mask[:X_cv.shape[1]]

    n_disc = int(discrete_mask.sum())
    n_cont = len(discrete_mask) - n_disc
    print(f"[discrete mask] {n_disc} discrete, {n_cont} continuous "
          f"of {len(discrete_mask)} total features")

    # Pad/trim to match X_cv width
    if len(X_names) < X_cv.shape[1]:
        extra = X_cv.shape[1] - len(X_names)
        X_names += [f'unknown_{i}' for i in range(extra)]
        X_groups += ['Unknown'] * extra
    else:
        X_names = X_names[:X_cv.shape[1]]
        X_groups = X_groups[:X_cv.shape[1]]

    print(f"[catalog] {len(X_names)} feature names for {X_cv.shape[1]} columns")
    return X_cv, y, groups, cv_tune, cv_eval, X_names, X_groups, discrete_mask, vt_discrete_mask


def main(data_path=None, skip_tuning=False):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Steps 1–4: data + features + CV setup ─────────────────────────────────

    ck = _load(DATA_CKPT)
    if ck is not None and "discrete_mask" in ck:
        X_cv, y, groups = ck["X_cv"], ck["y"], ck["groups"]
        cv_tune, cv_eval = ck["cv_tune"], ck["cv_eval"]
        X_names, X_groups = ck["X_names"], ck["X_groups"]
        discrete_mask    = ck["discrete_mask"]
        vt_discrete_mask = ck.get("vt_discrete_mask", discrete_mask)
    else:
        if ck is not None:
            print("\n── Checkpoint outdated (missing discrete mask), re-featurizing ──")
        else:
            print("\n── Featurizing from data file ──")
        X_cv, y, groups, cv_tune, cv_eval, X_names, X_groups, discrete_mask, \
            vt_discrete_mask = _featurize_data(data_path)

        joblib.dump({"X_cv": X_cv, "y": y, "groups": groups,
                     "cv_tune": cv_tune, "cv_eval": cv_eval,
                     "X_names": X_names, "X_groups": X_groups,
                     "discrete_mask": discrete_mask,
                     "vt_discrete_mask": vt_discrete_mask}, DATA_CKPT)
        print(f"[checkpoint] saved {DATA_CKPT}")

    # ── Wire discrete mask into models (Fix 1) ───────────────────────────────
    import models as _models_mod
    _models_mod.ORIGINAL_DISCRETE_MASK = discrete_mask
    print(f"[discrete mask] set ORIGINAL_DISCRETE_MASK: "
          f"{int(discrete_mask.sum())} discrete / "
          f"{len(discrete_mask) - int(discrete_mask.sum())} continuous")

    # ── MI cliff plot (always regenerated from cached mask — fast) ───────────
    print("── MI cliff diagnostic (discrete vs continuous) ──")
    mi_cliff_scores = run_mi_diagnostic(X_cv, y, discrete_mask=discrete_mask)
    plot_mi_cliff(mi_cliff_scores, discrete_mask=discrete_mask)

    # ── Steps 5–6: Optuna tuning (one study per pipeline variant) ─────────────
    _REQUIRED_KEYS = {
        "best_xgb_mi_params", "best_rf_mi_params",
        "best_xgb_cl_mi_params", "best_rf_cl_mi_params",
        "best_xgb_cl_only_params", "best_rf_cl_only_params",
    }
    ck_params = _load(PARAMS_CKPT)
    if ck_params is not None and _REQUIRED_KEYS.issubset(ck_params.keys()):
        best_xgb_mi_params      = ck_params["best_xgb_mi_params"]
        best_rf_mi_params       = ck_params["best_rf_mi_params"]
        best_xgb_cl_mi_params   = ck_params["best_xgb_cl_mi_params"]
        best_rf_cl_mi_params    = ck_params["best_rf_cl_mi_params"]
        best_xgb_cl_only_params = ck_params["best_xgb_cl_only_params"]
        best_rf_cl_only_params  = ck_params["best_rf_cl_only_params"]
        if skip_tuning:
            print("\n── Using existing hyperparameters (--skip-tuning) ──")
            print("   NOTE: If you changed MI_K, CL_EMB_DIM, or discrete mask, "
                  "consider re-tuning (run without --skip-tuning) and deleting "
                  "checkpoints/best_params.pkl + checkpoints/optuna.db")
    elif skip_tuning:
        raise RuntimeError(
            "No tuned hyperparameters found (checkpoints/best_params.pkl missing). "
            "Run without --skip-tuning first to tune hyperparameters."
        )
    else:
        def _run_study(study_name, objective_fn, n_trials):
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                storage=OPTUNA_DB,
                study_name=study_name,
                load_if_exists=True,
            )
            completed = len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE])
            remaining = max(0, n_trials - completed)
            if completed > 0:
                print(f"[checkpoint] {study_name}: {completed} trials already done, "
                      f"running {remaining} more")
            if remaining > 0:
                study.optimize(lambda t: objective_fn(t, X_cv, y, cv_tune, groups),
                               n_trials=remaining, callbacks=[progress_callback])
            return study

        print("\n── Tuning XGB | MI only ──────────────────────────────────────────────────")
        study_xgb_mi      = _run_study("xgb_mi_study",      objective_xgb,         n_trials=100)
        print("\n── Tuning RF  | MI only ──────────────────────────────────────────────────")
        study_rf_mi       = _run_study("rf_mi_study",        objective_rf,          n_trials=100)
        print("\n── Tuning XGB | CL + MI ──────────────────────────────────────────────────")
        study_xgb_cl_mi   = _run_study("xgb_cl_mi_study",   objective_xgb_cl_mi,   n_trials=100)
        print("\n── Tuning RF  | CL + MI ──────────────────────────────────────────────────")
        study_rf_cl_mi    = _run_study("rf_cl_mi_study",     objective_rf_cl_mi,    n_trials=100)
        print("\n── Tuning XGB | CL only ──────────────────────────────────────────────────")
        study_xgb_cl_only = _run_study("xgb_cl_only_study",  objective_xgb_cl_only, n_trials=100)
        print("\n── Tuning RF  | CL only ──────────────────────────────────────────────────")
        study_rf_cl_only  = _run_study("rf_cl_only_study",   objective_rf_cl_only,  n_trials=100)

        best_xgb_mi_params      = {k: v for k, v in study_xgb_mi.best_params.items()
                                   if k in XGB_TUNED_KEYS}
        best_rf_mi_params       = study_rf_mi.best_params
        best_xgb_cl_mi_params   = {k: v for k, v in study_xgb_cl_mi.best_params.items()
                                   if k in XGB_TUNED_KEYS}
        best_rf_cl_mi_params    = study_rf_cl_mi.best_params
        best_xgb_cl_only_params = {k: v for k, v in study_xgb_cl_only.best_params.items()
                                   if k in XGB_TUNED_KEYS}
        best_rf_cl_only_params  = study_rf_cl_only.best_params

        joblib.dump({
            "best_xgb_mi_params":      best_xgb_mi_params,
            "best_rf_mi_params":        best_rf_mi_params,
            "best_xgb_cl_mi_params":   best_xgb_cl_mi_params,
            "best_rf_cl_mi_params":     best_rf_cl_mi_params,
            "best_xgb_cl_only_params": best_xgb_cl_only_params,
            "best_rf_cl_only_params":   best_rf_cl_only_params,
        }, PARAMS_CKPT)
        print(f"[checkpoint] saved {PARAMS_CKPT}")

    pipe_rf_mi              = make_rf_pipe(best_rf_mi_params,       with_cl=False)
    pipe_rf_cl_mi           = make_rf_pipe(best_rf_cl_mi_params,    with_cl=True)
    pipe_xgb_mi             = make_xgb_pipe(best_xgb_mi_params,     with_cl=False)
    pipe_xgb_cl_mi          = make_xgb_pipe(best_xgb_cl_mi_params,  with_cl=True)
    pipe_rf_cl_only         = make_rf_pipe_cl_only(best_rf_cl_only_params)
    pipe_xgb_cl_only        = make_xgb_pipe_cl_only(best_xgb_cl_only_params)

    pipelines = [
        ("RF  | MI only",         pipe_rf_mi,       1),
        ("RF  | CL + MI",         pipe_rf_cl_mi,    1),
        ("XGB | MI only",         pipe_xgb_mi,      1),
        ("XGB | CL + MI",         pipe_xgb_cl_mi,   1),
        ("RF  | CL only (triplet)", pipe_rf_cl_only,  1),
        ("XGB | CL only (triplet)", pipe_xgb_cl_only, 1),
    ]

    # 8. Evaluate
    print("\n─── FINAL COMPARISON ──────────────────────────────────")
    for name, pipe, n_jobs in pipelines:
        eval_pipe(name, pipe, X_cv, y, cv_eval, groups, scoring_ordinal, n_jobs=n_jobs)

    # 9. Plots
    plot_roc_prc(pipelines, X_cv, y, cv_eval, groups)
    plot_learning_curves(pipelines, X_cv, y, cv_eval, groups, scoring_ordinal)
    plot_confusion_matrices(pipelines, X_cv, y, cv_eval, groups)

    # Pre-fit each pipeline once on the full dataset so SHAP can reuse the
    # fitted instance rather than cloning and re-fitting from scratch.
    from sklearn.base import clone as _clone
    _shap_targets = [
        ("XGB | MI only",           pipe_xgb_mi),
        ("XGB | CL + MI",           pipe_xgb_cl_mi),
        ("XGB | CL only (triplet)", pipe_xgb_cl_only),
        ("RF  | MI only",           pipe_rf_mi),
        ("RF  | CL + MI",           pipe_rf_cl_mi),
        ("RF  | CL only (triplet)", pipe_rf_cl_only),
    ]
    print("\n─── Pre-fitting pipelines for SHAP (full dataset) ──────────────────")
    _fitted_for_shap = {}
    for _lbl, _pipe in _shap_targets:
        print(f"  Fitting: {_lbl}")
        _fp = _clone(_pipe)
        _fp.fit(X_cv, y)
        _fitted_for_shap[_lbl] = _fp

    for _shap_label, _shap_pipe in _shap_targets:
        run_shap_featurized(_shap_label, _shap_pipe, X_cv, y, X_names, X_groups,
                            top_n=15, fitted_pipe=_fitted_for_shap[_shap_label])

# ── Bayesian Optimization ────────────────────────────────────────────────────

def _load_bo_data(data_path=None):
    """Load data from checkpoint + raw 0-9 labels for BO simulate/batch modes."""
    ck = _load(DATA_CKPT)
    if ck is not None:
        X_cv = ck["X_cv"]
        y_remapped = ck["y"]  # 3-class labels
    else:
        raise RuntimeError(
            "Data checkpoint not found. Run the classification pipeline first "
            "(python main.py) to generate checkpoints/data.pkl."
        )

    # We also need the raw 0-9 scores. Reload from source.
    from data_processing import load_data as _ld, build_inventory, merge_data, fix_missingness
    df = _ld(data_path or "data/Experiments_with_Calculated_Properties_no_linker.xlsx")
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    df_merged = fix_missingness(df_merged)

    import pandas as _pd
    y_raw_full = _pd.to_numeric(df_merged["pxrd_score"], errors="coerce").to_numpy()
    mask = np.isfinite(y_raw_full)
    y_raw = y_raw_full[mask].astype(float)

    if len(y_raw) != X_cv.shape[0]:
        raise RuntimeError(
            f"Shape mismatch: y_raw={len(y_raw)}, X_cv={X_cv.shape[0]}. "
            "Rerun the full pipeline to regenerate data.pkl."
        )

    return X_cv, y_raw, y_remapped, df_merged, mask


def _featurize_fresh(data_path=None):
    """Re-run the full featurization pipeline from the Excel file.

    Used by recommend mode so that newly added experiments are included.
    Returns (X_cv, y_raw, df_merged, mask, process_cols_present).
    """
    import pandas as _pd
    from data_processing import load_data as _ld, build_inventory, merge_data, fix_missingness
    from dimensionality import (prepare_labels, remap_score, apply_variance_threshold,
                                run_mi_diagnostic, build_process_interactions,
                                assemble_cv_matrix)

    print("[recommend] Featurizing from data file (includes any new experiments)...")
    df = _ld(data_path or "data/Experiments_with_Calculated_Properties_no_linker.xlsx")
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    df_merged = fix_missingness(df_merged)

    (X_final, df_merged, fp_cols, num_descriptors, calc,
     linker_col, mod_col, process_cols_present, X_process,
     X_linker, X_modulator, mod_eq, X_precursor_perlig,
     Xinventorynumeric) = assemble_features(df_merged, df_inventory)

    X_raw, y_int, mask = prepare_labels(df_merged, X_final)
    y_raw = y_int.astype(float)  # raw 0-9 scores (before remap)

    X_vt, vt_pre = apply_variance_threshold(X_raw)
    y_remapped = np.array([remap_score(s) for s in y_int])
    Xprocnorm, interactions, _ = build_process_interactions(df_merged, mask, process_cols_present)
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)

    print(f"[recommend] {X_cv.shape[0]} experiments, {X_cv.shape[1]} features, "
          f"best score in data: {y_raw.max():.0f}")

    return X_cv, y_raw, y_remapped, df_merged, mask, process_cols_present


def _resolve_surrogate(surrogate_name, params, ranking_target=False):
    """Create a RegressionSurrogate from surrogate name + hyperparams.

    Maps each --bo-surrogate choice to the matching Optuna-tuned hyperparams:
      rf_mi       → best_rf_mi_params
      xgb_mi      → best_xgb_mi_params
      rf_cl_mi    → best_rf_cl_mi_params
      xgb_cl_mi   → best_xgb_cl_mi_params
      rf_cl_only  → best_rf_cl_only_params
      xgb_cl_only → best_xgb_cl_only_params
    """
    from models import (make_rf_regressor_pipe, make_xgb_regressor_pipe,
                        make_rf_regressor_pipe_cl_only, make_xgb_regressor_pipe_cl_only)
    from bo_core import (RegressionSurrogate, XGBoostBootstrapEnsemble,
                         RankingRegressionSurrogate)

    _RF_FALLBACK = {"n_estimators": 300, "max_depth": 10,
                    "min_samples_split": 5, "min_samples_leaf": 3, "max_features": "sqrt"}
    _XGB_FALLBACK = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                     "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 5.0}

    # Pick the right hyperparams checkpoint key for each variant
    param_key_map = {
        "rf_mi":       "best_rf_mi_params",
        "xgb_mi":      "best_xgb_mi_params",
        "rf_cl_mi":    "best_rf_cl_mi_params",
        "xgb_cl_mi":   "best_xgb_cl_mi_params",
        "rf_cl_only":  "best_rf_cl_only_params",
        "xgb_cl_only": "best_xgb_cl_only_params",
    }
    param_key = param_key_map.get(surrogate_name)
    is_rf = surrogate_name.startswith("rf")
    fallback = _RF_FALLBACK if is_rf else _XGB_FALLBACK
    hp = params.get(param_key, fallback) if param_key else fallback

    cl_only = surrogate_name.endswith("cl_only")
    with_cl = "cl_mi" in surrogate_name

    SurrClass = RankingRegressionSurrogate if ranking_target else RegressionSurrogate
    if is_rf:
        if cl_only:
            pipe = make_rf_regressor_pipe_cl_only(hp)
        else:
            pipe = make_rf_regressor_pipe(hp, with_cl=with_cl)
        return SurrClass(pipe, model_type="rf")
    else:
        if cl_only:
            pipe = make_xgb_regressor_pipe_cl_only(hp)
        else:
            pipe = make_xgb_regressor_pipe(hp, with_cl=with_cl)
        surr = SurrClass(pipe, model_type="xgb")
        surr.bootstrap_ensemble = XGBoostBootstrapEnsemble(hp)
        return surr


def run_bo(args):
    """Run Bayesian Optimization in the specified mode."""
    from bo_core import BOLoop, BOCheckpointer, RegressionSurrogate
    from bo_metrics import (SimulationMetrics, plot_convergence, plot_average_score,
                            plot_topk_curves, plot_af_ef_comparison,
                            save_simulation_results, save_full_history,
                            plot_simple_regret, compute_surrogate_calibration,
                            plot_calibration)

    from config import BO_OPTIONAL_PARAMS

    print("\n" + "=" * 70)
    print("  BAYESIAN OPTIMIZATION")
    print("=" * 70)

    controllable = list(BO_CONTROLLABLE_PARAMS.keys())
    if args.bo_include_mlr:
        controllable += list(BO_OPTIONAL_PARAMS.keys())
    print(f"  Controllable params: {controllable}")

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)

    ck_params = _load(PARAMS_CKPT) or {}
    surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                   ranking_target=args.bo_ranking_target)

    # Optional classifier pipeline for PI-ordinal baseline
    classifier_pipeline = None
    if args.bo_acquisition == "pi_ordinal":
        from models import make_rf_pipe
        rf_params = ck_params.get("best_rf_mi_params", {"n_estimators": 300, "max_depth": 10,
                                   "min_samples_split": 5, "min_samples_leaf": 3, "max_features": "sqrt"})
        classifier_pipeline = make_rf_pipe(rf_params, with_cl=False)
        classifier_pipeline.fit(X_cv, y_remapped)

    bo = BOLoop(
        surrogate=surrogate,
        acquisition_name=args.bo_acquisition,
        batch_strategy=args.bo_batch_strategy,
        batch_size=args.bo_batch_size,
        n_iterations=args.bo_iterations,
        classifier_pipeline=classifier_pipeline,
        random_state=RANDOM_STATE,
        bore_adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
    )

    checkpointer = BOCheckpointer()

    if args.bo_mode == "simulate":
        print(f"\n── Simulation: {args.bo_acquisition} | {args.bo_surrogate} "
              f"| {args.bo_iterations} iters ──")
        history = bo.run_simulation(X_cv, y_raw, init_fraction=BO_INIT_FRACTION)
        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)

        print(f"\n── Results ──")
        print(f"  AF:           {summary['AF']:.2f}")
        print(f"  EF:           {summary['EF']:.2f}")
        print(f"  Top-5% found: {summary['Top_percent_final']*100:.1f}%")
        print(f"  Best score:   {summary['best_score_final']:.0f}")
        print(f"  Simple regret (final): {summary['simple_regret_final']:.2f}")

        label = f"{args.bo_acquisition}_{args.bo_surrogate}"
        checkpointer.save(f"sim_{label}", history)
        save_full_history(history, label)
        plot_convergence([history], [label], y_raw)
        plot_average_score([history], [label],
                           save_path=f"docs/bo_avg_score_{label}.png")
        plot_topk_curves([history], [label], y_raw)
        plot_simple_regret([history], [label], y_raw,
                           save_path=f"docs/bo_simple_regret_{label}.png")

        # ── Surrogate calibration check ──────────────────────────────────────
        # Refit surrogate on the initial training split, then evaluate calibration
        # on the held-out pool.  This is a clean snapshot of how well sigma
        # estimates the actual prediction error before any BO queries are made.
        print(f"\n── Surrogate Calibration ({args.bo_surrogate}) ──")
        init_idx = np.array(history["init_indices"])
        pool_idx = np.array(history["pool_indices"])
        surrogate.fit(X_cv[init_idx], y_raw[init_idx])
        cal = compute_surrogate_calibration(surrogate, X_cv[pool_idx], y_raw[pool_idx])

        if "error" in cal:
            print(f"  WARNING: {cal['error']}")
        else:
            print(f"  n_test={cal['n_valid']}, n_zero_sigma={cal['n_zero_sigma']}")
            print(f"  z-score mean: {cal['mean_z']:+.3f}  (ideal:  0.000)")
            print(f"  z-score std:  {cal['std_z']:.3f}   (ideal:  1.000)")
            print(f"  Coverage within 1σ: {cal['fraction_within_1sigma']*100:.1f}%  "
                  f"(expected: 68.3%)")
            print(f"  Coverage within 2σ: {cal['fraction_within_2sigma']*100:.1f}%  "
                  f"(expected: 95.4%)")
            print(f"  Mean calibration error: {cal['calibration_error']:.4f}  "
                  f"(0 = perfect)")
            if cal["calibration_error"] > 0.10:
                print("  NOTE: calibration error > 0.10 — sigma estimates are "
                      "unreliable. EI/LCB acquisition scores may be misleading.")
        plot_calibration(cal, surrogate_name=args.bo_surrogate,
                         save_path=f"docs/bo_calibration_{label}.png")

    elif args.bo_mode == "batch":
        print(f"\n── Batch simulation: {args.bo_acquisition} | {args.bo_surrogate} "
              f"| batch_size={args.bo_batch_size} | {args.bo_batch_strategy} ──")
        history = bo.run_batch(X_cv, y_raw, init_fraction=BO_INIT_FRACTION)
        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)

        print(f"\n── Results ──")
        print(f"  AF:          {summary['AF']:.2f}")
        print(f"  EF:          {summary['EF']:.2f}")
        print(f"  Top-5% found: {summary['Top_percent_final']*100:.1f}%")
        print(f"  Best score:   {summary['best_score_final']:.0f}")

        label = f"batch_{args.bo_acquisition}_{args.bo_surrogate}_{args.bo_batch_strategy}"
        checkpointer.save(f"batch_{label}", history)
        save_full_history(history, label)

    elif args.bo_mode == "recommend":
        _run_recommend(args)
        return

    else:
        raise ValueError(f"Unknown --bo-mode: {args.bo_mode}")


def _run_recommend(args):
    """Persistent recommendation loop.

    Each invocation:
      1. Re-featurizes the (possibly updated) data file
      2. Loads BO history from checkpoint
      3. Fits surrogate on the full current dataset
      4. Generates candidates (optionally within a trust region)
      5. Scores with acquisition function
      6. Outputs top recommendations
      7. Saves updated BO state

    Trust region mode (--bo-precursor + --bo-linker provided)
    ---------------------------------------------------------
    When target chemistry SMILES are supplied, a NeighborhoodTemplateSelector
    finds structurally similar past experiments (by linker AND precursor
    Tanimoto similarity) and computes a similarity×score-weighted centroid of
    their process conditions.  A TuRBO-style trust region is initialised at
    that centroid and adapts across iterations:
      - Expands after 3 consecutive improvements (new best pxrd_score found)
      - Shrinks after 3 consecutive failures
    This avoids cold-starting from an unrelated chemistry while still
    exploring the full space if no similar experiments exist.

    The user's workflow (chemistry-targeted):
      python main.py --bo --bo-mode recommend \\
          --bo-precursor <SMILES> --bo-linker <SMILES>
      → Synthesize top candidate → add result to Excel → run again

    Global workflow (no chemistry target, full-space search):
      python main.py --bo --bo-mode recommend
    """
    from bo_core import (BOLoop, BOCheckpointer, SearchSpace,
                         CandidateFeaturizer, _compute_acquisition,
                         NeighborhoodTemplateSelector, TrustRegion,
                         FeasibilityScorer)
    from config import BO_OPTIONAL_PARAMS

    print("\n" + "=" * 70)
    print("  BO RECOMMENDATION (persistent loop)")
    print("=" * 70)

    # 1. Re-featurize from the current data file
    X_cv, y_raw, y_remapped, df_merged, mask, process_cols_present = (
        _featurize_fresh(args.data)
    )

    # 2. Load BO state
    checkpointer = BOCheckpointer()
    state = checkpointer.load("recommend_state") or {
        "iteration": 0,
        "surrogate_name": args.bo_surrogate,
        "acquisition_name": args.bo_acquisition,
        "recommendations": [],
        "n_data_at_each_iter": [],
        "trust_region": None,   # persisted TrustRegion state dict
    }

    iteration = state["iteration"]
    f_best    = float(y_raw.max())

    print(f"  BO iteration:    {iteration} (previous recommendations: {iteration})")
    print(f"  Current dataset: {len(y_raw)} experiments")
    print(f"  Best score:      {f_best:.0f}")
    print(f"  Surrogate:       {args.bo_surrogate}")
    print(f"  Acquisition:     {args.bo_acquisition}")

    if state["n_data_at_each_iter"]:
        prev_n = state["n_data_at_each_iter"][-1]
        new_n  = len(y_raw) - prev_n
        if new_n > 0:
            print(f"  New experiments since last run: {new_n}")
        else:
            print("  No new experiments added since last run.")

    # 3. Build surrogate and fit on full dataset
    ck_params = _load(PARAMS_CKPT) or {}
    surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                   ranking_target=args.bo_ranking_target)
    surrogate.fit(X_cv, y_raw)

    # 4. Build search space
    extra_params = BO_OPTIONAL_PARAMS if args.bo_include_mlr else None
    controllable = list(BO_CONTROLLABLE_PARAMS.keys())
    if args.bo_include_mlr:
        controllable += list(BO_OPTIONAL_PARAMS.keys())
    print(f"  Controllable params: {controllable}")

    _ck_data = _load(DATA_CKPT) or {}
    X_names  = _ck_data.get("X_names",  [f"f_{i}" for i in range(X_cv.shape[1])])
    X_groups = _ck_data.get("X_groups", ["Unknown"] * X_cv.shape[1])

    from cosmo_features import CosmoMixer
    cosmo_mixer = CosmoMixer(
        index_path=os.path.join("data", "VT-2005_Sigma_Profile_Database_Index_v2.xlsx"),
        cosmo_folder=os.path.join("data", "solvent_cosmo"),
    )

    search_space = SearchSpace(
        train_df=df_merged[mask], solvent_mixer=cosmo_mixer, extra_params=extra_params
    )

    # ── Trust region logic ────────────────────────────────────────────────────
    using_trust_region = (args.bo_precursor is not None and
                          args.bo_linker    is not None)
    trust_region = None
    override_bounds = None
    ref_idx = None   # chemistry template — nearest neighbor in dataset

    if using_trust_region:
        print(f"\n  [TrustRegion] Target precursor: {args.bo_precursor[:40]}...")
        print(f"  [TrustRegion] Target linker:    {args.bo_linker[:40]}...")

        # Always run the two-stage selector so we get ref_idx for the template.
        selector = NeighborhoodTemplateSelector(
            df_train=df_merged[mask],
            X_cv=X_cv,
            X_groups=X_groups,
            linker_col=COLMAP["linker1"],
            precursor_col=COLMAP["precursor"],
        )
        center, spread, neighbors, ref_idx = selector.select(
            target_linker_smiles=args.bo_linker,
            target_precursor_smiles=args.bo_precursor,
            search_bounds=search_space.bounds,
        )

        if state.get("trust_region") is not None and iteration > 0:
            # Restore existing trust region and update with latest f_best
            trust_region = TrustRegion.from_dict(state["trust_region"])
            trust_region.update(f_best)
            print(f"  [TrustRegion] Restored | length={trust_region.length:.3f}")

            # Recenter on the best experiment currently in the dataset
            best_idx = int(np.argmax(y_raw))
            best_row = df_merged[mask].iloc[best_idx]
            new_center = {}
            for param in search_space.bounds:
                col = NeighborhoodTemplateSelector._PARAM_TO_COL.get(param, param)
                val = pd.to_numeric(best_row.get(col, np.nan), errors="coerce")
                lo, hi = search_space.bounds[param]
                if np.isfinite(val):
                    new_center[param] = float(np.clip(val, lo, hi))
            trust_region.recenter(new_center)
            print(f"  [TrustRegion] Recentered on best observed experiment.")

        else:
            # First iteration — use selector output to initialise trust region
            if center is None:
                print("  [TrustRegion] No similar neighbors found. "
                      "Using global search space.")
                using_trust_region = False
            else:
                spread_lengths = []
                for param, (lo, hi) in search_space.bounds.items():
                    full_range = hi - lo
                    if full_range > 0:
                        spread_lengths.append(
                            min(1.0, 2.0 * spread.get(param, full_range / 4) / full_range)
                        )
                init_length = float(np.clip(np.mean(spread_lengths)
                                            if spread_lengths else 0.8,
                                            0.3, 0.8))
                # Compute per-parameter scales from neighbour spread.
                # Tightly-clustered params → scale < 1 (narrow search).
                # Widely-spread params    → scale > 1 (broader search).
                param_scales = {}
                for param, (lo, hi) in search_space.bounds.items():
                    full_range = hi - lo
                    if full_range > 0 and spread and param in spread:
                        normalized_spread = spread[param] / max(full_range / 4.0, 1e-9)
                        param_scales[param] = float(np.clip(normalized_spread, 0.5, 2.0))
                    else:
                        param_scales[param] = 1.0

                trust_region = TrustRegion(
                    center=center,
                    full_bounds=search_space.bounds,
                    length=init_length,
                    param_scales=param_scales,
                )
                print(f"  [TrustRegion] Initialised | length={init_length:.3f} | "
                      f"param_scales={{{', '.join(f'{p}:{s:.2f}' for p,s in param_scales.items())}}}")

        if trust_region is not None:
            override_bounds = trust_region.get_bounds()
            print("  [TrustRegion] Search bounds:")
            for p, (lo, hi) in override_bounds.items():
                print(f"    {p}: [{lo:.3g}, {hi:.3g}]")

    # ── Chemistry template for featurizer ────────────────────────────────────
    # Use the nearest-neighbor row (ref_idx) from the two-stage selector as the
    # chemistry template.  This gives the surrogate the correct molecular context
    # (metal / linker / modulator features) for the target chemistry, rather than
    # defaulting to the dataset-median which may belong to a different metal family.
    if using_trust_region and ref_idx is not None:
        template_row = X_cv[ref_idx]
        print(f"  [Template] Using nearest neighbor idx={ref_idx} as chemistry template.")
    else:
        template_row = np.nanmedian(X_cv, axis=0)

    # 5. Generate candidates (within trust region if active)
    candidates = search_space.generate_lhs_candidates(
        seed=RANDOM_STATE + iteration,
        override_bounds=override_bounds,
    )

    featurizer = CandidateFeaturizer(
        template_row=template_row,
        X_names=X_names,
        X_cv=X_cv,
        process_cols_present=process_cols_present,
        cosmo_mixer=cosmo_mixer,
        total_volume_ml=TOTAL_VOLUME_ML,
    )
    X_candidates = featurizer.featurize(candidates)

    # 6. Score candidates
    mu, sigma = surrogate.predict(X_candidates)

    acq_kwargs = {
        "f_best": f_best,
        "gamma": BO_BORE_GAMMA,
        "random_state": RANDOM_STATE + iteration,
        "bore_adaptive_gamma": BO_BORE_ADAPTIVE_GAMMA,
    }
    acq_vals = _compute_acquisition(
        args.bo_acquisition, surrogate,
        X_cv, y_raw, X_candidates, **acq_kwargs
    )

    # Apply synthesis feasibility prior (if requested)
    if args.bo_feasibility:
        feas_scorer = FeasibilityScorer()
        feas_scores = feas_scorer.score(candidates)
        acq_vals    = acq_vals * feas_scores
        n_penalized = int((feas_scores < 0.99).sum())
        if n_penalized > 0:
            print(f"  [Feasibility] {n_penalized}/{len(candidates)} candidates "
                  f"penalized (T > solvent BP - margin).")

    results = candidates.copy()
    results["pxrd_predicted"] = mu
    results["uncertainty"]     = sigma
    results["acquisition_value"] = acq_vals
    results = results.sort_values("acquisition_value", ascending=False)

    # 7. Output
    os.makedirs("docs", exist_ok=True)
    out_path = f"docs/bo_recommendations_iter{iteration}.csv"
    results.head(100).to_csv(out_path, index=False)

    print(f"\n── Iteration {iteration} — Top recommendations ──")
    top_cols = ["equivalents", "temperature_k", "metal_over_linker_ratio",
                "total_conc", "solvent_1", "pxrd_predicted", "uncertainty",
                "acquisition_value"]
    display_cols = [c for c in top_cols if c in results.columns]
    print(results[display_cols].head(10).to_string(index=False))
    print(f"\n  Full results saved → {out_path}")

    # 8. Save updated state
    top_rec = results.head(args.bo_batch_size)[display_cols].to_dict("records")
    state["recommendations"].append({
        "iteration":     iteration,
        "n_data":        len(y_raw),
        "f_best":        f_best,
        "top_candidates": top_rec,
        "trust_region_active": using_trust_region,
    })
    state["n_data_at_each_iter"].append(len(y_raw))
    state["iteration"]        = iteration + 1
    state["surrogate_name"]   = args.bo_surrogate
    state["acquisition_name"] = args.bo_acquisition
    state["trust_region"]     = (trust_region.to_dict()
                                 if trust_region is not None else None)
    checkpointer.save("recommend_state", state)

    tr_flag = " --bo-precursor <SMILES> --bo-linker <SMILES>" if using_trust_region else ""
    print(f"\n  State saved. Next: synthesize top candidates, add results to data file,")
    print(f"  then run again:  python main.py --bo --bo-mode recommend "
          f"--bo-surrogate {args.bo_surrogate}{tr_flag}")


def run_bo_ablation(args):
    """Structured ablation study.

    Design rationale:
      - BORE, random, pi_ordinal do not use the regression surrogate for
        acquisition scoring, so varying the surrogate with these methods
        produces identical results.  They are run once per seed with a
        fixed surrogate (args.bo_surrogate).
      - EI, LCB, Thompson directly consume surrogate (mu, sigma), so they
        are crossed with all six surrogates × three seeds.
      - Calibration is evaluated once per surrogate using the seed=42 init
        split from the EI runs (EI uses sigma, so its init split is the
        most relevant reference).

    Total runs: 3 agnostic × 3 seeds  +  3 sensitive × 6 surrogates × 3 seeds
               + 2 batch strategies  =  9 + 54 + 2 = 65 runs
    (vs 108 in the old design, which wasted 50% on redundant BORE/random combos)
    """
    from bo_core import BOLoop
    from bo_metrics import (SimulationMetrics, plot_convergence, plot_average_score,
                            plot_topk_curves, plot_af_ef_comparison,
                            plot_seed_aggregated_comparison, plot_sensitive_heatmap,
                            plot_seed_averaged_convergence,
                            save_simulation_results, plot_batch_comparison,
                            save_full_history, plot_simple_regret,
                            compute_surrogate_calibration, plot_calibration)

    print("\n" + "=" * 70)
    print("  BO ABLATION STUDY")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}

    # Acquisitions that do not use regression surrogate mu/sigma for scoring.
    # "lfbo" and "lfbo_ssl" are BORE variants: same classifier approach, but
    # lfbo recovers EI (not PI) and lfbo_ssl adds semi-supervised regularisation.
    SURROGATE_AGNOSTIC = ["bore", "lfbo", "lfbo_ssl", "random", "pi_ordinal"]
    # Acquisitions that consume surrogate mu/sigma — cross with all surrogates
    SURROGATE_SENSITIVE = ["ei", "lcb", "thompson"]

    surrogates = ["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                  "rf_cl_only", "xgb_cl_only"]
    batch_strategies = ["constant_liar", "kriging_believer"]
    seeds = [42, 123, 456]

    all_histories = []
    all_labels = []
    all_summaries = []

    # ── 1. Surrogate-agnostic acquisitions ────────────────────────────────────
    # pi_ordinal uses a separate classifier pipeline, fit once here.
    pi_classifier = None
    from models import make_rf_pipe as _make_rf_pipe
    rf_params = ck_params.get("best_rf_mi_params", {
        "n_estimators": 300, "max_depth": 10,
        "min_samples_split": 5, "min_samples_leaf": 3, "max_features": "sqrt"})
    pi_classifier = _make_rf_pipe(rf_params, with_cl=False)
    pi_classifier.fit(X_cv, y_remapped)
    print(f"\n── Surrogate-agnostic acquisitions (fixed surrogate: {args.bo_surrogate}) ──")

    for acq in SURROGATE_AGNOSTIC:
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
        clf = pi_classifier if acq == "pi_ordinal" else None
        for seed in seeds:
            label = f"{acq}|seed={seed}"
            print(f"\n── {label} ──")
            bo = BOLoop(
                surrogate=surrogate,
                acquisition_name=acq,
                n_iterations=args.bo_iterations,
                classifier_pipeline=clf,
                random_state=seed,
                bore_adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
            )
            history = bo.run_simulation(X_cv, y_raw)
            metrics = SimulationMetrics(y_raw)
            summary = metrics.summary(history)
            all_histories.append(history)
            all_labels.append(label)
            all_summaries.append((label, summary))
            print(f"  AF={summary['AF']:.2f}  EF={summary['EF']:.2f}  "
                  f"Top-5%={summary['Top_percent_final']*100:.1f}%")

    # ── 2. Surrogate-sensitive acquisitions ───────────────────────────────────
    print(f"\n── Surrogate-sensitive acquisitions (EI / LCB / Thompson) ──")

    # Track one history per surrogate (seed=42, ei) for calibration later
    calibration_histories = {}

    for acq in SURROGATE_SENSITIVE:
        for surr_name in surrogates:
            surrogate = _resolve_surrogate(surr_name, ck_params)
            for seed in seeds:
                label = f"{acq}|{surr_name}|seed={seed}"
                print(f"\n── {label} ──")
                bo = BOLoop(
                    surrogate=surrogate,
                    acquisition_name=acq,
                    n_iterations=args.bo_iterations,
                    random_state=seed,
                    bore_adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
                )
                history = bo.run_simulation(X_cv, y_raw)
                metrics = SimulationMetrics(y_raw)
                summary = metrics.summary(history)
                all_histories.append(history)
                all_labels.append(label)
                all_summaries.append((label, summary))
                print(f"  AF={summary['AF']:.2f}  EF={summary['EF']:.2f}  "
                      f"Top-5%={summary['Top_percent_final']*100:.1f}%")

                # Keep seed=42 EI run per surrogate for calibration reference
                if acq == "ei" and seed == 42:
                    calibration_histories[surr_name] = history

    # ── 3. Batch strategy comparison ──────────────────────────────────────────
    # Pick best acquisition by mean AF across seeds (from agnostic + sensitive)
    print("\n── Batch strategy comparison ──")
    acq_af = {}
    for label, summary in all_summaries:
        acq_name = label.split("|")[0]
        acq_af.setdefault(acq_name, []).append(summary["AF"])
    best_acq = max(acq_af, key=lambda a: np.mean(acq_af[a]))
    print(f"  Best acquisition by mean AF: {best_acq}")

    # For batch, use args.bo_surrogate (sensible for both BORE and sensitive acqs)
    for strat in batch_strategies:
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
        clf = pi_classifier if best_acq == "pi_ordinal" else None
        bo = BOLoop(
            surrogate=surrogate,
            acquisition_name=best_acq,
            batch_strategy=strat,
            batch_size=args.bo_batch_size,
            n_iterations=args.bo_iterations,
            classifier_pipeline=clf,
            random_state=RANDOM_STATE,
            bore_adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
        )
        history = bo.run_batch(X_cv, y_raw)
        label = f"batch|{best_acq}|{strat}"
        all_histories.append(history)
        all_labels.append(label)
        metrics = SimulationMetrics(y_raw)
        all_summaries.append((label, metrics.summary(history)))

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    # Split into agnostic / sensitive subsets for per-group raw plots
    agnostic_mask  = [l.split("|")[0] in SURROGATE_AGNOSTIC  for l in all_labels]
    sensitive_mask = [l.split("|")[0] in SURROGATE_SENSITIVE for l in all_labels]

    h_agnostic  = [h for h, m in zip(all_histories, agnostic_mask)  if m]
    l_agnostic  = [l for l, m in zip(all_labels,   agnostic_mask)  if m]
    h_sensitive = [h for h, m in zip(all_histories, sensitive_mask) if m]
    l_sensitive = [l for l, m in zip(all_labels,   sensitive_mask) if m]

    # ── Primary comparison figures (seed-aggregated, readable) ────────────────
    # 1. Mean ± std AF / EF / Top-5% per method — the main summary chart
    plot_seed_aggregated_comparison(
        all_summaries,
        save_path="docs/bo_ablation_seed_aggregated.png",
    )

    # 2. Heatmap: acquisition × surrogate mean AF and EF
    plot_sensitive_heatmap(
        all_summaries, sensitive_acquisitions=SURROGATE_SENSITIVE,
        metric="AF", save_path="docs/bo_ablation_heatmap_AF.png",
    )
    plot_sensitive_heatmap(
        all_summaries, sensitive_acquisitions=SURROGATE_SENSITIVE,
        metric="EF", save_path="docs/bo_ablation_heatmap_EF.png",
    )

    # 3. Seed-averaged convergence bands — one shaded line per unique method
    plot_seed_averaged_convergence(
        all_histories, all_labels, y_raw, metric="avg_score",
        save_path="docs/bo_ablation_convergence_bands.png",
    )
    plot_seed_averaged_convergence(
        all_histories, all_labels, y_raw, metric="simple_regret",
        save_path="docs/bo_ablation_regret_bands.png",
    )

    # ── Per-group raw plots (all individual seed runs) ─────────────────────────
    plot_topk_curves(h_agnostic, l_agnostic, y_raw,
                     save_path="docs/bo_ablation_topk_agnostic.png")
    plot_topk_curves(h_sensitive, l_sensitive, y_raw,
                     save_path="docs/bo_ablation_topk_sensitive.png")

    # ── Results CSV ───────────────────────────────────────────────────────────
    results_df = save_simulation_results(all_histories, all_labels, y_raw,
                                         save_path="docs/bo_ablation_results.csv")
    print(f"\n── Ablation complete. {len(all_summaries)} runs. ──")
    print(results_df.to_string(index=False))

    # ── 5. Surrogate calibration ───────────────────────────────────────────────
    # Evaluate each surrogate using its seed=42 EI init split — EI is the most
    # relevant reference since it directly relies on sigma.
    print("\n── Surrogate Calibration Summary ──")
    for surr_name in surrogates:
        surrogate = _resolve_surrogate(surr_name, ck_params)
        ref_history = calibration_histories.get(surr_name)
        if ref_history is None:
            print(f"  {surr_name}: no reference history found, skipping")
            continue
        init_idx = np.array(ref_history["init_indices"])
        pool_idx = np.array(ref_history["pool_indices"])
        surrogate.fit(X_cv[init_idx], y_raw[init_idx])
        cal = compute_surrogate_calibration(surrogate, X_cv[pool_idx], y_raw[pool_idx])
        if "error" in cal:
            print(f"  {surr_name}: {cal['error']}")
        else:
            print(
                f"  {surr_name}: z_mean={cal['mean_z']:+.2f}  z_std={cal['std_z']:.2f}  "
                f"1σ_cov={cal['fraction_within_1sigma']*100:.0f}%  "
                f"2σ_cov={cal['fraction_within_2sigma']*100:.0f}%  "
                f"cal_err={cal['calibration_error']:.3f}"
            )
            plot_calibration(cal, surrogate_name=surr_name,
                             save_path=f"docs/bo_ablation_calibration_{surr_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LVMOF-Surrogate: MOF Synthesis Prediction + Bayesian Optimization"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to experiment data Excel file")

    # BO arguments
    parser.add_argument("--bo", action="store_true",
                        help="Run Bayesian Optimization instead of classification pipeline")
    parser.add_argument("--bo-mode", type=str, default="simulate",
                        choices=["simulate", "recommend", "batch"],
                        help="BO operating mode")
    parser.add_argument("--bo-surrogate", type=str, default="rf_mi",
                        choices=["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                                 "rf_cl_only", "xgb_cl_only"],
                        help="BO regression surrogate (matches classification pipeline variants)")
    parser.add_argument("--bo-acquisition", type=str, default=BO_DEFAULT_ACQUISITION,
                        choices=["bore", "lfbo", "lfbo_ssl",
                                 "ei", "lcb", "pi_ordinal", "thompson", "random"],
                        help="Acquisition function. "
                             "bore=original BORE (recovers PI); "
                             "lfbo=LFBO-EI weighted classifier (Song et al. ICML 2022, recovers EI); "
                             "lfbo_ssl=LFBO-EI + semi-supervised pseudo-labeling (DRE-BO-SSL 2023).")
    parser.add_argument("--bo-batch-strategy", type=str, default="constant_liar",
                        choices=["constant_liar", "kriging_believer"],
                        help="Batch selection strategy")
    parser.add_argument("--bo-batch-size", type=int, default=BO_BATCH_SIZE,
                        help="Batch size for batch BO mode")
    parser.add_argument("--bo-iterations", type=int, default=BO_N_ITERATIONS,
                        help="Number of BO iterations")
    parser.add_argument("--bo-ablation", action="store_true",
                        help="Run full BO ablation study")
    parser.add_argument("--bo-include-mlr", action="store_true",
                        help="Include metal_over_linker_ratio as a controllable BO parameter (off by default)")
    parser.add_argument("--bo-ranking-target", action="store_true",
                        help="Train surrogate on rank-normalised targets instead of raw 0-9 scores. "
                             "Better for ordinal objectives where relative ordering matters more "
                             "than exact magnitude (APL Machine Learning, 2024).")
    parser.add_argument("--bo-feasibility", action="store_true",
                        help="Apply synthesis feasibility prior to acquisition scores in recommend "
                             "mode: penalises candidates with temperature above solvent boiling "
                             "point (Griffiths et al., Digital Discovery 2022).")

    # Classification pipeline options
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Re-evaluate pipelines with existing hyperparams (skip Optuna tuning)")

    # Recommendation mode args
    parser.add_argument("--bo-precursor", type=str, default=None,
                        help="Precursor SMILES for recommend mode")
    parser.add_argument("--bo-linker", type=str, default=None,
                        help="Linker SMILES for recommend mode")
    parser.add_argument("--bo-modulator", type=str, default=None,
                        help="Modulator SMILES for recommend mode")

    args = parser.parse_args()

    if args.bo:
        if args.bo_ablation:
            run_bo_ablation(args)
        else:
            run_bo(args)
    else:
        main(data_path=args.data, skip_tuning=args.skip_tuning)
