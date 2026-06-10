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
                    BO_LFBO_GAMMA, BO_CHECKPOINT_DIR, BO_DEFAULT_SURROGATE,
                    BO_DEFAULT_ACQUISITION, BO_CONTROLLABLE_PARAMS,
                    BO_HIT_THRESHOLD,
                    TARGET_METALS, METAL_BLOCK_DIM, COLIGAND_BLOCK_DIM,
                    COMPLEX_BLOCK_DIM, TOTAL_VOLUME_ML,
                    BO_LFBO_ADAPTIVE_GAMMA, BO_LINKER_UMOL_BOUNDS,
                    BO_EI_XI)
from data_processing import (load_data, build_inventory, merge_data,
                              run_process_variable_audit, fix_missingness,
                              add_solvent_cosmo_features)
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
                             apply_variance_threshold_no_kmeans,
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
                         plot_confusion_matrices, plot_top_k_precision,
                         run_shap_featurized,
                         _patch_xgb_base_score_for_shap)
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
FEATURES_CKPT  = os.path.join(CHECKPOINT_DIR, "features.pkl")
OPTUNA_DB      = f"sqlite:///{CHECKPOINT_DIR}/optuna.db"


def _trace_feature_names_through_pipeline(pipeline, X_names):
    """Best-effort trace of which X_names survive preprocessing before the tree.

    Mirrors VT / MI support masks. TripletTrainer embedding columns (when
    concat_original=True) are appended with generic ``CL_emb_{i}`` labels;
    cl-only pipelines produce pure ``CL_emb_{i}`` labels. Returns a list of
    length equal to the tree's input feature count, or None if tracing fails.
    """
    try:
        names = list(X_names)
        for step_name, step in pipeline.steps:
            cls = type(step).__name__
            if cls in ("RandomForestRegressor", "XGBRegressor"):
                break
            if hasattr(step, "get_support"):
                mask = step.get_support()
                if len(mask) == len(names):
                    names = [n for n, k in zip(names, mask) if bool(k)]
            elif cls == "TripletTrainer":
                emb_dim = getattr(step, "embedding_dim", 0) or 0
                if getattr(step, "concat_original", True):
                    names = list(names) + [f"CL_emb_{i}" for i in range(emb_dim)]
                else:
                    names = [f"CL_emb_{i}" for i in range(emb_dim)]
        return names
    except Exception:
        return None


def _shap_for_batch(surrogate, X_candidates, batch_indices, X_names,
                     candidates_df, top_n=6):
    """Compute SHAP feature attributions for the selected batch rows.

    Returns a list of dicts {rank, pxrd_predicted, contributions:[{name,
    shap_value, feature_value}, ...]} or None if SHAP isn't available / fails.
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
    except Exception:
        return None

    try:
        pipeline = surrogate.pipeline
        Xt_full = surrogate._transform_features(np.asarray(X_candidates))
        reg = None
        for _, step in pipeline.steps:
            if isinstance(step, (RandomForestRegressor, XGBRegressor)):
                reg = step
                break
        if reg is None:
            return None

        traced = _trace_feature_names_through_pipeline(pipeline, X_names)
        if traced is None or len(traced) != Xt_full.shape[1]:
            traced = [f"feat_{i}" for i in range(Xt_full.shape[1])]

        batch_idx = list(batch_indices)
        Xt_batch = Xt_full[batch_idx]
        X_raw_batch = np.asarray(X_candidates)[batch_idx]

        _patch_xgb_base_score_for_shap(reg)
        explainer = shap.TreeExplainer(reg)
        sv = explainer.shap_values(Xt_batch)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]

        out = []
        for row_i, cand_i in enumerate(batch_idx):
            contribs = np.asarray(sv[row_i], dtype=float)
            order = np.argsort(-np.abs(contribs))[:top_n]
            entries = []
            for j in order:
                name = traced[j] if j < len(traced) else f"feat_{j}"
                raw_val = None
                # Only the original X_names slots have a meaningful raw value;
                # post-MI traced columns that map to an original slot do, others
                # (CL embeddings, synthetic) leave raw_val=None.
                if name in X_names:
                    try:
                        col = X_names.index(name)
                        raw_val = float(X_raw_batch[row_i, col])
                    except Exception:
                        raw_val = None
                entries.append({
                    "name":          str(name),
                    "shap_value":    float(contribs[j]),
                    "feature_value": raw_val,
                })
            # Resolve human-friendly row descriptor from the candidates DataFrame.
            row_desc = {}
            if candidates_df is not None:
                for c in ("temperature_k", "equivalents", "linker_conc",
                          "metal_over_linker_ratio", "phi_1",
                          "solvent_1", "solvent_2"):
                    if c in candidates_df.columns:
                        try:
                            row_desc[c] = candidates_df.iloc[cand_i][c]
                        except Exception:
                            pass
            out.append({
                "rank":          row_i + 1,
                "contributions": entries,
                "row_descriptor": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                                    for k, v in row_desc.items()},
            })
        return out
    except Exception as e:
        print(f"  [SHAP] Skipped batch explanation: {type(e).__name__}: {e}")
        return None


def _feature_layout_hash(X_names, X_groups, n_features):
    """Stable hash of (X_names, X_groups, n_features) for checkpoint consistency.

    Any change in feature ordering or count invalidates the positional
    assumptions that NeighborhoodTemplateSelector and CandidateFeaturizer rely
    on; compare this hash across checkpoints to detect the mismatch explicitly
    instead of silently falling back to a positional chemistry/process split.
    """
    import hashlib
    payload = "|".join([
        str(int(n_features)),
        ",".join(map(str, X_names or [])),
        ",".join(map(str, X_groups or [])),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
    df = add_solvent_cosmo_features(df)
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

    # Leakage-free: no pre-CV KMeans. Cluster-OHE is appended fold-locally
    # inside the surrogate pipeline via KMeansFeatureAugmenter.
    X_vt, _fitted_vt = apply_variance_threshold_no_kmeans(X_raw)
    n_chem_features = X_vt.shape[1]
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

    vt_mask = _fitted_vt["vt"].get_support()
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
        kmeans_prepended_to_vt=False,
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
        kmeans_prepended_to_vt=False,
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

    print(f"[catalog] {len(X_names)} feature names for {X_cv.shape[1]} columns "
          f"(chem={n_chem_features}, fold-local KMeans inside pipeline)")
    return (X_cv, y, groups, cv_tune, cv_eval, X_names, X_groups,
            discrete_mask, vt_discrete_mask, n_chem_features)


def main(data_path=None, skip_tuning=False):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Steps 1–4: data + features + CV setup ─────────────────────────────────

    ck = _load(DATA_CKPT)
    if (ck is not None and "discrete_mask" in ck
            and "n_chem_features" in ck):
        X_cv, y, groups = ck["X_cv"], ck["y"], ck["groups"]
        cv_tune, cv_eval = ck["cv_tune"], ck["cv_eval"]
        X_names, X_groups = ck["X_names"], ck["X_groups"]
        discrete_mask    = ck["discrete_mask"]
        vt_discrete_mask = ck.get("vt_discrete_mask", discrete_mask)
        n_chem_features  = ck["n_chem_features"]
    else:
        if ck is not None:
            print("\n── Checkpoint outdated (missing n_chem_features / discrete mask), "
                  "re-featurizing ──")
        else:
            print("\n── Featurizing from data file ──")
        (X_cv, y, groups, cv_tune, cv_eval, X_names, X_groups,
         discrete_mask, vt_discrete_mask,
         n_chem_features) = _featurize_data(data_path)

        layout_hash = _feature_layout_hash(X_names, X_groups, X_cv.shape[1])
        joblib.dump({"X_cv": X_cv, "y": y, "groups": groups,
                     "cv_tune": cv_tune, "cv_eval": cv_eval,
                     "X_names": X_names, "X_groups": X_groups,
                     "discrete_mask": discrete_mask,
                     "vt_discrete_mask": vt_discrete_mask,
                     "n_chem_features": n_chem_features,
                     "feature_layout_hash": layout_hash}, DATA_CKPT)
        print(f"[checkpoint] saved {DATA_CKPT} (layout={layout_hash}, "
              f"chem={n_chem_features})")

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
                study.optimize(
                    lambda t: objective_fn(t, X_cv, y, cv_tune, groups,
                                           n_chem_features=n_chem_features),
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

    pipe_rf_mi              = make_rf_pipe(best_rf_mi_params,       with_cl=False,
                                           n_chem_features=n_chem_features)
    pipe_rf_cl_mi           = make_rf_pipe(best_rf_cl_mi_params,    with_cl=True,
                                           n_chem_features=n_chem_features)
    pipe_xgb_mi             = make_xgb_pipe(best_xgb_mi_params,     with_cl=False,
                                            n_chem_features=n_chem_features)
    pipe_xgb_cl_mi          = make_xgb_pipe(best_xgb_cl_mi_params,  with_cl=True,
                                            n_chem_features=n_chem_features)
    pipe_rf_cl_only         = make_rf_pipe_cl_only(best_rf_cl_only_params,
                                                   n_chem_features=n_chem_features)
    pipe_xgb_cl_only        = make_xgb_pipe_cl_only(best_xgb_cl_only_params,
                                                    n_chem_features=n_chem_features)

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
    plot_top_k_precision(pipelines, X_cv, y, cv_eval, groups, positive_class=2)

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
    """Load featurized data for BO simulate/batch modes.

    Always reflects the current data file — new experiments are picked up
    automatically.  Results are cached in checkpoints/features.pkl and only
    recomputed when the data file changes.
    """
    X_cv, y_raw, y_remapped, df_merged, mask, _ = _featurize_fresh(data_path)
    return X_cv, y_raw, y_remapped, df_merged, mask


def _load_n_chem_features():
    """Read n_chem_features from the features cache for fold-local KMeans.

    Returns None if the cache predates the v2 schema — callers should treat
    None as "legacy / no in-pipeline KMeans" (i.e. expect X_cv to carry
    pre-fit cluster-OHE columns). After re-featurization the cache always
    contains n_chem_features, so this should only be None on stale caches.
    """
    cached = _load(FEATURES_CKPT)
    if cached is None:
        return None
    return cached.get("n_chem_features")


def _data_file_fingerprint(data_path):
    """Return (mtime, size) for the data file — used to detect changes."""
    try:
        st = os.stat(data_path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def _featurize_fresh(data_path=None):
    """Featurize from the Excel file, with a file-change cache.

    If the data file hasn't changed since the last run (same mtime + size),
    the cached result in checkpoints/features.pkl is returned immediately.
    Otherwise featurization runs and the cache is updated.

    Returns (X_cv, y_raw, y_remapped, df_merged, mask, process_cols_present).
    """
    from data_processing import (load_data as _ld, build_inventory, merge_data,
                                  fix_missingness,
                                  add_solvent_cosmo_features as _add_cosmo)
    from dimensionality import (prepare_labels, remap_score,
                                apply_variance_threshold_no_kmeans,
                                build_process_interactions, assemble_cv_matrix)

    # Cache schema version: bump to invalidate caches built with the leaky
    # pre-CV KMeans path. Surrogates trained with v2 X_cv expect raw chemistry
    # features (no pre-fit cluster OHE) and rely on the in-pipeline
    # KMeansFeatureAugmenter for fold-local cluster-OHE injection.
    CACHE_SCHEMA = "v2_no_kmeans"

    resolved_path = data_path or "data/Experiments_with_Calculated_Properties_no_linker.xlsx"
    fingerprint = _data_file_fingerprint(resolved_path)

    # Check cache (require both fingerprint *and* schema match)
    cached = _load(FEATURES_CKPT)
    if (cached is not None
            and fingerprint is not None
            and cached.get("fingerprint") == fingerprint
            and cached.get("schema") == CACHE_SCHEMA):
        print(f"[features] Cache hit — data file unchanged, skipping featurization.")
        return (cached["X_cv"], cached["y_raw"], cached["y_remapped"],
                cached["df_merged"], cached["mask"], cached["process_cols_present"])
    if cached is not None and cached.get("schema") != CACHE_SCHEMA:
        print(f"[features] Cache invalidated: schema "
              f"'{cached.get('schema', 'v1_leaky')}' → '{CACHE_SCHEMA}' "
              f"(fold-local KMeans). Re-featurizing.")

    print("[features] Featurizing from data file (includes any new experiments)...")
    df = _ld(resolved_path)
    df = _add_cosmo(df)
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    df_merged = fix_missingness(df_merged)

    (X_final, df_merged, fp_cols, num_descriptors, calc,
     linker_col, mod_col, process_cols_present, X_process,
     X_linker, X_modulator, mod_eq, X_precursor_perlig,
     Xinventorynumeric) = assemble_features(df_merged, df_inventory)

    X_raw, y_int, mask = prepare_labels(df_merged, X_final)
    y_raw = y_int.astype(float)  # raw 0-9 scores (before remap)

    # Leakage-free VT: no KMeans fit on the full dataset. Fold-local
    # cluster-OHE is added inside the surrogate pipeline.
    X_vt, fitted_vt = apply_variance_threshold_no_kmeans(X_raw)
    n_chem_features = X_vt.shape[1]

    y_remapped = np.array([remap_score(s) for s in y_int])
    Xprocnorm, interactions, _ = build_process_interactions(df_merged, mask, process_cols_present)
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)

    print(f"[features] {X_cv.shape[0]} experiments, {X_cv.shape[1]} features "
          f"(chem={n_chem_features}, proc+interact={X_cv.shape[1] - n_chem_features}); "
          f"best score in data: {y_raw.max():.0f}")

    # Persist SMILES-level feature cache (only writes if new entries were added)
    from smiles_cache import get_smiles_cache
    get_smiles_cache().flush()

    # Save featurization cache. Extras (df_inventory, fitted_vt, n_raw_features,
    # n_chem_features) enable BO recommend-mode to re-featurize a novel target
    # chemistry and project it into the same X_cv feature space used at
    # training time. fitted_vt now contains only {"vt": ...} — no KMeans/OHE.
    joblib.dump({
        "fingerprint": fingerprint,
        "schema": CACHE_SCHEMA,
        "X_cv": X_cv,
        "y_raw": y_raw,
        "y_remapped": y_remapped,
        "df_merged": df_merged,
        "mask": mask,
        "process_cols_present": process_cols_present,
        "df_inventory": df_inventory,
        "fitted_vt": fitted_vt,
        "n_raw_features": X_raw.shape[1],
        "n_chem_features": n_chem_features,
    }, FEATURES_CKPT)
    print(f"[features] Cache saved -> {FEATURES_CKPT}")

    return X_cv, y_raw, y_remapped, df_merged, mask, process_cols_present


def _resolve_surrogate(surrogate_name, params, ranking_target=False,
                       n_chem_features=None):
    """Create a RegressionSurrogate from surrogate name + hyperparams.

    Maps each --bo-surrogate choice to the matching Optuna-tuned hyperparams:
      rf_mi       → best_rf_mi_params
      xgb_mi      → best_xgb_mi_params
      rf_cl_mi    → best_rf_cl_mi_params
      xgb_cl_mi   → best_xgb_cl_mi_params
      rf_cl_only  → best_rf_cl_only_params
      xgb_cl_only → best_xgb_cl_only_params

    n_chem_features (optional): number of leading chemistry-feature columns
    in X_cv. When provided, fold-local KMeansFeatureAugmenter is inserted
    in the pipeline so cluster-OHE is fit per-fold (leakage-free). When None,
    falls back to the value persisted in the features cache by the v2
    schema; only stays None on stale (v1 leaky) caches.
    """
    if n_chem_features is None:
        n_chem_features = _load_n_chem_features()
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
            pipe = make_rf_regressor_pipe_cl_only(hp,
                                                  n_chem_features=n_chem_features)
        else:
            pipe = make_rf_regressor_pipe(hp, with_cl=with_cl,
                                          n_chem_features=n_chem_features)
        return SurrClass(pipe, model_type="rf")
    else:
        if cl_only:
            pipe = make_xgb_regressor_pipe_cl_only(hp,
                                                   n_chem_features=n_chem_features)
        else:
            pipe = make_xgb_regressor_pipe(hp, with_cl=with_cl,
                                           n_chem_features=n_chem_features)
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

    bo = BOLoop(
        surrogate=surrogate,
        acquisition_name=args.bo_acquisition,
        batch_strategy=args.bo_batch_strategy,
        batch_size=args.bo_batch_size,
        n_iterations=args.bo_iterations,
        random_state=RANDOM_STATE,
        lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
    )

    checkpointer = BOCheckpointer()

    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)

    if args.bo_mode == "simulate":
        print(f"\n── Simulation: {args.bo_acquisition} | {args.bo_surrogate} "
              f"| {args.bo_iterations} iters ──")
        history = bo.run_simulation(X_cv, y_raw, init_fraction=BO_INIT_FRACTION,
                                    groups=groups)
        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)

        # Hit rate vs baseline
        y_sel = y_raw[history["selected_indices"]]
        hit_rate = float((y_sel >= BO_HIT_THRESHOLD).mean()) if len(y_sel) > 0 else 0.0
        baseline_hit = float((y_raw >= BO_HIT_THRESHOLD).mean())

        af_s = "n/a (no hits in pool)" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
        ef_s = "n/a (no hits in pool)" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
        print(f"\n── Results (hit threshold = score >= {BO_HIT_THRESHOLD:.0f}) ──")
        print(f"  AF:           {af_s}")
        print(f"  EF:           {ef_s}")
        print(f"  Hit rate:     {hit_rate*100:.1f}%  (baseline: {baseline_hit*100:.1f}%)")
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
                      "unreliable. EI acquisition scores may be misleading.")
        plot_calibration(cal, surrogate_name=args.bo_surrogate,
                         save_path=f"docs/bo_calibration_{label}.png")

    elif args.bo_mode == "batch":
        print(f"\n── Batch simulation: {args.bo_acquisition} | {args.bo_surrogate} "
              f"| batch_size={args.bo_batch_size} | {args.bo_batch_strategy} ──")
        history = bo.run_batch(X_cv, y_raw, init_fraction=BO_INIT_FRACTION, groups=groups)
        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)

        y_sel = y_raw[history["selected_indices"]]
        hit_rate = float((y_sel >= BO_HIT_THRESHOLD).mean()) if len(y_sel) > 0 else 0.0
        baseline_hit = float((y_raw >= BO_HIT_THRESHOLD).mean())

        af_s = "n/a (no hits in pool)" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
        ef_s = "n/a (no hits in pool)" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
        print(f"\n── Results (hit threshold = score >= {BO_HIT_THRESHOLD:.0f}) ──")
        print(f"  AF:          {af_s}")
        print(f"  EF:          {ef_s}")
        print(f"  Hit rate:    {hit_rate*100:.1f}%  (baseline: {baseline_hit*100:.1f}%)")
        print(f"  Best score:   {summary['best_score_final']:.0f}")

        label = f"batch_{args.bo_acquisition}_{args.bo_surrogate}_{args.bo_batch_strategy}"
        checkpointer.save(f"batch_{label}", history)
        save_full_history(history, label)

    elif args.bo_mode == "recommend":
        _run_recommend(args)
        return

    elif args.bo_mode == "evaluate":
        _run_evaluate(args)
        return

    elif args.bo_mode == "loco":
        _run_loco(args)
        return

    elif args.bo_mode == "loco-evaluate":
        _run_loco_evaluate(args)
        return

    elif args.bo_mode == "learning-curve":
        _run_learning_curve(args)
        return

    else:
        raise ValueError(f"Unknown --bo-mode: {args.bo_mode}")


def _run_evaluate(args):
    """Multi-seed per-cluster BO evaluation.

    Runs the BO simulation with N different random seeds, computing per-cluster
    threshold-based AF / EF / Hit% for each run. Reports mean ± std across
    seeds per cluster and aggregate, and generates grouped bar charts.
    """
    from bo_core import BOLoop
    from bo_metrics import (SimulationMetrics, plot_per_cluster_bar,
                            plot_evaluate_hit_rate, save_simulation_results,
                            plot_convergence, plot_average_score,
                            plot_topk_curves, plot_simple_regret,
                            save_full_history, compute_surrogate_calibration,
                            plot_calibration)

    print("\n" + "=" * 70)
    print("  BO EVALUATION (multi-seed, per-cluster)")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}
    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)
    n_clusters = int(groups.max()) + 1

    seeds = [42 + i * 111 for i in range(args.bo_eval_seeds)]
    print(f"  Seeds: {seeds}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Acquisition: {args.bo_acquisition} | Surrogate: {args.bo_surrogate}")

    all_cluster_stats = []   # list of per_cluster_summary dicts
    all_summaries = []       # list of aggregate summary dicts
    all_histories = []       # for convergence plots
    first_surrogate = None   # for calibration (first seed only)

    for i_seed, seed in enumerate(seeds):
        print(f"\n── Seed {seed} ──")
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                       ranking_target=args.bo_ranking_target)

        bo = BOLoop(
            surrogate=surrogate,
            acquisition_name=args.bo_acquisition,
            n_iterations=args.bo_iterations,
            random_state=seed,
            lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
        )
        history = bo.run_simulation(X_cv, y_raw, groups=groups)

        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)
        cluster_stats = metrics.per_cluster_summary(history, groups)

        all_summaries.append(summary)
        all_cluster_stats.append(cluster_stats)
        all_histories.append(history)
        if i_seed == 0:
            first_surrogate = surrogate

        # Hit rate for per-seed printout
        y_sel = y_raw[history["selected_indices"]]
        hit = float((y_sel >= BO_HIT_THRESHOLD).mean()) if len(y_sel) > 0 else 0.0
        baseline = float((y_raw >= BO_HIT_THRESHOLD).mean())
        af_s = "n/a" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
        ef_s = "n/a" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
        print(f"  AF={af_s}  EF={ef_s}  "
              f"Hit={hit*100:.1f}% (base={baseline*100:.1f}%)")

    # ── Aggregate results ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY (mean ± std across seeds)  "
          f"hit threshold = score >= {BO_HIT_THRESHOLD:.0f}")
    print("=" * 70)

    def _fmt(vals, scale=1.0, width=5, prec=2):
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        if not valid:
            return f"{'  n/a':>{width*2 + 1}}"
        m = np.mean(valid) * scale
        s = np.std(valid)  * scale
        return f"{m:>{width}.{prec}f}±{s:<{width}.{prec}f}"

    # Per-cluster table
    header = (f"{'Cluster':>8}  {'AF':>12}  {'EF':>12}  "
              f"{'Hit%':>12}  {'Base%':>8}  {'n_pool':>6}  {'n_hits':>6}")
    print(header)
    print("-" * len(header))

    for cid in range(n_clusters):
        afs   = [s[cid]["AF"]           for s in all_cluster_stats if cid in s]
        efs   = [s[cid]["EF"]           for s in all_cluster_stats if cid in s]
        hits  = [s[cid]["hit_rate"]     for s in all_cluster_stats if cid in s]
        bases = [s[cid]["baseline_hit"] for s in all_cluster_stats if cid in s]
        n_pool   = all_cluster_stats[0][cid]["n_pool"]      if cid in all_cluster_stats[0] else 0
        n_hits_p = all_cluster_stats[0][cid].get("n_pool_hits", 0) \
            if cid in all_cluster_stats[0] else 0
        print(f"{'C' + str(cid):>8}  "
              f"{_fmt(afs)}  "
              f"{_fmt(efs)}  "
              f"{_fmt(hits, scale=100, prec=1)}%  "
              f"{np.mean(bases)*100:>6.1f}%  "
              f"{n_pool:>6d}  {n_hits_p:>6d}")

    agg_afs  = [s["AF"] for s in all_summaries]
    agg_efs  = [s["EF"] for s in all_summaries]
    agg_hits = [s["hit_rate"] for s in all_summaries]
    agg_base = [s["baseline_hit_rate"] for s in all_summaries]
    print("-" * len(header))
    print(f"{'Agg':>8}  "
          f"{_fmt(agg_afs)}  "
          f"{_fmt(agg_efs)}  "
          f"{_fmt(agg_hits, scale=100, prec=1)}%  "
          f"{np.mean(agg_base)*100:>6.1f}%")

    # ── Per-cluster bar charts ───────────────────────────────────────────
    for metric in ["AF", "EF"]:
        plot_per_cluster_bar(all_cluster_stats, metric=metric)
    plot_evaluate_hit_rate(all_cluster_stats)

    # ── Convergence & diagnostic plots (all seeds overlaid) ───────────
    label = f"{args.bo_acquisition}_{args.bo_surrogate}"
    seed_labels = [f"seed_{s}" for s in seeds]
    plot_convergence(all_histories, seed_labels, y_raw,
                     save_path=f"docs/bo_convergence_{label}.png")
    plot_average_score(all_histories, seed_labels,
                       save_path=f"docs/bo_avg_score_{label}.png")
    plot_topk_curves(all_histories, seed_labels, y_raw,
                     save_path=f"docs/bo_topk_{label}.png")
    plot_simple_regret(all_histories, seed_labels, y_raw,
                       save_path=f"docs/bo_simple_regret_{label}.png")
    save_full_history(all_histories[0], label)

    # ── Surrogate calibration (first seed's init/pool split) ──────────
    print(f"\n── Surrogate Calibration ({args.bo_surrogate}) ──")
    init_idx = np.array(all_histories[0]["init_indices"])
    pool_idx = np.array(all_histories[0]["pool_indices"])
    first_surrogate.fit(X_cv[init_idx], y_raw[init_idx])
    cal = compute_surrogate_calibration(first_surrogate, X_cv[pool_idx], y_raw[pool_idx])

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
                  "unreliable. EI acquisition scores may be misleading.")
    plot_calibration(cal, surrogate_name=args.bo_surrogate,
                     save_path=f"docs/bo_calibration_{label}.png")


def _run_loco(args):
    """Leave-one-cluster-out BO evaluation.

    For each cluster: train on ALL other clusters, use the held-out cluster
    as the pool.  Tests pure generalization to unseen chemistry.
    """
    from bo_core import BOLoop
    from bo_metrics import SimulationMetrics, plot_loco_bar, plot_loco_hit_rate

    print("\n" + "=" * 70)
    print("  BO LEAVE-ONE-CLUSTER-OUT (LOCO) EVALUATION")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}
    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)
    n_clusters = int(groups.max()) + 1

    print(f"  Acquisition: {args.bo_acquisition} | Surrogate: {args.bo_surrogate}")
    print(f"  Clusters: {n_clusters}")

    MIN_POOL_LOCO = 20  # skip clusters too small for meaningful evaluation
    loco_results = {}

    for cid in range(n_clusters):
        pool_idx = np.where(groups == cid)[0]
        if len(pool_idx) < MIN_POOL_LOCO:
            print(f"\n── Held-out cluster {cid} ── SKIPPED (pool={len(pool_idx)} < {MIN_POOL_LOCO})")
            continue

        print(f"\n── Held-out cluster {cid} ──")
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                       ranking_target=args.bo_ranking_target)

        bo = BOLoop(
            surrogate=surrogate,
            acquisition_name=args.bo_acquisition,
            n_iterations=args.bo_iterations,
            random_state=RANDOM_STATE,
            lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
        )
        history = bo.run_simulation_loco(X_cv, y_raw, groups, held_out_cluster=cid)

        # Compute metrics on the held-out cluster only
        y_cluster = y_raw[pool_idx]
        cluster_metrics = SimulationMetrics(y_cluster)

        # Remap global selected indices → local indices within held-out cluster
        global_to_local = {g: l for l, g in enumerate(pool_idx)}
        local_selected = [global_to_local[idx] for idx in history["selected_indices"]
                          if idx in global_to_local]
        y_sel = np.array([float(y_raw[idx]) for idx in history["selected_indices"]
                          if idx in global_to_local])
        local_history = {
            "selected_indices": local_selected,
            "y_selected": y_sel.tolist(),
            "best_so_far": history["best_so_far"],
            "init_indices": [],   # no init within this cluster
        }
        summary = cluster_metrics.summary(local_history)

        # Hit rate: fraction of BO selections with score >= 7 (crystalline)
        hit_rate = float((y_sel >= BO_HIT_THRESHOLD).mean()) if len(y_sel) > 0 else 0.0
        # Baseline hit rate: fraction of cluster with score >= 7
        baseline_hit = float((y_cluster >= BO_HIT_THRESHOLD).mean())

        loco_results[cid] = {
            "AF": summary["AF"],
            "EF": summary["EF"],
            "hit_discovery_rate": summary["hit_discovery_rate"],
            "hit_rate": hit_rate,
            "baseline_hit": baseline_hit,
            "n_pool": len(pool_idx),
            "n_selected": len(local_selected),
            "best_score": summary["best_score_final"],
        }
        af_str = "n/a" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
        ef_str = "n/a" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
        print(f"  AF={af_str}  EF={ef_str}  "
              f"HitDisc={summary['hit_discovery_rate']*100:.1f}%  "
              f"hit_rate={hit_rate*100:.0f}% (baseline={baseline_hit*100:.0f}%)  "
              f"pool={len(pool_idx)}  selected={len(local_selected)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LOCO SUMMARY")
    print("=" * 70)

    evaluated_cids = sorted(loco_results.keys())
    skipped = n_clusters - len(evaluated_cids)
    if skipped:
        print(f"  ({skipped} cluster(s) skipped: pool < {MIN_POOL_LOCO})")

    print(f"  hit threshold = score >= {BO_HIT_THRESHOLD:.0f}\n")
    header = (f"{'Cluster':>8}  {'AF':>8}  {'EF':>8}  {'HitDisc%':>9}  "
              f"{'Hit%':>6}  {'Base%':>6}  {'n_pool':>6}  {'selected':>8}")
    print(header)
    print("-" * len(header))
    for cid in evaluated_cids:
        r = loco_results[cid]
        name = group_names[cid] if cid < len(group_names) else f"C{cid}"
        af_s = "    n/a" if np.isnan(r["AF"]) else f"{r['AF']:>8.2f}"
        ef_s = "    n/a" if np.isnan(r["EF"]) else f"{r['EF']:>8.2f}"
        print(f"{name[:8]:>8}  {af_s}  {ef_s}  "
              f"{r['hit_discovery_rate']*100:>8.1f}%  "
              f"{r['hit_rate']*100:>5.0f}%  {r['baseline_hit']*100:>5.0f}%  "
              f"{r['n_pool']:>6d}  {r['n_selected']:>8d}")

    # Pool-size weighted mean across clusters with defined metrics
    def _weighted(key):
        pairs = [(loco_results[c][key], loco_results[c]["n_pool"])
                 for c in evaluated_cids
                 if not (isinstance(loco_results[c][key], float)
                         and np.isnan(loco_results[c][key]))]
        if not pairs:
            return float("nan")
        total = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / max(total, 1)

    w_af   = _weighted("AF")
    w_ef   = _weighted("EF")
    w_disc = _weighted("hit_discovery_rate")
    w_hit  = _weighted("hit_rate")
    w_base = _weighted("baseline_hit")
    af_s = "    n/a" if np.isnan(w_af) else f"{w_af:>8.2f}"
    ef_s = "    n/a" if np.isnan(w_ef) else f"{w_ef:>8.2f}"
    print("-" * len(header))
    print(f"{'W.Mean':>8}  {af_s}  {ef_s}  "
          f"{w_disc*100:>8.1f}%  {w_hit*100:>5.0f}%  {w_base*100:>5.0f}%")

    # Plots
    label = f"{args.bo_acquisition}_{args.bo_surrogate}"
    for metric in ["AF", "EF"]:
        plot_loco_bar(loco_results, metric=metric,
                      save_path=f"docs/bo_loco_{metric}_{label}.png")
    plot_loco_hit_rate(loco_results,
                       save_path=f"docs/bo_loco_hit_rate_{label}.png")


def _run_loco_evaluate(args):
    """Multi-seed leave-one-cluster-out BO evaluation.

    Runs the LOCO simulation across N random seeds and reports per-cluster
    AF / EF / Hit% as mean ± std, plus a pool-size weighted aggregate.
    Produces error-bar bar charts so cross-acquisition / cross-surrogate
    comparisons can be made with uncertainty taken into account.

    Mirrors the structure of _run_evaluate, but with the LOCO train/pool
    split (train on all-other-clusters, pool = held-out cluster) instead
    of the random init/pool split.
    """
    from bo_core import BOLoop
    from bo_metrics import (SimulationMetrics,
                            plot_loco_bar_multiseed,
                            plot_loco_hit_rate_multiseed)

    print("\n" + "=" * 70)
    print("  BO LOCO EVALUATION (multi-seed, per-cluster, with error bars)")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}
    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)
    n_clusters = int(groups.max()) + 1

    seeds = [42 + i * 111 for i in range(args.bo_eval_seeds)]
    print(f"  Seeds: {seeds}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Acquisition: {args.bo_acquisition} | Surrogate: {args.bo_surrogate}")

    MIN_POOL_LOCO = 20
    loco_results_by_seed = []   # list[dict[cid -> per-cluster results]]

    for i_seed, seed in enumerate(seeds):
        print(f"\n── Seed {seed} ──")
        seed_results = {}

        for cid in range(n_clusters):
            pool_idx = np.where(groups == cid)[0]
            if len(pool_idx) < MIN_POOL_LOCO:
                if i_seed == 0:
                    print(f"  Cluster {cid} SKIPPED "
                          f"(pool={len(pool_idx)} < {MIN_POOL_LOCO})")
                continue

            surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                           ranking_target=args.bo_ranking_target)
            bo = BOLoop(
                surrogate=surrogate,
                acquisition_name=args.bo_acquisition,
                n_iterations=args.bo_iterations,
                random_state=seed,
                lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
            )
            history = bo.run_simulation_loco(X_cv, y_raw, groups,
                                             held_out_cluster=cid)

            y_cluster = y_raw[pool_idx]
            cluster_metrics = SimulationMetrics(y_cluster)

            # Remap global selected indices → local indices within the pool.
            global_to_local = {g: l for l, g in enumerate(pool_idx)}
            local_selected = [global_to_local[idx]
                              for idx in history["selected_indices"]
                              if idx in global_to_local]
            y_sel = np.array([float(y_raw[idx])
                              for idx in history["selected_indices"]
                              if idx in global_to_local])
            local_history = {
                "selected_indices": local_selected,
                "y_selected": y_sel.tolist(),
                "best_so_far": history["best_so_far"],
                "init_indices": [],
            }
            summary = cluster_metrics.summary(local_history)
            hit_rate = float((y_sel >= BO_HIT_THRESHOLD).mean()) \
                if len(y_sel) > 0 else 0.0
            baseline_hit = float((y_cluster >= BO_HIT_THRESHOLD).mean())

            seed_results[cid] = {
                "AF": summary["AF"],
                "EF": summary["EF"],
                "hit_discovery_rate": summary["hit_discovery_rate"],
                "hit_rate": hit_rate,
                "baseline_hit": baseline_hit,
                "n_pool": len(pool_idx),
                "n_selected": len(local_selected),
                "best_score": summary["best_score_final"],
            }
            af_s = "n/a" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
            ef_s = "n/a" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
            print(f"  C{cid}: AF={af_s}  EF={ef_s}  "
                  f"Hit={hit_rate*100:.0f}% (base={baseline_hit*100:.0f}%)  "
                  f"pool={len(pool_idx)} sel={len(local_selected)}")

        loco_results_by_seed.append(seed_results)

    # ── Aggregate across seeds ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  LOCO EVALUATION SUMMARY (mean ± std across "
          f"{len(seeds)} seeds)  hit threshold = score >= "
          f"{BO_HIT_THRESHOLD:.0f}")
    print("=" * 70)

    def _fmt(vals, scale=1.0, width=5, prec=2):
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        if not valid:
            return f"{'  n/a':>{width*2 + 1}}"
        m = np.mean(valid) * scale
        s = np.std(valid)  * scale
        return f"{m:>{width}.{prec}f}±{s:<{width}.{prec}f}"

    evaluated_cids = sorted({cid for r in loco_results_by_seed
                             for cid in r.keys()})
    skipped = n_clusters - len(evaluated_cids)
    if skipped:
        print(f"  ({skipped} cluster(s) skipped: pool < {MIN_POOL_LOCO})\n")

    header = (f"{'Cluster':>8}  {'AF':>12}  {'EF':>12}  "
              f"{'Hit%':>12}  {'Base%':>8}  {'n_pool':>6}")
    print(header)
    print("-" * len(header))

    for cid in evaluated_cids:
        afs   = [r[cid]["AF"]           for r in loco_results_by_seed if cid in r]
        efs   = [r[cid]["EF"]           for r in loco_results_by_seed if cid in r]
        hits  = [r[cid]["hit_rate"]     for r in loco_results_by_seed if cid in r]
        bases = [r[cid]["baseline_hit"] for r in loco_results_by_seed if cid in r]
        n_pool = next((r[cid]["n_pool"] for r in loco_results_by_seed
                       if cid in r), 0)
        print(f"{'C' + str(cid):>8}  "
              f"{_fmt(afs)}  "
              f"{_fmt(efs)}  "
              f"{_fmt(hits, scale=100, prec=1)}%  "
              f"{np.mean(bases)*100:>6.1f}%  "
              f"{n_pool:>6d}")

    # Pool-size weighted aggregate per seed → mean ± std across seeds.
    def _weighted_per_seed(seed_results, key):
        pairs = [(r[key], r["n_pool"]) for r in seed_results.values()
                 if key in r and not (isinstance(r[key], float)
                                      and np.isnan(r[key]))]
        if not pairs:
            return float("nan")
        total = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / max(total, 1)

    agg_afs = [_weighted_per_seed(r, "AF")  for r in loco_results_by_seed]
    agg_efs = [_weighted_per_seed(r, "EF")  for r in loco_results_by_seed]
    agg_hits = [_weighted_per_seed(r, "hit_rate") for r in loco_results_by_seed]
    agg_base = [_weighted_per_seed(r, "baseline_hit") for r in loco_results_by_seed]

    print("-" * len(header))
    print(f"{'W.Mean':>8}  "
          f"{_fmt(agg_afs)}  "
          f"{_fmt(agg_efs)}  "
          f"{_fmt(agg_hits, scale=100, prec=1)}%  "
          f"{np.mean([b for b in agg_base if not np.isnan(b)])*100:>6.1f}%")

    # ── Plots ───────────────────────────────────────────────────────────────
    label = f"{args.bo_acquisition}_{args.bo_surrogate}"
    for metric in ["AF", "EF"]:
        plot_loco_bar_multiseed(
            loco_results_by_seed, metric=metric,
            save_path=f"docs/bo_loco_{metric}_{label}_multiseed.png")
    plot_loco_hit_rate_multiseed(
        loco_results_by_seed,
        save_path=f"docs/bo_loco_hit_rate_{label}_multiseed.png")


def _run_learning_curve(args):
    """BO learning curve: sweep init_fraction to find the cold-start budget.

    For each init fraction (5% to 50%), runs multi-seed BO simulation and
    measures threshold-based AF / EF / Hit%. The resulting plot shows how
    many initial experiments are needed before the BO becomes useful
    (AF > ~2 over the y >= BO_HIT_THRESHOLD criterion).
    """
    from bo_core import BOLoop
    from bo_metrics import SimulationMetrics, plot_learning_curve

    print("\n" + "=" * 70)
    print("  BO LEARNING CURVE (hit rate vs number of initial experiments)")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}
    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)

    # Baseline hit rate: fraction of entire pool with score >= 7 (random guessing)
    baseline_hit = float((y_raw >= BO_HIT_THRESHOLD).mean())
    print(f"  Global baseline hit rate (score≥7): {baseline_hit*100:.1f}%")

    init_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    seeds = [42 + i * 111 for i in range(args.bo_eval_seeds)]

    print(f"  Acquisition: {args.bo_acquisition} | Surrogate: {args.bo_surrogate}")
    print(f"  Seeds: {len(seeds)} | Fractions: {init_fractions}")

    lc_results = []

    for frac in init_fractions:
        seed_afs, seed_efs, seed_hits, seed_ninits = [], [], [], []

        for seed in seeds:
            surrogate = _resolve_surrogate(args.bo_surrogate, ck_params,
                                           ranking_target=args.bo_ranking_target)
            bo = BOLoop(
                surrogate=surrogate,
                acquisition_name=args.bo_acquisition,
                n_iterations=args.bo_iterations,
                random_state=seed,
                lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
            )
            history = bo.run_simulation(X_cv, y_raw, init_fraction=frac,
                                        groups=groups)
            metrics = SimulationMetrics(y_raw)
            summary = metrics.summary(history)

            # Hit rate of BO selections
            y_sel = y_raw[history["selected_indices"]]
            hit = float((y_sel >= BO_HIT_THRESHOLD).mean()) if len(y_sel) > 0 else 0.0

            seed_afs.append(summary["AF"])
            seed_efs.append(summary["EF"])
            seed_hits.append(hit)
            seed_ninits.append(len(history["init_indices"]))

        result = {
            "init_frac":     frac,
            "n_init_mean":   np.mean(seed_ninits),
            "AF_mean":       np.mean(seed_afs),
            "AF_std":        np.std(seed_afs),
            "EF_mean":       np.mean(seed_efs),
            "EF_std":        np.std(seed_efs),
            "hit_mean":      np.mean(seed_hits),
            "hit_std":       np.std(seed_hits),
            "baseline_hit":  baseline_hit,
        }
        lc_results.append(result)

        print(f"  init={frac:.0%} ({result['n_init_mean']:.0f} expts) | "
              f"AF={result['AF_mean']:.2f}±{result['AF_std']:.2f}  "
              f"EF={result['EF_mean']:.2f}±{result['EF_std']:.2f}  "
              f"Hit={result['hit_mean']*100:.1f}±{result['hit_std']*100:.1f}% "
              f"(base={baseline_hit*100:.1f}%)")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LEARNING CURVE SUMMARY")
    print("=" * 70)
    header = (f"{'Frac':>6}  {'n_init':>6}  {'AF':>12}  "
              f"{'EF':>12}  {'Hit%':>12}  {'Base%':>6}")
    print(header)
    print("-" * len(header))
    for r in lc_results:
        print(f"{r['init_frac']:>5.0%}  {r['n_init_mean']:>6.0f}  "
              f"{r['AF_mean']:>5.2f}±{r['AF_std']:<5.2f}  "
              f"{r['EF_mean']:>5.2f}±{r['EF_std']:<5.2f}  "
              f"{r['hit_mean']*100:>5.1f}±{r['hit_std']*100:<4.1f}%  "
              f"{r['baseline_hit']*100:>5.1f}%")

    # Find the knee — smallest fraction where hit rate > 1.5× baseline
    for r in lc_results:
        if r["hit_mean"] > 1.5 * r["baseline_hit"]:
            print(f"\n  → Recommendation: ~{r['n_init_mean']:.0f} initial experiments "
                  f"({r['init_frac']:.0%}) needed for hit rate > 1.5× baseline")
            break
    else:
        print("\n  → Hit rate never reached 1.5× baseline in this sweep. "
              "Consider increasing --bo-iterations or trying different surrogates.")

    plot_learning_curve(lc_results)


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

    Chemistry-targeted workflow (recommended)
    -----------------------------------------
    Provide your target linker, precursor, and/or modulator SMILES.
    The NeighborhoodTemplateSelector finds the most similar past experiments
    (by Morgan FP + chemistry-feature cosine similarity) and uses the nearest
    neighbor's molecular feature row as the fixed chemistry template.  The BO
    then only searches over process conditions (temperature, concentration,
    solvent ratio, equivalents) for that specific chemistry.

    When both --bo-precursor AND --bo-linker are provided, a TuRBO-style trust
    region is also activated to focus the process-parameter search near
    conditions that worked for similar chemistry:
      - Expands after 3 consecutive improvements
      - Shrinks after 3 consecutive failures

      python main.py --bo --bo-mode recommend \\
          --bo-linker <SMILES> --bo-precursor <SMILES> --bo-modulator <SMILES> \\
          --bo-surrogate rf_cl_mi --bo-batch-size 3
      → Synthesize top candidate → add result to data file → run again

    Global workflow (no chemistry target, full-space search):
      python main.py --bo --bo-mode recommend --bo-surrogate rf_cl_mi --bo-batch-size 3
    """
    from bo_core import (BOLoop, BOCheckpointer, SearchSpace,
                         CandidateFeaturizer, _compute_acquisition,
                         NeighborhoodTemplateSelector, TrustRegion,
                         FeasibilityScorer, BatchSelector,
                         compute_stoichiometric_ratio)
    from config import BO_OPTIONAL_PARAMS

    print("\n" + "=" * 70)
    print("  BO RECOMMENDATION (persistent loop)")
    print("=" * 70)

    # Meta collected across the run and written as a sidecar JSON next to the
    # recommendations CSV. The Streamlit page reads it to render banners
    # (extrapolation warning, calibration quality, layout staleness) and the
    # per-candidate SHAP explanation panel.
    meta = {
        "similarity":    None,   # {max_sim, level}
        "calibration":   None,   # {sigma_scale, quality}
        "layout":        {"ok": True, "warning": None},
        "refeaturize":   {"used": False, "warnings": []},
        "shap_batch":    None,   # {rows: [{rank, contributions: [...]}, ...]}
        "acq_scope":     None,   # {scope: "global"|"neighborhood", n, f_best}
        "warnings":      [],     # free-text user-visible notes
    }

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

    # Sigma is auto-calibrated inside RegressionSurrogate.fit via K-fold CV.
    # A well-calibrated scalar sits near 1.0; far-from-1.0 values mean raw
    # inter-tree sigma is chronically under- or over-estimating residual std.
    _sig_scale = getattr(surrogate, "sigma_scale_", None)
    if _sig_scale is not None and np.isfinite(_sig_scale):
        if 0.7 <= _sig_scale <= 1.8:
            _sig_quality = "good"
        elif 0.4 <= _sig_scale <= 3.0:
            _sig_quality = "ok"
        else:
            _sig_quality = "poor"
        meta["calibration"] = {
            "sigma_scale": float(_sig_scale),
            "quality": _sig_quality,
        }

    # Warn if thompson is used with an XGB surrogate — it degrades to
    # deterministic predict() because XGB has no tree estimators_ attribute.
    if args.bo_acquisition == "thompson" and args.bo_surrogate.startswith("xgb"):
        print("  WARNING: thompson sampling requires a RandomForest surrogate. "
              "With XGB it falls back to deterministic predict() (no exploration). "
              "Switch to an rf_* surrogate or a different acquisition function.")

    # 4. Build search space
    extra_params = BO_OPTIONAL_PARAMS if args.bo_include_mlr else None
    controllable = list(BO_CONTROLLABLE_PARAMS.keys())
    if args.bo_include_mlr:
        controllable += list(BO_OPTIONAL_PARAMS.keys())
    print(f"  Controllable params: {controllable}")

    _ck_data = _load(DATA_CKPT) or {}
    X_names  = _ck_data.get("X_names")
    X_groups = _ck_data.get("X_groups")
    _stored_layout_hash = _ck_data.get("feature_layout_hash")
    n_feat = X_cv.shape[1]

    # DATA_CKPT is only written by the full main() training path. In
    # recommend-only workflows it may be missing or stale. Without a correct
    # X_groups, the NeighborhoodTemplateSelector cannot distinguish chemistry
    # from process columns, so Stage-2 cosine similarity degenerates to zero.
    #
    # Fallback: reconstruct a minimal catalog by column POSITION. build_feature_catalog
    # appends the normalised process block and process-interaction block at the
    # very end of X_cv, so the trailing tail is process; everything before it
    # is chemistry (with a handful of raw-process columns embedded inside,
    # which is fine for chemistry-neighbour cosine similarity).
    _layout_stale = (X_groups is None or X_names is None
                     or len(X_groups) != n_feat)
    _hash_mismatch = False
    if not _layout_stale and _stored_layout_hash is not None:
        _expected = _feature_layout_hash(X_names, X_groups, n_feat)
        if _expected != _stored_layout_hash:
            _hash_mismatch = True
            print(f"[catalog] WARNING: DATA_CKPT layout hash mismatch "
                  f"({_stored_layout_hash} vs computed {_expected}). Cache "
                  f"inconsistent — chemistry/process split may be wrong.")

    if _layout_stale or _stored_layout_hash is None:
        from feature_assembly import _INTERACTION_NAMES
        n_proc = len(process_cols_present)
        n_int  = len(_INTERACTION_NAMES)
        n_tail = n_proc + n_int
        if n_tail >= n_feat:
            n_tail = 0   # guard against degenerate tiny matrices
        print(f"[catalog] DATA_CKPT missing/stale/unhashed — using positional "
              f"fallback (n_chem={n_feat - n_tail}, n_process_tail={n_tail}). "
              f"Run `python main.py` once to cache the full catalog.")
        X_groups = (["Chemistry"] * (n_feat - n_tail)
                    + ["Process Variables"] * n_proc
                    + ["Process Interactions"] * n_int)
        X_names  = [f"chem_{i}" for i in range(n_feat - n_tail)] \
                 + [f"proc:{c}" for c in process_cols_present] \
                 + list(_INTERACTION_NAMES)
        meta["layout"]["ok"] = False
        meta["layout"]["warning"] = (
            "Feature catalog missing or stale — chemistry/process split was "
            "inferred from column positions. Chemistry-neighbor similarity "
            "may be inaccurate. Fix: run `python main.py` once to refresh "
            "checkpoints/data.pkl."
        )
    elif _hash_mismatch:
        meta["layout"]["ok"] = False
        meta["layout"]["warning"] = (
            "Feature-layout hash mismatch between checkpoints/data.pkl and "
            "the current features. The chemistry/process split may be "
            "mislabeled. Fix: delete checkpoints/data.pkl and "
            "checkpoints/features.pkl, then run `python main.py`."
        )

    X_names  = list(X_names)
    X_groups = list(X_groups)

    # Final length alignment safeguard
    if len(X_groups) < n_feat:
        extra = n_feat - len(X_groups)
        X_names  += [f"chem_pad_{i}" for i in range(extra)]
        X_groups += ["Chemistry"] * extra
    elif len(X_groups) > n_feat:
        X_names  = X_names[:n_feat]
        X_groups = X_groups[:n_feat]

    from cosmo_features import CosmoMixer
    cosmo_mixer = CosmoMixer(
        index_path=os.path.join("data", "VT-2005_Sigma_Profile_Database_Index_v2.xlsx"),
        cosmo_folder=os.path.join("data", "solvent_cosmo"),
    )

    search_space = SearchSpace(
        train_df=df_merged[mask], solvent_mixer=cosmo_mixer, extra_params=extra_params,
        observed_pairs_only=args.bo_observed_pairs,
        linker_umol_bounds=BO_LINKER_UMOL_BOUNDS,
        total_volume_ml=TOTAL_VOLUME_ML,
    )

    # ── Chemistry template + trust region logic ───────────────────────────────
    # Chemistry template: find the nearest-neighbor row in the dataset whenever
    # any SMILES input is provided (linker, precursor, or modulator).  This
    # fixes the molecular features of the candidate matrix to the user's target
    # chemistry.  Trust region (narrowing the process-parameter search window)
    # is activated only when BOTH precursor AND linker are provided.
    has_chemistry_input = any(x is not None for x in
                              [args.bo_precursor, args.bo_linker, args.bo_modulator])
    using_trust_region  = (args.bo_precursor is not None and
                           args.bo_linker    is not None)
    trust_region = None
    override_bounds = None
    ref_idx = None   # chemistry template — nearest neighbor in dataset
    fixed_ratio = None  # stoichiometric metal/linker ratio from phosphine counts

    # Acquisition-scope defaults (global). Overridden below in fixed-chemistry
    # mode so EI's f_best and LFBO's tau reflect what's achievable for the
    # target chemistry rather than the global dataset maximum.
    X_train_acq   = X_cv
    y_train_acq   = y_raw
    f_best_scoped = f_best
    acq_scope_note = "global"

    if has_chemistry_input:
        linker_str    = args.bo_linker    or ""
        precursor_str = args.bo_precursor or ""
        modulator_str = args.bo_modulator or None

        print(f"\n  [Chemistry] Target linker:    "
              f"{linker_str[:60] or '(not specified)'}")
        print(f"  [Chemistry] Target precursor: "
              f"{precursor_str[:60] or '(not specified)'}")
        if modulator_str:
            print(f"  [Chemistry] Target modulator: {modulator_str[:60]}")

        # Compute stoichiometric metal/linker ratio from phosphine counts.
        if linker_str and precursor_str:
            fixed_ratio = compute_stoichiometric_ratio(precursor_str, linker_str)
            if fixed_ratio is not None:
                print(f"  [Chemistry] Fixed metal/linker ratio = {fixed_ratio:.4g} "
                      f"(from SMILES phosphine counts)")

        selector = NeighborhoodTemplateSelector(
            df_train=df_merged[mask],
            X_cv=X_cv,
            X_groups=X_groups,
            linker_col=COLMAP["linker1"],
            precursor_col=COLMAP["precursor"],
            modulator_col=COLMAP["modulator"],
        )
        center, spread, neighbors, ref_idx = selector.select(
            target_linker_smiles=linker_str,
            target_precursor_smiles=precursor_str,
            search_bounds=search_space.bounds,
            target_modulator_smiles=modulator_str,
        )

        # ── Scope acquisition to the chemistry neighborhood ───────────────────
        # When both precursor + linker are provided, the user is asking
        # "maximize crystallinity for THIS chemistry". A global f_best / LFBO
        # tau means the acquisition compares against the best pxrd score
        # anywhere in the dataset — often unattainable for the target family,
        # which collapses EI toward zero and makes LFBO labels come from
        # unrelated chemistries. Restrict the acquisition's reference pool to
        # the selected neighborhood so EI and LFBO calibrate against what's
        # achievable for this chemistry.  Require a minimum neighborhood size
        # so the LFBO classifier has enough data to train; otherwise fall
        # back to global scope.
        if using_trust_region and neighbors is not None and len(neighbors) > 0:
            _MIN_NB_FOR_SCOPED_ACQ = 8
            _nb_positions = neighbors.index.tolist()
            _y_nb = y_raw[_nb_positions]
            if len(_nb_positions) >= _MIN_NB_FOR_SCOPED_ACQ and np.ptp(_y_nb) > 0:
                X_train_acq   = X_cv[_nb_positions]
                y_train_acq   = _y_nb
                f_best_scoped = float(_y_nb.max())
                acq_scope_note = (f"neighborhood (n={len(_nb_positions)}, "
                                  f"f_best={f_best_scoped:.0f})")
                print(f"  [Acquisition] Scoped to chemistry neighborhood: "
                      f"n={len(_nb_positions)}, f_best={f_best_scoped:.0f} "
                      f"(global f_best={f_best:.0f}).")
                meta["acq_scope"] = {
                    "scope": "neighborhood",
                    "n": int(len(_nb_positions)),
                    "f_best": f_best_scoped,
                    "global_f_best": float(f_best),
                }
            else:
                acq_scope_note = (f"global (neighborhood too small: "
                                  f"n={len(_nb_positions)})")
                print(f"  [Acquisition] Using global scope — neighborhood "
                      f"has only {len(_nb_positions)} rows "
                      f"(need >= {_MIN_NB_FOR_SCOPED_ACQ}).")
                meta["acq_scope"] = {
                    "scope": "global",
                    "n": int(len(_nb_positions)),
                    "f_best": float(f_best),
                    "reason": "neighborhood_too_small",
                }

        # ── Trust region (only when both precursor + linker are given) ────────
        # Solvent restriction: when the recenter gate fires on exact-chem
        # successes, restrict the candidate pool to solvents those successes
        # used. Prevents the BO from sampling chemically unrelated solvents
        # (e.g. DCM) when the anchor success used a polar aprotic (e.g. DMF).
        allowed_solvent_set = None
        if using_trust_region:
            if state.get("trust_region") is not None and iteration > 0:
                # Restore existing trust region and update with latest f_best
                trust_region = TrustRegion.from_dict(state["trust_region"])
                trust_region.update(f_best)
                print(f"  [TrustRegion] Restored | length={trust_region.length:.3f}")

                # Recenter on the best experiment within the chemistry
                # neighborhood — not the global best, which may belong to a
                # completely different molecule and pull the TR to irrelevant
                # process conditions.
                #
                # Policy: prefer exact-canonical-chemistry rows when at least
                # one such row is a moderate success (score >= 5). This avoids
                # anchoring on a high-scoring cross-chemistry analog when we
                # have direct evidence for the target chemistry. Falls back to
                # argmax over all neighbors when same-chemistry data has no
                # successes (or no exact matches in the top-K).
                _EXACT_MATCH_SCORE_FLOOR = 5
                if neighbors is not None and len(neighbors) > 0:
                    nb_positions = neighbors.index.tolist()
                    df_nb        = df_merged[mask].iloc[nb_positions]

                    def _canon(s):
                        try:
                            from rdkit import Chem
                            m = Chem.MolFromSmiles(str(s))
                            return Chem.MolToSmiles(m) if m is not None else str(s).strip()
                        except Exception:
                            return str(s).strip()

                    tgt_l = _canon(linker_str)
                    tgt_p = _canon(precursor_str)
                    nb_linker_c    = df_nb[COLMAP["linker1"]].apply(_canon)
                    nb_precursor_c = df_nb[COLMAP["precursor"]].apply(_canon)
                    exact_mask = (nb_linker_c == tgt_l) & (nb_precursor_c == tgt_p)
                    exact_positions = [
                        p for p, m_ in zip(nb_positions, exact_mask) if m_
                    ]

                    exact_y_max = (
                        float(np.max(y_raw[exact_positions]))
                        if exact_positions else -np.inf
                    )

                    if exact_positions and exact_y_max >= _EXACT_MATCH_SCORE_FLOOR:
                        best_local = int(np.argmax(y_raw[exact_positions]))
                        best_idx   = exact_positions[best_local]
                        print(f"  [TrustRegion] Recentered on exact-chemistry match "
                              f"(idx={best_idx}, score={y_raw[best_idx]:.0f}, "
                              f"{len(exact_positions)} exact matches in top-K, "
                              f"max score={exact_y_max:.0f} >= "
                              f"{_EXACT_MATCH_SCORE_FLOOR}).")

                        # Build allowed solvent set from exact-chem rows
                        # scoring >= the floor. Includes both solvent_1 and
                        # solvent_2 cols so binary mixtures are allowed.
                        _success_rows = df_merged[mask].iloc[[
                            p for p in exact_positions
                            if y_raw[p] >= _EXACT_MATCH_SCORE_FLOOR
                        ]]
                        _sol_set = set()
                        for _col in ("solvent_1", "solvent_2"):
                            if _col in _success_rows.columns:
                                for _v in _success_rows[_col].dropna().astype(str):
                                    _v = _v.strip().upper()
                                    if _v and _v not in ("NAN", "NONE", "NA"):
                                        _sol_set.add(_v)
                        allowed_solvent_set = _sol_set if _sol_set else None
                        if allowed_solvent_set:
                            print(f"  [TrustRegion] Restricting candidate pool to "
                                  f"solvents from exact-chem successes: "
                                  f"{sorted(allowed_solvent_set)}")
                    else:
                        best_local = int(np.argmax(y_raw[nb_positions]))
                        best_idx   = nb_positions[best_local]
                        if exact_positions:
                            print(f"  [TrustRegion] {len(exact_positions)} exact-chem "
                                  f"matches in top-K but max score "
                                  f"{exact_y_max:.0f} < {_EXACT_MATCH_SCORE_FLOOR}; "
                                  f"falling back to cross-chemistry argmax.")
                        print(f"  [TrustRegion] Recentered on best chemistry-neighborhood "
                              f"experiment (idx={best_idx}, score={y_raw[best_idx]:.0f}).")
                else:
                    best_idx = int(np.argmax(y_raw))
                    print(f"  [TrustRegion] No neighbors found — recentered on global "
                          f"best (idx={best_idx}, score={y_raw[best_idx]:.0f}).")
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
    # When both target precursor AND linker SMILES are provided, re-featurize
    # the target chemistry from scratch through the full 12-block pipeline.
    # This makes the surrogate score the USER'S molecule rather than a
    # nearest-neighbor proxy — required for novel-chemistry discovery.
    #
    # Falls back to nearest-neighbor template if:
    #   - Only one of (precursor, linker) is provided (insufficient to re-featurize)
    #   - Re-featurization fails (bad SMILES, unparseable precursor, etc.)
    #   - Required cache entries (df_inventory, fitted_vt) are absent (older cache)
    template_row = None
    used_refeat  = False
    if (args.bo_precursor is not None and args.bo_linker is not None):
        _ck_features = _load(FEATURES_CKPT) or {}
        _df_inventory = _ck_features.get("df_inventory")
        _fitted_vt    = _ck_features.get("fitted_vt")

        if _df_inventory is not None and _fitted_vt is not None:
            from bo_featurize import (featurize_target_chemistry,
                                      chemistry_similarity_to_training)
            try:
                print("\n  [ReFeaturize] Computing target chemistry features from SMILES...")
                template_row, refeat_warns = featurize_target_chemistry(
                    precursor_smiles=args.bo_precursor,
                    linker_smiles=args.bo_linker,
                    modulator_smiles=args.bo_modulator,
                    df_merged_train=df_merged,
                    df_inventory_train=_df_inventory,
                    fitted_vt=_fitted_vt,
                    n_X_cv_features=X_cv.shape[1],
                )
                used_refeat = True
                print("  [ReFeaturize] Target featurized successfully "
                      "(surrogate will score THIS molecule, not a proxy).")

                # Extrapolation warning via chemistry-feature cosine similarity.
                max_sim = chemistry_similarity_to_training(template_row, X_cv)
                if max_sim < 0.30:
                    _sim_level = "low"
                    print(f"  [ReFeaturize] WARNING: low similarity to training "
                          f"({max_sim:.3f}). Surrogate is extrapolating — "
                          f"recommendations are informed guesses at best.")
                elif max_sim < 0.60:
                    _sim_level = "medium"
                    print(f"  [ReFeaturize] Moderate similarity to training "
                          f"({max_sim:.3f}). Some extrapolation risk.")
                else:
                    _sim_level = "high"
                    print(f"  [ReFeaturize] Target close to training distribution "
                          f"({max_sim:.3f}) — surrogate predictions should be reliable.")

                meta["similarity"] = {"max_sim": float(max_sim),
                                       "level": _sim_level}
                meta["refeaturize"]["used"] = True
                meta["refeaturize"]["warnings"] = [str(w) for w in refeat_warns]

                for w in refeat_warns:
                    print(f"  [ReFeaturize] NOTE: {w}")

            except Exception as e:
                print(f"  [ReFeaturize] FAILED: {type(e).__name__}: {e}")
                print(f"  [ReFeaturize] Falling back to nearest-neighbor template.")
                template_row = None
        else:
            print("  [ReFeaturize] Cache is from an older pipeline version "
                  "(missing df_inventory or fitted_vt). Delete features.pkl and "
                  "re-run 'python main.py' to enable target re-featurization.")

    if template_row is None:
        if ref_idx is not None:
            template_row = X_cv[ref_idx]
            print(f"  [Template] Using nearest neighbor idx={ref_idx} as chemistry template.")
        else:
            template_row = np.nanmedian(X_cv, axis=0)
            print("  [Template] Using dataset-median template (no chemistry SMILES provided).")

    # ── Resolve fixed metal/linker ratio (when not a BO parameter) ──────────
    # Fallback chain: explicit SMILES → ref row SMILES → ref row data → median
    if not args.bo_include_mlr and fixed_ratio is None:
        df_train = df_merged[mask]
        if ref_idx is not None:
            ref_row = df_train.iloc[ref_idx]
            fixed_ratio = compute_stoichiometric_ratio(
                ref_row.get('smiles_precursor', ''),
                ref_row.get('smiles_linker_1', ''),
            )
            if fixed_ratio is not None:
                print(f"  [Chemistry] Fixed metal/linker ratio = {fixed_ratio:.4g} "
                      f"(from nearest-neighbor phosphine counts)")
        if fixed_ratio is None and ref_idx is not None:
            existing = ref_row.get('metal_over_linker_ratio', np.nan)
            if pd.notna(existing):
                fixed_ratio = float(existing)
                print(f"  [Chemistry] Fixed metal/linker ratio = {fixed_ratio:.4g} "
                      f"(from nearest-neighbor experimental data)")
        if fixed_ratio is None:
            fixed_ratio = float(df_train['metal_over_linker_ratio'].median())
            print(f"  [Chemistry] Fixed metal/linker ratio = {fixed_ratio:.4g} "
                  f"(dataset median)")

    # 5. Generate candidates (within trust region if active)
    candidates = search_space.generate_lhs_candidates(
        seed=RANDOM_STATE + iteration,
        override_bounds=override_bounds,
    )

    # Restrict to allowed-solvent set when the recenter gate fired.
    #   strict     : both solvent_1 AND solvent_2 (if present) must be in the
    #                anchor set. Pure anchor solvents and pure-anchor binaries
    #                only — preserves the original conservative behavior.
    #   permissive : pure singles must use an anchor solvent; binary mixtures
    #                pass if at least one component is an anchor. Enables
    #                anchor + co-solvent exploration (e.g., DMF/DCM, DMF/THF
    #                when DMF was the anchor) without re-introducing pure-DCM
    #                style pure-non-anchor candidates the surrogate already
    #                considers failure regions.
    #   off        : no filter (skip this block entirely; controlled upstream
    #                by setting allowed_solvent_set = None when off).
    if allowed_solvent_set and args.bo_solvent_filter != "off":
        _s1 = candidates["solvent_1"].astype(str).str.strip().str.upper()
        _s2 = candidates["solvent_2"].astype(str).str.strip().str.upper()
        _s2_empty = _s2.isin(["", "NAN", "NONE", "NA"])
        _s1_in    = _s1.isin(allowed_solvent_set)
        _s2_in    = _s2.isin(allowed_solvent_set)

        if args.bo_solvent_filter == "strict":
            # Both components must be anchor (or solvent_2 must be empty).
            _keep = _s1_in & (_s2_empty | _s2_in)
        else:  # permissive
            # Pure singles: solvent_1 must be anchor.
            # Binary mixtures: at least one component must be anchor.
            _keep = (
                (_s2_empty & _s1_in)
                | (~_s2_empty & (_s1_in | _s2_in))
            )

        _n_before = len(candidates)
        candidates = candidates[_keep].reset_index(drop=True)
        print(f"  [TrustRegion] Solvent filter ({args.bo_solvent_filter}) kept "
              f"{len(candidates)}/{_n_before} candidates.")
        if len(candidates) == 0:
            print(f"  [TrustRegion] WARNING: solvent filter removed every "
                  f"candidate. Falling back to unfiltered pool.")
            candidates = search_space.generate_lhs_candidates(
                seed=RANDOM_STATE + iteration,
                override_bounds=override_bounds,
            )

    # Apply fixed ratio to all candidates when it is not a BO search parameter.
    if not args.bo_include_mlr and fixed_ratio is not None:
        candidates["metal_over_linker_ratio"] = fixed_ratio

    # linker_conc bounds already intersected with BO_LINKER_UMOL_BOUNDS inside
    # SearchSpace, so no post-LHS clip is needed.

    # Snap phi_1 to the experimentally realizable set BEFORE featurization so
    # the surrogate scores the configuration we will actually display, not a
    # continuous LHS fraction we never intended to propose. Pure-solvent rows
    # (empty or NA solvent_2) are forced to phi_1=1.0 so the direct
    # solvent_1/2_fraction feature cells match the recipe the user will see.
    if "phi_1" in candidates.columns:
        _phi_allowed = np.array([0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
        candidates["phi_1"] = candidates["phi_1"].apply(
            lambda v: float(_phi_allowed[np.argmin(np.abs(_phi_allowed - v))])
        )
        if "solvent_2" in candidates.columns:
            _no_sol2 = (
                candidates["solvent_2"].isna()
                | (candidates["solvent_2"].astype(str).str.strip() == "")
                | (candidates["solvent_2"].astype(str).str.upper() == "NA")
            )
            candidates.loc[_no_sol2, "phi_1"] = 1.0

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
        "f_best": f_best_scoped,
        "gamma": BO_LFBO_GAMMA,
        "random_state": RANDOM_STATE,
        "lfbo_adaptive_gamma": BO_LFBO_ADAPTIVE_GAMMA,
    }
    acq_vals = _compute_acquisition(
        args.bo_acquisition, surrogate,
        X_train_acq, y_train_acq, X_candidates, **acq_kwargs
    )

    # ── Degenerate-acquisition fallback chain ──────────────────────────────────
    # LFBO / LFBO-SSL output P(score >= tau) in [0, 1]. When the scoped
    # neighborhood's f_best comes from a solvent/chemistry class no longer in
    # the pool (e.g. solvent filter restricts to DMF but the highest-scoring
    # row used DCM), the surrogate predicts mu << tau across every candidate.
    # The classifier collapses to P(z=1) ~ 0 uniformly and the diverse-greedy
    # batch selector falls back to picking by diversity alone, surfacing
    # unreliable recommendations.
    #
    # Fallback chain (preserves the LFBO family before switching acquisition
    # type, so LFBO-SSL remains the headline algorithm in the vast majority of
    # iterations and the ablation finding stands):
    #
    #   lfbo_ssl → lfbo → ei      (pseudo-label dilution may be the only
    #                              issue; plain LFBO often still has
    #                              non-degenerate output)
    #   lfbo     → ei              (classifier itself is the bottleneck)
    #   consensus → ei             (consensus already uses LFBO internally)
    #
    # meta["acq_fallback"]["chain"] records every rung tried per iteration so
    # the ablation can measure how often each level fires.
    _ACQ_DEGENERATE_THRESHOLD = 1e-4

    def _is_degenerate(vals):
        if vals is None or len(vals) == 0:
            return True
        return (
            float(np.max(vals)) < _ACQ_DEGENERATE_THRESHOLD
            or float(np.ptp(vals)) < _ACQ_DEGENERATE_THRESHOLD
        )

    def _summarize(name, vals):
        return {
            "name":  name,
            "max":   float(np.max(vals)) if len(vals) else 0.0,
            "range": float(np.ptp(vals)) if len(vals) else 0.0,
        }

    _primary = args.bo_acquisition
    _chain   = [_summarize(_primary, acq_vals)]

    if _primary in ("lfbo", "lfbo_ssl", "consensus") and _is_degenerate(acq_vals):
        from bo_core import EIAcquisition

        # Rung 2: lfbo_ssl → lfbo (skip for lfbo/consensus primaries)
        if _primary == "lfbo_ssl":
            lfbo_vals = _compute_acquisition(
                "lfbo", surrogate,
                X_train_acq, y_train_acq, X_candidates, **acq_kwargs
            )
            _chain.append(_summarize("lfbo", lfbo_vals))
            if not _is_degenerate(lfbo_vals):
                print(f"  [Acquisition] lfbo_ssl degenerate "
                      f"(max={_chain[0]['max']:.4g}, range={_chain[0]['range']:.4g}); "
                      f"falling back to LFBO "
                      f"(max={_chain[1]['max']:.4g}, range={_chain[1]['range']:.4g}).")
                acq_vals = lfbo_vals
            else:
                # Rung 3: lfbo also degenerate → EI
                ei_vals = EIAcquisition(xi=BO_EI_XI).score(mu, sigma, f_best_scoped)
                _chain.append(_summarize("ei", ei_vals))
                print(f"  [Acquisition] lfbo_ssl AND lfbo both degenerate "
                      f"(lfbo_ssl max={_chain[0]['max']:.4g}, "
                      f"lfbo max={_chain[1]['max']:.4g}); "
                      f"falling back to EI "
                      f"(max={_chain[2]['max']:.4g}, range={_chain[2]['range']:.4g}).")
                acq_vals = ei_vals
        else:
            # Primary was lfbo or consensus — skip lfbo rung, go to EI
            ei_vals = EIAcquisition(xi=BO_EI_XI).score(mu, sigma, f_best_scoped)
            _chain.append(_summarize("ei", ei_vals))
            print(f"  [Acquisition] {_primary} degenerate "
                  f"(max={_chain[0]['max']:.4g}, range={_chain[0]['range']:.4g}); "
                  f"falling back to EI "
                  f"(max={_chain[1]['max']:.4g}, range={_chain[1]['range']:.4g}).")
            acq_vals = ei_vals

    meta["acq_fallback"] = {
        "triggered": len(_chain) > 1,
        "primary":   _primary,
        "final":     _chain[-1]["name"],
        "chain":     _chain,
    }

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

    # Select a batch from the candidate pool, then append remaining candidates
    # sorted by acquisition value so the full CSV is still informative.
    if args.bo_batch_strategy == "diverse_greedy":
        print(f"  [DiverseGreedy] Selecting {args.bo_batch_size} candidates "
              f"(lambda={args.bo_diversity_lambda}) ...")
        batch_indices, combined_scores = BatchSelector.diverse_greedy(
            surrogate, X_train_acq, y_train_acq, X_candidates,
            candidates, args.bo_acquisition, args.bo_batch_size,
            diversity_lambda=args.bo_diversity_lambda,
            acq_vals=acq_vals,
            **acq_kwargs,
        )
        results["diversity_combined_score"] = combined_scores
    elif args.bo_batch_strategy == "kriging_believer":
        print(f"  [KrigingBeliever] Selecting {args.bo_batch_size} candidates...")
        batch_indices = BatchSelector.kriging_believer(
            surrogate, X_train_acq, y_train_acq, X_candidates,
            None, args.bo_acquisition, args.bo_batch_size,
            **acq_kwargs,
        )
    else:  # constant_liar
        print(f"  [ConstantLiar] Selecting {args.bo_batch_size} candidates...")
        batch_indices = BatchSelector.constant_liar(
            surrogate, X_train_acq, y_train_acq, X_candidates,
            None, args.bo_acquisition, args.bo_batch_size,
            f_best_scoped, **acq_kwargs,
        )

    kb_mask = np.zeros(len(candidates), dtype=bool)
    kb_mask[batch_indices] = True
    results_batch = results.iloc[batch_indices].copy()
    results_batch["batch_rank"] = range(1, len(batch_indices) + 1)
    _rest = results[~kb_mask].copy()
    _rest["_pxrd_r"] = _rest["pxrd_predicted"].round(2)
    results_rest = _rest.sort_values(["_pxrd_r", "uncertainty"], ascending=[False, True]).drop(columns=["_pxrd_r"])
    results = pd.concat([results_batch, results_rest], ignore_index=True)

    # Round controllable params to experimentally sensible precision
    if "equivalents" in results.columns:
        results["equivalents"] = results["equivalents"].round(0).astype(int)
    if "linker_conc" in results.columns:
        results["linker_conc"] = results["linker_conc"].round(2)
    if "temperature_k" in results.columns:
        results["temperature_k"] = results["temperature_k"].round(0).astype(int)

    # phi_1 snap and pure-solvent forcing now happen BEFORE featurization
    # (see the block above `featurizer = CandidateFeaturizer(...)`), so results
    # are already experimentally-realizable. Below is display-only: mark
    # effectively single-solvent rows as "NA" in solvent_2 for the CSV.
    if "solvent_2" in results.columns:
        no_sol2 = (
            results["solvent_2"].isna()
            | (results["solvent_2"].astype(str).str.strip() == "")
        )
        results.loc[no_sol2, "solvent_2"] = "NA"
        if "phi_1" in results.columns:
            results.loc[results["phi_1"] == 1.0, "solvent_2"] = "NA"
    elif "phi_1" in results.columns:
        # No solvent_2 column at all — all rows are single-solvent
        results["solvent_2"] = "NA"
        results["phi_1"] = 1.0

    # Compute linker and modulator amounts for a 2 mL synthesis
    _display_vol_ml = 2.0
    if "linker_conc" in results.columns:
        _ratio = results["metal_over_linker_ratio"].values if "metal_over_linker_ratio" in results.columns else np.ones(len(results))
        _ratio = np.where(_ratio > 0, _ratio, 1.0)
        _equiv = results["equivalents"].values.astype(float) if "equivalents" in results.columns else np.ones(len(results))
        _umol_linker = results["linker_conc"].values * _display_vol_ml
        _umol_metal  = _ratio * _umol_linker
        _umol_mod    = _equiv * _umol_linker
        results["precursor_umol"] = np.round(_umol_metal, 2)
        results["linker_umol"]    = np.round(_umol_linker, 2)
        results["modulator_umol"] = np.round(_umol_mod, 2)

    # 7. Output
    os.makedirs("docs", exist_ok=True)
    out_path = f"docs/bo_recommendations_iter{iteration}.csv"
    results.head(100).to_csv(out_path, index=False)

    # Compute SHAP attributions for the selected batch so chemists can sanity-
    # check WHY the surrogate recommended each row.
    _shap_rows = _shap_for_batch(
        surrogate, X_candidates, batch_indices, X_names, candidates, top_n=6
    )
    if _shap_rows is not None:
        meta["shap_batch"] = {"rows": _shap_rows}

    # Write meta sidecar JSON for the Streamlit UI.
    import json
    meta_path = out_path.replace(".csv", "_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"  Meta sidecar saved → {meta_path}")
    except Exception as e:
        print(f"  [Meta] Could not write sidecar: {type(e).__name__}: {e}")

    print(f"\n── Iteration {iteration} — Top recommendations ──")
    print(f"   (precursor_umol, linker_umol, modulator_umol are for {_display_vol_ml:.0f} mL of solvent)")
    top_cols = ["batch_rank",
                "solvent_1", "solvent_2", "phi_1",
                "temperature_k", "equivalents",
                "linker_conc", "metal_over_linker_ratio",
                "precursor_umol", "linker_umol", "modulator_umol",
                "pxrd_predicted", "uncertainty", "acquisition_value"]
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

    chem_flags = ""
    if args.bo_linker:
        chem_flags += f" --bo-linker '{args.bo_linker}'"
    if args.bo_precursor:
        chem_flags += f" --bo-precursor '{args.bo_precursor}'"
    if args.bo_modulator:
        chem_flags += f" --bo-modulator '{args.bo_modulator}'"
    print(f"\n  State saved. Next: synthesize top candidates, add results to data file,")
    print(f"  then run again:  python main.py --bo --bo-mode recommend "
          f"--bo-surrogate {args.bo_surrogate} --bo-batch-size {args.bo_batch_size}"
          f"{chem_flags}")


def run_bo_ablation(args):
    """Structured ablation study.

    Design rationale:
      - LFBO, random do not use the regression surrogate for
        acquisition scoring, so varying the surrogate with these methods
        produces identical results.  They are run once per seed with a
        fixed surrogate (args.bo_surrogate).
      - EI, Thompson, Consensus directly consume surrogate (mu, sigma),
        so they are crossed with all six surrogates × three seeds.
        Consensus runs both EI and LFBO, intersects their top-K picks,
        and falls back to pure LFBO when no overlap exists.
      - Calibration is evaluated once per surrogate using the seed=42 init
        split from the EI runs (EI uses sigma, so its init split is the
        most relevant reference).

    Total runs: 2 agnostic × 3 seeds  +  3 sensitive × 6 surrogates × 3 seeds
               + 1-2 batch strategies  =  6 + 54 + 1..2 = 61..62 runs
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
    from bo_core import compute_chemistry_groups
    groups, group_names = compute_chemistry_groups(df_merged)

    # Acquisitions that do not use regression surrogate mu/sigma for scoring.
    SURROGATE_AGNOSTIC = ["lfbo", "random"]
    # Acquisitions that consume surrogate mu/sigma — cross with all surrogates.
    # lfbo_ssl uses surrogate (mu, sigma) to pseudo-label the pool, so it is
    # surrogate-sensitive even though its scoring head is a classifier.
    SURROGATE_SENSITIVE = ["ei", "thompson", "consensus", "lfbo_ssl"]

    surrogates = ["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                  "rf_cl_only", "xgb_cl_only"]

    methods_filter = getattr(args, "bo_ablation_methods", None)
    if methods_filter:
        SURROGATE_AGNOSTIC = [a for a in SURROGATE_AGNOSTIC if a in methods_filter]
        SURROGATE_SENSITIVE = [a for a in SURROGATE_SENSITIVE if a in methods_filter]
        if not SURROGATE_AGNOSTIC and not SURROGATE_SENSITIVE:
            raise ValueError(
                f"--bo-ablation-methods {methods_filter} matched no known "
                f"acquisitions. Choose from lfbo, random, ei, thompson, "
                f"consensus, lfbo_ssl."
            )
        print(f"  [filter] Restricted methods: agnostic={SURROGATE_AGNOSTIC}, "
              f"sensitive={SURROGATE_SENSITIVE}")

    surrogates_filter = getattr(args, "bo_ablation_surrogates", None)
    if surrogates_filter:
        surrogates = [s for s in surrogates if s in surrogates_filter]
        if not surrogates:
            raise ValueError(
                f"--bo-ablation-surrogates {surrogates_filter} matched no known "
                f"surrogates."
            )
        print(f"  [filter] Restricted surrogates: {surrogates}")
    from bo_core import VALID_BATCH_STRATEGIES
    # Seed count is configurable so a defensible Wilcoxon test (n>=10) is
    # possible.  Default 3 preserves prior ablation behaviour.
    n_seeds = getattr(args, "bo_ablation_n_seeds", 3)
    seeds = [42 + i * 111 for i in range(n_seeds)]

    all_histories = []
    all_labels = []
    all_summaries = []

    # ── 1. Surrogate-agnostic acquisitions ────────────────────────────────────
    print(f"\n── Surrogate-agnostic acquisitions (fixed surrogate: {args.bo_surrogate}) ──")

    for acq in SURROGATE_AGNOSTIC:
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
        for seed in seeds:
            label = f"{acq}|seed={seed}"
            print(f"\n── {label} ──")
            bo = BOLoop(
                surrogate=surrogate,
                acquisition_name=acq,
                n_iterations=args.bo_iterations,
                random_state=seed,
                lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
            )
            history = bo.run_simulation(X_cv, y_raw, groups=groups)
            metrics = SimulationMetrics(y_raw)
            summary = metrics.summary(history)
            all_histories.append(history)
            all_labels.append(label)
            all_summaries.append((label, summary))
            af_s = "n/a" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
            ef_s = "n/a" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
            print(f"  AF={af_s}  EF={ef_s}  "
                  f"HitDisc={summary['hit_discovery_rate']*100:.1f}%  "
                  f"Hit={summary['hit_rate']*100:.1f}%")

    # ── 2. Surrogate-sensitive acquisitions ───────────────────────────────────
    print(f"\n── Surrogate-sensitive acquisitions (EI / Thompson) ──")

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
                    lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
                )
                history = bo.run_simulation(X_cv, y_raw, groups=groups)
                metrics = SimulationMetrics(y_raw)
                summary = metrics.summary(history)
                all_histories.append(history)
                all_labels.append(label)
                all_summaries.append((label, summary))
                af_s = "n/a" if np.isnan(summary["AF"]) else f"{summary['AF']:.2f}"
                ef_s = "n/a" if np.isnan(summary["EF"]) else f"{summary['EF']:.2f}"
                print(f"  AF={af_s}  EF={ef_s}  "
                      f"HitDisc={summary['hit_discovery_rate']*100:.1f}%  "
                      f"Hit={summary['hit_rate']*100:.1f}%")

                # Keep seed=42 EI run per surrogate for calibration reference
                if acq == "ei" and seed == 42:
                    calibration_histories[surr_name] = history

    # ── 3. Batch strategy comparison ──────────────────────────────────────────
    # Pick best acquisition by mean AF across seeds (from agnostic + sensitive)
    print("\n── Batch strategy comparison ──")
    acq_af = {}
    for label, summary in all_summaries:
        acq_name = label.split("|")[0]
        if not np.isnan(summary["AF"]):
            acq_af.setdefault(acq_name, []).append(summary["AF"])
    if not acq_af:
        print("  [skip] No non-NaN AF values — skipping batch comparison.")
        batch_strategies = []
        best_acq = None
    else:
        best_acq = max(acq_af, key=lambda a: np.mean(acq_af[a]))
        print(f"  Best acquisition by mean AF: {best_acq}")
        # For batch, use args.bo_surrogate — only valid strategies for best_acq
        batch_strategies = VALID_BATCH_STRATEGIES.get(best_acq, ["diverse_greedy"])
    for strat in batch_strategies:
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
        bo = BOLoop(
            surrogate=surrogate,
            acquisition_name=best_acq,
            batch_strategy=strat,
            batch_size=args.bo_batch_size,
            n_iterations=args.bo_iterations,
            random_state=RANDOM_STATE,
            lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
        )
        history = bo.run_batch(X_cv, y_raw, groups=groups)
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
    # 1. Mean ± std AF / EF / HitDisc% per method — the main summary chart
    plot_seed_aggregated_comparison(
        all_summaries,
        save_path="docs/bo_ablation_seed_aggregated.png",
    )

    # 2. Heatmap: acquisition × surrogate mean AF and EF
    if SURROGATE_SENSITIVE:
        plot_sensitive_heatmap(
            all_summaries, sensitive_acquisitions=SURROGATE_SENSITIVE,
            metric="AF", save_path="docs/bo_ablation_heatmap_AF.png",
        )
        plot_sensitive_heatmap(
            all_summaries, sensitive_acquisitions=SURROGATE_SENSITIVE,
            metric="EF", save_path="docs/bo_ablation_heatmap_EF.png",
        )
    else:
        print("  [skip] No surrogate-sensitive methods selected — skipping heatmaps.")

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
    if h_agnostic:
        plot_topk_curves(h_agnostic, l_agnostic, y_raw,
                         save_path="docs/bo_ablation_topk_agnostic.png")
    if h_sensitive:
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
                        choices=["simulate", "recommend", "batch",
                                 "evaluate", "loco", "loco-evaluate",
                                 "learning-curve"],
                        help="BO operating mode. evaluate=multi-seed per-cluster "
                             "metrics; loco=leave-one-cluster-out generalization; "
                             "loco-evaluate=multi-seed LOCO with mean±std error "
                             "bars; learning-curve=AF vs init fraction sweep")
    parser.add_argument("--bo-surrogate", type=str, default="rf_mi",
                        choices=["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                                 "rf_cl_only", "xgb_cl_only"],
                        help="BO regression surrogate (matches classification pipeline variants)")
    parser.add_argument("--bo-acquisition", type=str, default=BO_DEFAULT_ACQUISITION,
                        choices=["lfbo", "lfbo_ssl", "ei", "thompson", "random",
                                 "consensus"],
                        help="Acquisition function. "
                             "lfbo=LFBO-EI weighted classifier (Song et al. ICML 2022, recovers EI). "
                             "lfbo_ssl=LFBO-EI augmented with surrogate pseudo-labels on the pool. "
                             "consensus=EI∩LFBO rank-intersection, falls back to LFBO.")
    parser.add_argument("--bo-batch-strategy", type=str, default=None,
                        choices=["constant_liar", "kriging_believer", "diverse_greedy"],
                        help="Batch selection strategy (auto-selected if omitted). "
                             "EI supports constant_liar (pessimistic, more diverse) "
                             "and kriging_believer (optimistic, more exploitative). "
                             "LFBO/thompson/random use diverse_greedy automatically.")
    parser.add_argument("--bo-diversity-lambda", type=float, default=0.3,
                        help="Diversity weight for diverse_greedy batch selection "
                             "(0=pure quality, 1=pure diversity; default 0.3).")
    parser.add_argument("--bo-batch-size", type=int, default=BO_BATCH_SIZE,
                        help="Batch size for batch BO mode")
    parser.add_argument("--bo-iterations", type=int, default=BO_N_ITERATIONS,
                        help="Number of BO iterations")
    parser.add_argument("--bo-ablation", action="store_true",
                        help="Run full BO ablation study")
    parser.add_argument("--bo-ablation-n-seeds", type=int, default=3,
                        help="Number of seeds for --bo-ablation (default 3). "
                             "Use >=10 for a defensible paired Wilcoxon test "
                             "via bo_paired_compare.py.")
    parser.add_argument("--bo-ablation-methods", type=str, nargs="+", default=None,
                        choices=["lfbo", "random", "ei", "thompson",
                                 "consensus", "lfbo_ssl"],
                        help="Restrict --bo-ablation to a subset of acquisitions "
                             "(default: all six). Example: "
                             "`--bo-ablation-methods lfbo lfbo_ssl` for a focused "
                             "paired comparison.")
    parser.add_argument("--bo-ablation-surrogates", type=str, nargs="+", default=None,
                        choices=["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                                 "rf_cl_only", "xgb_cl_only"],
                        help="Restrict the surrogate sweep for surrogate-sensitive "
                             "acquisitions in --bo-ablation (default: all six).")
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
    parser.add_argument("--bo-eval-seeds", type=int, default=5,
                        help="Number of random seeds for --bo-mode evaluate / "
                             "loco-evaluate / learning-curve (default 5)")

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
    parser.add_argument("--bo-observed-pairs", action="store_true",
                        help="Restrict solvent search space to (sol1, sol2) pairs "
                             "observed together in training data. Default: all "
                             "combinations of individually-used solvents.")
    parser.add_argument("--bo-solvent-filter", type=str, default="permissive",
                        choices=["permissive", "strict", "off"],
                        help="How to restrict the candidate solvent pool when the "
                             "exact-chemistry recenter gate fires. "
                             "'permissive' (default): pure-single candidates must use "
                             "an anchor-success solvent, but binary mixtures pass if "
                             "at least one component is an anchor solvent — allows "
                             "co-solvent exploration (e.g., DMF/DCM if DMF anchored). "
                             "'strict': both components must be in the anchor set "
                             "(conservative — pure anchor solvents and their pairwise "
                             "mixtures only). "
                             "'off': no filter; all enumerated solvent pairs eligible.")

    args = parser.parse_args()

    # Auto-resolve batch strategy based on acquisition function
    if args.bo:
        from bo_core import resolve_batch_strategy
        args.bo_batch_strategy = resolve_batch_strategy(
            args.bo_acquisition, args.bo_batch_strategy
        )

    if args.bo:
        if args.bo_ablation:
            run_bo_ablation(args)
        else:
            run_bo(args)
    else:
        main(data_path=args.data, skip_tuning=args.skip_tuning)
