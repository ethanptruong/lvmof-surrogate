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

from config import (COLMAP, N_CLUSTERS, RANDOM_STATE, XGB_TUNED_KEYS,
                    BO_N_ITERATIONS, BO_BATCH_SIZE, BO_INIT_FRACTION,
                    BO_BORE_GAMMA, BO_CHECKPOINT_DIR, BO_DEFAULT_SURROGATE,
                    BO_DEFAULT_ACQUISITION, BO_CONTROLLABLE_PARAMS)
from data_processing import load_data, build_inventory, merge_data, run_process_variable_audit, fix_missingness
from feature_assembly import assemble_features
from dimensionality import (prepare_labels, remap_score, apply_variance_threshold,
                             build_umap_embedding, select_kmeans_groups,
                             run_mi_diagnostic, build_process_interactions,
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
                         plot_confusion_matrices, run_shap_analysis)
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

    Returns (X_cv, y, groups, cv_tune, cv_eval).
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
    mi_pre = run_mi_diagnostic(X_vt, y)
    Xprocnorm, interactions, _ = build_process_interactions(df_merged, mask, process_cols_present)
    X_for_umap = assemble_cv_matrix(mi_pre.transform(X_vt), Xprocnorm, interactions)
    X_2d = build_umap_embedding(X_for_umap)
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)
    groups, best_k, cv_tune, cv_eval = select_kmeans_groups(X_2d, y)

    return X_cv, y, groups, cv_tune, cv_eval


def main(data_path=None, skip_tuning=False):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Steps 1–4: data + features + CV setup ─────────────────────────────────

    ck = _load(DATA_CKPT)
    if ck is not None:
        X_cv, y, groups = ck["X_cv"], ck["y"], ck["groups"]
        cv_tune, cv_eval = ck["cv_tune"], ck["cv_eval"]
    else:
        print("\n── Featurizing from data file ──")
        X_cv, y, groups, cv_tune, cv_eval = _featurize_data(data_path)

        joblib.dump({"X_cv": X_cv, "y": y, "groups": groups,
                     "cv_tune": cv_tune, "cv_eval": cv_eval}, DATA_CKPT)
        print(f"[checkpoint] saved {DATA_CKPT}")

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
        ("RF  | CL only (supcon)", pipe_rf_cl_only,  1),
        ("XGB | CL only (supcon)", pipe_xgb_cl_only, 1),
    ]

    # 8. Evaluate
    print("\n─── FINAL COMPARISON ──────────────────────────────────")
    for name, pipe, n_jobs in pipelines:
        eval_pipe(name, pipe, X_cv, y, cv_eval, groups, scoring_ordinal, n_jobs=n_jobs)

    # 9. Plots
    plot_roc_prc(pipelines, X_cv, y, cv_eval, groups)
    plot_learning_curves(pipelines, X_cv, y, cv_eval, groups, scoring_ordinal)
    plot_confusion_matrices(pipelines, X_cv, y, cv_eval, groups)
    run_shap_analysis(
        [("XGB | MI only",          pipe_xgb_mi),
         ("XGB | CL + MI",          pipe_xgb_cl_mi),
         ("XGB | CL only (supcon)", pipe_xgb_cl_only),
         ("RF  | MI only",          pipe_rf_mi),
         ("RF  | CL + MI",          pipe_rf_cl_mi),
         ("RF  | CL only (supcon)", pipe_rf_cl_only)],
        X_cv, y
    )

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


def _resolve_surrogate(surrogate_name, params):
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
    from bo_core import RegressionSurrogate, XGBoostBootstrapEnsemble

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

    if is_rf:
        if cl_only:
            pipe = make_rf_regressor_pipe_cl_only(hp)
        else:
            pipe = make_rf_regressor_pipe(hp, with_cl=with_cl)
        return RegressionSurrogate(pipe, model_type="rf")
    else:
        if cl_only:
            pipe = make_xgb_regressor_pipe_cl_only(hp)
        else:
            pipe = make_xgb_regressor_pipe(hp, with_cl=with_cl)
        surr = RegressionSurrogate(pipe, model_type="xgb")
        surr.bootstrap_ensemble = XGBoostBootstrapEnsemble(hp)
        return surr


def run_bo(args):
    """Run Bayesian Optimization in the specified mode."""
    from bo_core import BOLoop, BOCheckpointer, RegressionSurrogate
    from bo_metrics import (SimulationMetrics, plot_convergence, plot_topk_curves,
                            plot_af_ef_comparison, save_simulation_results,
                            save_full_history)

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
    surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)

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
    )

    checkpointer = BOCheckpointer()

    if args.bo_mode == "simulate":
        print(f"\n── Simulation: {args.bo_acquisition} | {args.bo_surrogate} "
              f"| {args.bo_iterations} iters ──")
        history = bo.run_simulation(X_cv, y_raw, init_fraction=BO_INIT_FRACTION)
        metrics = SimulationMetrics(y_raw)
        summary = metrics.summary(history)

        print(f"\n── Results ──")
        print(f"  AF:          {summary['AF']:.2f}")
        print(f"  EF:          {summary['EF']:.2f}")
        print(f"  Top-5% found: {summary['Top_percent_final']*100:.1f}%")
        print(f"  Best score:   {summary['best_score_final']:.0f}")

        label = f"{args.bo_acquisition}_{args.bo_surrogate}"
        checkpointer.save(f"sim_{label}", history)
        save_full_history(history, label)
        plot_convergence([history], [label], y_raw)
        plot_topk_curves([history], [label], y_raw)

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
      4. Generates candidates, scores with acquisition function
      5. Outputs top recommendations
      6. Saves updated BO state

    The user's workflow:
      - Run recommend → get top conditions
      - Synthesize in the lab → add result row to Excel
      - Run recommend again → surrogate refits on updated data,
        acquisition function accounts for new observation
    """
    from bo_core import (BOLoop, BOCheckpointer, SearchSpace,
                         CandidateFeaturizer, _compute_acquisition)
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
        "recommendations": [],   # list of dicts per iteration
        "n_data_at_each_iter": [],
    }

    iteration = state["iteration"]
    print(f"  BO iteration:    {iteration} (previous recommendations: {iteration})")
    print(f"  Current dataset: {len(y_raw)} experiments")
    print(f"  Best score:      {y_raw.max():.0f}")
    print(f"  Surrogate:       {args.bo_surrogate}")
    print(f"  Acquisition:     {args.bo_acquisition}")

    # Show what happened since last iteration (new data added?)
    if state["n_data_at_each_iter"]:
        prev_n = state["n_data_at_each_iter"][-1]
        new_n = len(y_raw) - prev_n
        if new_n > 0:
            print(f"  New experiments since last run: {new_n}")
        elif new_n == 0:
            print("  No new experiments added since last run.")

    # 3. Build surrogate and fit on full dataset
    ck_params = _load(PARAMS_CKPT) or {}
    surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
    surrogate.fit(X_cv, y_raw)

    # 4. Generate candidates
    controllable = list(BO_CONTROLLABLE_PARAMS.keys())
    if args.bo_include_mlr:
        controllable += list(BO_OPTIONAL_PARAMS.keys())
    print(f"  Controllable params: {controllable}")

    extra_params = BO_OPTIONAL_PARAMS if args.bo_include_mlr else None
    search_space = SearchSpace(
        train_df=df_merged[mask], extra_params=extra_params
    )
    candidates = search_space.generate_lhs_candidates(
        seed=RANDOM_STATE + iteration  # different candidates each iteration
    )

    template_row = np.nanmedian(X_cv, axis=0)
    feature_columns = [f"f_{i}" for i in range(X_cv.shape[1])]
    featurizer = CandidateFeaturizer(template_row, feature_columns)
    X_candidates = featurizer.featurize(candidates)

    # 5. Score candidates
    mu, sigma = surrogate.predict(X_candidates)
    f_best = y_raw.max()

    acq_kwargs = {
        "f_best": f_best,
        "gamma": BO_BORE_GAMMA,
        "random_state": RANDOM_STATE + iteration,
    }
    acq_vals = _compute_acquisition(
        args.bo_acquisition, surrogate,
        X_cv, y_raw, X_candidates, **acq_kwargs
    )

    results = candidates.copy()
    results["pxrd_predicted"] = mu
    results["uncertainty"] = sigma
    results["acquisition_value"] = acq_vals
    results = results.sort_values("acquisition_value", ascending=False)

    # 6. Output
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

    # 7. Save updated state
    top_rec = results.head(args.bo_batch_size)[display_cols].to_dict("records")
    state["recommendations"].append({
        "iteration": iteration,
        "n_data": len(y_raw),
        "f_best": float(f_best),
        "top_candidates": top_rec,
    })
    state["n_data_at_each_iter"].append(len(y_raw))
    state["iteration"] = iteration + 1
    state["surrogate_name"] = args.bo_surrogate
    state["acquisition_name"] = args.bo_acquisition
    checkpointer.save("recommend_state", state)

    print(f"\n  State saved. Next: synthesize top candidates, add results to data file,")
    print(f"  then run again:  python main.py --bo --bo-mode recommend "
          f"--bo-surrogate {args.bo_surrogate}")


def run_bo_ablation(args):
    """Full ablation: acquisitions × surrogates × batch strategies × seeds."""
    from bo_core import BOLoop
    from bo_metrics import (SimulationMetrics, plot_convergence, plot_topk_curves,
                            plot_af_ef_comparison, save_simulation_results,
                            plot_batch_comparison, save_full_history)

    print("\n" + "=" * 70)
    print("  BO ABLATION STUDY")
    print("=" * 70)

    X_cv, y_raw, y_remapped, df_merged, mask = _load_bo_data(args.data)
    ck_params = _load(PARAMS_CKPT) or {}

    acquisitions = ["bore", "ei", "lcb", "pi_ordinal", "thompson", "random"]
    surrogates = ["rf_mi", "xgb_mi", "rf_cl_mi", "xgb_cl_mi",
                   "rf_cl_only", "xgb_cl_only"]
    batch_strategies = ["constant_liar", "kriging_believer"]
    seeds = [42, 123, 456]

    all_histories = []
    all_labels = []
    all_summaries = []

    # Sequential ablation
    for acq in acquisitions:
        for surr_name in surrogates:
            surrogate = _resolve_surrogate(surr_name, ck_params)

            classifier_pipeline = None
            if acq == "pi_ordinal":
                from models import make_rf_pipe
                rf_params = ck_params.get("best_rf_mi_params", {
                    "n_estimators": 300, "max_depth": 10,
                    "min_samples_split": 5, "min_samples_leaf": 3, "max_features": "sqrt"})
                classifier_pipeline = make_rf_pipe(rf_params, with_cl=False)
                classifier_pipeline.fit(X_cv, y_remapped)

            for seed in seeds:
                label = f"{acq}|{surr_name}|seed={seed}"
                print(f"\n── {label} ──")
                bo = BOLoop(
                    surrogate=surrogate,
                    acquisition_name=acq,
                    n_iterations=args.bo_iterations,
                    classifier_pipeline=classifier_pipeline,
                    random_state=seed,
                )
                history = bo.run_simulation(X_cv, y_raw)
                metrics = SimulationMetrics(y_raw)
                summary = metrics.summary(history)

                all_histories.append(history)
                all_labels.append(label)
                all_summaries.append((label, summary))
                print(f"  AF={summary['AF']:.2f}  EF={summary['EF']:.2f}  "
                      f"Top-5%={summary['Top_percent_final']*100:.1f}%")

    # Batch comparison (best acquisition only)
    print("\n── Batch strategy comparison ──")
    best_acq = max(
        [(l, s) for l, s in all_summaries if "seed=42" in l],
        key=lambda x: x[1]["AF"],
    )[0].split("|")[0]

    for strat in batch_strategies:
        surrogate = _resolve_surrogate(args.bo_surrogate, ck_params)
        bo = BOLoop(
            surrogate=surrogate,
            acquisition_name=best_acq,
            batch_strategy=strat,
            batch_size=args.bo_batch_size,
            n_iterations=args.bo_iterations,
            random_state=RANDOM_STATE,
        )
        history = bo.run_batch(X_cv, y_raw)
        label = f"batch_{best_acq}_{strat}"
        all_histories.append(history)
        all_labels.append(label)
        metrics = SimulationMetrics(y_raw)
        all_summaries.append((label, metrics.summary(history)))

    # Plots
    plot_convergence(all_histories, all_labels, y_raw,
                     save_path="docs/bo_ablation_convergence.png")
    plot_topk_curves(all_histories, all_labels, y_raw,
                     save_path="docs/bo_ablation_topk.png")
    plot_af_ef_comparison(all_summaries, save_path="docs/bo_ablation_af_ef.png")
    results_df = save_simulation_results(all_histories, all_labels, y_raw,
                                         save_path="docs/bo_ablation_results.csv")
    print(f"\n── Ablation complete. {len(all_summaries)} runs. ──")
    print(results_df.to_string(index=False))


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
                        choices=["bore", "ei", "lcb", "pi_ordinal", "thompson", "random"],
                        help="Acquisition function")
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
