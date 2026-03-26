"""
run_shap.py — Standalone SHAP analysis with labeled features.

Usage:
    python run_shap.py                  # run all 6 pipelines
    python run_shap.py --pipe "XGB | MI only"   # single pipeline

Loads data and hyperparams from checkpoints, rebuilds featurization
to get feature names, then calls run_shap_featurized() from evaluation.py.
Does NOT retrain Optuna or modify any checkpoints.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Ensure reproducibility
import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from config import (RANDOM_STATE, N_CLUSTERS, PROCESS_COLS, TARGET_METALS,
                    METAL_BLOCK_DIM, COLIGAND_BLOCK_DIM, COMPLEX_BLOCK_DIM)
from data_processing import load_data, build_inventory, merge_data, fix_missingness
from feature_assembly import (
    assemble_features, build_feature_catalog,
    build_chemberta_block, build_g14_features, build_ttp_features,
    build_linker_extra_features, build_halide_block, build_drfp_block,
    build_soap_block, build_mordred_rac_features, build_precursor_full_block,
    build_physicochem_features, build_tep_features, build_steric_features,
)
from dimensionality import (
    prepare_labels, remap_score, apply_variance_threshold,
    build_process_interactions, assemble_cv_matrix,
)
from models import (
    make_rf_pipe, make_xgb_pipe,
    make_rf_pipe_cl_only, make_xgb_pipe_cl_only,
)
from evaluation import run_shap_featurized


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DATA_CKPT      = os.path.join(CHECKPOINT_DIR, "data.pkl")
PARAMS_CKPT    = os.path.join(CHECKPOINT_DIR, "best_params.pkl")


def main():
    parser = argparse.ArgumentParser(description="Run SHAP analysis with labeled features")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to experiment data Excel file")
    parser.add_argument("--pipe", type=str, default=None,
                        help='Run only this pipeline (e.g. "XGB | MI only")')
    args = parser.parse_args()

    # ── Load hyperparams ──
    if not os.path.exists(PARAMS_CKPT):
        print("ERROR: No checkpoints/best_params.pkl found. Run the full pipeline first.")
        sys.exit(1)

    ck_params = joblib.load(PARAMS_CKPT)
    print(f"[shap] Loaded hyperparams from {PARAMS_CKPT}")

    # ── Featurize from source (we need intermediate arrays for name catalog) ──
    data_path = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "data", "Experiments_with_Calculated_Properties_no_linker.xlsx")
    print(f"[shap] Loading and featurizing data from {data_path}...")
    df = load_data(data_path)
    df_inventory = build_inventory(df)
    df_merged = merge_data(df, df_inventory)
    df_merged = fix_missingness(df_merged)

    (X_final, df_merged, fp_cols, num_descriptors, calc,
     linker_col, mod_col, process_cols_present, X_process,
     X_linker, X_modulator, mod_eq, X_precursor_perlig,
     Xinventorynumeric) = assemble_features(df_merged, df_inventory)

    X_raw, y_int, mask = prepare_labels(df_merged, X_final)
    y = np.array([remap_score(s) for s in y_int])

    X_vt, vt_pre = apply_variance_threshold(X_raw)
    Xprocnorm, interactions, _ = build_process_interactions(
        df_merged, mask, process_cols_present
    )
    X_cv = assemble_cv_matrix(X_vt, Xprocnorm, interactions)

    # ── Rebuild intermediate arrays needed for the catalog ──
    print("[shap] Rebuilding intermediate feature blocks for name catalog...")

    X_modulator_rac_aug, _, X_precursor_perlig_rac = \
        build_mordred_rac_features(df_merged, fp_cols, num_descriptors, calc)

    # Metal block (same as in assemble_features)
    from featurization import get_metal_descriptors, lookup_metal_descriptors
    metal_ohe_df = pd.get_dummies(
        df_merged['metal_atom'].fillna('Unknown'), prefix='metal_is'
    )
    for sym in TARGET_METALS:
        col = f'metal_is_{sym}'
        if col not in metal_ohe_df.columns:
            metal_ohe_df[col] = 0
    ohe_cols = sorted([c for c in metal_ohe_df.columns if c.startswith('metal_is_')])
    metal_ohe_df = metal_ohe_df[ohe_cols]
    X_metal_ohe = metal_ohe_df.values.astype(float)

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
    X_metal_block = np.hstack([X_metal, X_metal_ohe])

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

    # ── Build feature name catalog ──
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

    print(f"[shap] Feature catalog: {len(X_names)} names, X_cv has {X_cv.shape[1]} columns")
    if len(X_names) != X_cv.shape[1]:
        print(f"[WARN] Mismatch! Delta = {len(X_names) - X_cv.shape[1]}")
        # Pad or trim names to match
        if len(X_names) < X_cv.shape[1]:
            extra = X_cv.shape[1] - len(X_names)
            X_names += [f'unknown_{i}' for i in range(extra)]
            X_groups += ['Unknown'] * extra
        else:
            X_names = X_names[:X_cv.shape[1]]
            X_groups = X_groups[:X_cv.shape[1]]

    # ── Build pipelines ──
    pipe_defs = [
        ("XGB | MI only",          lambda p: make_xgb_pipe(p, with_cl=False),
         "best_xgb_mi_params"),
        ("XGB | CL + MI",          lambda p: make_xgb_pipe(p, with_cl=True),
         "best_xgb_cl_mi_params"),
        ("XGB | CL only (triplet)", lambda p: make_xgb_pipe_cl_only(p),
         "best_xgb_cl_only_params"),
        ("RF  | MI only",          lambda p: make_rf_pipe(p, with_cl=False),
         "best_rf_mi_params"),
        ("RF  | CL + MI",          lambda p: make_rf_pipe(p, with_cl=True),
         "best_rf_cl_mi_params"),
        ("RF  | CL only (triplet)", lambda p: make_rf_pipe_cl_only(p),
         "best_rf_cl_only_params"),
    ]

    for label, factory, param_key in pipe_defs:
        if args.pipe and args.pipe != label:
            continue

        hp = ck_params.get(param_key)
        if hp is None:
            print(f"[shap] Skipping {label}: no hyperparams for {param_key}")
            continue

        print(f"\n{'='*70}")
        print(f"  Running SHAP for: {label}")
        print(f"{'='*70}")

        pipe = factory(hp)
        run_shap_featurized(label, pipe, X_cv, y, X_names, X_groups, top_n=15)


if __name__ == "__main__":
    main()
