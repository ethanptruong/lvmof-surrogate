"""
bo_featurize.py - Target-chemistry re-featurization for BO recommend mode.

When the user provides a novel target linker + precursor (and optionally
modulator) SMILES at BO recommend time, this module runs the full 12-block
featurization pipeline on those SMILES and projects the result into the same
X_cv feature space the surrogate was trained on.

This replaces the previous "nearest-neighbor template" shortcut, which fed
the surrogate a proxy chemistry instead of the user's actual target - fine
for in-distribution targets, but wrong when the goal is novel-chemistry
discovery.

Pipeline:
    target SMILES → 1-row df_merged (schema matches training)
                  → 1-row df_inventory (target ligands)
                  → assemble_features(...) → X_final_target  (1 × n_raw)
                  → transform_variance_threshold(..., fitted) → X_vt_target
                  → pad process+interaction slots with zeros
                  → return (1, n_X_cv_features) row

The process/interaction slots are intentionally zero - CandidateFeaturizer
overwrites them per-candidate based on the BO-proposed knobs, so any value
placed here is ignored downstream.
"""

import numpy as np
import pandas as pd

from config import COLMAP, PROCESS_COLS
from data_processing import build_inventory, fix_missingness
from dimensionality import (transform_variance_threshold,
                            transform_variance_threshold_no_kmeans)
from feature_assembly import assemble_features


# Reasonable defaults for process columns in the target row. These are
# placeholders - CandidateFeaturizer overwrites them per BO candidate.
# They only need to be numeric and non-NaN so assemble_features doesn't choke.
_PROCESS_DEFAULTS = {
    "equivalents":             1.0,
    "total_solvent_volume_ml": 5.0,
    "solvent_1_fraction":      1.0,
    "solvent_2_fraction":      0.0,
    "solvent_3_fraction":      0.0,
    "temperature_k":            333.0,
    "metal_over_linker_ratio":  1.0,
    "reaction_hours":           72.0,
    "reaction_hours_missing":   0.0,
    "temperature_k_missing":    0.0,
    "linker_conc":              0.01,
    "metal_conc":               0.01,   # = ratio(1) * linker_conc
    "mod_conc":                 0.01,   # = equiv(1) * linker_conc
    "total_conc":               0.03,   # = linker_conc * (1 + ratio + equiv)
    # COSMO Mix_* placeholders - also overwritten by CandidateFeaturizer.
    "Mix_M0_Area":      0.0,
    "Mix_M1_NetCharge": 0.0,
    "Mix_M2_Polarity":  0.0,
    "Mix_M3_Asymmetry": 0.0,
    "Mix_M4_Kurtosis":  0.0,
    "Mix_M_HB_Acc":     0.0,
    "Mix_M_HB_Don":     0.0,
    "Mix_f_nonpolar":   0.0,
    "Mix_f_acc":        0.0,
    "Mix_f_don":        0.0,
    "Mix_sigma_std":    0.0,
    "Mix_Vcosmo":       0.0,
    "Mix_lnPvap":       0.0,
}


def _build_target_df_merged(
    precursor_smiles: str,
    linker_smiles: str,
    modulator_smiles: str | None,
    df_merged_train: pd.DataFrame,
    df_inventory_train: pd.DataFrame,
):
    """Construct a 1-row df_merged + 1-row df_inventory for the target chemistry.

    The schema (column names and order) is inherited from training so
    assemble_features produces an X_final with the same column count.

    Returns
    ---
    df_target_merged    : 1-row DataFrame with training schema + target values
    df_target_inventory : 1-row DataFrame for build_halide_block
    warnings            : list[str] of chemistry-alignment issues worth surfacing
    """
    warnings: list[str] = []

    # -- 1. Template row from training --
    # Inherit all columns from the first training row. Overwrite the parts
    # that depend on target chemistry; leave everything else as-is.
    template = df_merged_train.iloc[0].copy()
    target_row = template.copy()

    # -- 2. Set target SMILES --
    target_row[COLMAP["id"]]        = "BO_TARGET_ROW"
    target_row[COLMAP["precursor"]] = precursor_smiles
    target_row[COLMAP["linker1"]]   = linker_smiles
    if COLMAP.get("linker2") in target_row.index:
        target_row[COLMAP["linker2"]] = None
    target_row[COLMAP["modulator"]] = modulator_smiles if modulator_smiles else None

    # -- 3. Parse metal_atom from target precursor --
    # build_inventory does this same parsing; we replicate a minimal version
    # so metal_atom is set before merge.
    from data_processing import deconstruct_precursor
    metal, _ = deconstruct_precursor(precursor_smiles)
    if metal is None:
        warnings.append(
            f"Could not parse metal atom from precursor SMILES '{precursor_smiles[:60]}' "
            f"- falling back to template row's metal ({target_row.get('metal_atom')})."
        )
    else:
        target_row["metal_atom"] = metal

    # -- 4. Zero out all Total_* inventory columns on the target row --
    # They get filled in step 5 from the target's own inventory.
    total_cols = [c for c in df_merged_train.columns if str(c).startswith("Total_")]
    for c in total_cols:
        target_row[c] = 0.0

    # -- 5. Build target's inventory from its SMILES --
    # build_inventory expects a df with experiment_id, smiles_precursor,
    # smiles_modulator, equivalents columns.
    mini_df = pd.DataFrame([{
        COLMAP["id"]:         "BO_TARGET_ROW",
        COLMAP["precursor"]:  precursor_smiles,
        COLMAP["modulator"]:  modulator_smiles if modulator_smiles else np.nan,
        "equivalents":        target_row.get("equivalents", 1.0),
    }])
    try:
        df_target_inventory = build_inventory(mini_df)
    except Exception as e:
        warnings.append(f"build_inventory failed on target SMILES: {e}")
        df_target_inventory = pd.DataFrame([{
            COLMAP["id"]: "BO_TARGET_ROW",
            "metal_atom": metal or target_row.get("metal_atom"),
        }])

    # Copy target's Total_* values onto the target row, restricted to columns
    # that also exist in training (schema alignment). Novel ligands not seen
    # at training time are silently dropped - this is a real limitation worth
    # surfacing to the user.
    target_inv_row = df_target_inventory.iloc[0]
    missing_ligands = []
    for c in df_target_inventory.columns:
        if not str(c).startswith("Total_"):
            continue
        val = float(target_inv_row[c]) if pd.notna(target_inv_row[c]) else 0.0
        if val <= 0:
            continue
        if c in total_cols:
            target_row[c] = val
        else:
            missing_ligands.append((c.replace("Total_", ""), val))

    if missing_ligands:
        ligand_summary = ", ".join(f"{lig}×{int(v)}" for lig, v in missing_ligands[:5])
        warnings.append(
            f"Target contains {len(missing_ligands)} ligand(s) not present in the "
            f"training set ({ligand_summary}{'...' if len(missing_ligands) > 5 else ''}). "
            f"These contribute no per-ligand features - the surrogate may "
            f"extrapolate more than usual for this target."
        )

    # -- 6. Also ensure df_target_inventory has the training halide columns --
    # build_halide_block inside assemble_features looks for Total_I / Total_Br /
    # Total_Cl variants and only sums those it finds. Pre-populating all halide
    # columns avoids KeyError on missing columns.
    for c in total_cols:
        if c not in df_target_inventory.columns:
            df_target_inventory[c] = 0.0

    # -- 7. Set process/solvent/COSMO defaults on the target row --
    for col, default_val in _PROCESS_DEFAULTS.items():
        if col in target_row.index:
            target_row[col] = default_val

    # Solvent string columns (if present) - set to the template's solvent_1
    # so featurization doesn't see NaN. Actual solvents used by the BO
    # candidate are set later by CandidateFeaturizer.
    for sol_col in ("solvent_1", "solvent_2", "solvent_3"):
        if sol_col in target_row.index and pd.isna(target_row[sol_col]):
            target_row[sol_col] = template.get(sol_col, "")

    # -- 8. pxrd_score is not needed for featurization; null out to signal
    # "no label" - it's never consumed past prepare_labels which we skip. --
    if "pxrd_score" in target_row.index:
        target_row["pxrd_score"] = np.nan

    # -- 9. Build the 1-row df_merged --
    df_target_merged = pd.DataFrame([target_row], columns=df_merged_train.columns)

    # fix_missingness adds missing-flag columns if they don't exist yet.
    # Because we inherited the template, they already exist; run it anyway
    # to be idempotent with training-time treatment.
    df_target_merged = fix_missingness(df_target_merged)

    return df_target_merged, df_target_inventory, warnings


def featurize_target_chemistry(
    precursor_smiles: str,
    linker_smiles: str,
    modulator_smiles: str | None,
    df_merged_train: pd.DataFrame,
    df_inventory_train: pd.DataFrame,
    fitted_vt: dict,
    n_X_cv_features: int,
):
    """Featurize the user's target chemistry and project into X_cv space.

    Parameters
    ---
    precursor_smiles    : str
    linker_smiles       : str
    modulator_smiles    : str or None
    df_merged_train     : training df_merged (provides column schema)
    df_inventory_train  : training df_inventory (halide block ligand set)
    fitted_vt           : dict from apply_variance_threshold (scaler/kmeans/ohe/vt)
    n_X_cv_features     : int - training X_cv.shape[1], for output sizing

    Returns
    ---
    template_row : np.ndarray  shape (n_X_cv_features,)
                   Target chemistry projected into X_cv feature space.
                   Process/interaction slots are zero-padded - CandidateFeaturizer
                   overwrites them per-candidate.
    warnings     : list[str]  chemistry-alignment issues worth surfacing
    """
    # 1. Build 1-row df_merged + df_inventory for the target.
    df_target_merged, df_target_inventory, warns = _build_target_df_merged(
        precursor_smiles, linker_smiles, modulator_smiles,
        df_merged_train, df_inventory_train,
    )

    # 2. Run the full 12-block featurization on the 1-row df.
    # assemble_features rebuilds fp_cols from df_merged.columns, so as long
    # as we preserved the training schema the output column count matches.
    (X_final_target, _df_out, _fp_cols, _num_desc, _calc,
     _linker_col, _mod_col, _process_cols_present,
     _X_process, _X_linker, _X_modulator, _mod_eq,
     _X_precursor_perlig, _X_inv_num) = assemble_features(
        df_target_merged, df_target_inventory
    )

    if X_final_target.shape[0] != 1:
        raise RuntimeError(
            f"Expected 1-row X_final_target, got shape {X_final_target.shape}"
        )

    # 3. Validate alignment with training's fitted VT.
    # v2 schema (fold-local KMeans) only persists {"vt": ...}; v1 also had
    # scaler/kmeans/ohe. Probe whichever is present to read the expected
    # raw-feature width.
    is_v2_no_kmeans = "scaler" not in fitted_vt
    if is_v2_no_kmeans:
        expected_n_raw = fitted_vt["vt"].n_features_in_
    else:
        expected_n_raw = fitted_vt["scaler"].n_features_in_
    got_n_raw      = X_final_target.shape[1]
    if got_n_raw != expected_n_raw:
        warns.append(
            f"Feature-count mismatch: target produced {got_n_raw} raw features, "
            f"training expected {expected_n_raw}. This usually means the target "
            f"introduces ligand columns not present at training time - predictions "
            f"should be treated as extrapolation."
        )
        # Pad or truncate to match training. This is defensive - in practice
        # we want the schemas to agree exactly.
        if got_n_raw < expected_n_raw:
            pad = np.zeros((1, expected_n_raw - got_n_raw), dtype=float)
            X_final_target = np.hstack([X_final_target, pad])
        else:
            X_final_target = X_final_target[:, :expected_n_raw]

    # 4. Project to X_vt space.
    # v2: VT only - fold-local KMeans inside the surrogate pipeline appends
    #     cluster-OHE at predict time, so the target row stays "raw" here.
    # v1 (legacy): StandardScaler → KMeans → OHE → VT using fitted transforms.
    if is_v2_no_kmeans:
        X_vt_target = transform_variance_threshold_no_kmeans(
            X_final_target, fitted_vt
        )
    else:
        X_vt_target = transform_variance_threshold(X_final_target, fitted_vt)

    # 5. Pad with zeros for the process-norm + interaction slots so the
    # final row matches n_X_cv_features. CandidateFeaturizer overwrites
    # these columns per candidate, so the zero values never reach the surrogate.
    pad_width = n_X_cv_features - X_vt_target.shape[1]
    if pad_width < 0:
        raise RuntimeError(
            f"X_vt_target has {X_vt_target.shape[1]} features but training "
            f"X_cv has only {n_X_cv_features}. Fitted-transformer mismatch."
        )
    if pad_width > 0:
        X_vt_target = np.hstack([X_vt_target, np.zeros((1, pad_width))])

    # Guard: finite values only.
    bad = ~np.isfinite(X_vt_target)
    if bad.any():
        X_vt_target[bad] = 0.0

    return X_vt_target[0], warns


def chemistry_similarity_to_training(
    template_row: np.ndarray,
    X_cv_train: np.ndarray,
    n_chem_features: int | None = None,
) -> float:
    """Cosine similarity between the target chemistry row and the nearest
    training row, restricted to chemistry features (everything before the
    process-norm + interaction slots).

    Low similarity (< ~0.3) suggests the surrogate is extrapolating on this
    target and BO recommendations should be treated as informed guesses.

    Parameters
    ---
    template_row     : np.ndarray  (n_X_cv_features,)  freshly featurized target
    X_cv_train       : np.ndarray  (n_train, n_X_cv_features)  training X_cv
    n_chem_features  : int or None - if None, uses all non-zero columns of
                       template_row (which effectively excludes the zero-
                       padded process slots).

    Returns
    ---
    max_cosine_sim : float  in [-1, 1], typically in [0, 1] for this feature set.
    """
    if n_chem_features is None:
        # Heuristic: chemistry columns are those where template_row is non-zero
        # and X_cv_train has variance. This matches the "pad process slots with
        # zero" convention above.
        nonzero_cols = np.nonzero(template_row)[0]
        if len(nonzero_cols) == 0:
            return 0.0
        cols = nonzero_cols
    else:
        cols = slice(0, n_chem_features)

    target_vec = template_row[cols].reshape(1, -1)
    train_mat  = X_cv_train[:, cols]

    # Cosine similarity
    tgt_norm   = np.linalg.norm(target_vec, axis=1, keepdims=True)
    train_norm = np.linalg.norm(train_mat,  axis=1, keepdims=True)
    denom = (tgt_norm * train_norm.T).clip(min=1e-12)
    sims = (target_vec @ train_mat.T) / denom

    return float(sims.max())
