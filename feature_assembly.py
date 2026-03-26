"""
feature_assembly.py — Functions to build and assemble the full feature matrix
for the LVMOF-Surrogate pipeline.
"""

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from featurization import (
    get_metal_descriptors, get_precursor_geometry,
    get_physicochem_10, get_tepid_value,
    get_mordred_racs_smiles_with_stats,
    get_phosphine_sterics, process_for_sterics, map_sterics_processed,
    chemberta_batch, get_ext_rdkit, get_3d_shape, get_vsa_descriptors,
    get_composition, get_maccs, get_key_fragments,
    get_g14_hub_topology, get_g14_smarts_features,
    get_ttp_features,
    get_atom_pair_fp, get_torsion_fp, get_graph_topo_descriptors, get_estate_fp,
    get_metal_center_block, get_coligand_block, get_complex_level_block,
    morgan_fp_numpy, canonicalize_smiles_keep,
    normalize_inventory_token,
    calc, num_descriptors,
    _normalize_inventory_token, get_mordred_racs_smiles_with_stats,
    finite, _f,
    BERT_DIM, _N_TOTAL, _EXT_RDKIT_BASE_NAMES,
    embed_organic_3d, run_soap_block, SOAP_DIM,
    parse_oxidation_state, get_d_electron_count, get_cbc,
    make_rxn_smiles
)
from data_processing import clean_smiles
from config import (
    TARGET_METALS, GEOMETRY_LABELS, PROCESS_COLS,
    METAL_BLOCK_DIM, COLIGAND_BLOCK_DIM, COMPLEX_BLOCK_DIM,
    G14_HUB_NAMES, ALL_G14_SMARTS_NAMES,
    TTP_DIM, TTP_FEATURE_NAMES,
    SHAPE_3D_NAMES, VSA_NAMES, COMPOSITION_NAMES, MACCS_NAMES, FRAGMENT_NAMES,
    HALIDE_FEAT_COLS,
)


def build_metal_features(df_merged):
    """
    Build metal descriptor features (Block A: mendeleev + OHE).
    Returns (X_metal_block_aug, metal_names_aug).
    """
    from featurization import get_metal_descriptors, lookup_metal_descriptors

    if 'metal_atom' not in df_merged.columns:
        raise KeyError("df_merged must have 'metal_atom' column (from df_inventory merge).")

    metal_descriptor_cache = {
        sym: get_metal_descriptors(sym) for sym in TARGET_METALS
    }
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

    metal_ohe_df = pd.get_dummies(
        df_merged['metal_atom'].fillna('Unknown'),
        prefix='metal_is'
    )

    for sym in TARGET_METALS:
        col = f'metal_is_{sym}'
        if col not in metal_ohe_df.columns:
            metal_ohe_df[col] = 0

    ohe_cols = sorted([c for c in metal_ohe_df.columns if c.startswith('metal_is_')])
    metal_ohe_df = metal_ohe_df[ohe_cols]
    X_metal_ohe = metal_ohe_df.values.astype(float)
    metal_names = metal_descriptor_names + ohe_cols

    df_merged['precursor_ox_state'] = df_merged['precursor_iupac_standardized'].apply(parse_oxidation_state)
    df_merged['d_electron_count'] = df_merged.apply(
        lambda r: get_d_electron_count(r['metal_atom'], r['precursor_ox_state']),
        axis=1
    )

    df_merged['precursor_nL'], df_merged['precursor_nX'] = zip(
        *df_merged['precursor_iupac_standardized'].apply(get_cbc)
    )

    geom_rows = df_merged.apply(
        lambda r: get_precursor_geometry(
            r.get('smiles_precursor_canon', r.get('smiles_precursor', '')),
            r['metal_atom'],
            r['d_electron_count']
        )[0],
        axis=1
    )
    df_geom = pd.DataFrame(list(geom_rows))

    for _col in [f'precgeom_{g}' for g in GEOMETRY_LABELS]:
        if _col not in df_geom.columns:
            df_geom[_col] = 0.0
    df_geom = df_geom[[f'precgeom_{g}' for g in GEOMETRY_LABELS]]

    X_prec_geom = df_geom.values.astype(float)
    X_d_electron = df_merged[['d_electron_count', 'precursor_ox_state']].values.astype(float)
    Xcbc = df_merged[['precursor_nL', 'precursor_nX']].fillna(0).values.astype(float)

    _d_cov = (
        df_merged['d_electron_count'].fillna(0).values
        + df_merged['precursor_ox_state'].fillna(0).values
    )
    _tec_cov = (
        _d_cov
        + 2.0 * df_merged['precursor_nL'].fillna(0).values
        + 1.0 * df_merged['precursor_nX'].fillna(0).values
    )

    _tec_known = (
        (df_merged['precursor_nL'].fillna(0) + df_merged['precursor_nX'].fillna(0)) > 0
    ).astype(float).values

    Xtec = np.column_stack([
        _tec_cov,
        18.0 - _tec_cov,
        16.0 - _tec_cov,
        (_tec_cov == 18.0).astype(float),
        (_tec_cov == 16.0).astype(float),
        ((_tec_cov != 18.0) & (_tec_cov != 16.0)).astype(float),
        _tec_known,
    ])

    X_metal_block = np.hstack([X_metal, X_metal_ohe])

    X_metal_block_aug = np.hstack([
        X_metal_block,
        X_d_electron,
        X_prec_geom,
        Xcbc,
        Xtec,
    ])

    d_electron_names = ['metal_d_electron_count', 'metal_precursor_ox_state']
    prec_geom_names = [f'precgeom_{g}' for g in GEOMETRY_LABELS]
    cbc_names = ['precursor_nL', 'precursor_nX']
    tec_names = [
        'tec_covalent', 'tec_delta18', 'tec_delta16',
        'tec_is_18e', 'tec_is_16e', 'tec_is_other', 'tec_cbc_known'
    ]

    metal_names_aug = metal_names + d_electron_names + prec_geom_names + cbc_names + tec_names
    return X_metal_block_aug, metal_names_aug



def build_precursor_auxiliary_features(df_merged):
    """
    Build precursor metal center, co-ligand, and complex-level feature blocks.
    Returns (Xprecursor_full,).
    """
    precursor_smiles_col = next(
        (c for c in ['smiles_precursor_canon', 'smiles_precursor_raw', 'smiles_precursor']
         if c in df_merged.columns), None)
    iupac_col = 'precursor_iupac_standardized'

    if precursor_smiles_col is None:
        raise KeyError("No precursor SMILES column found in df_merged.")

    # Block A: Metal center
    Xmetal_center = np.stack(
        [get_metal_center_block(
            row[precursor_smiles_col],
            row[iupac_col] if iupac_col in df_merged.columns else '')
         for _, row in df_merged.iterrows()])

    # Block B: Co-ligand inventory
    Xcoligand = np.stack(
        df_merged[precursor_smiles_col].apply(get_coligand_block).values)

    # Block C: Complex-level
    Xcomplex_level = np.stack(
        df_merged[precursor_smiles_col].apply(get_complex_level_block).values)

    for name, arr in [('Metal center', Xmetal_center),
                      ('Co-ligand',    Xcoligand),
                      ('Complex-level', Xcomplex_level)]:
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = 0.0

    Xprecursor_full = np.concatenate(
        [Xmetal_center, Xcoligand, Xcomplex_level], axis=1)

    return Xprecursor_full


def build_fingerprint_features(df_merged):
    """
    Build fixed-component Morgan fingerprints for linker, modulator, and modulator equivalents.
    Also sets df_merged canonical/feat SMILES columns.
    Returns (X_linker, X_modulator, mod_eq, linker_col, mod_col).
    """
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")

    for col in ["smiles_linker_1", "smiles_linker_2", "smiles_modulator", "smiles_precursor"]:
        raw = f"{col}_raw"
        can = f"{col}_canon"
        if raw not in df_merged.columns:
            df_merged[raw] = df_merged[col]
        df_merged[can] = df_merged[col].apply(canonicalize_smiles_keep)

    def pick_featurizable_smiles(row, base_col):
        can = row.get(f"{base_col}_canon", None)
        raw = row.get(f"{base_col}_raw", None)
        fp_can, ok_can = morgan_fp_numpy(can)
        if ok_can:
            return can
        fp_raw, ok_raw = morgan_fp_numpy(raw)
        return raw if ok_raw else None

    df_merged["smiles_linker_1_feat"] = df_merged.apply(lambda r: pick_featurizable_smiles(r, "smiles_linker_1"), axis=1)
    df_merged["smiles_modulator_feat"] = df_merged.apply(lambda r: pick_featurizable_smiles(r, "smiles_modulator"), axis=1)

    X_linker = np.stack([morgan_fp_numpy(s)[0] for s in df_merged["smiles_linker_1_feat"].tolist()])
    X_modulator = np.stack([morgan_fp_numpy(s)[0] for s in df_merged["smiles_modulator_feat"].tolist()])

    if "equivalents" in df_merged.columns:
        mod_eq = pd.to_numeric(df_merged["equivalents"], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
    else:
        mod_eq = np.zeros((len(df_merged), 1), dtype=float)

    def _pick_first_existing(cols):
        for c in cols:
            if c in df_merged.columns:
                return c
        return None

    linker_col = _pick_first_existing(["smiles_linker_1_feat", "smiles_linker_1", "smileslinker1feat", "smileslinker1"])
    mod_col    = _pick_first_existing(["smiles_modulator_feat", "smiles_modulator", "smilesmodulatorfeat", "smilesmodulator"])

    return X_linker, X_modulator, mod_eq, linker_col, mod_col


def build_mordred_rac_features(df_merged, fp_cols, num_descriptors, calc):
    """
    Build Mordred RAC descriptors for modulator and per-ligand precursor blocks.
    Returns (X_modulator_rac_aug, X_precursor_lig_rac_aug, X_precursor_perlig_rac).
    """
    def _pick_first_existing(cols):
        for c in cols:
            if c in df_merged.columns:
                return c
        return None

    mod_smiles_col = _pick_first_existing([
        "smiles_modulator_feat", "smiles_modulator_canon", "smiles_modulator"
    ])
    if mod_smiles_col is None:
        raise KeyError("Could not find a modulator SMILES column in df_merged.")

    # Modulator RACs (cached per SMILES)
    from smiles_cache import get_smiles_cache
    _smi_cache = get_smiles_cache()
    NS_MORD = "mordred_rac_v1"

    mod_out = []
    for smi in df_merged[mod_smiles_col]:
        cached = _smi_cache.get(NS_MORD, smi)
        if cached is None:
            cached = get_mordred_racs_smiles_with_stats(smi)
            _smi_cache.set(NS_MORD, smi, cached)
        mod_out.append(cached)

    X_modulator_rac = np.stack([t[0] for t in mod_out])
    mod_rac_any_bad = np.array([t[1] for t in mod_out], dtype=float).reshape(-1, 1)
    mod_rac_frac_bad = np.array([t[2] for t in mod_out], dtype=float).reshape(-1, 1)

    X_modulator_rac_aug = np.concatenate([X_modulator_rac, mod_rac_any_bad, mod_rac_frac_bad], axis=1)

    # Precursor-ligand RACs: weighted sum
    inventory_cols = [c for c in df_merged.columns if c.startswith("Total_")]
    df_merged[inventory_cols] = df_merged[inventory_cols].fillna(0.0)

    mod_smiles_unique = (
        df_merged[mod_smiles_col].fillna("").astype(str).map(str.strip).unique()
    )
    mod_total_cols = set(f"Total_{s}" for s in mod_smiles_unique if s and s.lower() != "none")
    precursor_ligand_cols = [c for c in inventory_cols if c not in mod_total_cols]

    token_to_vec = {}
    token_to_any = {}
    token_to_frac = {}

    n = len(df_merged)
    X_precursor_lig_rac = np.zeros((n, num_descriptors), dtype=float)

    prec_rac_missing_any_wsum = np.zeros((n, 1), dtype=float)
    prec_rac_missing_frac_wsum = np.zeros((n, 1), dtype=float)
    prec_total_counts = np.zeros((n, 1), dtype=float)

    for col in precursor_ligand_cols:
        token = col.replace("Total_", "")
        smi = _normalize_inventory_token(token)
        if smi is None:
            continue

        counts = pd.to_numeric(df_merged[col], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
        if np.allclose(counts, 0.0):
            continue

        if smi not in token_to_vec:
            cached = _smi_cache.get(NS_MORD, smi)
            if cached is None:
                cached = get_mordred_racs_smiles_with_stats(smi)
                _smi_cache.set(NS_MORD, smi, cached)
            v, a, f = cached
            token_to_vec[smi] = v
            token_to_any[smi] = a
            token_to_frac[smi] = f

        vec = token_to_vec[smi]
        X_precursor_lig_rac += counts * vec.reshape(1, -1)

        prec_total_counts += counts
        prec_rac_missing_any_wsum += counts * token_to_any[smi]
        prec_rac_missing_frac_wsum += counts * token_to_frac[smi]

    prec_rac_missing_frac_wavg = prec_rac_missing_frac_wsum / np.clip(prec_total_counts, 1.0, None)

    X_precursor_lig_rac_aug = np.concatenate(
        [X_precursor_lig_rac, prec_rac_missing_any_wsum, prec_rac_missing_frac_wavg],
        axis=1
    )

    # Per-ligand RAC block (indexed by fp_cols)
    n_fp_lig = len(fp_cols)
    D = num_descriptors

    if n_fp_lig > 0:
        rac_cache = {}
        for col in fp_cols:
            token = col.replace('Total_', '')
            smi   = normalize_inventory_token(token)

            if smi is None:
                rac_cache[col] = (np.zeros(D, dtype=float), 1.0, 1.0)
                continue

            cached = _smi_cache.get(NS_MORD, smi)
            if cached is None:
                cached = get_mordred_racs_smiles_with_stats(smi)
                _smi_cache.set(NS_MORD, smi, cached)
            vec, any_bad, frac_bad = cached
            rac_cache[col] = (vec, float(any_bad), float(frac_bad))

        count_matrix_rac = np.column_stack([
            pd.to_numeric(df_merged[col], errors='coerce')
              .fillna(0.0).to_numpy()
            for col in fp_cols
        ])

        rac_matrix  = np.vstack([rac_cache[col][0] for col in fp_cols])
        any_bad_arr = np.array([rac_cache[col][1] for col in fp_cols], dtype=float)
        frac_bad_arr= np.array([rac_cache[col][2] for col in fp_cols], dtype=float)

        presence = (count_matrix_rac > 0).astype(float)

        rac_expanded     = presence[:, :, np.newaxis] * rac_matrix[np.newaxis, :, :]

        any_bad_expanded  = presence * any_bad_arr[np.newaxis, :]
        frac_bad_expanded = presence * frac_bad_arr[np.newaxis, :]
        count_expanded    = count_matrix_rac

        per_lig_rac_3d = np.concatenate(
            [rac_expanded,
             any_bad_expanded[:, :, np.newaxis],
             frac_bad_expanded[:, :, np.newaxis],
             count_expanded[:, :, np.newaxis]],
            axis=2
        )

        X_precursor_perlig_rac = per_lig_rac_3d.reshape(n, n_fp_lig * (D + 3))
    else:
        X_precursor_perlig_rac = np.zeros((n, 0), dtype=float)

    bad_rac = ~np.isfinite(X_precursor_perlig_rac)
    if bad_rac.any():
        X_precursor_perlig_rac[bad_rac] = 0.0

    return X_modulator_rac_aug, X_precursor_lig_rac_aug, X_precursor_perlig_rac


def build_precursor_full_block(df_merged):
    """
    Build the full precursor block (Blocks A+B+C from cell 16).
    Returns Xprecursor_full.
    """
    return build_precursor_auxiliary_features(df_merged)


def build_physicochem_features(df_merged, linker_col, mod_col):
    """
    Build physicochemical descriptors for linker and modulator.
    Returns (X_linker_phys10, X_modulator_phys10).
    """
    X_linker_phys10 = np.stack(df_merged[linker_col].apply(get_physicochem_10).values)
    X_modulator_phys10 = np.stack(df_merged[mod_col].apply(get_physicochem_10).values)
    return X_linker_phys10, X_modulator_phys10


def build_tep_features(df_merged, linker_col, mod_col, fp_cols):
    """
    Build TEP (Tolman Electronic Parameter) features for modulator, linker,
    and per-ligand precursor blocks.
    Returns (X_modulator_tep, X_linker_tep, X_precursor_perlig_tep).
    """
    from smiles_cache import get_smiles_cache
    _smi_cache = get_smiles_cache()
    NS_TEP = "tep_v1"

    def _tep_cached(smi):
        val = _smi_cache.get(NS_TEP, smi)
        if val is None:
            val = get_tepid_value(smi)
            _smi_cache.set(NS_TEP, smi, val)
        return val

    X_modulator_tep = np.stack(df_merged[mod_col].apply(_tep_cached).values)
    X_linker_tep = np.stack(df_merged[linker_col].apply(_tep_cached).values)

    n = len(df_merged)
    n_fp_lig = len(fp_cols)

    if n_fp_lig > 0:
        tep_cache = {}
        for col in fp_cols:
            token = col.replace('Total_', '')
            smi   = normalize_inventory_token(token)

            if smi is None:
                tep_cache[col] = (0.0, 1.0)
                continue

            cached = _smi_cache.get(NS_TEP, smi)
            if cached is None:
                cached = get_tepid_value(smi)
                _smi_cache.set(NS_TEP, smi, cached)
            tep_cache[col] = (float(cached[0]), float(cached[1]))

        count_matrix_tep = np.column_stack([
            pd.to_numeric(df_merged[col], errors='coerce')
              .fillna(0.0).to_numpy()
            for col in fp_cols
        ])

        tep_vals  = np.array([tep_cache[col][0] for col in fp_cols], dtype=float)
        miss_flags = np.array([tep_cache[col][1] for col in fp_cols], dtype=float)

        presence = (count_matrix_tep > 0).astype(float)

        tep_block    = presence * tep_vals[np.newaxis, :]
        miss_block   = presence * miss_flags[np.newaxis, :]
        count_block  = count_matrix_tep

        per_lig_tep_3d = np.stack(
            [tep_block, miss_block, count_block], axis=2
        )
        X_precursor_perlig_tep = per_lig_tep_3d.reshape(n, n_fp_lig * 3)
    else:
        X_precursor_perlig_tep = np.zeros((n, 0), dtype=float)

    bad_tep = ~np.isfinite(X_precursor_perlig_tep)
    if bad_tep.any():
        X_precursor_perlig_tep[bad_tep] = 0.0

    return X_modulator_tep, X_linker_tep, X_precursor_perlig_tep


def build_steric_features(df_merged, fp_cols):
    """
    Build steric features (cone angle, buried volume) for modulator, linker,
    and per-ligand precursor blocks.
    Returns (df_merged_with_sterics, X_precursor_perlig_steric).
    """
    from sklearn.impute import SimpleImputer

    map_sterics_processed(df_merged, "smiles_modulator", "modulator", "modulator_sterics_smiles")
    map_sterics_processed(df_merged, "smiles_linker_1", "linker1", "linker1_sterics_smiles", extra_remove={"Sn", "Si"})
    map_sterics_processed(df_merged, "smiles_precursor", "precursor", "precursor_sterics_smiles")

    steric_cols = [
        "modulator_cone_angle", "modulator_buried_vol",
        "linker1_cone_angle", "linker1_buried_vol",
    ]

    all_nan_cols = df_merged[steric_cols].columns[df_merged[steric_cols].isna().all()].tolist()
    if all_nan_cols:
        df_merged[all_nan_cols] = 0.0

    cols_to_impute = [c for c in steric_cols if df_merged[c].isna().any()]
    if cols_to_impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        df_merged[cols_to_impute] = imputer.fit_transform(df_merged[cols_to_impute])

    # Per-ligand sterics block
    n = len(df_merged)
    n_fp_lig = len(fp_cols)

    if n_fp_lig > 0:
        from smiles_cache import get_smiles_cache
        _smi_cache = get_smiles_cache()
        NS_STERIC = "steric_v1"

        steric_cache = {}

        for col in fp_cols:
            token = col.replace('Total_', '')
            smi   = normalize_inventory_token(token)

            if smi is None:
                steric_cache[col] = (0.0, 0.0, 0.0)
                continue

            disk_val = _smi_cache.get(NS_STERIC, smi)
            if disk_val is not None:
                steric_cache[col] = disk_val
                continue

            processed_smi = process_for_sterics(smi, extra_remove=None)

            if processed_smi is None or ('P' not in processed_smi and 'p' not in processed_smi):
                result = (0.0, 0.0, 0.0)
            else:
                cone, buried = get_phosphine_sterics(processed_smi)
                if np.isnan(cone) or np.isnan(buried):
                    result = (0.0, 0.0, 1.0)
                else:
                    result = (float(cone), float(buried), 0.0)

            _smi_cache.set(NS_STERIC, smi, result)
            steric_cache[col] = result

        count_matrix_steric = np.column_stack([
            pd.to_numeric(df_merged[col], errors='coerce')
              .fillna(0.0).to_numpy()
            for col in fp_cols
        ])

        cone_arr    = np.array([steric_cache[col][0] for col in fp_cols], dtype=float)
        buried_arr  = np.array([steric_cache[col][1] for col in fp_cols], dtype=float)
        missing_arr = np.array([steric_cache[col][2] for col in fp_cols], dtype=float)

        presence = (count_matrix_steric > 0).astype(float)

        cone_block    = presence * cone_arr[np.newaxis, :]
        buried_block  = presence * buried_arr[np.newaxis, :]
        missing_block = presence * missing_arr[np.newaxis, :]
        count_block   = count_matrix_steric

        per_lig_steric_3d = np.stack(
            [cone_block, buried_block, missing_block, count_block],
            axis=2
        )

        X_precursor_perlig_steric = per_lig_steric_3d.reshape(n, n_fp_lig * 4)
    else:
        X_precursor_perlig_steric = np.zeros((n, 0), dtype=float)

    bad_steric = ~np.isfinite(X_precursor_perlig_steric)
    if bad_steric.any():
        X_precursor_perlig_steric[bad_steric] = 0.0

    return df_merged, X_precursor_perlig_steric


def build_chemberta_block(df_merged, linker_col, mod_col):
    """
    Build ChemBERTa-2 + extended RDKit feature block for linker and modulator.
    Returns (X_chemberta_block, chemberta_feature_names_all).

    Per-SMILES results are cached in checkpoints/smiles_cache.pkl so that
    transformer inference is only run for SMILES not seen in previous runs.
    """
    from smiles_cache import get_smiles_cache
    cache = get_smiles_cache()
    NS = "chemberta_v1"

    _ROLES = [("linker", linker_col), ("mod", mod_col)]
    chemberta_feature_names_all = []
    blocks = []

    for role, col in _ROLES:
        smi_list = df_merged[col].tolist()
        n = len(smi_list)

        cached_rows = [cache.get(NS, s) for s in smi_list]
        new_indices = [i for i, r in enumerate(cached_rows) if r is None]
        new_smiles  = [smi_list[i] for i in new_indices]

        if new_smiles:
            print(f"  [smiles_cache] chemberta/{role}: "
                  f"{len(new_smiles)} new, {n - len(new_smiles)} cached")
            bert_n  = chemberta_batch(new_smiles)
            extrd_n = np.stack([get_ext_rdkit(s) for s in new_smiles])
            shape_n = np.stack([get_3d_shape(s) for s in new_smiles])
            vsa_n   = np.stack([get_vsa_descriptors(s) for s in new_smiles])
            comp_n  = np.stack([get_composition(s) for s in new_smiles])
            maccs_n = np.stack([get_maccs(s) for s in new_smiles])
            frags_n = np.stack([get_key_fragments(s) for s in new_smiles])

            block_n = np.hstack([bert_n, extrd_n, shape_n, vsa_n,
                                  comp_n, maccs_n, frags_n])
            block_n = np.where(np.isfinite(block_n), block_n, 0.0)

            for local_i, (orig_i, smi) in enumerate(zip(new_indices, new_smiles)):
                row = block_n[local_i]
                cache.set(NS, smi, row)
                cached_rows[orig_i] = row
        else:
            print(f"  [smiles_cache] chemberta/{role}: all {n} from cache")

        block = np.stack(cached_rows)
        blocks.append(block)

        role_names = (
            [f"{role}_bert_{i}" for i in range(BERT_DIM)]
            + [f"{role}_{n_}"   for n_ in _EXT_RDKIT_BASE_NAMES]
            + [f"{role}_{n_}"   for n_ in SHAPE_3D_NAMES]
            + [f"{role}_{n_}"   for n_ in VSA_NAMES]
            + [f"{role}_{n_}"   for n_ in COMPOSITION_NAMES]
            + [f"{role}_{n_}"   for n_ in MACCS_NAMES]
            + [f"{role}_{n_}"   for n_ in FRAGMENT_NAMES]
        )
        chemberta_feature_names_all.extend(role_names)

    X_chemberta_block = np.hstack(blocks)
    return X_chemberta_block, chemberta_feature_names_all


def build_g14_features(df_merged, linker_col, mod_col):
    """
    Build G14 hub topology and SMARTS features for linker and modulator.
    Returns (X_g14_block, g14_feature_names).
    """
    from smiles_cache import get_smiles_cache
    _smi_cache = get_smiles_cache()
    NS_HUB    = "g14_hub_v1"
    NS_SMARTS = "g14_smarts_v1"

    def _hub_cached(smi):
        val = _smi_cache.get(NS_HUB, smi)
        if val is None:
            val = get_g14_hub_topology(smi)
            _smi_cache.set(NS_HUB, smi, val)
        return val

    def _smarts_cached(smi):
        val = _smi_cache.get(NS_SMARTS, smi)
        if val is None:
            val = get_g14_smarts_features(smi)
            _smi_cache.set(NS_SMARTS, smi, val)
        return val

    X_linker_g14hub = np.stack(df_merged[linker_col].apply(_hub_cached).values)
    X_mod_g14hub    = np.stack(df_merged[mod_col].apply(_hub_cached).values)
    X_linker_g14s   = np.stack(df_merged[linker_col].apply(_smarts_cached).values)
    X_mod_g14s      = np.stack(df_merged[mod_col].apply(_smarts_cached).values)

    X_g14_block = np.concatenate(
        [X_linker_g14hub, X_mod_g14hub, X_linker_g14s, X_mod_g14s], axis=1
    )

    g14_feature_names = (
        [f'linker_{n}' for n in G14_HUB_NAMES] +
        [f'mod_{n}'    for n in G14_HUB_NAMES] +
        [f'linker_{n}' for n in ALL_G14_SMARTS_NAMES] +
        [f'mod_{n}'    for n in ALL_G14_SMARTS_NAMES]
    )

    bad = ~np.isfinite(X_g14_block)
    if bad.any():
        X_g14_block[bad] = 0.0

    return X_g14_block, g14_feature_names


def build_ttp_features(df_merged, linker_col):
    """
    Build Tetratopic Phosphine (TTP) topology features for linker.
    Returns (Xlinker_ttp, ttp_feature_names).
    """
    _linker_col = next(
        (c for c in ['smiles_linker_1_feat', 'smiles_linker_1_canon', 'smiles_linker_1']
         if c in df_merged.columns), None
    )
    if _linker_col is None:
        _linker_col = linker_col

    from smiles_cache import get_smiles_cache
    _smi_cache = get_smiles_cache()
    NS_TTP = "ttp_v1"

    def _ttp_cached(smi):
        val = _smi_cache.get(NS_TTP, smi)
        if val is None:
            val = get_ttp_features(smi)
            _smi_cache.set(NS_TTP, smi, val)
        return val

    Xlinker_ttp = np.stack(df_merged[_linker_col].apply(_ttp_cached).values)

    bad = ~np.isfinite(Xlinker_ttp)
    if bad.any():
        Xlinker_ttp[bad] = 0.0

    return Xlinker_ttp, TTP_FEATURE_NAMES


def build_linker_extra_features(df_merged, linker_col):
    """
    Build extra linker features: EState FP, graph topological descriptors,
    topological torsion FP, and atom-pair FP.
    Returns (X_linker_extra, linker_extra_names).
    """
    _linker_col = next(
        (c for c in ['smiles_linker_1_feat', 'smiles_linker_1_canon', 'smiles_linker_1']
         if c in df_merged.columns), None
    )
    if _linker_col is None:
        _linker_col = linker_col

    Xlinker_topo = np.stack(df_merged[_linker_col].apply(get_graph_topo_descriptors).values)
    Xlinker_tor = np.stack(df_merged[_linker_col].apply(get_torsion_fp).values)
    Xlinker_ap = np.stack(df_merged[_linker_col].apply(get_atom_pair_fp).values)
    Xlinker_estate = np.stack(df_merged[_linker_col].apply(get_estate_fp).values)

    X_linker_extra = np.concatenate([Xlinker_estate, Xlinker_topo, Xlinker_tor, Xlinker_ap], axis=1)

    bad = ~np.isfinite(X_linker_extra)
    if bad.any():
        X_linker_extra[bad] = 0.0

    return X_linker_extra


def build_halide_block(df_merged, df_inventory):
    """
    Build halide feature block from inventory.
    Returns Xhalide_full.
    """
    I_cols  = ['Total_I', 'Total_[I]',  'Total_[I-]']
    Br_cols = ['Total_Br', 'Total_[Br-]']
    Cl_cols = ['Total_Cl', 'Total_[Cl-]']

    I_cols  = [c for c in I_cols  if c in df_inventory.columns]
    Br_cols = [c for c in Br_cols if c in df_inventory.columns]
    Cl_cols = [c for c in Cl_cols if c in df_inventory.columns]

    df_inventory.loc[:, 'halide_I_count']  = df_inventory[I_cols].sum(axis=1)  if I_cols  else 0.0
    df_inventory.loc[:, 'halide_Br_count'] = df_inventory[Br_cols].sum(axis=1) if Br_cols else 0.0
    df_inventory.loc[:, 'halide_Cl_count'] = df_inventory[Cl_cols].sum(axis=1) if Cl_cols else 0.0
    df_inventory.loc[:, 'halide_count']   = (df_inventory['halide_I_count'] +
                                              df_inventory['halide_Br_count'] +
                                              df_inventory['halide_Cl_count'])
    df_inventory.loc[:, 'halide_present'] = (df_inventory['halide_count'] > 0).astype(float)

    def _halide_type(row):
        if row['halide_I_count']  > 0: return 1.0
        if row['halide_Br_count'] > 0: return 2.0
        if row['halide_Cl_count'] > 0: return 3.0
        return 0.0

    df_inventory.loc[:, 'halide_type'] = df_inventory.apply(_halide_type, axis=1)

    Xhalide_full = (
        pd.DataFrame({'experiment_id': df_merged['experiment_id'].values})
        .merge(df_inventory[['experiment_id'] + HALIDE_FEAT_COLS],
               on='experiment_id', how='left')
        .fillna(0.0)[HALIDE_FEAT_COLS]
        .values.astype(float)
    )

    return Xhalide_full


def build_drfp_block(df_merged):
    """
    Build DRFP (Differential Reaction Fingerprint) block.
    Returns (X_drfp, drfp_feature_names).

    Results are cached per reaction-SMILES string so DrfpEncoder.encode is
    only called for reaction combinations not seen in previous runs.
    """
    from smiles_cache import get_smiles_cache
    cache = get_smiles_cache()
    NS = "drfp_v1"

    def make_rxn_smiles(row):
        parts = []
        for col in ['smiles_precursor', 'smiles_linker1', 'smiles_modulator']:
            s_can = row.get(f'{col}_canon')
            s_raw = row.get(col)
            val = s_can if pd.notna(s_can) and str(s_can).strip() else s_raw
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return '.'.join(parts) + '>>' if parts else '>>'

    rxn_smiles_list = df_merged.apply(make_rxn_smiles, axis=1).tolist()
    n = len(rxn_smiles_list)

    cached_rows = [cache.get(NS, rxn) for rxn in rxn_smiles_list]
    new_indices = [i for i, r in enumerate(cached_rows) if r is None]
    new_rxns    = [rxn_smiles_list[i] for i in new_indices]

    if new_rxns:
        print(f"  [smiles_cache] drfp: {len(new_rxns)} new, {n - len(new_rxns)} cached")
        new_fps = np.array(DrfpEncoder.encode(
            new_rxns, n_folded_length=2048, radius=3, rings=True
        ), dtype=np.float32)
        for local_i, (orig_i, rxn) in enumerate(zip(new_indices, new_rxns)):
            cache.set(NS, rxn, new_fps[local_i])
            cached_rows[orig_i] = new_fps[local_i]
    else:
        print(f"  [smiles_cache] drfp: all {n} from cache")

    X_drfp = np.stack(cached_rows).astype(np.float32)

    assert X_drfp.shape == (n, 2048), f"Unexpected DRFP shape: {X_drfp.shape}"
    assert np.isfinite(X_drfp).all(), "Non-finite values in X_drfp"

    drfp_feature_names = [f'drfp_{i}' for i in range(2048)]
    return X_drfp, drfp_feature_names


def _soap_with_cache(role: str, smiles_series: "pd.Series") -> tuple:
    """Run run_soap_block, skipping SMILES already in the persistent cache.

    Returns (X_soap, feature_names) with the same shape as the full series.
    """
    from smiles_cache import get_smiles_cache
    cache = get_smiles_cache()
    NS = f"soap_{role}_v1"
    NS_NAMES = f"soap_{role}_names_v1"

    smi_list = smiles_series.tolist()
    n = len(smi_list)

    cached_rows = [cache.get(NS, s) for s in smi_list]
    new_indices = [i for i, r in enumerate(cached_rows) if r is None]

    feature_names = cache.get(NS_NAMES, "__names__") or []

    if new_indices:
        new_series = pd.Series([smi_list[i] for i in new_indices])
        print(f"  [smiles_cache] soap/{role}: "
              f"{len(new_indices)} new, {n - len(new_indices)} cached")
        new_matrix, feature_names = run_soap_block(role, new_series)
        cache.set(NS_NAMES, "__names__", feature_names)
        for local_i, orig_i in enumerate(new_indices):
            cache.set(NS, smi_list[orig_i], new_matrix[local_i])
            cached_rows[orig_i] = new_matrix[local_i]
    else:
        print(f"  [smiles_cache] soap/{role}: all {n} from cache")

    return np.stack(cached_rows), feature_names


def build_soap_block(df_merged, linker_col):
    """
    Build SOAP (Smooth Overlap of Atomic Positions) 3D descriptor block
    for precursor sterics and linker.
    Returns (X_soap_precursor, X_soap_linker, soap_all_names).

    Per-SMILES SOAP vectors are cached in checkpoints/smiles_cache.pkl.
    """
    slatm_src_col = 'precursor_sterics_smiles'
    X_soap_precursor, soap_precursor_names = _soap_with_cache(
        'precursor', df_merged[slatm_src_col]
    )
    X_soap_linker, soap_linker_names = _soap_with_cache(
        'linker', df_merged[linker_col]
    )
    soap_all_names = soap_precursor_names + soap_linker_names
    return X_soap_precursor, X_soap_linker, soap_all_names


def assemble_features(df_merged, df_inventory):
    """
    Master assembly function: calls all build_* functions in order,
    assembles and returns the full feature matrix X_final.

    Returns a tuple:
        (X_final, df_merged, fp_cols, num_descriptors, calc,
         linker_col, mod_col, process_cols_present, X_process,
         X_linker, X_modulator, mod_eq, X_precursor_perlig, Xinventorynumeric)
    """
    # ── 1. Fingerprint features + col selection ────────────────────────────
    X_linker, X_modulator, mod_eq, linker_col, mod_col = build_fingerprint_features(df_merged)

    # ── 2. Per-ligand fingerprint block setup ─────────────────────────────
    inventory_cols = [c for c in df_merged.columns if c.startswith('Total_')]
    df_merged[inventory_cols] = df_merged[inventory_cols].fillna(0)

    precursor_ligand_cols = inventory_cols

    FP_NBITS  = 2048
    fp_cols   = []
    num_cols  = []
    fp_cache  = {}

    for col in precursor_ligand_cols:
        token     = col.replace('Total_', '')
        smi_token = normalize_inventory_token(token)

        if smi_token is None:
            num_cols.append(col)
            continue

        fp_vec, ok = morgan_fp_numpy(smi_token, n_bits=FP_NBITS)
        if not ok:
            num_cols.append(col)
            continue

        fp_cache[col] = fp_vec
        fp_cols.append(col)

    n = len(df_merged)
    n_fp_lig = len(fp_cols)

    if n_fp_lig > 0:
        fp_matrix = np.vstack([fp_cache[col] for col in fp_cols])

        count_matrix = np.column_stack([
            pd.to_numeric(df_merged[col], errors='coerce').fillna(0.0).to_numpy()
            for col in fp_cols
        ])

        fp_presence = (count_matrix > 0).astype(float)
        fp_expanded = fp_presence[:, :, np.newaxis] * fp_matrix[np.newaxis, :, :]

        count_expanded = count_matrix[:, :, np.newaxis]
        per_ligand_block = np.concatenate(
            [fp_expanded, count_expanded], axis=2
        )

        X_precursor_perlig = per_ligand_block.reshape(n, n_fp_lig * (FP_NBITS + 1))
    else:
        X_precursor_perlig = np.zeros((n, 0), dtype=float)

    if len(num_cols) > 0:
        Xinventorynumeric = np.column_stack([
            pd.to_numeric(df_merged[col], errors='coerce').fillna(0.0).to_numpy()
            for col in num_cols
        ])
    else:
        Xinventorynumeric = np.zeros((n, 0), dtype=float)

    bad = ~np.isfinite(X_precursor_perlig)
    if bad.any():
        X_precursor_perlig[bad] = 0.0

    # ── 3. Process variables ──────────────────────────────────────────────
    from config import PROCESS_COLS
    process_cols = PROCESS_COLS
    process_cols_present = [c for c in process_cols if c in df_merged.columns]
    X_process = df_merged[process_cols_present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # ── 4. Initial X_features assembly ───────────────────────────────────
    X_features = np.concatenate(
        [X_linker, X_modulator, mod_eq,
         X_precursor_perlig,
         Xinventorynumeric,
         X_process],
        axis=1
    )

    # ── 5. Mordred RAC features ───────────────────────────────────────────
    X_modulator_rac_aug, X_precursor_lig_rac_aug, X_precursor_perlig_rac = \
        build_mordred_rac_features(df_merged, fp_cols, num_descriptors, calc)

    # ── 6. Initial X_final with metal block ───────────────────────────────
    X_metal_block_aug, metal_names_aug = build_metal_features(df_merged)

    # Keep X_metal_block (just continuous + OHE, without aug extras) for older concat
    from featurization import get_metal_descriptors, lookup_metal_descriptors

    metal_ohe_df = pd.get_dummies(
        df_merged['metal_atom'].fillna('Unknown'),
        prefix='metal_is'
    )
    for sym in TARGET_METALS:
        col = f'metal_is_{sym}'
        if col not in metal_ohe_df.columns:
            metal_ohe_df[col] = 0
    ohe_cols = sorted([c for c in metal_ohe_df.columns if c.startswith('metal_is_')])
    metal_ohe_df = metal_ohe_df[ohe_cols]
    X_metal_ohe = metal_ohe_df.values.astype(float)

    metal_descriptor_cache = {
    sym: get_metal_descriptors(sym) for sym in TARGET_METALS
    }
    zero_descriptor = get_metal_descriptors("XYZ")
    metal_descriptor_names = list(next(iter(metal_descriptor_cache.values())).keys())

    metal_ohe_df = pd.get_dummies(
        df_merged['metal_atom'].fillna('Unknown'),
        prefix='metal_is'
    )
    for sym in TARGET_METALS:
        col = f'metal_is_{sym}'
        if col not in metal_ohe_df.columns:
            metal_ohe_df[col] = 0
    ohe_cols = sorted([c for c in metal_ohe_df.columns if c.startswith('metal_is_')])
    metal_ohe_df = metal_ohe_df[ohe_cols]
    X_metal_ohe = metal_ohe_df.values.astype(float)

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

    X_final = np.concatenate([
        X_features,
        X_modulator_rac_aug,
        X_metal_block
    ], axis=1)

    # ── 7. Precursor full block ───────────────────────────────────────────
    Xprecursor_full = build_precursor_full_block(df_merged)
    X_final = np.concatenate([X_final, Xprecursor_full], axis=1)

    # ── 8. Per-ligand RAC ─────────────────────────────────────────────────
    X_final = np.concatenate([X_final, X_precursor_perlig_rac], axis=1)

    # ── 9. Physicochemical features ───────────────────────────────────────
    X_linker_phys10, X_modulator_phys10 = build_physicochem_features(df_merged, linker_col, mod_col)
    X_final = np.concatenate([X_final, X_linker_phys10, X_modulator_phys10], axis=1)

    # ── 10. TEP features ──────────────────────────────────────────────────
    X_modulator_tep, X_linker_tep, X_precursor_perlig_tep = \
        build_tep_features(df_merged, linker_col, mod_col, fp_cols)
    X_final = np.concatenate(
        [X_final, X_modulator_tep, X_linker_tep, X_precursor_perlig_tep],
        axis=1
    )

    # ── 11. Steric features ───────────────────────────────────────────────
    df_merged, X_precursor_perlig_steric = build_steric_features(df_merged, fp_cols)
    X_final = np.concatenate([X_final, X_precursor_perlig_steric], axis=1)

    # ── 12. ChemBERTa block ───────────────────────────────────────────────
    X_chemberta_block, _ = build_chemberta_block(df_merged, linker_col, mod_col)
    X_final = np.concatenate([X_final, X_chemberta_block], axis=1)

    # ── 13. G14 features ──────────────────────────────────────────────────
    X_g14_block, _ = build_g14_features(df_merged, linker_col, mod_col)
    X_final = np.concatenate([X_final, X_g14_block], axis=1)

    # ── 14. TTP features ──────────────────────────────────────────────────
    Xlinker_ttp, _ = build_ttp_features(df_merged, linker_col)
    X_final = np.concatenate([X_final, Xlinker_ttp], axis=1)

    # ── 15. Linker extra features ─────────────────────────────────────────
    X_linker_extra = build_linker_extra_features(df_merged, linker_col)
    X_final = np.concatenate([X_final, X_linker_extra], axis=1)

    # ── 16. Halide block ──────────────────────────────────────────────────
    Xhalide_full = build_halide_block(df_merged, df_inventory)
    X_final = np.concatenate([X_final, Xhalide_full], axis=1)

    # ── 17. DRFP block ────────────────────────────────────────────────────
    X_drfp, _ = build_drfp_block(df_merged)
    X_final = np.concatenate([X_final, X_drfp], axis=1)

    # ── 18. SOAP block ────────────────────────────────────────────────────
    X_soap_precursor, X_soap_linker, _ = build_soap_block(df_merged, linker_col)
    X_final = np.concatenate([X_final, X_soap_precursor, X_soap_linker], axis=1)

    # ── Final finiteness check ────────────────────────────────────────────
    bad = ~np.isfinite(X_final)
    if bad.any():
        X_final[bad] = 0.0

    print(f"X_final shape: {X_final.shape}")

    return (X_final, df_merged, fp_cols, num_descriptors, calc,
            linker_col, mod_col, process_cols_present, X_process,
            X_linker, X_modulator, mod_eq, X_precursor_perlig, Xinventorynumeric)


# ─────────────────────────────────────────────────────────────
# Feature name / group catalog for SHAP
# ─────────────────────────────────────────────────────────────

_PHYSCHEM_NAMES = [
    'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds',
    'NumHAcceptors', 'NumHDonors', 'MaxPartialCharge_clean',
    'HallKierAlpha', 'NumAromaticRings', 'MaxPartialCharge_missing',
]

_METAL_DESC_NAMES = [
    'metal_atomic_number', 'metal_period', 'metal_group',
    'metal_en_pauling', 'metal_en_allen',
    'metal_covalent_radius_pm', 'metal_atomic_radius_pm',
    'metal_vdw_radius_pm',
    'metal_ionization_energy1_eV', 'metal_ionization_energy2_eV',
    'metal_electron_affinity_eV', 'metal_nvalence',
    'metal_max_oxidation_state', 'metal_min_oxidation_state',
    'metal_common_oxidation_states_count', 'metal_mendeleev_number',
    'metal_hardness_eV', 'metal_dipole_polarizability_bohr3',
    'metal_missing_flag',
]

_TOPO_NAMES = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi1', 'Chi2n', 'Chi3n',
    'Kappa1', 'Kappa2', 'Kappa3', 'WienerIndex',
    'Fsp3', 'FractionCSP3', 'LabuteASA', 'topo_missing',
]

_INTERACTION_NAMES = [
    'proc_int:temp_x_metal_ratio',
    'proc_int:temp_x_rxn_hours',
    'proc_int:metal_ratio_x_rxn_hours',
    'proc_int:temp_sq',
    'proc_int:metal_ratio_sq',
    'proc_int:hightemp_flag',
]


def build_feature_catalog(
    X_final, X_linker, X_modulator, mod_eq,
    X_precursor_perlig, Xinventorynumeric, X_process,
    fp_cols, num_descriptors, ohe_cols,
    process_cols_present, n_clusters,
    X_modulator_rac_aug, X_metal_block,
    Xprecursor_full, X_precursor_perlig_rac,
    X_linker_phys10, X_modulator_phys10,
    X_modulator_tep, X_linker_tep, X_precursor_perlig_tep,
    X_precursor_perlig_steric,
    X_chemberta_block, chemberta_names,
    X_g14_block, g14_names,
    Xlinker_ttp,
    X_linker_extra,
    Xhalide_full,
    X_drfp,
    X_soap_precursor, X_soap_linker, soap_names,
    vt_mask,
):
    """Build feature name and group arrays for the full X_cv matrix.

    Parameters correspond to the arrays produced during assemble_features()
    and the subsequent dimensionality reduction steps.

    Returns
    -------
    names  : list[str]  — feature names for X_cv columns
    groups : list[str]  — feature group labels for X_cv columns
    """
    names = []
    groups = []

    def _push(n_list, g_label):
        names.extend(n_list)
        groups.extend([g_label] * len(n_list))

    n_fp_lig = len(fp_cols)

    # ── X_features block (concat step 4 in assemble_features) ──
    # 1. Linker Morgan FP
    _push([f'linker_fp_{i:04d}' for i in range(X_linker.shape[1])],
          "Linker Morgan FP")
    # 2. Modulator Morgan FP
    _push([f'mod_fp_{i:04d}' for i in range(X_modulator.shape[1])],
          "Modulator Morgan FP")
    # 3. Modulator equivalents
    _push(['mod_equivalents'], "Modulator Equiv.")
    # 4. Per-ligand precursor FP
    for j, col in enumerate(fp_cols):
        token = col.replace('Total_', '')
        _push([f'perlig_{token}_fp_{i:04d}' for i in range(2048)] + [f'perlig_{token}_count'],
              "Precursor Ligand FP")
    # 5. Inventory numeric
    _push([f'inv_num_{i}' for i in range(Xinventorynumeric.shape[1])],
          "Inventory Numeric")
    # 6. Process variables (raw, in X_features)
    _push([f'proc_raw:{c}' for c in process_cols_present],
          "Process Variables (raw)")

    # ── Mordred RAC block ──
    _push([f'mod_rac_{i}' for i in range(num_descriptors)]
          + ['mod_rac_any_bad', 'mod_rac_frac_bad'],
          "Modulator RAC")

    # ── Metal block (continuous descriptors + OHE) ──
    _push(_METAL_DESC_NAMES, "Metal Center (mendeleev)")
    _push(list(ohe_cols), "Metal Center (mendeleev)")

    # ── Precursor full block (metal center + coligand + complex) ──
    _push([f'prec_metal_center_{i}' for i in range(METAL_BLOCK_DIM)],
          "Metal Precursor Complex")
    _push([f'prec_coligand_{i}' for i in range(COLIGAND_BLOCK_DIM)],
          "Metal Precursor Complex")
    _push([f'prec_complex_{i}' for i in range(COMPLEX_BLOCK_DIM)],
          "Metal Precursor Complex")

    # ── Per-ligand RAC ──
    D = num_descriptors
    for j, col in enumerate(fp_cols):
        token = col.replace('Total_', '')
        _push([f'perlig_{token}_rac_{i}' for i in range(D)]
              + [f'perlig_{token}_rac_anybad', f'perlig_{token}_rac_fracbad',
                 f'perlig_{token}_rac_count'],
              "Precursor Ligand RAC")

    # ── Physicochem ──
    _push([f'linker_{n}' for n in _PHYSCHEM_NAMES], "Linker Physchem/FP")
    _push([f'mod_{n}' for n in _PHYSCHEM_NAMES], "Mod Physchem/FP")

    # ── TEP ──
    _push(['mod_tep_val', 'mod_tep_missing'], "Ligand TEP (Electronic)")
    _push(['linker_tep_val', 'linker_tep_missing'], "Ligand TEP (Electronic)")
    for j, col in enumerate(fp_cols):
        token = col.replace('Total_', '')
        _push([f'perlig_{token}_tep_val', f'perlig_{token}_tep_miss',
               f'perlig_{token}_tep_count'],
              "Ligand TEP (Electronic)")

    # ── Steric ──
    for j, col in enumerate(fp_cols):
        token = col.replace('Total_', '')
        _push([f'perlig_{token}_cone_angle', f'perlig_{token}_buried_vol',
               f'perlig_{token}_steric_miss', f'perlig_{token}_steric_count'],
              "Ligand Sterics")

    # ── ChemBERTa block ──
    # chemberta_names already has per-role names from build_chemberta_block()
    # Split into groups: linker bert / linker physchem-fp / mod bert / mod physchem-fp
    per_mol = len(chemberta_names) // 2
    for i, n in enumerate(chemberta_names[:per_mol]):
        names.append(n)
        groups.append("Linker ChemBERT" if i < BERT_DIM else "Linker Physchem/FP")
    for i, n in enumerate(chemberta_names[per_mol:]):
        names.append(n)
        groups.append("Mod ChemBERT" if i < BERT_DIM else "Mod Physchem/FP")

    # ── G14 ──
    n_hub = len(G14_HUB_NAMES)
    n_smarts = len(ALL_G14_SMARTS_NAMES)
    for i, n in enumerate(g14_names):
        names.append(n)
        # First 2*n_hub are hub features, rest are SMARTS
        groups.append("G14 Hub Topology" if i < 2 * n_hub else "G14 Hub SMARTS")

    # ── TTP ──
    _push(list(TTP_FEATURE_NAMES), "Linker TTP")

    # ── Linker extra (EState 79, Topo 14, Torsion 1024, AtomPair 2048) ──
    _push([f'linker_estate_{i}' for i in range(79)], "Linker EState")
    _push([f'linker_{n}' for n in _TOPO_NAMES], "Linker Topological")
    _push([f'linker_torsion_{i:04d}' for i in range(1024)], "Linker Torsion FP")
    _push([f'linker_atompair_{i:04d}' for i in range(2048)], "Linker Atom-Pair FP")

    # ── Halide ──
    _push(list(HALIDE_FEAT_COLS), "Halide Features")

    # ── DRFP ──
    _push([f'drfp_{i}' for i in range(2048)], "Reaction FP (DRFP)")

    # ── SOAP ──
    n_soap_prec = X_soap_precursor.shape[1]
    n_soap_link = X_soap_linker.shape[1]
    _push([f'soap_precursor_{i}' for i in range(n_soap_prec)], "3D SOAP (Precursor)")
    _push([f'soap_linker_{i}' for i in range(n_soap_link)], "3D SOAP (Linker)")

    # ── KMeans cluster OHE (appended by apply_variance_threshold) ──
    _push([f'kmeans_cluster_{i}' for i in range(n_clusters)], "KMeans Cluster OHE")

    # Sanity check vs X_final + cluster OHE
    n_xfinal_plus_ohe = X_final.shape[1] + n_clusters
    if len(names) != n_xfinal_plus_ohe:
        print(f"[WARN] Feature catalog has {len(names)} names but "
              f"X_final+OHE has {n_xfinal_plus_ohe} columns. "
              f"Delta = {len(names) - n_xfinal_plus_ohe}")

    # ── Apply VT mask ──
    names_arr = np.array(names, dtype=object)
    groups_arr = np.array(groups, dtype=object)
    vt_names = list(names_arr[vt_mask])
    vt_groups = list(groups_arr[vt_mask])

    # ── Append process (normalized) + interactions ──
    vt_names += [f'proc:{c}' for c in process_cols_present]
    vt_groups += ['Process Variables'] * len(process_cols_present)

    vt_names += _INTERACTION_NAMES
    vt_groups += ['Process Interactions'] * len(_INTERACTION_NAMES)

    return vt_names, vt_groups


def build_discrete_mask(
    X_linker, X_modulator, mod_eq,
    X_precursor_perlig, Xinventorynumeric, X_process,
    fp_cols, num_descriptors, ohe_cols,
    process_cols_present, n_clusters,
    X_modulator_rac_aug, X_metal_block,
    Xprecursor_full, X_precursor_perlig_rac,
    X_linker_phys10, X_modulator_phys10,
    X_modulator_tep, X_linker_tep, X_precursor_perlig_tep,
    X_precursor_perlig_steric,
    X_chemberta_block, chemberta_names,
    X_g14_block, g14_names,
    Xlinker_ttp,
    X_linker_extra,
    Xhalide_full,
    X_drfp,
    X_soap_precursor, X_soap_linker, soap_names,
    vt_mask,
):
    """Build a boolean mask: True = discrete/binary, False = continuous.

    Mirrors the exact block ordering in build_feature_catalog so index i
    in the mask corresponds to the same column as index i in the name array.
    After construction the VT mask is applied, then process-variable and
    interaction columns are appended (all continuous except the high-temp flag).

    Returns
    -------
    discrete : np.ndarray[bool]  — length matches X_cv columns
    """
    mask_parts: list[np.ndarray] = []

    def _d(n):
        """n discrete (True) entries."""
        mask_parts.append(np.ones(n, dtype=bool))

    def _c(n):
        """n continuous (False) entries."""
        mask_parts.append(np.zeros(n, dtype=bool))

    n_fp_lig = len(fp_cols)

    # ── X_features block ──
    # 1. Linker Morgan FP — binary
    _d(X_linker.shape[1])
    # 2. Modulator Morgan FP — binary
    _d(X_modulator.shape[1])
    # 3. Modulator equivalents — continuous
    _c(mod_eq.shape[1])
    # 4. Per-ligand precursor: 2048 binary FP + 1 continuous count each
    for _ in range(n_fp_lig):
        _d(2048)   # fingerprint bits
        _c(1)      # count
    # 5. Inventory numeric — continuous
    _c(Xinventorynumeric.shape[1])
    # 6. Process variables (raw) — continuous
    _c(len(process_cols_present))

    # ── Mordred RAC block — continuous descriptors + 2 QA cols ──
    _c(num_descriptors + 2)

    # ── Metal block: continuous descriptors + OHE (discrete) ──
    n_metal_desc = len(_METAL_DESC_NAMES)
    n_metal_ohe = len(ohe_cols)
    _c(n_metal_desc)   # mendeleev descriptors
    _d(n_metal_ohe)    # metal OHE

    # ── Precursor full block (metal center + coligand + complex) — continuous ──
    _c(METAL_BLOCK_DIM + COLIGAND_BLOCK_DIM + COMPLEX_BLOCK_DIM)

    # ── Per-ligand RAC — continuous (descriptors + QA + count) ──
    for _ in range(n_fp_lig):
        _c(num_descriptors + 3)

    # ── Physicochemical — continuous ──
    _c(X_linker_phys10.shape[1])
    _c(X_modulator_phys10.shape[1])

    # ── TEP — continuous ──
    _c(X_modulator_tep.shape[1])
    _c(X_linker_tep.shape[1])
    for _ in range(n_fp_lig):
        _c(3)   # tep_val, miss_flag, count

    # ── Steric — continuous ──
    for _ in range(n_fp_lig):
        _c(4)   # cone_angle, buried_vol, miss, count

    # ── ChemBERTa block (linker + modulator) ──
    # Per molecule: BERT_DIM(cont) + ExtRDKit(cont) + Shape3D(cont)
    #   + VSA(cont) + Composition(cont) + MACCS(discrete) + Fragments(discrete)
    from config import BERT_DIM
    _per_mol_cont = BERT_DIM + _N_TOTAL + len(SHAPE_3D_NAMES) + len(VSA_NAMES) + len(COMPOSITION_NAMES)
    _per_mol_disc = len(MACCS_NAMES) + len(FRAGMENT_NAMES)
    for _ in range(2):  # linker, modulator
        _c(_per_mol_cont)
        _d(_per_mol_disc)

    # ── G14 block ──
    # Hub topology features — mostly continuous (counts, fractions, etc.)
    # but binary flags (g14hub_present, isSi/Ge/Sn/Pb, missing) are discrete.
    # SMARTS pattern counts — discrete (integer match counts, mostly 0/1).
    # For simplicity: hub = continuous, SMARTS = discrete
    n_hub = len(G14_HUB_NAMES)
    n_smarts = len(ALL_G14_SMARTS_NAMES)
    _c(n_hub)      # linker hub
    _c(n_hub)      # modulator hub
    _d(n_smarts)   # linker SMARTS
    _d(n_smarts)   # modulator SMARTS

    # ── TTP — continuous ──
    _c(len(TTP_FEATURE_NAMES))

    # ── Linker extra: EState(79, discrete) + Topo(14, cont)
    #    + Torsion(1024, discrete) + AtomPair(2048, discrete) ──
    _d(79)     # EState FP
    _c(14)     # topological descriptors
    _d(1024)   # torsion FP
    _d(2048)   # atom-pair FP

    # ── Halide — continuous (counts & type encoding) ──
    _c(len(HALIDE_FEAT_COLS))

    # ── DRFP — discrete (binary fingerprint) ──
    _d(2048)

    # ── SOAP — continuous ──
    _c(X_soap_precursor.shape[1])
    _c(X_soap_linker.shape[1])

    # ── KMeans cluster OHE — discrete ──
    _d(n_clusters)

    # Concatenate all parts
    full_mask = np.concatenate(mask_parts)

    # Apply VT mask (same as feature catalog)
    vt_only_discrete = full_mask[vt_mask]

    # Full X_cv mask: VT features + process (continuous) + interactions
    cv_discrete = np.concatenate([
        vt_only_discrete,
        # Process (normalized) — continuous
        np.zeros(len(process_cols_present), dtype=bool),
        # Interactions — 5 continuous + 1 discrete (high-temp flag)
        np.array([False, False, False, False, False, True], dtype=bool),
    ])

    return cv_discrete, vt_only_discrete
