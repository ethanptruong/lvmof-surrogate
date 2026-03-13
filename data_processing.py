"""
data_processing.py — Data loading, cleaning, and inventory construction.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, rdBase, RDLogger
from collections import Counter

from config import COLMAP


# ── SMILES utilities ──────────────────────────────────────────────────────────

def canonicalize_smiles(smiles):
    """Converts a SMILES string to its unique, standard canonical form."""
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def deconstruct_precursor(smiles):
    """
    Deconstructs a precursor SMILES by removing metal atoms and counting the remaining ligand fragments.
    Explicitly handles coordination complexes by removing the metal node from the graph.
    """
    target_metals = {'Pd', 'Rh', 'Pt', 'Ag', 'Ir', 'Au', 'Cu', 'Co', 'Ni', 'Fe', 'Ru', 'Os'}

    if not isinstance(smiles, str):
        return None, {}

    # Disable RDKit error logs temporarily
    rdBase.DisableLog('rdApp.error')

    try:
        # Attempt to create molecule
        mol = Chem.MolFromSmiles(smiles)

        # Fallback to unsanitized if standard parsing fails
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)

        if mol is None:
            rdBase.EnableLog('rdApp.error')
            return None, {}

        # Convert to editable molecule
        rwmol = Chem.RWMol(mol)

        # Identify metal atoms
        metal_indices = []
        found_metal_symbol = None

        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() in target_metals:
                metal_indices.append(atom.GetIdx())
                if found_metal_symbol is None:
                    found_metal_symbol = atom.GetSymbol()

        # Remove metal atoms (descending order)
        metal_indices.sort(reverse=True)
        for idx in metal_indices:
            rwmol.RemoveAtom(idx)

        # Extract remaining fragments
        # Convert back to Mol to ensure stability for GetMolFrags
        frag_mol = rwmol.GetMol()
        fragments = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

        # Convert fragments to SMILES and count
        ligand_counts = Counter()
        for frag in fragments:
            try:
                lig_smiles = Chem.MolToSmiles(frag, canonical=True)
                if lig_smiles:
                    ligand_counts[lig_smiles] += 1
            except:
                pass

        return found_metal_symbol, dict(ligand_counts)

    finally:
        # Ensure logs are re-enabled even if an error occurs
        rdBase.EnableLog('rdApp.error')


def assert_required_columns(df: pd.DataFrame, colmap: dict, required_keys=("id","precursor","linker1","modulator")):
    missing_keys = [k for k in required_keys if k not in colmap]
    if missing_keys:
        raise KeyError(f"COLMAP is missing keys: {missing_keys}")

    missing_cols = [colmap[k] for k in required_keys if colmap[k] not in df.columns]
    if missing_cols:
        raise KeyError(f"DataFrame is missing required columns: {missing_cols}")

    id_col = colmap["id"]
    null_ids = df[id_col].isna().sum()
    dup_ids = df[id_col].duplicated().sum()

    if null_ids > 0:
        raise ValueError(f"{id_col} has {null_ids} null values (merge/key will break).")
    if dup_ids > 0:
        # not always fatal, but usually indicates you need a compound key
        print(f"WARNING: {id_col} has {dup_ids} duplicate values.")


def clean_smiles(x):
    """Normalize missing/placeholder values but DO NOT rename df columns."""
    if pd.isna(x) or not isinstance(x, str):
        return None
    s = x.strip()
    if not s or s.lower() in {"none", "nan"}:
        return None
    return s


def smiles_parse_ok(smiles):
    s = clean_smiles(smiles)
    if s is None:
        return False
    m = Chem.MolFromSmiles(s)
    if m is not None:
        return True
    # fallback: sometimes sanitization fails for weird coordination/ions
    m2 = Chem.MolFromSmiles(s, sanitize=False)
    return m2 is not None


def add_parse_diagnostics(df, colmap, keys=("precursor","linker1","modulator","linker2")):
    out = df.copy()
    for k in keys:
        if k not in colmap or colmap[k] not in out.columns:
            continue
        col = colmap[k]
        out[f"qa_{k}_smiles_clean"] = out[col].map(clean_smiles)
        out[f"qa_{k}_parse_ok"] = out[col].map(smiles_parse_ok)

        # Print quick stats
        n = len(out)
        ok = int(out[f"qa_{k}_parse_ok"].sum())
        print(f"{k}: parse_ok {ok}/{n} ({ok/n:.1%})")

        # Show top failing raw strings
        fails = out.loc[~out[f"qa_{k}_parse_ok"], col].value_counts(dropna=False).head(15)
        if len(fails) > 0:
            print(f"Top failing {k} strings:")
            print(fails)
            print("-"*60)
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """Load the experiment dataset from an Excel file."""
    df = pd.read_excel(file_path)
    return df


# ── Inventory construction ────────────────────────────────────────────────────

def build_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a ligand inventory DataFrame by deconstructing precursor SMILES
    and tracking modulator counts per experiment.
    """
    inventory_data = []

    # Loop through the dataset
    for index, row in df.iterrows():
        # Deconstruct precursor using the helper function
        metal, precursor_ligands = deconstruct_precursor(row['smiles_precursor'])

        # Process modulator
        mod_smiles_raw = row['smiles_modulator']
        modulator_smiles = None

        # Attempt to canonicalize if it's a string
        if isinstance(mod_smiles_raw, str):
            # Temporarily disable logs to avoid clutter from invalid SMILES
            rdBase.DisableLog('rdApp.error')
            canon = canonicalize_smiles(mod_smiles_raw)
            rdBase.EnableLog('rdApp.error')

            if canon:
                modulator_smiles = canon
            else:
                # Fallback to raw value if parsing fails
                modulator_smiles = mod_smiles_raw
        else:
            # Use raw value if not a string (e.g., NaN or other types)
            modulator_smiles = mod_smiles_raw

        # Retrieve equivalents
        try:
            if pd.notnull(row['equivalents']):
                mod_equiv = float(row['equivalents'])
            else:
                mod_equiv = 0.0
        except (ValueError, TypeError):
            mod_equiv = 0.0

        # Initialize row dict
        row_dict = {
            'experiment_id': row['experiment_id'],
            'metal_atom': metal
        }

        # Add precursor ligands
        for lig, count in precursor_ligands.items():
            key = f'Total_{lig}'
            row_dict[key] = row_dict.get(key, 0.0) + count

        # Add modulator if valid
        # Check if it is a non-empty string and not NaN (pandas treat NaN as float usually, but check notnull just in case)
        if isinstance(modulator_smiles, str) and modulator_smiles:
            key = f'Total_{modulator_smiles}'
            row_dict[key] = row_dict.get(key, 0.0) + mod_equiv

        inventory_data.append(row_dict)

    # Create DataFrame
    df_inventory = pd.DataFrame(inventory_data)

    # Fill NaNs with 0
    df_inventory = df_inventory.fillna(0)

    return df_inventory


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_data(df: pd.DataFrame, df_inventory: pd.DataFrame) -> pd.DataFrame:
    """Merge the main dataset with the ligand inventory on experiment_id."""
    df_merged = pd.merge(df, df_inventory, on="experiment_id", how="left")
    return df_merged


# ── Process variable audit ────────────────────────────────────────────────────

def run_process_variable_audit(df_merged: pd.DataFrame) -> None:
    """
    Run a comprehensive audit of process variables in df_merged, printing
    NaN counts, coercion failures, temperature sanity checks, and per-experiment
    breakdown of missing values.
    """
    process_cols = [
        'equivalents', 'total_solvent_volume_ml', 'solvent_1_fraction',
        'solvent_2_fraction', 'solvent_3_fraction', 'Min_Boiling_Point_K',
        'Max_Boiling_Point_K', 'Weighted_Boiling_Point_K', 'Weighted_AN_mole',
        'Weighted_DN_mole', 'Weighted_Dielectric_vol', 'Weighted_Polarity_vol',
        'Weighted_sig_h_vol', 'Weighted_sig_d_vol', 'Weighted_sig_p_vol',
        'Mix_M0_Area', 'Mix_M2_Polarity', 'Mix_M3_Asymmetry', 'Mix_M_HB_Acc',
        'Mix_M_HB_Don', 'temperature_k', 'metal_over_linker_ratio', 'reaction_hours'
    ]

    process_cols_present = [c for c in process_cols if c in df_merged.columns]
    process_df = df_merged[process_cols_present].apply(pd.to_numeric, errors='coerce')

    # ── 1. Global NaN summary ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  STEP 1 — NaN counts per column (raw, before any imputation)")
    print("=" * 65)
    nan_summary = process_df.isnull().sum()
    total_nan = nan_summary.sum()
    print(f"  Total NaN entries across all process vars: {total_nan}")
    print()
    for col, n in nan_summary.items():
        if n > 0:
            pct = 100 * n / len(process_df)
            print(f"  {col:<35} {n:>4} NaNs  ({pct:.1f}%)")
    if total_nan == 0:
        print("  ✅ No NaNs found — issue is NOT missing values from the source file.")

    # ── 2. Coercion failures — values that became NaN after to_numeric ─────────────
    print()
    print("=" * 65)
    print("  STEP 2 — Values coerced to NaN (non-numeric strings in source)")
    print("=" * 65)
    raw_df = df_merged[process_cols_present]
    coercion_failures = {}
    for col in process_cols_present:
        original = raw_df[col]
        coerced  = pd.to_numeric(original, errors='coerce')
        bad_mask = coerced.isnull() & original.notna()  # was something, became NaN
        if bad_mask.any():
            coercion_failures[col] = df_merged.loc[bad_mask, 'experiment_id'].tolist()
            print(f"\n  ⚠️  {col} — {bad_mask.sum()} coercion failure(s):")
            print(f"      Raw values: {original[bad_mask].unique().tolist()}")
            print(f"      Experiments: {coercion_failures[col][:10]}")

    if not coercion_failures:
        print("  ✅ No coercion failures — all values are numeric or truly NaN.")

    # ── 3. Temperature-specific check ─────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  STEP 3 — Temperature sanity check (expected: 298–393 K)")
    print("=" * 65)
    if 'temperature_k' in process_df.columns:
        temp = process_df['temperature_k']
        low_mask  = temp.notna() & (temp < 200)
        high_mask = temp.notna() & (temp > 500)
        null_mask = temp.isnull()
        print(f"  NaN:            {null_mask.sum()} experiments")
        print(f"  < 200 K (cold): {low_mask.sum()} experiments")
        print(f"  > 500 K (hot):  {high_mask.sum()} experiments")
        if low_mask.any():
            print(f"\n  🔴 Suspiciously cold experiments:")
            print(df_merged.loc[low_mask, ['experiment_id', 'temperature_k',
                                              'precursor_iupac_standardized',
                                              'metal_atom']].head(20))
        if null_mask.any():
            print(f"\n  🔴 Experiments with no temperature recorded:")
            print(df_merged.loc[null_mask, ['experiment_id', 'temperature_k',
                                               'precursor_iupac_standardized',
                                               'metal_atom']].head(20))

    # ── 4. Find ALL experiments with ANY bad process variable ─────────────────────
    print()
    print("=" * 65)
    print("  STEP 4 — All experiments with ≥1 NaN process variable")
    print("=" * 65)
    any_nan_mask = process_df.isnull().any(axis=1)
    print(f"  {any_nan_mask.sum()} / {len(process_df)} experiments have at least one NaN")

    if any_nan_mask.any():
        # Which columns are NaN for each bad experiment
        bad_df = df_merged.loc[any_nan_mask, ['experiment_id', 'metal_atom',
                                               'precursor_iupac_standardized']].copy()
        bad_df['nan_columns'] = process_df[any_nan_mask].apply(
            lambda row: [c for c in row.index if pd.isnull(row[c])], axis=1
        )
        bad_df['n_nan_cols'] = bad_df['nan_columns'].apply(len)
        bad_df = bad_df.sort_values('n_nan_cols', ascending=False)
        print(f"\n  Worst offenders (most missing process vars):")
        print(bad_df.head(30))

        # ── 5. Pattern — are bad experiments clustered by metal or source file? ──
        print()
        print("=" * 65)
        print("  STEP 5 — Are NaN experiments clustered by metal or source?")
        print("=" * 65)
        print("\n  Metal breakdown of NaN experiments:")
        print(df_merged.loc[any_nan_mask, 'metal_atom'].value_counts().to_string())
        if 'source_file' in df_merged.columns:
            print("\n  Source file breakdown of NaN experiments:")
            print(df_merged.loc[any_nan_mask, 'source_file'].value_counts().to_string())


# ── Worst experiments lookup ──────────────────────────────────────────────────

def get_worst_experiments(df_merged: pd.DataFrame, worst_ids: list) -> pd.DataFrame:
    """Return a subset of df_merged for the specified experiment IDs, showing solvent columns."""
    worst_mask = df_merged['experiment_id'].isin(worst_ids)
    solvent_cols = ['solvent_1', 'solvent_2', 'solvent_3',
                    'total_solvent_volume_ml', 'solvent_1_fraction',
                    'temperature_k', 'reaction_hours']
    return df_merged.loc[worst_mask, ['experiment_id'] + solvent_cols]
