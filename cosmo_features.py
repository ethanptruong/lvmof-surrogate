"""
cosmo_features.py — COSMO-RS sigma-profile featurizer for the LVMOF-Surrogate pipeline.

Reads VT-2005 sigma profile .txt files and computes sigma moments and surface
charge descriptors for each experiment's solvent (or solvent mixture), then
writes the enriched dataframe to an output Excel file.

Can be run as a standalone script:
    python cosmo_features.py
    python cosmo_features.py --data data/my_experiments.xlsx --output data/out.xlsx

Or imported and called programmatically:
    from cosmo_features import enrich_with_cosmo_features
    df_out = enrich_with_cosmo_features(df, index_path=..., cosmo_folder=...)

Computed columns (Mix_ prefix = mole-fraction-weighted mixture value):
  Sigma moments — from P(σ) (the 51-point sigma profile, σ in e/Å²):
    Mix_M0_Area       : zeroth moment = total COSMO surface area (Å²)
    Mix_M1_NetCharge  : first moment  = net surface charge (≈0 for neutral molecules)
    Mix_M2_Polarity   : second moment = polarity descriptor
    Mix_M3_Asymmetry  : third moment  = charge asymmetry / sigma-profile skewness
    Mix_M4_Kurtosis   : fourth moment = peakedness of the sigma profile

  Hydrogen-bonding moments — computed from P(σ) with an HB cutoff (σ_HB = 0.00854 e/Å²):
    Mix_M_HB_Acc      : ∫ P(σ) · max(0,  σ − σ_HB) dσ  (acceptor strength)
    Mix_M_HB_Don      : ∫ P(σ) · max(0, −σ − σ_HB) dσ  (donor strength)

  Surface-fraction descriptors:
    Mix_f_nonpolar    : fraction of surface area with |σ| < σ_HB  (nonpolar)
    Mix_f_acc         : fraction of surface area with  σ > σ_HB   (HB acceptor sites)
    Mix_f_don         : fraction of surface area with  σ < −σ_HB  (HB donor sites)

  Charge-distribution shape:
    Mix_sigma_std     : σ-weighted std of the charge distribution
                        (sqrt(M2/M0 - (M1/M0)²), width of sigma profile)

  From the VT-2005 index file (weighted by mole fraction):
    Mix_Vcosmo        : COSMO cavity volume (Å³), proxy for molecular size
    Mix_lnPvap        : reference ln(Pvap) from the index

References:
  Klamt, A. (1995). Conductor-like Screening Model for Real Solvents.
  VT-2005 Sigma Profile Database (Mullins et al., 2006).
  AMS COSMO-RS documentation (SCM).
"""

import os
import argparse
import sys

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

SIGMA_HB_CUTOFF = 0.00854   # e/Å²  — standard COSMO-RS HB cutoff

_DEFAULT_INDEX  = os.path.join("data", "VT-2005_Sigma_Profile_Database_Index_v2.xlsx")
_DEFAULT_COSMO  = os.path.join("data", "solvent_cosmo")
_DEFAULT_DATA   = os.path.join("data", "Experiments_with_Calculated_Properties_no_linker.xlsx")
_DEFAULT_OUTPUT = os.path.join("data", "Experiments_with_Calculated_Properties_no_linker.xlsx")

# Output column names (order is preserved)
COSMO_COLS = [
    "Mix_M0_Area",
    "Mix_M1_NetCharge",
    "Mix_M2_Polarity",
    "Mix_M3_Asymmetry",
    "Mix_M4_Kurtosis",
    "Mix_M_HB_Acc",
    "Mix_M_HB_Don",
    "Mix_f_nonpolar",
    "Mix_f_acc",
    "Mix_f_don",
    "Mix_sigma_std",
    "Mix_Vcosmo",
    "Mix_lnPvap",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_cosmo_index(index_path: str) -> tuple[dict, dict, dict, dict]:
    """
    Load the VT-2005 sigma profile database index.

    Returns
    -------
    index_map   : {COMPOUND_NAME_UPPER -> Index No. (int)}
    bp_map      : {COMPOUND_NAME_UPPER -> Temp. (K) (float)}
    vcosmo_map  : {COMPOUND_NAME_UPPER -> Vcosmo, A3 (float)}
    lnpvap_map  : {COMPOUND_NAME_UPPER -> ln Pvap (float)}
    """
    df = pd.read_excel(index_path)
    df["_name"] = df["Compound Name"].astype(str).str.strip().str.upper()

    index_map  = dict(zip(df["_name"], df["Index No."].astype(int)))
    bp_map     = dict(zip(df["_name"], pd.to_numeric(df["Temp. (K)"], errors="coerce")))
    vcosmo_map = dict(zip(df["_name"], pd.to_numeric(df["Vcosmo, A3"], errors="coerce")))
    lnpvap_map = dict(zip(df["_name"], pd.to_numeric(df["ln Pvap"],   errors="coerce")))

    return index_map, bp_map, vcosmo_map, lnpvap_map


def load_sigma_profile(index_no: int, cosmo_folder: str) -> pd.DataFrame | None:
    """
    Load a VT-2005 sigma profile file.  Returns a DataFrame with columns
    ['sigma', 'area'] (51 rows, σ from −0.025 to +0.025 e/Å²), or None on failure.
    """
    fname = f"VT2005-{index_no:04d}-PROF.txt"
    fpath = os.path.join(cosmo_folder, fname)
    if not os.path.exists(fpath):
        # Fallback: unpadded
        fpath = os.path.join(cosmo_folder, f"VT2005-{index_no}-PROF.txt")
        if not os.path.exists(fpath):
            return None
    try:
        df = pd.read_csv(fpath, sep=r"\s+", header=None, names=["sigma", "area"])
        if len(df) != 51:
            return None
        return df
    except Exception:
        return None


# ── Moment calculation ────────────────────────────────────────────────────────

def compute_sigma_moments(
    sigma: np.ndarray,
    area: np.ndarray,
    hb_cutoff: float = SIGMA_HB_CUTOFF,
) -> dict:
    """
    Compute all sigma moments and surface-fraction descriptors from a sigma profile.

    Parameters
    ----------
    sigma      : array of σ values (e/Å²), shape (51,)
    area       : array of P(σ) values (Å²), shape (51,)
    hb_cutoff  : hydrogen bond σ cutoff (e/Å²)

    Returns
    -------
    dict with keys matching COSMO_COLS (excluding Mix_Vcosmo, Mix_lnPvap which
    come from the index file).
    """
    m0 = float(np.sum(area))
    m1 = float(np.sum(area * sigma))
    m2 = float(np.sum(area * sigma ** 2))
    m3 = float(np.sum(area * sigma ** 3))
    m4 = float(np.sum(area * sigma ** 4))

    m_acc = float(np.sum(area * np.maximum(0.0,  sigma - hb_cutoff)))
    m_don = float(np.sum(area * np.maximum(0.0, -sigma - hb_cutoff)))

    # Surface fractions
    if m0 > 0:
        f_nonpolar = float(np.sum(area[np.abs(sigma) <  hb_cutoff])) / m0
        f_acc      = float(np.sum(area[ sigma         >  hb_cutoff])) / m0
        f_don      = float(np.sum(area[ sigma         < -hb_cutoff])) / m0
        mean_sigma = m1 / m0
        variance   = m2 / m0 - mean_sigma ** 2
        sigma_std  = float(np.sqrt(max(variance, 0.0)))
    else:
        f_nonpolar = f_acc = f_don = sigma_std = np.nan

    return {
        "Mix_M0_Area":      m0,
        "Mix_M1_NetCharge": m1,
        "Mix_M2_Polarity":  m2,
        "Mix_M3_Asymmetry": m3,
        "Mix_M4_Kurtosis":  m4,
        "Mix_M_HB_Acc":     m_acc,
        "Mix_M_HB_Don":     m_don,
        "Mix_f_nonpolar":   f_nonpolar,
        "Mix_f_acc":        f_acc,
        "Mix_f_don":        f_don,
        "Mix_sigma_std":    sigma_std,
    }


# ── Per-solvent lookup ────────────────────────────────────────────────────────

def _collect_solvents(row: pd.Series) -> list[dict]:
    """
    Parse solvent_1/2/3 + volume columns from an experiment row.
    Returns a list of dicts: [{name, vol}, ...] with only valid entries.
    """
    solvents = []
    for k in [1, 2, 3]:
        name = row.get(f"solvent_{k}")
        vol  = row.get(f"solvent_{k}_volume_ml")
        frac = row.get(f"solvent_{k}_fraction")

        if pd.isna(name) or str(name).strip().upper() in ("NAN", ""):
            continue

        clean = str(name).strip().upper()
        v_val = 0.0
        if pd.notna(vol):
            try:
                v_val = float(vol)
            except ValueError:
                pass
        elif pd.notna(frac):
            try:
                v_val = float(frac)
            except ValueError:
                pass

        if v_val > 0:
            solvents.append({"name": clean, "vol": v_val})

    return solvents


def _add_mole_fractions(
    solvents: list[dict],
    vcosmo_map: dict,
) -> bool:
    """
    Compute mole fractions in-place.  Uses volume / Vcosmo as a proxy for
    moles (since we lack MW/density here).  Returns True on success.

    Note: Vcosmo (Å³) ∝ molecular volume, which is approximately proportional
    to molar volume for similar molecule classes.  For a more rigorous approach
    the caller can supply MW + density; here we keep it self-contained within
    the COSMO index.
    """
    total_proxy = 0.0
    valid = True
    for s in solvents:
        vc = vcosmo_map.get(s["name"])
        if pd.notna(vc) and vc > 0:
            proxy = s["vol"] / vc  # volume / molecular_volume ∝ moles
            s["proxy_moles"] = proxy
            total_proxy += proxy
        else:
            valid = False
            s["proxy_moles"] = 0.0

    if valid and total_proxy > 0:
        for s in solvents:
            s["mole_frac"] = s["proxy_moles"] / total_proxy
        return True
    else:
        # Fall back to volume fractions
        total_vol = sum(s["vol"] for s in solvents)
        for s in solvents:
            s["mole_frac"] = s["vol"] / total_vol if total_vol > 0 else np.nan
        return False


# ── Main enrichment function ──────────────────────────────────────────────────

def enrich_with_cosmo_features(
    df: pd.DataFrame,
    index_path: str = _DEFAULT_INDEX,
    cosmo_folder: str = _DEFAULT_COSMO,
    hb_cutoff: float = SIGMA_HB_CUTOFF,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Add COSMO sigma-moment features to a copy of *df* and return it.

    Parameters
    ----------
    df           : experiment dataframe with solvent_1/2/3 + volume columns
    index_path   : path to VT-2005_Sigma_Profile_Database_Index_v2.xlsx
    cosmo_folder : folder containing VT2005-XXXX-PROF.txt files
    hb_cutoff    : hydrogen bond σ cutoff in e/Å² (default 0.00854)
    overwrite    : if True, overwrite existing COSMO columns; otherwise skip rows
                   that already have Mix_M0_Area populated.

    Returns
    -------
    df_out : enriched dataframe (copy)
    """
    df_out = df.copy()

    # Ensure output columns exist
    for col in COSMO_COLS:
        if col not in df_out.columns:
            df_out[col] = np.nan

    # Load index
    print("Loading COSMO index...")
    index_map, bp_map, vcosmo_map, lnpvap_map = load_cosmo_index(index_path)
    print(f"  {len(index_map)} compounds in index.")

    # Profile cache to avoid re-reading the same file for every row
    profile_cache: dict[int, pd.DataFrame | None] = {}

    missing_names: set[str] = set()
    missing_profiles: set[int] = set()
    n_success = n_fail = 0

    print(f"Processing {len(df_out)} rows...")
    for i, row in df_out.iterrows():
        if i % 100 == 0:
            print(f"  Row {i}...")

        if not overwrite and pd.notna(row.get("Mix_M0_Area")):
            continue

        solvents = _collect_solvents(row)
        if not solvents:
            continue

        total_vol = sum(s["vol"] for s in solvents)
        for s in solvents:
            s["vol_frac"] = s["vol"] / total_vol

        # Mole fractions (Vcosmo-based proxy)
        _add_mole_fractions(solvents, vcosmo_map)

        # ── Build weighted sigma profile ──────────────────────────────────────
        mix_area: np.ndarray | None = None
        sigma_axis: np.ndarray | None = None
        cosmo_ok = True

        for s in solvents:
            idx = index_map.get(s["name"])
            if idx is None:
                missing_names.add(s["name"])
                cosmo_ok = False
                continue

            if idx not in profile_cache:
                profile_cache[idx] = load_sigma_profile(idx, cosmo_folder)

            prof = profile_cache[idx]
            if prof is None:
                missing_profiles.add(idx)
                cosmo_ok = False
                continue

            if sigma_axis is None:
                sigma_axis = prof["sigma"].values.copy()
                mix_area   = np.zeros(51)

            mix_area += prof["area"].values * s["mole_frac"]

        if cosmo_ok and sigma_axis is not None and mix_area is not None:
            moments = compute_sigma_moments(sigma_axis, mix_area, hb_cutoff)
            for col, val in moments.items():
                df_out.at[i, col] = val

            # Index-based properties (mole-fraction weighted)
            mix_vc   = sum(
                vcosmo_map.get(s["name"], np.nan) * s["mole_frac"]
                for s in solvents
                if pd.notna(vcosmo_map.get(s["name"]))
            )
            mix_lnp  = sum(
                lnpvap_map.get(s["name"], np.nan) * s["mole_frac"]
                for s in solvents
                if pd.notna(lnpvap_map.get(s["name"]))
            )
            df_out.at[i, "Mix_Vcosmo"]  = mix_vc  if mix_vc  else np.nan
            df_out.at[i, "Mix_lnPvap"]  = mix_lnp if mix_lnp else np.nan

            n_success += 1
        else:
            n_fail += 1

    print(f"\nDone. {n_success} rows enriched, {n_fail} skipped (missing data).")
    if missing_names:
        print(f"  Solvent names not found in index ({len(missing_names)}): {sorted(missing_names)}")
    if missing_profiles:
        print(f"  Profile files missing for index numbers: {sorted(missing_profiles)}")

    return df_out


# ── Standalone CLI entry point ────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich experiment data with COSMO-RS sigma-moment features."
    )
    p.add_argument("--data",   default=_DEFAULT_DATA,   help="Input experiments Excel file")
    p.add_argument("--index",  default=_DEFAULT_INDEX,  help="VT-2005 index Excel file")
    p.add_argument("--cosmo",  default=_DEFAULT_COSMO,  help="Folder containing PROF.txt files")
    p.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output Excel file path")
    p.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Skip rows that already have Mix_M0_Area populated"
    )
    p.add_argument(
        "--hb-cutoff", type=float, default=SIGMA_HB_CUTOFF,
        help=f"HB sigma cutoff in e/Å² (default {SIGMA_HB_CUTOFF})"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.index):
        print(f"Error: index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.cosmo):
        print(f"Error: COSMO folder not found: {args.cosmo}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading experiments from: {args.data}")
    df = pd.read_excel(args.data)
    print(f"  {len(df)} rows loaded.")

    df_out = enrich_with_cosmo_features(
        df,
        index_path=args.index,
        cosmo_folder=args.cosmo,
        hb_cutoff=args.hb_cutoff,
        overwrite=args.overwrite,
    )

    print(f"\nSaving to: {args.output}")
    df_out.to_excel(args.output, index=False)
    print("Saved.")

    # Quick summary of computed columns
    print("\nColumn summary (non-null counts):")
    for col in COSMO_COLS:
        if col in df_out.columns:
            n = df_out[col].notna().sum()
            print(f"  {col:<22}: {n}/{len(df_out)}")


if __name__ == "__main__":
    main()
