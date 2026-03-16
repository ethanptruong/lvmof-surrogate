"""
featurization.py — All featurization functions for the LVMOF-Surrogate pipeline.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter, deque
from rdkit import Chem, rdBase, DataStructs, RDLogger
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors, rdmolops,
                         MACCSkeys, Fragments, GraphDescriptors)
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.EState.Fingerprinter import FingerprintMol as EStateFP
import mendeleev
import joblib
import urllib.request
import lightgbm as lgb
import torch
from transformers import AutoTokenizer, AutoModel

from config import (
    TARGET_METALS, ROMAN_TO_INT, GROUP11_METALS, PRECURSOR_CBC, GEOMETRY_LABELS,
    ION_MAP, CHEMBERTA_MODEL, BERT_DIM, TEP_MODEL_URL,
    _COLIGAND_LOOKUP, _COLIGAND_FEATURE_NAMES, _COLIGAND_DIM,
    _METAL_OX_STATE_FALLBACK, METAL_BLOCK_DIM, COLIGAND_BLOCK_DIM, COMPLEX_BLOCK_DIM,
    GROUP14_SYMBOLS, COORD_SYMBOLS, G14_ENEG, G14_COVRAD, G14_PERIOD,
    G14_HUB_NAMES, G14_SMARTS_RAW, ALL_G14_SMARTS_NAMES,
    TTP_DIM, TTP_FEATURE_NAMES,
    SHAPE_3D_NAMES, VSA_NAMES, COMPOSITION_NAMES, MACCS_NAMES, FRAGMENT_NAMES,
    _SMARTS_RAW, _COORD_KEYS,
    ELEMENT_TO_Z, Z_TO_ELEMENT, SOAP_SPECIES,
    HALIDE_FEAT_COLS,
)
from data_processing import clean_smiles


# ── Numpy patches for mordred compatibility ────────────────────────────────────
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'product'):
    np.product = np.prod
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

from mordred import Calculator, descriptors as mordred_descriptors

calc = Calculator(mordred_descriptors.Autocorrelation)
num_descriptors = len(calc.descriptors)


# ── Helpers ────────────────────────────────────────────────────────────────────

def finite(x, default=0.0):
    """Cast to float; replace NaN/inf/errors with default."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _f(x, default=0.0):
    """Safe float cast; replaces NaN/Inf with default."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ── Metal descriptors (mendeleev) ─────────────────────────────────────────────

def get_metal_descriptors(symbol: str) -> dict:
    """
    Fetch atomic descriptors for a metal element using mendeleev.
    Returns a flat dict of float values. Missing values → 0.0.
    All values are physically meaningful scalars — no hallucination risk.
    """
    try:
        el = mendeleev.element(symbol)
    except Exception:
        # Unknown symbol — return all zeros with missing flag
        return {
            'metal_atomic_number':       0.0,
            'metal_period':              0.0,
            'metal_group':               0.0,
            'metal_en_pauling':          0.0,
            'metal_en_allen':            0.0,
            'metal_covalent_radius_pm':  0.0,
            'metal_atomic_radius_pm':    0.0,
            'metal_vdw_radius_pm':       0.0,
            'metal_ionization_energy1_eV': 0.0,
            'metal_ionization_energy2_eV': 0.0,
            'metal_electron_affinity_eV': 0.0,
            'metal_nvalence':            0.0,
            'metal_max_oxidation_state': 0.0,
            'metal_min_oxidation_state': 0.0,
            'metal_common_oxidation_states_count': 0.0,
            'metal_mendeleev_number':    0.0,
            'metal_hardness_eV':         0.0,
            'metal_dipole_polarizability_bohr3': 0.0,
            'metal_missing_flag':        1.0,
        }

    # --- Helper: safe float conversion ---
    def safe(val):
        try:
            v = float(val)
            return v if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    # --- Ionization energies (1st and 2nd) ---
    try:
        ie_dict = el.ionenergies  # {1: float, 2: float, ...}
        ie1 = safe(ie_dict.get(1, 0.0))
        ie2 = safe(ie_dict.get(2, 0.0))
    except Exception:
        ie1, ie2 = 0.0, 0.0

    # --- Oxidation states (main only) ---
    try:
        ox_states = [os.oxidation_state for os in el.oxidation_states
                     if os.category == 'main']
        if not ox_states:
            # Fallback to all if 'main' is empty
            ox_states = [os.oxidation_state for os in el.oxidation_states]
        max_ox = safe(max(ox_states)) if ox_states else 0.0
        min_ox = safe(min(ox_states)) if ox_states else 0.0
        n_ox   = float(len(ox_states))
    except Exception:
        max_ox, min_ox, n_ox = 0.0, 0.0, 0.0

    # --- Number of valence electrons ---
    try:
        nval = safe(el.nvalence())
    except Exception:
        nval = 0.0

    # --- Hardness (Pearson absolute hardness = (IE1 - EA) / 2) ---
    #     mendeleev computes this as el.hardness (eV)
    try:
        hardness = safe(el.hardness())
    except Exception:
        # Manual fallback
        ea = safe(el.electron_affinity) if el.electron_affinity is not None else 0.0
        hardness = (ie1 - ea) / 2.0 if ie1 > 0 else 0.0

    # --- Covalent radius: prefer Pyykko single-bond, fallback Bragg ---
    try:
        cov_r = safe(el.covalent_radius_pyykko)
        if cov_r == 0.0:
            cov_r = safe(el.covalent_radius_bragg)
    except Exception:
        cov_r = 0.0

    return {
        'metal_atomic_number':           safe(el.atomic_number),
        'metal_period':                  safe(el.period),
        'metal_group':                   safe(el.group_id),
        'metal_en_pauling':              safe(el.en_pauling),
        'metal_en_allen':                safe(el.en_allen),
        'metal_covalent_radius_pm':      cov_r,
        'metal_atomic_radius_pm':        safe(el.atomic_radius),
        'metal_vdw_radius_pm':           safe(el.vdw_radius),
        'metal_ionization_energy1_eV':   ie1,
        'metal_ionization_energy2_eV':   ie2,
        'metal_electron_affinity_eV':    safe(el.electron_affinity),
        'metal_nvalence':                nval,
        'metal_max_oxidation_state':     max_ox,
        'metal_min_oxidation_state':     min_ox,
        'metal_common_oxidation_states_count': n_ox,
        'metal_mendeleev_number':        safe(el.mendeleev_number),
        'metal_hardness_eV':             hardness,
        'metal_dipole_polarizability_bohr3': safe(el.dipole_polarizability),
        'metal_missing_flag':            0.0,
    }


def lookup_metal_descriptors(metal_symbol, metal_descriptor_cache, zero_descriptor):
    """Look up cached descriptors, return zero vector if unknown."""
    if pd.isna(metal_symbol) or not isinstance(metal_symbol, str):
        return zero_descriptor
    sym = str(metal_symbol).strip()
    if sym in metal_descriptor_cache:
        return metal_descriptor_cache[sym]
    # Attempt live lookup for any metal not in TARGET_METALS
    try:
        desc = get_metal_descriptors(sym)
        metal_descriptor_cache[sym] = desc  # Cache for reuse
        return desc
    except Exception:
        return zero_descriptor


# ── Oxidation state and CBC ───────────────────────────────────────────────────

def parse_oxidation_state(iupac_name: str) -> float:
    if not isinstance(iupac_name, str):
        return 0.0
    # Match trailing Roman numerals or '0' at end of name
    m = re.search(r'((?:VIII|VII|VI|IV|V|III|II|I|0))$', iupac_name.strip())
    if m:
        return float(ROMAN_TO_INT.get(m.group(1).upper(), 0))
    return 0.0  # default: assume 0 if unparseable


def get_d_electron_count(metal_symbol: str, oxidation_state: float) -> float:
    """
    Returns the d-electron count for a transition metal in a given oxidation state.

    Formula:
      - Group 3-10:  d^n = group_number - oxidation_state
      - Group 11:    d^n = 10 - oxidation_state  (d10s1 neutral; s1 excluded)

    Args:
        metal_symbol:    Element symbol, e.g. 'Pd', 'Cu', 'Rh'
        oxidation_state: Numeric oxidation state, e.g. 0, 1, 2

    Returns:
        d-electron count as float, clamped to [0, 10].
    """
    try:
        el = mendeleev.element(metal_symbol)
        group = el.group_id          # e.g., Pd→10, Rh→9, Fe→8, Cu→11

        if metal_symbol in GROUP11_METALS:
            # d10s1 neutral: effective CBC d-count is 10 minus oxidation state
            # BUG FIXED: NameError — 'oxidationstate' should be 'oxidation_state' (missing underscore, mismatches function parameter name)
            d_count = 11 - int(oxidation_state) if oxidation_state >= 1 else 10
        else:
            # Standard CBC formula for groups 3-10 and 12
            d_count = float(group - int(oxidation_state))

        return float(max(0.0, min(10.0, d_count)))   # physically bounded [0,10]

    except Exception:
        return 0.0


def get_cbc(iupac_name):
    """Returns (nL, nX) for a precursor. Defaults to (0, 0) if unknown."""
    return PRECURSOR_CBC.get(str(iupac_name).strip(), (0, 0))


# ── Geometry ──────────────────────────────────────────────────────────────────

def get_precursor_geometry(smiles: str, metal_symbol: str, dcount: float):
    result = {f'precgeom_{g}': 0.0 for g in GEOMETRY_LABELS}  # ← uses GEOMETRY_LABELS
    coordn = 0.0

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['precgeom_unknown'] = 1.0
            return result, coordn

        # ── Step 1: locate metal and check its degree ──────────────────────
        metal_degree = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == metal_symbol:
                metal_degree = atom.GetDegree()
                break

        if metal_degree is None:
            result['precgeom_unknown'] = 1.0
            return result, coordn

        # ── Step 2: choose coordination number source ───────────────────────
        if metal_degree > 0:
            coordn = float(metal_degree)
        else:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            n_ligand_frags = sum(
                1 for frag in frags
                if not any(a.GetSymbol() == metal_symbol for a in frag.GetAtoms())
            )
            coordn = float(n_ligand_frags)

        # ── Step 3: assign geometry ─────────────────────────────────────────
        # FIX 1: strings now use underscores to match GEOMETRY_LABELS keys
        # FIX 2: dcount == 8.0 only (not >= 8.0) — d10 Pd(0)/Ni(0) are tetrahedral
        cn = int(coordn)
        if cn == 2:
            geom = 'linear'
        elif cn == 3:
            geom = 'trigonal_planar'
        elif cn == 4:
            geom = 'square_planar' if dcount == 8.0 else 'tetrahedral'
        elif cn == 5:
            geom = 'trigonal_bipyramidal'
        elif cn == 6:
            geom = 'octahedral'
        else:
            geom = 'unknown'

        result[f'precgeom_{geom}'] = 1.0

    except Exception:
        result['precgeom_unknown'] = 1.0

    return result, coordn


# ── Morgan fingerprint ────────────────────────────────────────────────────────

def generate_morgan_fp(smiles, n_bits=2048):
    """
    Generates a Morgan fingerprint (radius=2) for a given SMILES string.
    Returns a numpy array of bits. Returns a zero vector if SMILES is invalid.
    """
    # Handle missing or non-string input
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return np.zeros(n_bits, dtype=int)

    # Attempt to create molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Handle failed molecule creation
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    # Generate Morgan fingerprint
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        # Convert to numpy array
        return np.array(fp)
    except:
        return np.zeros(n_bits, dtype=int)


def canonicalize_smiles_keep(smiles):
    """
    Return canonical SMILES if RDKit can parse; otherwise return the original string (NOT None).
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    s = smiles.strip()
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s  # keep original
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return s  # keep original


def normalize_inventory_token(token):
    """
    Inventory column suffix -> either a SMILES (string) or None (meaning: don't fingerprint it).
    """
    if token is None:
        return None
    t = str(token).strip()
    if t in ION_MAP:
        return ION_MAP[t]
    # crude filter: if it contains spaces or clearly isn't SMILES-like, skip fingerprinting
    if " " in t or len(t) == 0:
        return None
    return t


def morgan_fp_numpy(smiles, n_bits=2048, radius=2):
    """
    Returns (fp_array, ok_bool). If parse fails, returns (zeros, False).
    """
    arr = np.zeros((n_bits,), dtype=np.int8)
    if smiles is None or (not isinstance(smiles, str)) or smiles.strip() == "":
        return arr, False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return arr, False
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, True
    except Exception:
        return arr, False


def pick_featurizable_smiles(row, base_col):
    can = row.get(f"{base_col}_canon", None)
    raw = row.get(f"{base_col}_raw", None)
    # Try canonical first
    fp_can, ok_can = morgan_fp_numpy(can)
    if ok_can:
        return can
    # Fall back to raw
    fp_raw, ok_raw = morgan_fp_numpy(raw)
    return raw if ok_raw else None


def fp_zero_report(df, colmap, key="linker1", nbits=2048):
    col = colmap[key]
    fps = []
    ok_flags = []
    for s in df[col].tolist():
        v, ok = morgan_fp_numpy(s, n_bits=nbits)
        fps.append(v)
        ok_flags.append(ok)
    X = np.stack(fps)
    ok_flags = np.array(ok_flags, dtype=bool)

    nnz = X.sum(axis=1)
    zero = (nnz == 0)
    print(f"{key}: fp_parse_ok {ok_flags.sum()}/{len(ok_flags)}")
    print(f"{key}: all-zero fingerprints {int(zero.sum())}/{len(zero)} ({zero.mean():.1%})")

    # show a few problematic rows
    bad_idx = np.where(zero)[0][:10]
    if len(bad_idx) > 0:
        print("Example all-zero rows:")
        print(df.iloc[bad_idx][[colmap['id'], col]])

    return X, ok_flags, zero


# ── Mordred RAC descriptors ───────────────────────────────────────────────────

def _normalize_inventory_token(token: str):
    """Map obvious ions and return a SMILES-like string (or None if not usable)."""
    if token is None:
        return None
    t = str(token).strip()
    if t == "" or " " in t:
        return None
    return ION_MAP.get(t, t)


def _mordred_result_to_vec_and_stats(results):
    """
    Returns:
      vec_clean: shape (num_descriptors,), non-finite -> 0.0
      any_bad:   1.0 if any non-finite/failed entry occurred else 0.0
      frac_bad:  fraction of entries that were non-finite/failed
    """
    vec = np.zeros(num_descriptors, dtype=float)
    n_bad = 0

    for i, r in enumerate(results):
        try:
            v = float(r)  # float(np.nan) succeeds, so we must check isfinite
        except (ValueError, TypeError):
            v = np.nan

        if not np.isfinite(v):
            n_bad += 1
            v = 0.0

        vec[i] = v

    any_bad = 1.0 if n_bad > 0 else 0.0
    frac_bad = n_bad / float(num_descriptors) if num_descriptors else 0.0
    return vec, any_bad, frac_bad


def get_mordred_racs_smiles_with_stats(smiles):
    """
    Compute Autocorrelation descriptors for a single SMILES.
    Returns (vec_clean, any_bad, frac_bad).
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return np.zeros(num_descriptors, dtype=float), 1.0, 1.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(num_descriptors, dtype=float), 1.0, 1.0

    try:
        results = calc(mol)
        return _mordred_result_to_vec_and_stats(results)
    except Exception:
        return np.zeros(num_descriptors, dtype=float), 1.0, 1.0


# ── Metal center block (Block 16) ─────────────────────────────────────────────

_METAL_PROPS_CACHE = {}

def _get_mendeleev_props(symbol: str) -> dict:
    if symbol in _METAL_PROPS_CACHE:
        return _METAL_PROPS_CACHE[symbol]
    try:
        el = mendeleev.element(symbol)
        def safe(v, default=0.0):
            try:
                f = float(v)
                return f if np.isfinite(f) else default
            except Exception:
                return default

        props = {
            'atomic_number':   safe(el.atomic_number),
            'period':          safe(el.period),
            'group':           safe(el.group_id, 0.0),
            'en_pauling':      safe(el.en_pauling),
            'en_allen':        safe(el.en_allen),
            'atomic_radius':   safe(el.atomic_radius),
            'cov_radius':      safe(el.covalent_radius_pyykko),
            'vdw_radius':      safe(el.vdw_radius),
            'ie1':             safe(el.ionization_energies[0]
                                    if el.ionization_energies else 0),
            'ie2':             safe(el.ionization_energies[1]
                                    if len(el.ionization_energies) > 1 else 0),
            'electron_affinity': safe(el.electron_affinity),
            'dipole_polarizability': safe(el.dipole_polarizability),
            # d/f electron counts
            'd_electrons':     safe(el.d_electrons),
            'f_electrons':     safe(el.f_electrons),
            'valence_electrons': safe(el.valence_electrons),
            # block: s=0, p=1, d=2, f=3
            'block_enc':       {'s': 0.0, 'p': 1.0,
                                'd': 2.0, 'f': 3.0}.get(el.block, 2.0),
        }
    except Exception:
        props = {k: 0.0 for k in [
            'atomic_number','period','group','en_pauling','en_allen',
            'atomic_radius','cov_radius','vdw_radius','ie1','ie2',
            'electron_affinity','dipole_polarizability',
            'd_electrons','f_electrons','valence_electrons','block_enc']}
    _METAL_PROPS_CACHE[symbol] = props
    return props


_MENDELEEV_PROP_KEYS = [
    'atomic_number','period','group','en_pauling','en_allen',
    'atomic_radius','cov_radius','vdw_radius','ie1','ie2',
    'electron_affinity','dipole_polarizability',
    'd_electrons','f_electrons','valence_electrons','block_enc'
]


def _parse_oxidation_state(iupac_name: str, metal_symbol: str) -> float:
    """Extract Roman numeral oxidation state from IUPAC name."""
    if not isinstance(iupac_name, str):
        return float(_METAL_OX_STATE_FALLBACK.get(metal_symbol, 0))
    roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4,
             'V': 5, 'VI': 6, '0': 0}
    # Match pattern like "rhodium(I)" or "palladium(0)"
    match = re.search(r'\(([IVX]+|0)\)', iupac_name, re.IGNORECASE)
    if match:
        return float(roman.get(match.group(1).upper(),
                    _METAL_OX_STATE_FALLBACK.get(metal_symbol, 0)))
    return float(_METAL_OX_STATE_FALLBACK.get(metal_symbol, 0))


def get_metal_center_block(smiles: str,
                           iupac_name: str = '') -> np.ndarray:
    out = np.zeros(METAL_BLOCK_DIM, dtype=float)
    out[31] = 1.0  # assume missing

    TARGET_METALS_SET = {'Pd','Rh','Pt','Ag','Ir','Au','Cu',
                     'Co','Ni','Fe','Ru','Os','Re','W','Mo'}
    if not isinstance(smiles, str) or not smiles.strip():
        return out

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return out

    metal_atoms = [a for a in mol.GetAtoms()
                   if a.GetSymbol() in TARGET_METALS_SET]
    if not metal_atoms:
        return out

    # Use the first unique metal found (these are all mononuclear or
    # homo-metallic precursors)
    metal_sym = metal_atoms[0].GetSymbol()
    props = _get_mendeleev_props(metal_sym)

    # Fill mendeleev properties (slots 0–15)
    for i, key in enumerate(_MENDELEEV_PROP_KEYS):
        out[i] = props[key]

    # Oxidation state (slot 16)
    ox = _parse_oxidation_state(iupac_name, metal_sym)
    out[16] = ox

    # d-electron count in complex: d^n = group - oxidation_state
    # e.g. Rh(I) is group 9, d8; Pd(0) is group 10, d10
    group = props['group']
    d_count = group - ox  # approximate, works for groups 8–12
    d_count = max(0.0, min(10.0, d_count))
    out[17] = d_count

    # d-count flags
    out[18] = 1.0 if abs(d_count - 8)  < 0.5 else 0.0  # d8
    out[19] = 1.0 if abs(d_count - 10) < 0.5 else 0.0  # d10
    out[20] = 1.0 if abs(d_count - 6)  < 0.5 else 0.0  # d6

    # Estimated coordination number from fragment count
    # (number of non-metal heavy-atom neighbors across all metal atoms)
    total_bonds = sum(a.GetDegree() for a in metal_atoms)
    coord_num = total_bonds / max(len(metal_atoms), 1)
    out[21] = coord_num

    # Geometry flags from d-count + coord num heuristic
    is_sq = (abs(d_count - 8) < 0.5 and coord_num >= 3.5)
    is_tet = (abs(d_count - 10) < 0.5 and coord_num < 5)
    is_oct = (coord_num >= 5.5)
    out[22] = 1.0 if is_sq  else 0.0  # square planar
    out[23] = 1.0 if is_tet else 0.0  # tetrahedral
    out[24] = 1.0 if is_oct else 0.0  # octahedral

    # Period flags (3d/4d/5d)
    period = int(props['period'])
    out[25] = 1.0 if period == 4 else 0.0  # 3d metals (period 4)
    out[26] = 1.0 if period == 5 else 0.0  # 4d metals (period 5)
    out[27] = 1.0 if period == 6 else 0.0  # 5d metals (period 6)

    # Group flags (relevant for these precursors)
    out[28] = 1.0 if abs(group - 9)  < 0.5 else 0.0   # group 9 (Rh, Ir, Co)
    out[29] = 1.0 if abs(group - 10) < 0.5 else 0.0   # group 10 (Pd, Pt, Ni)
    out[30] = 1.0 if abs(group - 11) < 0.5 else 0.0   # group 11 (Cu, Ag, Au)

    out[31] = 0.0  # success
    bad = ~np.isfinite(out)
    out[bad] = 0.0
    return out


def get_coligand_block(smiles: str) -> np.ndarray:
    """
    Characterizes co-ligands attached to the metal in the precursor.
    Separates simple ligands (CO, halide) using lookup and
    complex ligands (phosphines) using known structural flags.
    """
    out = np.zeros(COLIGAND_BLOCK_DIM, dtype=float)
    if not isinstance(smiles, str) or not smiles.strip():
        return out

    # Split on '.' to get individual fragments (SMILES disconnected)
    frags = smiles.split('.')
    TARGET_METALS_SET = {'Pd','Rh','Pt','Ag','Ir','Au','Cu',
                     'Co','Ni','Fe','Ru','Os'}

    co_vec      = np.zeros(_COLIGAND_DIM)
    halide_vec  = np.zeros(_COLIGAND_DIM)
    phos_count  = 0
    other_count = 0

    for frag in frags:
        frag = frag.strip()
        if not frag:
            continue
        # Skip fragments that contain a metal atom
        mol_frag = Chem.MolFromSmiles(frag, sanitize=False)
        if mol_frag and any(a.GetSymbol() in TARGET_METALS_SET
                            for a in mol_frag.GetAtoms()):
            continue

        # Simple lookup
        if frag in _COLIGAND_LOOKUP:
            props = np.array(_COLIGAND_LOOKUP[frag])
            if props[5] > 0:   # is_carbonyl
                co_vec += props
            else:              # is_halide
                halide_vec += props
            continue

        # Phosphine detection: must contain P with aryl substituents
        if 'P' in frag or 'p' in frag:
            if mol_frag is not None:
                has_P = any(a.GetSymbol() == 'P'
                            for a in mol_frag.GetAtoms())
                if has_P:
                    phos_count += 1
                    continue

        other_count += 1

    # Pack: 6 CO props, 6 halide props, phos_count, other_count
    out[0:6]   = co_vec
    out[6:12]  = halide_vec
    out[12]    = float(phos_count)
    out[13]    = float(other_count)

    # Scalar summary counts
    out[14] = float(co_vec[5])       # n_CO (summed carbonyl flag)
    out[15] = float(halide_vec[4])   # n_halide (summed halide flag)
    out[16] = float(phos_count)
    out[17] = float(other_count)

    # Electronic balance: σ-donor sum - π-acceptor sum
    # High = electron-rich metal center, low = electron-poor
    sigma_sum = co_vec[1] + halide_vec[1]
    pi_sum    = co_vec[2] + halide_vec[2]
    out[18]   = sigma_sum
    out[19]   = pi_sum
    out[20]   = sigma_sum - pi_sum  # net donor character

    # Net charge of co-ligands (from lookup charges)
    out[21] = co_vec[3] + halide_vec[3]

    # Flags
    out[22] = 1.0 if co_vec[5] > 0 else 0.0    # has_CO
    out[23] = 1.0 if halide_vec[4] > 0 else 0.0 # has_halide
    out[24] = 1.0 if 'Cl' in smiles else 0.0
    out[25] = 1.0 if 'Br' in smiles else 0.0
    out[26] = 1.0 if 'I'  in smiles and 'Ir' not in smiles else 0.0
    out[27] = 1.0 if phos_count > 0 else 0.0

    bad = ~np.isfinite(out)
    out[bad] = 0.0
    return out


def get_complex_level_block(smiles: str) -> np.ndarray:
    out = np.zeros(COMPLEX_BLOCK_DIM, dtype=float)
    out[11] = 1.0
    if not isinstance(smiles, str) or not smiles.strip():
        return out

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return out

    TARGET_METALS_SET = {'Pd','Rh','Pt','Ag','Ir','Au','Cu',
                     'Co','Ni','Fe','Ru','Os'}
    metal_atoms = [a for a in mol.GetAtoms()
                   if a.GetSymbol() in TARGET_METALS_SET]
    n_metals = len(metal_atoms)
    if n_metals == 0:
        return out

    out[0] = 1.0 if n_metals > 1 else 0.0   # is_dimer/cluster
    out[1] = float(n_metals)

    # Total bonds from metal atoms
    total_bonds = sum(a.GetDegree() for a in metal_atoms)
    out[2] = float(total_bonds)

    # Homoleptic: all non-metal fragments are the same SMILES
    frags = smiles.split('.')
    lig_frags = [f for f in frags if not any(
        m in f for m in TARGET_METALS_SET)]
    out[3] = 1.0 if len(set(lig_frags)) <= 1 else 0.0

    # Net charge estimate (count [X-] fragments)
    neg_count = sum(1 for f in frags if f.strip().startswith('[')
                    and f.strip().endswith('-]'))
    out[4] = float(-neg_count)

    # Unique ligand types
    out[5] = float(len(set(f.strip() for f in lig_frags if f.strip())))

    # Bridging halide (present in dimers like [Rh(CO)2Cl]2)
    has_bridge_X = (n_metals > 1 and
                    any(s in smiles for s in ['Cl', 'Br', 'I', 'F'])
                    and 'CO' in smiles)
    out[6] = 1.0 if has_bridge_X else 0.0

    # M-M bond flag (check for direct metal-metal bond in graph)
    has_mm = any(
        n.GetSymbol() in TARGET_METALS_SET
        for a in metal_atoms
        for n in a.GetNeighbors()
        if n.GetIdx() != a.GetIdx()
    )
    out[7] = 1.0 if has_mm else 0.0

    # Lability proxy: d10 metals with no CO are labile (easily displace PPh3)
    d_count = (mendeleev.element(metal_atoms[0].GetSymbol()).group_id or 10) - \
              _METAL_OX_STATE_FALLBACK.get(metal_atoms[0].GetSymbol(), 0)
    has_co = 'CO' in smiles or '[CO]' in smiles or 'OC' in smiles
    out[8] = 1.0 if (d_count >= 9 and not has_co) else 0.0   # labile
    out[9] = 1.0 if (abs(d_count - 8) < 1 and has_co) else 0.0  # robust

    # Approx MW
    try:
        from rdkit.Chem import Descriptors as _Desc
        mol_san = Chem.MolFromSmiles(smiles)
        if mol_san:
            out[10] = float(_Desc.MolWt(mol_san))
    except Exception:
        out[10] = 0.0

    out[11] = 0.0
    bad = ~np.isfinite(out)
    out[bad] = 0.0
    return out


# ── Physicochemical descriptors ───────────────────────────────────────────────

def get_physicochem_10(smiles):
    """
    9 physchem descriptors + 1 missing indicator.
    Patched: throwOnParamFailure=False to handle Si/Sn/Ge gracefully.
    """
    out = np.zeros(10, dtype=float)
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        out[9] = 1.0
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[9] = 1.0
        return out

    out[0] = finite(Descriptors.MolWt(mol))
    out[1] = finite(Descriptors.MolLogP(mol))
    out[2] = finite(Descriptors.TPSA(mol))
    out[3] = finite(Descriptors.NumRotatableBonds(mol))
    out[4] = finite(Descriptors.NumHAcceptors(mol))
    out[5] = finite(Descriptors.NumHDonors(mol))
    out[7] = finite(Descriptors.HallKierAlpha(mol))
    out[8] = finite(Descriptors.NumAromaticRings(mol))

    missing = 0.0
    try:
        # KEY FIX: throwOnParamFailure=False — Si/Sn/Ge will get charge=0, not crash
        AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=False)
        v = finite(Descriptors.MaxPartialCharge(mol))
        if not np.isfinite(v):
            missing = 1.0
            v = 0.0
    except Exception:
        missing = 1.0
        v = 0.0

    out[6] = v
    out[9] = missing
    return out


# ── TEP (Tolman Electronic Parameter) ────────────────────────────────────────

_tepid_model = None
_tep_features = None


def _load_tepid_model():
    global _tepid_model, _tep_features
    if _tepid_model is None:
        urllib.request.urlretrieve(TEP_MODEL_URL, "LGBMReg_model.pkl")
        _tepid_model = joblib.load("LGBMReg_model.pkl")
        try:
            _tep_features = _tepid_model.feature_name_
        except AttributeError:
            _tep_features = _tepid_model.booster_.feature_name()
    return _tepid_model, _tep_features


def get_tepid_value(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return [0.0, 1.0]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0, 1.0]

    tepid_model, tep_features = _load_tepid_model()

    try:
        all_descs = Descriptors.CalcMolDescriptors(mol)
        x_input = np.array(
            [[all_descs.get(feat, 0.0) for feat in tep_features]],
            dtype=np.float64
        )   # shape (1, 19)

        # Use booster_ directly — bypasses the broken sklearn wrapper
        tep_val = tepid_model.booster_.predict(x_input)[0]
        return [float(tep_val), 0.0]

    except Exception as e:
        print(f" ERROR [{type(e).__name__}]: {e}")
        return [0.0, 1.0]


# ── Sterics (Morfeus) ─────────────────────────────────────────────────────────

DEBUG_STERICS = True
_debug_errors = []  # collects tuples like (stage, smiles, exception_str)


def process_for_sterics(smiles, extra_remove=None):
    """
    Convert a raw SMILES into a representative ligand fragment for sterics.
    Removes metals (and optionally Sn/Si handles), fragments, and returns the most common P-containing fragment.
    """
    target_metals = {'Pd','Rh','Pt','Ag','Ir','Au','Cu','Co','Ni','Fe','Ru','Os'}
    remove = set(target_metals)
    if extra_remove:
        remove |= set(extra_remove)

    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        rwmol = Chem.RWMol(mol)
        idxs = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetSymbol() in remove]
        for idx in sorted(idxs, reverse=True):
            rwmol.RemoveAtom(idx)

        frag_mol = rwmol.GetMol()
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

        cands = []
        for f in frags:
            try:
                s = Chem.MolToSmiles(f, canonical=True)
                if s:
                    cands.append(s)
            except Exception:
                pass

        if not cands:
            return None

        p_cands = [s for s in cands if ('P' in s or 'p' in s)]
        if p_cands:
            return Counter(p_cands).most_common(1)[0][0]
        return Counter(cands).most_common(1)[0][0]

    except Exception as e:
        if DEBUG_STERICS:
            _debug_errors.append(("process_for_sterics", smiles, repr(e)))
        return None


def _heavy_atom_elements_coords(mol_with_H):
    conf = mol_with_H.GetConformer()
    heavy_idxs = [a.GetIdx() for a in mol_with_H.GetAtoms() if a.GetSymbol() != "H"]
    elements = [mol_with_H.GetAtomWithIdx(i).GetSymbol() for i in heavy_idxs]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idxs], dtype=float)
    return heavy_idxs, elements, coords


def get_phosphine_sterics(smiles):
    from morfeus import ConeAngle, BuriedVolume
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return 0.0, 0.0
    if ('P' not in smiles) and ('p' not in smiles):
        return 0.0, 0.0

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan, np.nan

        p_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
        if len(p_atoms) == 0:
            return 0.0, 0.0

        molH = Chem.AddHs(mol)

        ps = AllChem.ETKDGv3()
        ps.useRandomCoords = True
        ps.maxIterations = 2000
        if AllChem.EmbedMolecule(molH, ps) == -1:
            return np.nan, np.nan
        try:
            AllChem.UFFOptimizeMolecule(molH, maxIters=200)
        except Exception:
            pass

        conf = molH.GetConformer()

        # find a P atom index in the H-added molecule (same indexing as original heavy atoms)
        p_idx = next(a.GetIdx() for a in molH.GetAtoms() if a.GetSymbol() == "P")
        p_atom = molH.GetAtomWithIdx(p_idx)

        # build lone-pair direction from neighbors
        p_pos = np.array(conf.GetAtomPosition(p_idx))
        neighbors = p_atom.GetNeighbors()
        if not neighbors:
            return np.nan, np.nan

        vec_sum = np.zeros(3)
        for n in neighbors:
            vec_sum += (np.array(conf.GetAtomPosition(n.GetIdx())) - p_pos)

        norm = np.linalg.norm(vec_sum)
        if norm < 1e-8:
            return np.nan, np.nan

        lp_vec = -vec_sum / norm

        # heavy-atom-only set for Morfeus
        heavy_idxs, elements_heavy, coords_heavy = _heavy_atom_elements_coords(molH)

        # map P index -> heavy-atom coordinate index
        try:
            p_heavy_i = heavy_idxs.index(p_idx)
        except ValueError:
            return np.nan, np.nan

        # try increasing dummy distances to avoid vdW collisions
        for dist in [2.28, 2.60, 3.00, 3.50, 4.00]:
            dummy_pos = p_pos + lp_vec * dist

            elements = elements_heavy + ["Pd"]
            coords = np.vstack([coords_heavy, dummy_pos])

            # Morfeus uses 1-indexed central atom index [web:6]
            metal_idx = len(elements)  # 1-indexed central atom (dummy Pd is last)
            try:
              ca = ConeAngle(elements, coords, metal_idx, method="internal")  # or omit method=
              bv = BuriedVolume(elements, coords, metal_idx, radius=3.5)
              return float(ca.cone_angle), float(bv.fraction_buried_volume) * 100.0
            except Exception as e:
              if DEBUG_STERICS:
                _debug_errors.append(("morfeus", smiles, f"dist={dist}", repr(e)))
              continue


        return np.nan, np.nan

    except Exception:
        return np.nan, np.nan


def map_sterics_processed(df, src_col, prefix, processed_col, extra_remove=None):
    from sklearn.impute import SimpleImputer
    if src_col not in df.columns:
        raise KeyError(f"{src_col} not in df.columns")

    df[processed_col] = df[src_col].apply(lambda s: process_for_sterics(s, extra_remove=extra_remove))

    targets = (
        df[processed_col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    targets = targets[targets != ""].unique()

    print(f"{src_col}: raw unique={df[src_col].nunique(dropna=False)}; processed unique={len(targets)}")

    cache = {t: get_phosphine_sterics(t) for t in targets}

    df[f"{prefix}_cone_angle"] = df[processed_col].map(lambda x: cache.get(x, (0.0, 0.0))[0])
    df[f"{prefix}_buried_vol"] = df[processed_col].map(lambda x: cache.get(x, (0.0, 0.0))[1])


# ── ChemBERTa-2 ───────────────────────────────────────────────────────────────

print(f"Loading {CHEMBERTA_MODEL} …")
_cb_tok = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
_cb_mod = AutoModel.from_pretrained(CHEMBERTA_MODEL)
_cb_mod.eval()


def chemberta_batch(smiles_list, batch_size=32):
    """
    Returns CLS-token embeddings as np.ndarray (n, 384).
    Invalid / missing SMILES → zero vector (NOT flagged separately;
    the zero vector is a natural out-of-distribution signal).
    """
    n   = len(smiles_list)
    out = np.zeros((n, BERT_DIM), dtype=float)

    for i in range(0, n, batch_size):
        raw   = smiles_list[i : i + batch_size]
        clean = [s if (isinstance(s, str) and s.strip()) else "C"
                 for s in raw]                         # fallback = methane

        enc = _cb_tok(
            clean,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            h = _cb_mod(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        out[i : i + len(clean)] = h

    return out   # (n, 384)


def chemberta_feature_names(prefix):
    return [f"{prefix}_bert_{i}" for i in range(BERT_DIM)]


# ── Extended RDKit descriptors ────────────────────────────────────────────────

# Pre-compile patterns once
_SMARTS = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS_RAW.items()}
_N_SMARTS = len(_SMARTS)  # 20

_N_BASE   = 28          # indices 0–27
_N_TOTAL  = _N_BASE + _N_SMARTS + 2   # 28 + 20 + 2 = 50


def get_ext_rdkit(smiles):
    out = np.zeros(_N_TOTAL, dtype=float)
    missing_idx   = _N_TOTAL - 1          # index 49
    charge_rng_idx = _N_TOTAL - 2         # index 48

    if not isinstance(smiles, str) or not smiles.strip():
        out[missing_idx] = 1.0
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[missing_idx] = 1.0
        return out

    # ── Complexity / surface ────────────────────────────────────────────────
    out[0]  = _f(Descriptors.BertzCT(mol))
    out[1]  = _f(Descriptors.MolMR(mol))
    out[2]  = _f(Descriptors.LabuteASA(mol))
    out[3]  = _f(Descriptors.FractionCSP3(mol))

    # ── Valence connectivity (electronic topology) ─────────────────────────
    out[4]  = _f(Descriptors.Chi0v(mol))
    out[5]  = _f(Descriptors.Chi1v(mol))
    out[6]  = _f(Descriptors.Chi2v(mol))
    out[7]  = _f(Descriptors.Chi3v(mol))
    out[8]  = _f(Descriptors.Chi4v(mol))

    # ── Shape indices ──────────────────────────────────────────────────────
    out[9]  = _f(Descriptors.Kappa1(mol))
    out[10] = _f(Descriptors.Kappa2(mol))
    out[11] = _f(Descriptors.Kappa3(mol))

    # ── Ring system ───────────────────────────────────────────────────────
    out[12] = _f(rdMolDescriptors.CalcNumAliphaticRings(mol))
    out[13] = _f(rdMolDescriptors.CalcNumSaturatedRings(mol))
    out[14] = _f(Descriptors.RingCount(mol))
    out[15] = _f(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    out[16] = _f(rdMolDescriptors.CalcNumSpiroAtoms(mol))

    # ── Partial charge distribution ───────────────────────────────────────
    try:
        AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
        mn  = _f(Descriptors.MinPartialCharge(mol))
        mx  = _f(Descriptors.MaxPartialCharge(mol))
        out[17] = mn
        out[18] = _f(Descriptors.MaxAbsPartialCharge(mol))
        out[19] = _f(Descriptors.MinAbsPartialCharge(mol))
        out[charge_rng_idx] = mx - mn     # charge asymmetry / polarity spread
    except Exception:
        pass   # leave as 0 (no missing flag; Gasteiger failure is common for heteroatom-rich mols)

    # ── Atom composition ──────────────────────────────────────────────────
    out[20] = _f(rdMolDescriptors.CalcNumHeteroatoms(mol))
    out[21] = _f(Descriptors.NumRadicalElectrons(mol))
    out[22] = _f(Descriptors.NumValenceElectrons(mol))
    out[23] = _f(Descriptors.NOCount(mol))
    out[24] = _f(Descriptors.NHOHCount(mol))

    # ── Normalized / derived ─────────────────────────────────────────────
    mw   = _f(Descriptors.MolWt(mol))
    tpsa = _f(Descriptors.TPSA(mol))
    out[25] = tpsa / mw if mw > 0 else 0.0              # polarity per unit mass

    n_atoms  = mol.GetNumAtoms()
    n_arom   = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    out[26]  = n_arom / n_atoms if n_atoms > 0 else 0.0  # aromatic fraction

    # ── Coordination / binding site total (denticity proxy) ──────────────
    coord_total = 0
    for k in _COORD_KEYS:
        patt = _SMARTS.get(k)
        if patt is not None:
            coord_total += len(mol.GetSubstructMatches(patt))
    out[27] = float(coord_total)

    # ── Individual SMARTS counts ──────────────────────────────────────────
    for j, (name, patt) in enumerate(_SMARTS.items()):
        if patt is not None:
            out[_N_BASE + j] = float(len(mol.GetSubstructMatches(patt)))

    return out


# Build matching feature names for SHAP
def ext_rdkit_feature_names(prefix):
    base_names = [
        "bertzCT", "molMR", "labuteASA", "fracCSP3",
        "chi0v", "chi1v", "chi2v", "chi3v", "chi4v",
        "kappa1", "kappa2", "kappa3",
        "nAliphaticRings", "nSaturatedRings", "ringCount",
        "nBridgeheadAtoms", "nSpiroAtoms",
        "minPartialCharge", "maxAbsPartialCharge", "minAbsPartialCharge",
        "nHeteroatoms", "nRadicalElectrons", "nValenceElectrons",
        "NOCount", "NHOHCount",
        "TPSA_per_MW", "aromaticFrac", "nCoordSites",
    ]
    smarts_names = [f"smarts_{k}" for k in _SMARTS.keys()]
    tail = ["chargeRange", "ext_missing"]
    return [f"{prefix}_{n}" for n in base_names + smarts_names + tail]


_EXT_RDKIT_BASE_NAMES = [
    n.replace("x_", "") for n in ext_rdkit_feature_names("x")
]


assert len(ext_rdkit_feature_names("x")) == _N_TOTAL, \
    f"Name count mismatch: {len(ext_rdkit_feature_names('x'))} vs {_N_TOTAL}"


# ── 3D Shape ──────────────────────────────────────────────────────────────────

def get_3d_shape(smiles, n_conf=1):
    """
    3D shape descriptors with multi-strategy embedding for large macrocycles.
    Returns zeros with flag=-1.0 in last slot on failure.
    Output: [NPR1, NPR2, Asphericity, Eccentricity, InertialShapeFactor,
             PMI1, PMI2, PMI3, SpherocityIndex, missing_flag]
    """
    out = np.full(10, 0.0, dtype=float)
    out[9] = -1.0  # assume failure initially

    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    molH = Chem.AddHs(mol)

    # Strategy 1: Standard ETKDGv3
    ps = AllChem.ETKDGv3()
    ps.useRandomCoords = True
    ps.randomSeed = 42
    ps.maxIterations = 2000
    ps.numThreads = 1
    result = AllChem.EmbedMolecule(molH, ps)

    # Strategy 2: Disable pruning for macrocycles with large rings
    if result == -1:
        ps2 = AllChem.ETKDGv3()
        ps2.useRandomCoords = True
        ps2.randomSeed = 0
        ps2.maxIterations = 5000
        ps2.pruneRmsThresh = -1.0
        ps2.numThreads = 1
        result = AllChem.EmbedMolecule(molH, ps2)

    # Strategy 3: Distance geometry fallback
    if result == -1:
        try:
            result = AllChem.EmbedMolecule(molH, AllChem.ETKDG())
        except Exception:
            result = -1

    if result == -1:
        return out  # all zeros, missing_flag=-1

    try:
        AllChem.UFFOptimizeMolecule(molH, maxIters=500)
    except Exception:
        pass

    try:
        out[0] = finite(rdMolDescriptors.CalcNPR1(molH))
        out[1] = finite(rdMolDescriptors.CalcNPR2(molH))
        out[2] = finite(rdMolDescriptors.CalcAsphericity(molH))
        out[3] = finite(rdMolDescriptors.CalcEccentricity(molH))
        out[4] = finite(rdMolDescriptors.CalcInertialShapeFactor(molH))
        pmi = rdMolDescriptors.CalcPMI(molH)
        out[5] = finite(pmi[0])
        out[6] = finite(pmi[1])
        out[7] = finite(pmi[2])
        out[8] = finite(rdMolDescriptors.CalcSpherocityIndex(molH))
        out[9] = 0.0  # success
    except Exception:
        out[9] = -1.0

    return out


# ── VSA descriptors ───────────────────────────────────────────────────────────

def get_vsa_descriptors(smiles):
    """34 VSA features: PEOE_VSA1-14, SlogP_VSA1-10, SMR_VSA1-10."""
    out = np.zeros(34, dtype=float)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    for i in range(1, 15):
        fn = getattr(Descriptors, f"PEOE_VSA{i}", None)
        if fn: out[i - 1]      = _f(fn(mol))

    for i in range(1, 11):
        fn = getattr(Descriptors, f"SlogP_VSA{i}", None)
        if fn: out[13 + i]     = _f(fn(mol))   # indices 14–23

    for i in range(1, 11):
        fn = getattr(Descriptors, f"SMR_VSA{i}", None)
        if fn: out[23 + i]     = _f(fn(mol))   # indices 24–33

    return out


# ── Composition ───────────────────────────────────────────────────────────────

def get_composition(smiles):
    """8 features: n_heavy, DBE, N/C, O/C, H/C, S_present, halogen_frac, MW_per_heavy."""
    out = np.zeros(8, dtype=float)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    counts = {}
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        counts[s] = counts.get(s, 0) + 1

    C       = counts.get("C",  0)
    N       = counts.get("N",  0)
    O       = counts.get("O",  0)
    S       = counts.get("S",  0)
    P       = counts.get("P",  0)
    halogens = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))

    mol_h = Chem.AddHs(mol)
    H = sum(1 for a in mol_h.GetAtoms() if a.GetSymbol() == "H")

    n_heavy = mol.GetNumHeavyAtoms()
    dbe     = (2*C + 2 + N + P - H - halogens) / 2.0

    out[0] = float(n_heavy)
    out[1] = max(0.0, _f(dbe))
    out[2] = N / C          if C > 0 else float(N)
    out[3] = O / C          if C > 0 else float(O)
    out[4] = H / C          if C > 0 else float(H)
    out[5] = float(S > 0)                                   # sulfur present?
    out[6] = halogens / n_heavy if n_heavy > 0 else 0.0
    out[7] = _f(Descriptors.MolWt(mol)) / n_heavy if n_heavy > 0 else 0.0
    return out


# ── MACCS keys + fragments ────────────────────────────────────────────────────

def get_maccs(smiles):
    out = np.zeros(167, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out


_SELECTED_FRAGS = {
    "fr_COO":             Fragments.fr_COO,
    "fr_pyridine":        Fragments.fr_pyridine,
    "fr_benzene":         Fragments.fr_benzene,
    "fr_ether":           Fragments.fr_ether,
    "fr_ester":           Fragments.fr_ester,
    "fr_amide":           Fragments.fr_amide,
    "fr_NH0":             Fragments.fr_NH0,
    "fr_NH1":             Fragments.fr_NH1,
    "fr_NH2":             Fragments.fr_NH2,
    "fr_Ar_NH":           Fragments.fr_Ar_NH,
    "fr_phenol":          Fragments.fr_phenol,
    "fr_imide":           Fragments.fr_imide,
    "fr_sulfone":         Fragments.fr_sulfone,
    "fr_nitro":           Fragments.fr_nitro,
    "fr_urea":            Fragments.fr_urea,
}


def get_key_fragments(smiles):
    """15 RDKit fr_* fragment counts for MOF-relevant functional groups."""
    out = np.zeros(len(_SELECTED_FRAGS), dtype=float)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    for i, fn in enumerate(_SELECTED_FRAGS.values()):
        try:
            out[i] = float(fn(mol))
        except Exception:
            pass
    return out


# ── G14 hub topology ──────────────────────────────────────────────────────────

def _arms_from_hub(mol, hub_idx):
    """
    For each direct neighbor of hub_idx, trace a BFS arm until a
    coordination-site atom (P, N, O) is found.
    Returns list of (coord_symbol, bond_path_length, coord_atom_idx).
    One entry per arm that terminates at a coord site.
    """
    hub_atom = mol.GetAtomWithIdx(hub_idx)
    arms = []

    for root_nb in hub_atom.GetNeighbors():
        rn_idx = root_nb.GetIdx()
        rn_sym = root_nb.GetSymbol()

        # Direct neighbor is already a coord site (e.g. Si-P bond)
        if rn_sym in COORD_SYMBOLS:
            arms.append((rn_sym, 1, rn_idx))
            continue

        # BFS through this arm only — don't cross back through hub
        visited = {hub_idx, rn_idx}
        queue   = deque([(rn_idx, 1)])
        found   = None

        while queue:
            idx, depth = queue.popleft()
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            if sym in COORD_SYMBOLS:
                found = (sym, depth, idx)
                break                       # stop at first coord site in arm
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx not in visited:
                    visited.add(nb_idx)
                    queue.append((nb_idx, depth + 1))

        if found:
            arms.append(found)

    return arms


def _path_composition(mol, hub_idx, target_idx):
    """
    Characterise the bond path from hub to target.
    Returns (has_aromatic, has_triple, has_only_sp3_single).
    """
    try:
        path = rdmolops.GetShortestPath(mol, hub_idx, target_idx)
    except Exception:
        return False, False, False

    n_arom, n_triple, n_sp3 = 0, 0, 0
    for i in range(len(path) - 1):
        bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
        if bond.GetIsAromatic():
            n_arom += 1
        bt = bond.GetBondTypeAsDouble()
        if bt == 3.0:
            n_triple += 1
        if bt == 1.0 and not bond.GetIsAromatic() and not bond.IsInRing():
            n_sp3 += 1

    has_aromatic = n_arom > 0
    has_triple   = n_triple > 0
    # "alkyl" arm: only sp3 single bonds, no aromatic, no triple
    only_sp3     = (n_arom == 0 and n_triple == 0 and n_sp3 > 0)
    return has_aromatic, has_triple, only_sp3


def get_g14_hub_topology(smiles: str) -> np.ndarray:
    """
    25 features encoding the G14 hub and its arm topology.

    idx  name                  meaning
    ---  --------------------  -------------------------------------------------
    0    hub_present           1 if a G14 atom exists in molecule
    1    hub_degree            bonds on hub (degree 4 → tetratopic)
    2    hub_nCoordArms        arms reaching any coord site (topicity)
    3    hub_nP_arms           arms reaching a P atom
    4    hub_nN_arms           arms reaching an N atom
    5    hub_nO_arms           arms reaching an O atom
    6    hub_armLen_min        shortest arm (bonds, hub → coord site)
    7    hub_armLen_max        longest arm
    8    hub_armLen_mean       mean arm length
    9    hub_armLen_std        std of arm lengths (0 = perfectly symmetric)
    10   hub_armFrac_alkynyl   fraction of arms containing a triple bond
    11   hub_armFrac_aromatic  fraction of arms with ≥1 aromatic bond
    12   hub_armFrac_alkyl     fraction of arms with only sp3 single bonds
    13   hub_eccentricity      max graph distance from hub to any atom
    14   hub_centrality        1/eccentricity (high → hub is topological center)
    15   hub_elem_eneg         Pauling electronegativity of hub element
    16   hub_elem_covrad_pm    covalent radius (pm) — sets binding-site spacing
    17   hub_elem_period       row in periodic table (3=Si, 4=Ge, 5=Sn, 6=Pb)
    18   hub_isSi              OHE identity flags
    19   hub_isGe
    20   hub_isSn
    21   hub_isPb
    22   hub_nG14total         total G14 heavy atoms in molecule
    23   hub_fracG14           fraction of heavy atoms that are G14
    24   hub_missing           1 if SMILES invalid or G14 not found
    """
    out = np.zeros(25, dtype=float)

    if not isinstance(smiles, str) or not smiles.strip():
        out[24] = 1.0
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[24] = 1.0
        return out

    g14_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() in GROUP14_SYMBOLS]

    if not g14_atoms:
        return out  # all zeros, no missing flag — molecule simply has no G14

    out[0] = 1.0

    # Total G14 count / fraction
    out[22] = float(len(g14_atoms))
    n_heavy  = mol.GetNumHeavyAtoms()
    out[23]  = len(g14_atoms) / n_heavy if n_heavy > 0 else 0.0

    # ── Hub selection: highest-degree G14 atom (most arms) ─────────────────
    hub      = max(g14_atoms, key=lambda a: a.GetDegree())
    hub_idx  = hub.GetIdx()
    hub_sym  = hub.GetSymbol()

    out[1]  = float(hub.GetDegree())
    out[15] = G14_ENEG.get(hub_sym, 0.0)
    out[16] = float(G14_COVRAD.get(hub_sym, 0))
    out[17] = float(G14_PERIOD.get(hub_sym, 0))
    for i, sym in enumerate(['Si', 'Ge', 'Sn', 'Pb']):
        out[18 + i] = float(hub_sym == sym)

    # ── Graph eccentricity of hub ──────────────────────────────────────────
    try:
        dist_row = rdmolops.GetDistanceMatrix(mol)[hub_idx]
        ecc      = float(np.max(dist_row))
        out[13]  = ecc
        out[14]  = 1.0 / ecc if ecc > 0 else 0.0
    except Exception:
        pass

    # ── Arm topology ──────────────────────────────────────────────────────
    arms = _arms_from_hub(mol, hub_idx)

    if arms:
        out[2] = float(len(arms))
        out[3] = float(sum(1 for sym, _, _ in arms if sym == 'P'))
        out[4] = float(sum(1 for sym, _, _ in arms if sym == 'N'))
        out[5] = float(sum(1 for sym, _, _ in arms if sym == 'O'))

        lengths = [ln for _, ln, _ in arms]
        out[6] = float(min(lengths))
        out[7] = float(max(lengths))
        out[8] = float(np.mean(lengths))
        out[9] = float(np.std(lengths))

        # Arm backbone composition
        n_alkynyl = n_aromatic = n_alkyl = 0
        for _, _, coord_idx in arms:
            has_arom, has_triple, only_sp3 = _path_composition(
                mol, hub_idx, coord_idx
            )
            n_alkynyl  += int(has_triple)
            n_aromatic += int(has_arom)
            n_alkyl    += int(only_sp3)

        n_arms     = len(arms)
        out[10] = n_alkynyl  / n_arms
        out[11] = n_aromatic / n_arms
        out[12] = n_alkyl    / n_arms

    return out


assert len(G14_HUB_NAMES) == 25


# ── G14 SMARTS features ───────────────────────────────────────────────────────

G14_SMARTS = {k: Chem.MolFromSmarts(v) for k, v in G14_SMARTS_RAW.items()}
N_G14_SMARTS = len(G14_SMARTS)
G14_SMARTS_NAMES = list(G14_SMARTS_RAW.keys())


def get_g14_smarts_features(smiles: str) -> np.ndarray:
    """
    Returns match counts for each G14 SMARTS pattern + 1 missing flag.
    Total length: N_G14_SMARTS + 1
    """
    out = np.zeros(N_G14_SMARTS + 1, dtype=float)
    missing_idx = N_G14_SMARTS

    if not isinstance(smiles, str) or not smiles.strip():
        out[missing_idx] = 1.0
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[missing_idx] = 1.0
        return out

    for i, (name, patt) in enumerate(G14_SMARTS.items()):
        if patt is not None:
            out[i] = float(len(mol.GetSubstructMatches(patt)))

    return out


# ── TTP features ──────────────────────────────────────────────────────────────

_HUB_PROPS = {
    'Si': {'eneg': 1.90, 'cov_rad': 111, 'period': 3},
    'Ge': {'eneg': 2.01, 'cov_rad': 122, 'period': 4},
    'Sn': {'eneg': 1.96, 'cov_rad': 139, 'period': 5},
    'C':  {'eneg': 2.55, 'cov_rad': 77,  'period': 2},
}
_G14_SYMS = {'Si', 'Ge', 'Sn'}

# Precompiled SMARTS for structural motifs
_MOTIF_SMARTS = {
    'biphenyl':        Chem.MolFromSmarts('c1ccccc1-c1ccccc1'),
    'spirobifluorene': Chem.MolFromSmarts('C12(c3ccccc31)c1ccccc12'),
    'adamantane':      Chem.MolFromSmarts('C1C2CC3CC1CC(C2)C3'),
}

assert len(TTP_FEATURE_NAMES) == TTP_DIM


def _arm_backbone_type(mol, hub_idx, p_idx):
    """Classify hub→P arm as aryl/alkynyl/alkyl/vinyl."""
    try:
        path = list(rdmolops.GetShortestPath(mol, hub_idx, p_idx))
        has_triple = has_arom = False
        all_sp3_single = True
        for i in range(len(path) - 1):
            bond = mol.GetBondBetweenAtoms(path[i], path[i+1])
            if bond.GetBondTypeAsDouble() == 3.0:
                has_triple = True
                all_sp3_single = False
            if bond.GetIsAromatic():
                has_arom = True
                all_sp3_single = False
            if bond.GetBondTypeAsDouble() != 1.0 and not bond.GetIsAromatic():
                all_sp3_single = False
        if has_triple:   return 'alkynyl'
        if has_arom:     return 'aryl'
        if all_sp3_single: return 'alkyl'
        return 'vinyl'
    except Exception:
        return 'unknown'


def _arms_from_hub_ttp(mol, hub_idx):
    """BFS from hub: return list of (path_len, P_idx) for each P-containing arm."""
    hub_atom = mol.GetAtomWithIdx(hub_idx)
    arms = []
    for root_nb in hub_atom.GetNeighbors():
        rn_idx = root_nb.GetIdx()
        visited = {hub_idx, rn_idx}
        queue = deque([(rn_idx, 1)])
        found = None
        while queue:
            idx, depth = queue.popleft()
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'P':
                found = (depth, idx)
                break
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx not in visited:
                    visited.add(nb_idx)
                    queue.append((nb_idx, depth + 1))
        if found:
            arms.append(found)
    return arms


def _count_PPh2_groups(mol):
    """Count PPh2 groups: P connected to >=2 phenyl rings."""
    patt = Chem.MolFromSmarts('P(c1ccccc1)c1ccccc1')
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def get_ttp_features(smiles: str) -> np.ndarray:
    """
    52-feature vector for tetratopic phosphine linkers.
    Reliable for Si/Ge/Sn/C hubs — uses only 2D descriptors,
    no Gasteiger charges, no 3D embedding.
    """
    out = np.zeros(TTP_DIM, dtype=float)

    if not isinstance(smiles, str) or not smiles.strip():
        out[47] = 1.0
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[47] = 1.0
        return out

    # --- Identify hub atom ---
    g14_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() in _G14_SYMS]

    if not g14_atoms:
        # Check for quaternary carbon hub (tetrakis-aryl methane type)
        for a in mol.GetAtoms():
            if a.GetSymbol() == 'C' and a.GetDegree() == 4 and not a.GetIsAromatic():
                test_arms = _arms_from_hub_ttp(mol, a.GetIdx())
                if len(test_arms) >= 3:
                    g14_atoms = [a]
                    break

    if not g14_atoms:
        return out   # Not this linker type — all zeros, no missing flag

    hub = max(g14_atoms, key=lambda a: a.GetDegree())
    hub_idx = hub.GetIdx()
    hub_sym = hub.GetSymbol()

    out[0] = 1.0
    for oi, sym in [(1,'Si'),(2,'Ge'),(3,'Sn'),(4,'C')]:
        if hub_sym == sym:
            out[oi] = 1.0

    hp = _HUB_PROPS.get(hub_sym, {'eneg':0.,'cov_rad':0,'period':0})
    out[5]  = hp['eneg']
    out[6]  = float(hp['cov_rad'])
    out[7]  = float(hp['period'])
    out[8]  = float(hub.GetDegree())

    # --- Arms ---
    arms = _arms_from_hub_ttp(mol, hub_idx)
    n_arms = len(arms)
    out[9] = float(n_arms)

    if arms:
        lengths  = [a[0] for a in arms]
        p_indices = [a[1] for a in arms]
        out[11] = float(min(lengths))
        out[12] = float(max(lengths))
        out[13] = float(np.mean(lengths))
        out[14] = float(np.std(lengths)) if n_arms > 1 else 0.0

        types = [_arm_backbone_type(mol, hub_idx, pi) for pi in p_indices]
        n = len(types)
        out[10] = float(len(set(t for t in types if t != 'unknown')))
        out[15] = sum(1 for t in types if t == 'aryl')    / n
        out[16] = sum(1 for t in types if t == 'alkynyl') / n
        out[17] = sum(1 for t in types if t == 'alkyl')   / n
        out[18] = sum(1 for t in types if t == 'vinyl')   / n

        if n_arms == 2: out[19] = 1.0
        elif n_arms == 3: out[20] = 1.0
        elif n_arms >= 4: out[21] = 1.0

        out[22] = 1.0 if len(set(lengths)) == 1 else 0.0

        for plen, _ in arms:
            if 1 <= plen <= 8:
                out[39 + plen - 1] += 1.0

    # --- Hub graph properties ---
    try:
        dist_row = rdmolops.GetDistanceMatrix(mol)[hub_idx]
        ecc = float(np.max(dist_row))
        out[23] = ecc
        out[24] = 1.0 / ecc if ecc > 0 else 0.0
    except Exception:
        pass

    # --- Global 2D descriptors ---
    n_heavy = mol.GetNumHeavyAtoms()
    out[25] = float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'P'))
    out[26] = float(n_heavy)
    try:
        out[27] = finite(Descriptors.MolWt(mol))
        out[28] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        out[29] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        out[30] = float(rdMolDescriptors.CalcNumRings(mol))
        n_arom  = sum(1 for a in mol.GetAtoms()
                      if a.GetIsAromatic() and a.GetAtomicNum() > 1)
        out[31] = n_arom / n_heavy if n_heavy > 0 else 0.0
        out[32] = finite(Descriptors.TPSA(mol))
        out[33] = finite(Descriptors.MolLogP(mol))
        out[34] = finite(Descriptors.HallKierAlpha(mol))
    except Exception:
        pass

    # --- Structural motifs ---
    try:
        if _MOTIF_SMARTS['biphenyl'] and mol.HasSubstructMatch(_MOTIF_SMARTS['biphenyl']):
            out[35] = 1.0
        if out[28] >= 6:          # rough terphenyl proxy: ≥6 arene rings
            out[36] = 1.0
        if _MOTIF_SMARTS['spirobifluorene'] and mol.HasSubstructMatch(_MOTIF_SMARTS['spirobifluorene']):
            out[37] = 1.0
        if _MOTIF_SMARTS['adamantane'] and mol.HasSubstructMatch(_MOTIF_SMARTS['adamantane']):
            out[38] = 1.0
    except Exception:
        pass

    # --- Diagnostics ---
    out[48] = float(len(g14_atoms))
    out[49] = out[25] / n_heavy if n_heavy > 0 else 0.0
    try:
        n_pph2 = _count_PPh2_groups(mol)
        out[51] = float(n_pph2)
        # Ph2P-aryl arm: PPh2 group directly on an arene that connects to hub
        pph2_aryl_patt = Chem.MolFromSmarts('P(c1ccccc1)(c1ccccc1)c1ccccc1')
        out[50] = float(len(mol.GetSubstructMatches(pph2_aryl_patt))) if pph2_aryl_patt else 0.0
    except Exception:
        pass

    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0

    return out


# ── Additional linker fingerprints ────────────────────────────────────────────

def get_atom_pair_fp(smiles, n_bits=2048):
    """Hashed atom pair fingerprint — captures P···P through-bond distance."""
    out = np.zeros(n_bits, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    try:
        fp = Pairs.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out


def get_torsion_fp(smiles, n_bits=1024):
    out = np.zeros(n_bits, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    try:
        fp = Torsions.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, out)
    except Exception:
        pass
    return out


def get_graph_topo_descriptors(smiles):
    """
    14 descriptors: graph indices + EState summaries + flexibility metrics.
    Output: [Wiener, Ipc, BalabanJ, BertzCT, Chi0, Chi1, Chi2n, Chi3n,
             Kappa1, Kappa2, Kappa3, Fsp3, FractionCSP3, missing_flag]
    """
    out = np.zeros(14, dtype=float)
    if not isinstance(smiles, str) or not smiles.strip():
        out[13] = 1.0
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out[13] = 1.0
        return out
    try:
        out[0]  = finite(GraphDescriptors.BalabanJ(mol))
        out[1]  = finite(GraphDescriptors.BertzCT(mol))
        out[2]  = finite(GraphDescriptors.Chi0(mol))
        out[3]  = finite(GraphDescriptors.Chi1(mol))
        out[4]  = finite(GraphDescriptors.Chi2n(mol))
        out[5]  = finite(GraphDescriptors.Chi3n(mol))
        out[6]  = finite(GraphDescriptors.Kappa1(mol))
        out[7]  = finite(GraphDescriptors.Kappa2(mol))
        out[8]  = finite(GraphDescriptors.Kappa3(mol))
        # Wiener index via distance matrix
        dm = rdmolops.GetDistanceMatrix(mol)
        out[9]  = float(dm.sum() / 2.0)
        # Flexibility
        n_heavy = mol.GetNumHeavyAtoms()
        n_sp3_c = sum(
            1 for a in mol.GetAtoms()
            if a.GetSymbol() == 'C'
            and a.GetHybridization().name == 'SP3'
        )
        out[10] = n_sp3_c / n_heavy if n_heavy > 0 else 0.0  # Fsp3
        out[11] = finite(rdMolDescriptors.CalcFractionCSP3(mol))
        # Labute ASA approximation
        out[12] = finite(rdMolDescriptors.CalcLabuteASA(mol))
        out[13] = 0.0
    except Exception:
        out[13] = 1.0
    bad = ~np.isfinite(out)
    out[bad] = 0.0
    return out


def get_estate_fp(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return np.zeros(79, dtype=float)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(79, dtype=float)
    try:
        maxs, sums = EStateFP(mol)
        out = np.array(sums, dtype=float)
        out = np.where(np.isfinite(out), out, 0.0)
        return out  # length 79
    except Exception:
        return np.zeros(79, dtype=float)


# ── DRFP reaction fingerprint ─────────────────────────────────────────────────

def make_rxn_smiles(row):
    """
    Reactant side: precursor + linker + modulator (whatever is present).
    Product side: empty — we are predicting the outcome, not encoding it.
    Prefers canonical SMILES; falls back to raw if canonical is unavailable.
    """
    parts = []
    for col in ['smiles_precursor', 'smiles_linker1', 'smiles_modulator']:
        s_can = row.get(f'{col}_canon')
        s_raw = row.get(col)
        val = s_can if pd.notna(s_can) and str(s_can).strip() else s_raw
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return '.'.join(parts) + '>>' if parts else '>>'


# ── SOAP 3D descriptors ───────────────────────────────────────────────────────

from dscribe.descriptors import SOAP
from ase import Atoms

soap_species_set = set(SOAP_SPECIES)

soap = SOAP(
    species=SOAP_SPECIES,
    r_cut=6.0,
    n_max=8,
    l_max=6,
    average='outer',   # one vector per molecule
    sparse=False
)
# +1 for the per-row "soap_available" flag appended by run_soap_block
SOAP_DIM = soap.get_number_of_features() + 1


# Hardcoded geometries for small/rigid molecules that defeat ETKDGv3.
# Keys are canonical SMILES; values are (coords_Å, atomic_numbers).
_HARDCODED_3D = {
    # Carbon monoxide: linear, experimental C≡O bond length 1.128 Å
    "C#O":        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]]), np.array([6, 8])),
    "[C-]#[O+]":  (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]]), np.array([6, 8])),
}


def embed_organic_3d(smiles):
    """
    Multi-strategy ETKDGv3 + UFF embedding for organic (metal-free) SMILES.
    Returns (heavy_atom_coords, nuclear_charges) or (None, None) on failure.

    Falls back to _HARDCODED_3D for small/rigid molecules (e.g. C#O) that
    ETKDGv3 cannot embed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None

    # Check hardcoded table first (covers CO and its resonance form)
    if smiles in _HARDCODED_3D:
        coords, charges = _HARDCODED_3D[smiles]
        return coords.copy(), charges.copy()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        molH = Chem.AddHs(mol)

        # Strategy 1: Standard ETKDGv3
        ps = AllChem.ETKDGv3()
        ps.useRandomCoords = True
        ps.randomSeed = 42
        ps.maxIterations = 2000
        ps.numThreads = 1
        result = AllChem.EmbedMolecule(molH, ps)

        # Strategy 2: Disable RMSD pruning for large/macrocyclic molecules
        if result == -1:
            ps2 = AllChem.ETKDGv3()
            ps2.useRandomCoords = True
            ps2.randomSeed = 0
            ps2.maxIterations = 5000
            ps2.pruneRmsThresh = -1.0
            ps2.numThreads = 1
            result = AllChem.EmbedMolecule(molH, ps2)

        # Strategy 3: Classic ETKDG fallback
        if result == -1:
            result = AllChem.EmbedMolecule(molH, AllChem.ETKDG())

        if result == -1:
            return None, None

        try:
            AllChem.UFFOptimizeMolecule(molH, maxIters=500)
        except Exception:
            pass

        conf = molH.GetConformer()
        coords, charges = [], []
        for atom in molH.GetAtoms():
            if atom.GetSymbol() == 'H':
                continue  # heavy atoms only
            sym = atom.GetSymbol()
            charges.append(ELEMENT_TO_Z.get(sym, atom.GetAtomicNum()))
            coords.append(list(conf.GetAtomPosition(atom.GetIdx())))

        if not coords:
            return None, None

        return np.array(coords, dtype=float), np.array(charges, dtype=int)

    except Exception:
        return None, None


def run_soap_block(label, smiles_series):
    """
    Given a label (for printing) and a pandas Series of SMILES,
    returns (X_soap, feature_names) ready to append to X_final.
    """
    unique_smiles = [
        s for s in smiles_series.dropna().astype(str).map(str.strip).unique()
        if s
    ]
    print(f"\n-- {label}: {len(unique_smiles)} unique structures --")

    # Step 1: Generate conformers
    conform_cache = {}
    for smi in unique_smiles:
        coords, charges = embed_organic_3d(smi)
        if coords is not None and len(coords) >= 2:
            # Validate all elements are in SOAP_SPECIES
            unknown = {Z_TO_ELEMENT.get(z, f'Z={z}') for z in charges
                       if Z_TO_ELEMENT.get(z, f'Z={z}') not in soap_species_set}
            if unknown:
                print(f"  WARN {smi[:45]}: unknown elements {unknown} → zero vector")
                conform_cache[smi] = None
            else:
                conform_cache[smi] = (coords, charges)
                print(f"  OK   {smi[:55]} ({len(charges)} heavy atoms)")
        else:
            conform_cache[smi] = None
            print(f"  FAIL {smi[:55]}")

    n_ok = sum(1 for v in conform_cache.values() if v is not None)
    print(f"  Conformers: {n_ok}/{len(unique_smiles)} succeeded")

    # Step 2: Generate SOAP vectors (raw descriptor only, flag appended in Step 4)
    _soap_raw_dim = soap.get_number_of_features()
    vec_cache = {}
    for smi in unique_smiles:
        entry = conform_cache[smi]
        if entry is None:
            vec_cache[smi] = np.zeros(_soap_raw_dim, dtype=np.float32)
            continue
        try:
            coords, charges = entry
            symbols = [Z_TO_ELEMENT[z] for z in charges]
            atoms = Atoms(symbols=symbols, positions=coords)
            vec = soap.create(atoms).astype(np.float32)
            vec = np.where(np.isfinite(vec), vec, 0.0)
            vec_cache[smi] = vec
        except Exception as e:
            print(f"  SOAP FAILED {smi[:45]}: {e}")
            vec_cache[smi] = np.zeros(_soap_raw_dim, dtype=np.float32)

    # Step 3: Map to full dataset
    _zero_raw = np.zeros(_soap_raw_dim, dtype=np.float32)

    X_soap = np.stack(smiles_series.apply(
        lambda s: vec_cache.get(
            str(s).strip() if pd.notna(s) else '',
            _zero_raw
        )
    ).values)

    # Step 4: Append "soap_available" flag (1 = conformer succeeded, 0 = zero-padded)
    available_flag = smiles_series.apply(
        lambda s: float(conform_cache.get(str(s).strip() if pd.notna(s) else '', None) is not None)
    ).values.reshape(-1, 1).astype(np.float32)

    X = np.hstack([X_soap, available_flag])

    n_zero = (X_soap.sum(axis=1) == 0).sum()
    bad = ~np.isfinite(X)
    if bad.any():
        print(f"  WARNING: {bad.sum()} non-finite values → replacing with 0.0")
        X[bad] = 0.0
    print(f"  Output shape: {X.shape}  |  zero-vector rows: {n_zero}/{len(X_soap)}")

    names = [f'soap_{label.lower()}_{i}' for i in range(_soap_raw_dim)] + [f'soap_{label.lower()}_available']
    return X, names
