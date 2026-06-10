"""SMILES validation + 2D structure rendering helpers.

The Recommend page calls these on every keystroke (within Streamlit's
session-scoped cache) so chemists see immediate feedback when they paste a
typo.  This is the UI-layer fix for the silent-fallback bug at
``bo_core.py:1644-1655`` where an invalid SMILES would silently degrade to
median features.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# Import the existing canonicalizer rather than re-deriving it.
from data_processing import canonicalize_smiles


@dataclass(frozen=True)
class SmilesCheck:
    """Result of validating one SMILES string.

    ``raw``       : the string the user typed (stripped, never None).
    ``canonical`` : RDKit-canonical form, or None if parsing failed.
    ``ok``        : True if RDKit successfully parsed the molecule.
    ``error``     : human-readable explanation, only set when ``ok`` is False.
    """

    raw: str
    canonical: Optional[str]
    ok: bool
    error: Optional[str]

    @property
    def is_blank(self) -> bool:
        return not self.raw.strip()


def validate(smiles: Optional[str]) -> SmilesCheck:
    """Validate one SMILES string.

    Blank input is treated as a no-op (``ok=True``, ``canonical=None``) so
    that optional fields (e.g. modulator) don't error.
    """
    if smiles is None:
        return SmilesCheck(raw="", canonical=None, ok=True, error=None)

    raw = smiles.strip()
    if not raw:
        return SmilesCheck(raw="", canonical=None, ok=True, error=None)

    canon = canonicalize_smiles(raw)
    if canon is None:
        return SmilesCheck(
            raw=raw,
            canonical=None,
            ok=False,
            error="RDKit could not parse this SMILES. Check for typos or "
                  "stray whitespace.",
        )
    return SmilesCheck(raw=raw, canonical=canon, ok=True, error=None)


@lru_cache(maxsize=256)
def _render_png_bytes(canonical_smiles: str, size_px: int) -> Optional[bytes]:
    """Cache 2D PNG renders by canonical SMILES (LRU, in-process)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(size_px, size_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_png(check: SmilesCheck, size_px: int = 240) -> Optional[bytes]:
    """Return PNG bytes of the 2D structure, or None if blank/invalid."""
    if not check.ok or not check.canonical:
        return None
    return _render_png_bytes(check.canonical, size_px)
