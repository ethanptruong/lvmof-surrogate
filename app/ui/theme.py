"""Theme tokens and friendly column-name mappings.

The friendly names are the **only** place where pipeline column identifiers
are translated into chemist-readable labels.  Both the Recommend and Record
pages import from here so that a rename only ever has to happen once.
"""

from __future__ import annotations

# ── Color palette (kept in sync with .streamlit/config.toml) ─────────────────
PRIMARY        = "#2563eb"   # blue accent
BACKGROUND     = "#ffffff"
SECONDARY_BG   = "#f5f5f7"
TEXT           = "#1c1c1e"
SUCCESS        = "#16a34a"
WARNING        = "#f59e0b"
ERROR          = "#dc2626"

# Highlight color used to mark the top batch_size rows in result tables.
TOP_PICK_BG    = "#dcfce7"   # very light green

# ── Column → friendly label mapping ──────────────────────────────────────────
# Used by both the Recommend results panel and the Record-result form.
FRIENDLY_LABELS: dict[str, str] = {
    # Recommendation outputs
    "batch_rank":              "Pick #",
    "pxrd_predicted":          "Predicted crystallinity (0-9)",
    "uncertainty":             "Model uncertainty (\u00b1)",
    "acquisition_value":       "BO score",
    # Process / synthesis variables
    "temperature_k":           "Temperature (K)",
    "temperature_c":           "Temperature (\u00b0C)",
    "equivalents":             "Modulator equivalents",
    "linker_conc":             "Linker concentration (\u00b5mol/mL)",
    "metal_over_linker_ratio": "Metal:linker ratio",
    "reaction_hours":          "Reaction time (h)",
    # Solvent system
    "solvent_1":               "Solvent 1",
    "solvent_2":               "Solvent 2",
    "solvent_3":               "Solvent 3",
    "phi_1":                   "Solvent 1 fraction (v/v)",
    "solvent_1_fraction":      "Solvent 1 fraction (v/v)",
    "solvent_2_fraction":      "Solvent 2 fraction (v/v)",
    "solvent_3_fraction":      "Solvent 3 fraction (v/v)",
    "solvent_1_volume_ml":     "Solvent 1 volume (mL)",
    "solvent_2_volume_ml":     "Solvent 2 volume (mL)",
    "solvent_3_volume_ml":     "Solvent 3 volume (mL)",
    "total_solvent_volume_ml": "Total solvent volume (mL)",
    # Amounts (computed for the displayed synthesis volume)
    "precursor_umol":          "Metal Precursor (\u00b5mol)",
    "linker_umol":             "Linker (\u00b5mol)",
    "modulator_umol":          "Modulator (\u00b5mol)",
    # Identifiers / SMILES
    "experiment_id":           "Experiment ID",
    "smiles_precursor":        "Metal Precursor SMILES",
    "smiles_linker_1":         "Linker SMILES",
    "smiles_modulator":        "Modulator SMILES",
    "pxrd_score":              "PXRD score (0-9)",
    "pxrd_comments":           "Notes",
}


def friendly(col: str) -> str:
    """Return the chemist-friendly label for a pipeline column name."""
    return FRIENDLY_LABELS.get(col, col)


def friendly_columns(cols) -> dict[str, str]:
    """Build a rename mapping for a list of columns (only known cols change)."""
    return {c: FRIENDLY_LABELS[c] for c in cols if c in FRIENDLY_LABELS}


# ── PXRD score key ────────────────────────────────────────────────────────────
# Shown next to the PXRD input on the Record-result page.
PXRD_KEY = (
    "**0\u20132** \u2014 amorphous  \u00a0\u00a0"
    "**3\u20136** \u2014 partially crystalline  \u00a0\u00a0"
    "**7\u20139** \u2014 crystalline"
)
