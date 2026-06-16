"""Small reusable Streamlit fragments shared across pages."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from app.services.smiles_validator import SmilesCheck, render_png, validate
from app.ui.theme import friendly_columns

# -- Lab branding ---
# Resolve the lab logo once at import time. The components module lives at
# ``app/ui/components.py``, so the project root is two parents up.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
LAB_LOGO_PATH = os.path.join(_PROJECT_ROOT, "docs", "Picture2.png")


def page_header(title: str, *, caption: Optional[str] = None) -> None:
    """Render the lab logo + page title (+ optional caption) side-by-side.

    Used at the top of every page so the lab branding is consistent.  The
    logo lives at ``docs/Picture2.png`` under the project root; if it's
    missing the header still renders without it.
    """
    cols = st.columns([1, 6], vertical_alignment="center")
    with cols[0]:
        if os.path.exists(LAB_LOGO_PATH):
            st.image(LAB_LOGO_PATH, width=110)
    with cols[1]:
        st.title(title)
        if caption:
            st.caption(caption)


def smiles_input(label: str, key: str, *, required: bool, help_text: str = "") -> SmilesCheck:
    """Render one labelled SMILES box plus a 2D structure preview.

    Returns the SmilesCheck so the calling page can decide whether to enable
    its action button.
    """
    cols = st.columns([3, 1])
    with cols[0]:
        value = st.text_input(label, key=key, help=help_text or None,
                              placeholder="Paste SMILES here")
        check = validate(value)

        if check.is_blank:
            if required:
                st.caption(":red[Required]")
            else:
                st.caption("Optional - leave blank if not used.")
        elif not check.ok:
            st.error(check.error)
        else:
            st.success(f"Parsed: `{check.canonical}`")

    with cols[1]:
        png = render_png(check, size_px=200)
        if png is not None:
            st.image(png, caption="2D structure", width="stretch")
        elif check.is_blank:
            st.caption("(no structure)")

    return check


def status_pill(label: str, *, ok: bool) -> None:
    """Small colored badge - green for OK, amber for warning."""
    color = "#16a34a" if ok else "#f59e0b"
    st.markdown(
        f'<span style="background:{color}22;color:{color};'
        f'padding:4px 12px;border-radius:999px;font-weight:600;'
        f'border:1px solid {color}55;">{label}</span>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "") -> None:
    """Three-line stat card used on the Home page."""
    st.markdown(
        f"""
        <div style="background:#f5f5f7;border-radius:12px;padding:16px 20px;
                    border:1px solid #e5e5ea;height:100%;">
            <div style="font-size:0.85rem;color:#6b7280;margin-bottom:4px;">{label}</div>
            <div style="font-size:1.6rem;font-weight:700;color:#1c1c1e;">{value}</div>
            <div style="font-size:0.85rem;color:#6b7280;margin-top:4px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations_table(
    csv_path: str,
    top_n: int = 10,
    *,
    editor_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Editable BO recommendations table.

    Renders the recommendations CSV inside an ``st.data_editor`` so the
    chemist can tweak any synthesis condition or drop picks they don't
    intend to try (use the trash icon at the right of each row).  BO meta
    columns (rank, predicted score, uncertainty, acquisition value) are
    locked read-only so they can't drift away from what the surrogate
    actually said.

    The returned DataFrame reflects every edit and deletion the chemist
    made and keeps the original pipeline column names so callers can
    serialize it back to disk without re-translating headers.  Returns
    ``None`` if the CSV path doesn't exist.
    """
    import os
    if not csv_path or not os.path.exists(csv_path):
        st.info("No recommendation file to display yet.")
        return None

    df = pd.read_csv(csv_path).head(top_n)

    # Add a Celsius column next to temperature_k for easier reading.
    if "temperature_k" in df.columns and "temperature_c" not in df.columns:
        try:
            df.insert(
                df.columns.get_loc("temperature_k") + 1,
                "temperature_c",
                (df["temperature_k"].astype(float) - 273.15).round(1),
            )
        except Exception:
            pass

    # Build column_config so the editor shows friendly labels but the
    # underlying machine-readable column names stay intact for round-tripping.
    label_map = friendly_columns(df.columns)
    READ_ONLY_COLS = {
        "batch_rank", "pxrd_predicted", "uncertainty",
        "acquisition_value", "diversity_combined_score", "temperature_c",
    }
    column_config: dict = {}
    for col in df.columns:
        label = label_map.get(col, col)
        if col in READ_ONLY_COLS:
            column_config[col] = st.column_config.Column(label=label, disabled=True)
        else:
            column_config[col] = st.column_config.Column(label=label)

    key = editor_key or f"rec_editor_{os.path.basename(csv_path)}"
    edited = st.data_editor(
        df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config=column_config,
        key=key,
    )

    # Recompute the read-only Celsius column from any user edit to K so the
    # two stay in sync after the next rerun.
    if "temperature_k" in edited.columns and "temperature_c" in edited.columns:
        try:
            edited["temperature_c"] = (
                pd.to_numeric(edited["temperature_k"], errors="coerce") - 273.15
            ).round(1)
        except Exception:
            pass

    return edited


def about_panel() -> None:
    """Chemist-facing walkthrough of what this app does and how.

    Rendered on the landing page and the Home page so a new user can
    open the app and understand the project without leaving Streamlit.
    Kept in sync with `README.md` but written for a lab audience rather
    than a software audience.
    """
    with st.expander("About this app", expanded=False):
        st.markdown(
            "### The problem\n"
            "Low-Valent Metal-Organic Frameworks (LVMOFs) are built from "
            "electron-rich, low-oxidation-state metals like Pd(0), Rh(I), "
            "and Ir(I). They're notoriously hard to reproduce - small "
            "changes in solvent ratio, temperature, stoichiometry, or "
            "modulator can flip the product from fully crystalline to "
            "completely amorphous. The Cohen Lab has accumulated hundreds "
            "of LVMOF synthesis experiments over the years, but the "
            "chemical space of metal precursors, linkers, modulators, and "
            "process conditions is combinatorially huge, so pure "
            "trial-and-error is slow and expensive.\n\n"
            "### What this app does\n"
            "It wraps a machine-learning **surrogate model** of "
            "crystallinity around a **Bayesian Optimization (BO)** loop. "
            "Given the ~750 historical experiments in "
            "`data/Experiments_with_Calculated_Properties_no_linker.xlsx`, "
            "the surrogate learns to predict the PXRD crystallinity "
            "score (0-9, remapped to Amorphous / Partial / Crystalline) "
            "from molecular + process descriptors. BO then uses those "
            "predictions - together with the surrogate's uncertainty - "
            "to suggest which experiment to run next.\n\n"
            "### How the surrogate is built\n"
            "- **Features (~10,000 raw, then MI-filtered):** metal-centre "
            "descriptors, RDKit + Mordred ligand descriptors, DRFP "
            "reaction fingerprints, ChemBERTa-2 embeddings, COSMO-RS "
            "solvent-mixture moments, and process variables "
            "(temperature, concentrations, equivalents, solvent "
            "fractions, reaction time).\n"
            "- **Selection:** variance threshold → mutual-information "
            "selection with separate discrete/continuous budgets → "
            "SMOTE inside each training fold only.\n"
            "- **Cross-validation:** RepeatedStratifiedGroupKFold with "
            "groups from KMeans on a 2D UMAP embedding, so chemically "
            "similar experiments don't leak across folds.\n"
            "- **Models:** six Frank-Hall ordinal pipelines "
            "(RF / XGB × MI-only / CL+MI / CL-only). Hyperparameters "
            "are tuned with Optuna (TPE, maximizing quadratic weighted "
            "kappa).\n\n"
            "### How Bayesian Optimization closes the loop\n"
            "1. **Recommend** - you pick a linker/metal precursor/modulator "
            "on the Recommend page; the surrogate scores a generated "
            "candidate pool of process conditions and ranks the top "
            "batch by the chosen acquisition (LFBO / EI / Thompson).\n"
            "2. **Run the experiment** in the lab.\n"
            "3. **Record Result** - enter the PXRD outcome; the "
            "Excel dataset is updated in place.\n"
            "4. **Retrain Model** - refresh the surrogate on the new "
            "data (re-evaluate is minutes, full Optuna retune is "
            "hours).\n"
            "5. Repeat.\n\n"
            "### Where to look for what\n"
            "- **Recommend** - get next-experiment picks for a given "
            "linker/metal precursor/modulator.\n"
            "- **Record Result** - log the PXRD outcome of a completed "
            "experiment.\n"
            "- **Model Confidence** - surrogate calibration, "
            "convergence, and BO performance plots.\n"
            "- **BO Tools** - simulate (1 seed, fast) or evaluate "
            "(multi-seed, per-cluster mean ± std) a BO configuration, "
            "reset recommend state, inspect checkpoints.\n"
            "- **Retrain Model** - re-featurize or retrain after new "
            "data has been recorded.\n\n"
            "For the full technical writeup - feature blocks, "
            "dimensionality reduction, ordinal Frank-Hall decomposition, "
            "evaluation metrics, and SHAP analysis - see `README.md` in "
            "the repo."
        )


def show_log_panel(log_lines: list[str], *, max_lines: int = 200, height: int = 280) -> None:
    """Render the streaming stdout buffer in a fixed-height scroll box."""
    text = "\n".join(log_lines[-max_lines:]) or "(no output yet)"
    st.code(text, language="text")
    st.caption(f"showing last {min(len(log_lines), max_lines)} of {len(log_lines)} lines")
    _ = height  # reserved for future use; st.code already auto-scrolls
