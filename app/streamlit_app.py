"""Streamlit entry point for the LVMOF-Surrogate chemist GUI.

Run from the project root with::

    streamlit run app/streamlit_app.py

Streamlit auto-discovers any file under ``app/pages/`` that starts with a
number, so this entry point only needs to set page config, fix sys.path so
the existing modules (``main.py``, ``data_processing.py`` ...) import
cleanly, and render the landing screen.
"""

from __future__ import annotations

import os
import sys

# -- Path bootstrap ---
# Streamlit launches this file from any working directory, so push the
# project root onto sys.path before importing anything that lives at the
# repo root (config, main, data_processing, ...).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Also chdir so relative paths in main.py (e.g. "data/...", "checkpoints/...")
# resolve correctly.
os.chdir(_PROJECT_ROOT)

import streamlit as st  # noqa: E402  (after sys.path bootstrap)

from app.services.status import (dataset_summary, iteration_history,
                                  model_status)  # noqa: E402
from app.ui.components import (about_panel, metric_card, page_header,
                                status_pill)  # noqa: E402

# -- Page config ---
st.set_page_config(
    page_title="COMPASS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Sidebar ---
with st.sidebar:
    st.markdown("### COMPASS")
    st.caption("**C**losed-loop **O**ptimization of **M**OF **P**rocess "
               "**a**nd **S**olvent **S**ynthesis.")

    st.divider()
    st.caption("**Pages**")
    st.markdown(
        "- Home\n"
        "- Recommend\n"
        "- Record Result\n"
        "- Model Confidence\n"
        "- BO Tools\n"
        "- Retrain Model\n"
        "- Model Documentation"
    )

    st.divider()
    if st.button("Refresh status", width="stretch"):
        st.rerun()

# -- Landing page (also reachable via 1_Home) ---
page_header(
    "COMPASS",
    caption=("**C**losed-loop **O**ptimization of **M**OF **P**rocess "
             "**a**nd **S**olvent **S**ynthesis. Get next-experiment "
             "recommendations \u00b7 record results \u00b7 close the loop."),
)

about_panel()

dataset = dataset_summary()
history = iteration_history()
model   = model_status()

# -- Stat cards row ---
c1, c2 = st.columns(2)
with c1:
    metric_card(
        "Experiments recorded",
        f"{dataset.n_experiments:,}",
        sub=(f"updated {dataset.last_modified:%Y-%m-%d %H:%M}"
             if dataset.last_modified else ""),
    )
with c2:
    metric_card(
        "BO iterations run",
        f"{history.iteration}",
        sub=(f"last run {history.last_run_mtime:%Y-%m-%d %H:%M}"
             if history.last_run_mtime else "no runs yet"),
    )

st.write("")

# -- Model status pill ---
if model.tuned:
    status_pill(
        f"Model tuned ({model.tuned_at:%Y-%m-%d})" if model.tuned_at
        else "Model tuned",
        ok=True,
    )
else:
    status_pill("Model not yet tuned - using default hyperparameters", ok=False)

st.write("")
st.divider()

# -- Action buttons ---
st.subheader("What would you like to do?")
b1, b2 = st.columns(2)
with b1:
    st.markdown(
        "#### Get experiment recommendations\n"
        "Provide a target linker / metal precursor and the model will suggest the next "
        "synthesis conditions to try."
    )
    if st.button("Open Recommend page", type="primary", width="stretch"):
        try:
            st.switch_page("pages/2_Recommend.py")
        except Exception:
            st.info("Use the sidebar to open the **Recommend** page.")
with b2:
    st.markdown(
        "#### Record an experiment result\n"
        "Save the PXRD outcome of an experiment so the next round of "
        "recommendations learns from it."
    )
    if st.button("Open Record Result page", width="stretch"):
        try:
            st.switch_page("pages/3_Record_Result.py")
        except Exception:
            st.info("Use the sidebar to open the **Record Result** page.")

st.write("")
st.caption(
    "Data file: `" + os.path.relpath(dataset.df_path, _PROJECT_ROOT) + "` "
    "\u00b7 Project root: `" + _PROJECT_ROOT + "`"
)
