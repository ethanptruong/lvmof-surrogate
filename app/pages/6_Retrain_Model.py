"""Retrain Model page.

Dedicated, chemist-facing entry point for refreshing the surrogate model
after new experiments have been recorded.  For heavier/destructive tools
(BO simulation, reset state, checkpoint inventory) see the Admin page.
"""

from __future__ import annotations

import os
import sys
import time

# Path bootstrap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import streamlit as st  # noqa: E402

from app.services import bo_runner, trainer  # noqa: E402
from app.services.status import model_status  # noqa: E402
from app.ui.components import page_header, show_log_panel, status_pill  # noqa: E402

st.set_page_config(page_title="Retrain Model \u00b7 COMPASS",
                   layout="wide")

page_header(
    "Retrain Model",
    caption="Refresh the surrogate model after new experiments have been recorded.",
)

model = model_status()
if model.tuned:
    status_pill(
        f"Model tuned ({model.tuned_at:%Y-%m-%d})" if model.tuned_at
        else "Model tuned",
        ok=True,
    )
else:
    status_pill("Model not yet tuned - using default hyperparameters", ok=False)

st.write("")


def _stream_until_done(status: bo_runner.RunStatus, *, max_seconds: int = 6 * 60 * 60) -> None:
    progress = st.empty()
    log_box  = st.empty()
    started  = time.time()
    while status.running:
        progress.progress(0.5, text=status.phase)
        with log_box.container():
            show_log_panel(status.log, max_lines=400)
        time.sleep(2.0)
        if time.time() - started > max_seconds:
            break
    if status.error is None:
        progress.progress(1.0, text="Done")
    else:
        progress.progress(1.0, text="Failed")
        st.error(f"Job failed: {status.error}")
    with log_box.container():
        show_log_panel(status.log, max_lines=400)
    st.caption(f"Elapsed: {status.elapsed_seconds:.0f} s")


# ── Re-featurize ─────────────────────────────────────────────────────────────
st.subheader("Re-featurize from data file")
st.caption("Rebuilds `checkpoints/features.pkl` + `checkpoints/data.pkl` from "
           "the current Excel file. Fast (seconds to minutes). Run this after "
           "the chemist edits the Excel directly, or after dropping/adding "
           "columns.")
if st.button("Re-featurize"):
    st.session_state["retrain_refeat"] = trainer.start_refeaturize()

if st.session_state.get("retrain_refeat") is not None:
    _stream_until_done(st.session_state["retrain_refeat"], max_seconds=900)
    if not st.session_state["retrain_refeat"].running:
        if st.button("Clear re-featurize log"):
            st.session_state["retrain_refeat"] = None
            st.rerun()

st.divider()

# ── Retrain (re-evaluate only) ───────────────────────────────────────────────
st.subheader("Retrain (re-evaluate with existing hyperparameters)")
st.caption("Re-fits the surrogate with the current `best_params.pkl` and "
           "regenerates the evaluation plots. Much faster than a full retune "
           "(minutes, not hours). Use this as the default after adding a "
           "handful of new experiments.")
if st.button("Retrain model", type="primary"):
    st.session_state["retrain_reeval"] = trainer.start_retune(skip_tuning=True)

if st.session_state.get("retrain_reeval") is not None:
    _stream_until_done(st.session_state["retrain_reeval"])
    if not st.session_state["retrain_reeval"].running:
        if st.button("Clear retrain log"):
            st.session_state["retrain_reeval"] = None
            st.rerun()

st.divider()

# ── Full retune ──────────────────────────────────────────────────────────────
st.subheader("Full retune (Optuna hyperparameter search)")
st.caption("Runs the full Optuna hyperparameter search across every pipeline "
           "and then re-evaluates. **This typically takes several hours.** "
           "Only run this occasionally, or when the dataset has grown "
           "significantly.")
if st.button("Full retune"):
    st.session_state["retrain_full"] = trainer.start_retune(skip_tuning=False)

if st.session_state.get("retrain_full") is not None:
    _stream_until_done(st.session_state["retrain_full"])
    if not st.session_state["retrain_full"].running:
        if st.button("Clear full-retune log"):
            st.session_state["retrain_full"] = None
            st.rerun()
