"""Retrain Model page.

Chemist-facing entry point for refreshing the surrogate model after new
experiments have been recorded.  One primary action (update with the latest
data); the granular building blocks live under "Advanced".  For
heavier/destructive tools (BO simulation, reset state, checkpoint inventory)
see the Admin page.
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

st.set_page_config(page_title="Retrain Model · COMPASS",
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

st.info(
    "**COMPASS retrains itself automatically** on the first Friday evening of "
    "January, April, July and October: the full hyperparameter search re-runs "
    "on the complete dataset and every plot on this site is refreshed. "
    "You normally never need this page.\n\n"
    "Use the button below only if you've just recorded a batch of new "
    "experiments and want the recommendations and plots to include them "
    "**now**, without waiting for the next scheduled retrain.",
    icon="\U0001F916",
)

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


# ── Primary action: update with the latest experiments ──────────────────────
st.subheader("Update model with the latest experiments")
st.caption(
    "Re-reads the experiment spreadsheet, re-fits the model with its current "
    "tuned settings, and regenerates every evaluation plot on this site. "
    "Takes roughly 15–45 minutes; you can leave this page and come back."
)
if st.button("Update model now", type="primary"):
    st.session_state["retrain_update"] = trainer.start_update()

if st.session_state.get("retrain_update") is not None:
    _stream_until_done(st.session_state["retrain_update"])
    if not st.session_state["retrain_update"].running:
        if st.button("Clear update log"):
            st.session_state["retrain_update"] = None
            st.rerun()

st.divider()

# ── Advanced: the individual building blocks ─────────────────────────────────
with st.expander("Advanced (individual steps — most users never need these)"):
    st.markdown(
        "The update button above runs the first two steps together, which is "
        "almost always what you want. These are exposed separately for "
        "debugging. **Note the trap:** “Re-fit” alone reuses the cached "
        "features, so it will *not* see newly recorded experiments unless "
        "you re-featurize first."
    )

    st.subheader("1 · Re-featurize from data file")
    st.caption("Rebuilds `checkpoints/features.pkl` + `checkpoints/data.pkl` "
               "from the current Excel file. Run after editing the Excel "
               "directly or adding/dropping columns.")
    if st.button("Re-featurize"):
        st.session_state["retrain_refeat"] = trainer.start_refeaturize()

    if st.session_state.get("retrain_refeat") is not None:
        _stream_until_done(st.session_state["retrain_refeat"], max_seconds=900)
        if not st.session_state["retrain_refeat"].running:
            if st.button("Clear re-featurize log"):
                st.session_state["retrain_refeat"] = None
                st.rerun()

    st.subheader("2 · Re-fit with existing hyperparameters")
    st.caption("Re-fits the surrogate using the cached features and the "
               "current `best_params.pkl`, then regenerates the evaluation "
               "plots. Does NOT pick up new experiments by itself.")
    if st.button("Re-fit model"):
        st.session_state["retrain_reeval"] = trainer.start_retune(skip_tuning=True)

    if st.session_state.get("retrain_reeval") is not None:
        _stream_until_done(st.session_state["retrain_reeval"])
        if not st.session_state["retrain_reeval"].running:
            if st.button("Clear re-fit log"):
                st.session_state["retrain_reeval"] = None
                st.rerun()

    st.subheader("3 · Full retune (Optuna hyperparameter search)")
    st.caption("Re-runs the full hyperparameter search (600 Optuna trials). "
               "**Takes many hours on this server and slows the site for "
               "everyone.** Prefer the automatic quarterly retrain, or "
               "trigger it early from GitHub: Actions → “Quarterly "
               "model retrain” → Run workflow — that runs on a "
               "faster machine and leaves this site untouched.")
    if st.button("Full retune (hours)"):
        st.session_state["retrain_full"] = trainer.start_retune(skip_tuning=False)

    if st.session_state.get("retrain_full") is not None:
        _stream_until_done(st.session_state["retrain_full"])
        if not st.session_state["retrain_full"].running:
            if st.button("Clear full-retune log"):
                st.session_state["retrain_full"] = None
                st.rerun()
