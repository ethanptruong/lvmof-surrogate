"""Bayesian Optimization tools page.

Exposes BO-specific operations:
  - Simulate the BO loop (single seed, convergence + calibration plots)
  - Evaluate the BO loop (multi-seed, per-cluster AF/EF/hit-rate)
  - Reset BO recommend state
  - Inspect checkpoint inventory
  - Browse recommendation history

For retraining / re-featurizing the surrogate model, see the
**Retrain Model** page.
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

from app.services import bo_runner, trainer  # noqa: E402  (trainer used for reset_recommend_state)
from app.services.status import (CHECKPOINT_DIR, checkpoint_inventory,
                                  list_recommendation_csvs)  # noqa: E402
from app.ui.components import page_header, show_log_panel  # noqa: E402

st.set_page_config(page_title="BO Tools \u00b7 COMPASS",
                   layout="wide")

page_header(
    "Bayesian Optimization Tools",
    caption="Simulate or evaluate the BO loop, inspect checkpoints, and reset recommend state.",
)

# Common log helper for any of the buttons below.
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


# ── Simulate vs Evaluate ─────────────────────────────────────────────────────
st.subheader("Benchmark the BO loop")
st.markdown(
    "- **Simulate** — single random seed, fast. Produces convergence, "
    "top-k, simple-regret, and surrogate calibration plots in `docs/`. "
    "Use this to eyeball a specific surrogate/acquisition combo.\n"
    "- **Evaluate** — same loop, but repeated over N seeds with per "
    "chemistry-cluster AF/EF/hit-rate reported as mean ± std. Slower "
    "(roughly N× a single simulate) but gives statistically meaningful "
    "comparisons between configurations."
)

_SURROGATES = ["rf_mi", "rf_cl_mi", "rf_cl_only",
               "xgb_mi", "xgb_cl_mi", "xgb_cl_only"]
_ACQUISITIONS = ["lfbo", "lfbo_ssl", "ei", "thompson"]

sim_tab, eval_tab = st.tabs(["Simulate (1 seed)", "Evaluate (multi-seed)"])

with sim_tab:
    st.caption("Runs `main.py --bo --bo-mode simulate`. Outputs convergence + "
               "calibration plots used by the Model Confidence page.")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sim_surrogate = st.selectbox(
            "Surrogate", _SURROGATES, index=0, key="admin_sim_surrogate",
        )
    with col_s2:
        sim_acq = st.selectbox(
            "Acquisition", _ACQUISITIONS, index=0, key="admin_sim_acq",
        )
    if st.button("Run simulation", type="primary", key="admin_sim_btn"):
        st.session_state["admin_sim"] = bo_runner.start_simulate(
            surrogate=sim_surrogate,
            acquisition=sim_acq,
        )

    if st.session_state.get("admin_sim") is not None:
        _stream_until_done(st.session_state["admin_sim"], max_seconds=3600)
        if not st.session_state["admin_sim"].running:
            if st.button("Clear simulation log", key="admin_sim_clear"):
                st.session_state["admin_sim"] = None
                st.rerun()

with eval_tab:
    st.caption("Runs `main.py --bo --bo-mode evaluate`. Per-cluster AF/EF/hit "
               "tables + grouped bar charts across multiple seeds.")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        eval_surrogate = st.selectbox(
            "Surrogate", _SURROGATES, index=0, key="admin_eval_surrogate",
        )
    with col_e2:
        eval_acq = st.selectbox(
            "Acquisition", _ACQUISITIONS, index=0, key="admin_eval_acq",
        )
    with col_e3:
        eval_seeds = st.number_input(
            "Seeds", min_value=2, max_value=30, value=5, step=1,
            key="admin_eval_seeds",
            help="Number of random seeds. Runtime scales linearly.",
        )
    if st.button("Run evaluation", type="primary", key="admin_eval_btn"):
        st.session_state["admin_eval"] = bo_runner.start_evaluate(
            surrogate=eval_surrogate,
            acquisition=eval_acq,
            n_seeds=int(eval_seeds),
        )

    if st.session_state.get("admin_eval") is not None:
        _stream_until_done(st.session_state["admin_eval"], max_seconds=6 * 3600)
        if not st.session_state["admin_eval"].running:
            if st.button("Clear evaluation log", key="admin_eval_clear"):
                st.session_state["admin_eval"] = None
                st.rerun()

st.divider()

# ── Reset BO state ───────────────────────────────────────────────────────────
st.subheader("Reset BO recommend state")
st.caption("Deletes `checkpoints/recommend_state.pkl` so the iteration counter "
           "starts at 0. Existing experiment data is **not** touched.")
typed = st.text_input("Type RESET to confirm", key="admin_reset_typed")
if st.button("Reset BO state", type="secondary", disabled=(typed != "RESET")):
    if trainer.reset_recommend_state():
        st.success("BO recommend state cleared.")
    else:
        st.info("There was no BO recommend state to clear.")

st.divider()

# ── Checkpoint inventory ─────────────────────────────────────────────────────
st.subheader("Checkpoint inventory")
inventory = checkpoint_inventory()
if not inventory:
    st.info("`checkpoints/` is empty.")
else:
    import pandas as pd
    df = pd.DataFrame([
        {
            "Name":      e["name"],
            "Type":      "dir" if e["is_dir"] else "file",
            "Size (KB)": round(e["size"] / 1024, 1),
            "Modified":  e["mtime"].strftime("%Y-%m-%d %H:%M"),
        }
        for e in inventory
    ])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"Path: `{CHECKPOINT_DIR}`")

st.divider()

# ── Recommendation file inventory ────────────────────────────────────────────
st.subheader("Recommendation history")
csvs = list_recommendation_csvs()
if not csvs:
    st.info("No recommendation CSVs in `docs/` yet.")
else:
    for path in csvs[:10]:
        rel = os.path.relpath(path, _PROJECT_ROOT)
        st.write(f"- `{rel}`")
