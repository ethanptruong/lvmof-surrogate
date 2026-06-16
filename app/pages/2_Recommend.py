"""Recommend page - the primary chemist workflow.

Wizard layout:
  Step 1 - target chemistry (linker / precursor / modulator SMILES)
  Step 2 - batch size slider
  Step 3 - advanced options (collapsible)
  Then: Get Recommendations button -> background BO run -> friendly results table.
"""

from __future__ import annotations

import os
import re
import sys
import time

# Path bootstrap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.services import bo_runner, data_writer  # noqa: E402
from app.ui.components import (page_header, render_recommendations_table,
                                show_log_panel, smiles_input)  # noqa: E402
from config import (BO_BATCH_SIZE, BO_DEFAULT_ACQUISITION,  # noqa: E402
                    BO_DEFAULT_SURROGATE)

st.set_page_config(page_title="Recommend \u00b7 COMPASS",
                   layout="wide")

page_header(
    "Recommend Experiments",
    caption="Get the next batch of synthesis conditions",
)

# -- Step 1: target chemistry ---
st.subheader("Target Chemistry")
st.caption("Paste SMILES strings below. The 2D structures appear next to each "
           "field")

linker_check    = smiles_input("Linker SMILES",    key="rec_linker",
                                required=True,
                                help_text="The organic linker (e.g. a phosphine ligand).")
precursor_check = smiles_input("Metal Precursor SMILES", key="rec_precursor",
                                required=True,
                                help_text="The metal precursor complex.")
modulator_check = smiles_input("Modulator SMILES", key="rec_modulator",
                                required=False,
                                help_text="Optional - leave blank for none.")

chemistry_inputs_ok = (
    (linker_check.is_blank or linker_check.ok)
    and (precursor_check.is_blank or precursor_check.ok)
    and (modulator_check.is_blank or modulator_check.ok)
)
any_chemistry = not (linker_check.is_blank
                     and precursor_check.is_blank
                     and modulator_check.is_blank)

st.divider()

# -- Step 2: batch size ---
st.subheader("Number of Recommendations")
batch_size = st.slider("Batch size", min_value=1, max_value=10,
                       value=int(BO_BATCH_SIZE), step=1,
                       help="The model will recommend this many synthesis "
                            "conditions to try in parallel.")

st.divider()

# -- Step 3: advanced options ---
with st.expander("Advanced Options", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        surrogate = st.selectbox(
            "Surrogate model",
            options=["rf_mi", "rf_cl_mi", "xgb_mi", "xgb_cl_mi",
                     "rf_cl_only", "xgb_cl_only"],
            index=["rf_mi", "rf_cl_mi", "xgb_mi", "xgb_cl_mi",
                   "rf_cl_only", "xgb_cl_only"].index(BO_DEFAULT_SURROGATE)
                  if BO_DEFAULT_SURROGATE in
                     ["rf_mi", "rf_cl_mi", "xgb_mi", "xgb_cl_mi",
                      "rf_cl_only", "xgb_cl_only"] else 0,
            help="Random forest variants are recommended for ranking quality.",
        )
        _acq_options = ["lfbo", "lfbo_ssl", "ei", "thompson", "consensus",
                        "random"]
        acquisition = st.selectbox(
            "Acquisition function",
            options=_acq_options,
            index=_acq_options.index(BO_DEFAULT_ACQUISITION)
                  if BO_DEFAULT_ACQUISITION in _acq_options else 0,
            help="LFBO is the default and recovers Expected Improvement. "
                 "lfbo_ssl pseudo-labels the candidate pool with the surrogate "
                 "to boost the classifier when observed data is small.",
        )
    with col_b:
        feasibility = st.checkbox(
            "Use feasibility prior",
            value=False,
            help="Penalises recommendations whose temperature exceeds the "
                 "solvent boiling point.",
        )
        observed_pairs = st.checkbox(
            "Restrict to observed solvent pairs",
            value=True,
            help="Limit the search to solvent pairs that appear together in "
                 "the training data. This is recommended because COMPASS can recommend solvents that we know we have",
        )
        include_mlr = st.checkbox(
            "Include metal:linker ratio in search",
            value=False,
            help="Otherwise the metal:linker ratio is fixed by the SMILES "
                 "phosphine count.",
        )

st.divider()

# -- Action button ---
disabled_reason = None
if not chemistry_inputs_ok:
    disabled_reason = "Fix the SMILES errors above to continue."
elif not any_chemistry:
    disabled_reason = ("Provide at least one of linker / metal precursor / modulator "
                       "to target a specific chemistry.")

go = st.button("Get recommendations", type="primary",
               disabled=bool(disabled_reason), width="stretch")
if disabled_reason:
    st.caption(f":orange[{disabled_reason}]")

# -- Background job state ---
if "rec_status" not in st.session_state:
    st.session_state["rec_status"] = None
if "rec_csv" not in st.session_state:
    st.session_state["rec_csv"] = None
if "rec_meta" not in st.session_state:
    st.session_state["rec_meta"] = None

if go:
    st.session_state["rec_csv"] = None
    st.session_state["rec_meta"] = None
    st.session_state["rec_status"] = bo_runner.start_recommend(
        linker_smiles=linker_check.canonical or linker_check.raw or None,
        precursor_smiles=precursor_check.canonical or precursor_check.raw or None,
        modulator_smiles=modulator_check.canonical or modulator_check.raw or None,
        batch_size=batch_size,
        surrogate=surrogate,
        acquisition=acquisition,
        feasibility=feasibility,
        observed_pairs=observed_pairs,
        include_mlr=include_mlr,
    )

status: bo_runner.RunStatus | None = st.session_state.get("rec_status")

if status is not None:
    st.divider()
    st.subheader("Run progress")
    progress_box = st.empty()
    log_box      = st.empty()
    elapsed_box  = st.empty()

    # Soft progress bar based on the phase counter.
    PHASE_ORDER = [
        "Loading data...", "Loading features...", "Featurizing dataset...",
        "Building feature catalog...", "Fitting surrogate...",
        "Starting recommendation...", "Generating candidate pool...",
        "Sampling candidate pool...", "Scoring candidates...",
        "Selecting batch...", "Finalizing recommendations...", "Done",
    ]

    def _phase_index(phase: str) -> int:
        try:
            return PHASE_ORDER.index(phase)
        except ValueError:
            return 1

    # Wall-clock cap for the foreground polling loop. The background thread
    # keeps running regardless - this just stops the Streamlit render from
    # blocking the server indefinitely. A Recommend run that has to re-
    # featurize the full Excel after a Record-result write can easily take
    # 20-40 min (ChemBERTa-2 reload + COSMO enrichment + assemble_features +
    # candidate-pool scoring), so 10 min was too tight and produced false
    # "Done" reports.
    POLL_CAP_SECONDS = 30 * 60
    poll_started = time.time()
    timed_out = False
    while status.running:
        idx = _phase_index(status.phase)
        progress_box.progress(
            min(0.05 + 0.95 * idx / max(1, len(PHASE_ORDER) - 1), 0.99),
            text=status.phase,
        )
        with log_box.container():
            show_log_panel(status.log)
        elapsed_box.caption(f"Elapsed: {status.elapsed_seconds:.0f} s")
        time.sleep(1.0)
        if time.time() - poll_started > POLL_CAP_SECONDS:
            timed_out = True
            break

    # Final render after the job ends - or after the polling cap fires.
    # Critically, do NOT call this "Done" when the cap fired while the
    # background thread is still working; that's the trap that made the
    # page lie about success on long runs.
    if status.running:
        # Cap fired, job still alive in the background.
        progress_box.progress(0.99, text=f"Still running: {status.phase}")
        with log_box.container():
            show_log_panel(status.log)
        elapsed_box.caption(f"Elapsed: {status.elapsed_seconds:.0f} s")
        st.warning(
            "**This run is taking longer than the page's foreground polling "
            "window (30 min)** - it is still going in the background. "
            "Recommend re-featurizes the entire Excel when new experiments "
            "have been recorded, which can take 20-40 min for "
            "ChemBERTa-2 + COSMO enrichment + candidate-pool scoring. "
            "Click **Resume watching** to reconnect to the same job, or "
            "leave this tab alone - the run will finish in the background "
            "and the next time you load the page the latest CSV will "
            "appear under **Recommendations** below."
        )
        if st.button("Resume watching", key="rec_resume"):
            st.rerun()
    elif status.error is None:
        progress_box.progress(1.0, text="Done")
        with log_box.container():
            show_log_panel(status.log)
        elapsed_box.caption(f"Elapsed: {status.elapsed_seconds:.0f} s")
    else:
        progress_box.progress(1.0, text="Failed")
        with log_box.container():
            show_log_panel(status.log)
        elapsed_box.caption(f"Elapsed: {status.elapsed_seconds:.0f} s")
        st.error(f"Recommendation failed: {status.error}")

    # Only publish the result back to the session when the job actually
    # finished. Reading status.result while status.running is True was the
    # second half of the bug: the page rendered no CSV because the thread
    # hadn't reached the save step yet.
    if not status.running and status.error is None and status.result is not None:
        st.session_state["rec_csv"] = status.result.get("csv_path")
        st.session_state["rec_meta"] = status.result.get("meta")
    elif not status.running and status.error is None and status.result is None:
        st.warning(
            "The Recommend job finished but did not return a CSV path. "
            "This usually means the run errored after the save step or "
            "was interrupted. Check the run log above for stack traces."
        )
    # If the cap fired (timed_out and status.running is True), don't touch
    # rec_csv - leave the previous batch visible if there is one, rather
    # than blanking the results section.
    _ = timed_out

# -- Meta banners (similarity, calibration, layout) ---
meta = st.session_state.get("rec_meta")
if meta:
    st.divider()
    sim = meta.get("similarity")
    if sim:
        _ms = sim.get("max_sim")
        _lvl = sim.get("level")
        if _lvl == "low":
            st.error(
                f"**Extrapolation warning** - target chemistry has low "
                f"similarity to training data (max cosine = {_ms:.3f}). "
                f"The surrogate is extrapolating; these recommendations are "
                f"informed guesses and should be treated as exploratory."
            )
        elif _lvl == "medium":
            st.warning(
                f"Moderate similarity to training data "
                f"(max cosine = {_ms:.3f}). Some extrapolation risk - expect "
                f"wider error bars than the surrogate reports."
            )
        else:
            st.success(
                f"Target chemistry is close to training data "
                f"(max cosine = {_ms:.3f}) - surrogate predictions should be reliable."
            )
    cal = meta.get("calibration")
    if cal:
        _scale = cal.get("sigma_scale")
        _q = cal.get("quality")
        if _q == "poor":
            st.error(
                f"**Uncertainty poorly calibrated** (σ-scale = {_scale:.2f}). "
                f"The uncertainty column in the table is not trustworthy - "
                f"exploration vs. exploitation trade-off may be off. "
                f"Consider retraining the surrogate on more data."
            )
        elif _q == "ok":
            st.info(
                f"Uncertainty is roughly calibrated (σ-scale = {_scale:.2f}). "
                f"Treat the uncertainty column as an order-of-magnitude estimate."
            )
        # "good" → no banner; default state is fine.
    layout = meta.get("layout") or {}
    if not layout.get("ok", True):
        st.warning(layout.get("warning") or "Feature layout may be stale.")

# -- Results panel ---
csv_path = st.session_state.get("rec_csv")
if csv_path:
    st.divider()
    st.subheader("Recommendations")
    st.caption(
        "Edit any cell to override a synthesis condition, or use the trash "
        "icon (right edge of each row) to drop picks you won't pursue. "
        "Click **Save edits** to persist your changes \u2014 the Record-result "
        "page will then show the same edited batch."
    )
    edited_df = render_recommendations_table(csv_path, top_n=10)
    if edited_df is not None:
        # Iteration number is encoded in the CSV filename
        # (e.g. ``bo_recommendations_iter32.csv`` -> 32). Needed so we can
        # mirror edits back into ``recommend_state.pkl``.
        m = re.search(r"iter(\d+)", os.path.basename(csv_path))
        iteration = int(m.group(1)) if m else None

        col_save, col_dl = st.columns(2)
        with col_save:
            save_clicked = st.button(
                "Save edits",
                type="primary",
                width="stretch",
                disabled=iteration is None,
                help=("Overwrites the recommendations CSV with your edited "
                      "rows and updates the BO checkpoint so the same edits "
                      "appear on the Record-result page."),
            )
        with col_dl:
            st.download_button(
                "Download CSV",
                edited_df.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(csv_path),
                mime="text/csv",
                width="stretch",
            )

        if save_clicked and iteration is not None:
            try:
                # Diff dropped picks against the on-disk CSV BEFORE we
                # overwrite it so we know which ranks the chemist removed.
                original = pd.read_csv(csv_path)
                original_ranks: set[int] = set()
                if "batch_rank" in original.columns:
                    original_ranks = {
                        int(r) for r in original["batch_rank"].dropna().tolist()
                    }
                kept_ranks: set[int] = set()
                if "batch_rank" in edited_df.columns:
                    kept_ranks = {
                        int(r) for r in edited_df["batch_rank"].dropna().tolist()
                    }
                dropped_ranks = original_ranks - kept_ranks

                # 1) Save the edited rows over the original CSV.
                edited_df.to_csv(csv_path, index=False)
                # 2) Mirror the edits into recommend_state.pkl so the
                #    Record-result cards reflect them too.
                data_writer.update_recommendation_batch(
                    iteration, edited_df.to_dict(orient="records")
                )
                # 3) Mark every dropped pick as discarded so it disappears
                #    from the Record-result page even after a refresh.
                for r in dropped_ranks:
                    data_writer.discard_recommendation(iteration, r)
            except data_writer.WriteError as exc:
                st.error(f"Could not save edits: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save edits: {exc}")
            else:
                msg = f"Saved {len(edited_df)} edited recommendation(s)."
                if dropped_ranks:
                    msg += f" Discarded pick(s): {sorted(dropped_ranks)}."
                st.success(msg)

# -- SHAP explanation panel ---
_shap_meta = (meta or {}).get("shap_batch") if meta else None
if _shap_meta and _shap_meta.get("rows"):
    st.divider()
    st.subheader("Why these picks? (SHAP feature attribution)")
    st.caption(
        "For each recommendation, the features that most pushed the predicted "
        "crystallinity up (**+**, green) or down (**-**, red). SHAP values are "
        "in the same units as the crystallinity score."
    )
    for _row in _shap_meta["rows"]:
        _rank = _row.get("rank", "?")
        _desc = _row.get("row_descriptor") or {}
        _label_bits = []
        if "temperature_k" in _desc:
            _label_bits.append(f"T={_desc['temperature_k']}K")
        if "linker_conc" in _desc:
            _label_bits.append(f"linker_conc={_desc['linker_conc']}")
        if "solvent_1" in _desc:
            _s1 = _desc["solvent_1"]
            _s2 = _desc.get("solvent_2", "")
            _label_bits.append(f"{_s1}" + (f"/{_s2}" if _s2 and _s2 != "NA" else ""))
        _summary = " · ".join(str(b) for b in _label_bits)
        with st.expander(f"Rank {_rank} - {_summary}", expanded=(_rank == 1)):
            _contribs = _row.get("contributions") or []
            if not _contribs:
                st.caption("No attributions available.")
                continue
            _rows = []
            for c in _contribs:
                _rows.append({
                    "Feature":    c.get("name", "?"),
                    "SHAP":       c.get("shap_value", 0.0),
                    "Feature value": (f"{c['feature_value']:.3g}"
                                       if c.get("feature_value") is not None
                                       else "-"),
                })
            _df = pd.DataFrame(_rows)
            st.bar_chart(_df.set_index("Feature")["SHAP"])
            st.dataframe(_df, hide_index=True, width="stretch")
