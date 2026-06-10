"""Record an experiment result.

Closes the BO loop without forcing the chemist to hand-edit the Excel file.
The page shows every recommendation from the latest BO batch as its own
editable card.  For each pick the chemist can:

  - Edit any synthesis condition that was recommended (the BO values are
    only defaults; nothing is locked in).
  - Enter the PXRD outcome and notes.
  - Click "Save this experiment" to append a row to the dataset, OR
  - Click "Discard this pick" to dismiss it without recording.

A separate "Manual entry" tab handles off-recommendation experiments.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Path bootstrap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import streamlit as st  # noqa: E402

from app.services import data_writer  # noqa: E402
from app.services.status import iteration_history  # noqa: E402
from app.ui.components import page_header, smiles_input  # noqa: E402
from app.ui.theme import PXRD_KEY  # noqa: E402

st.set_page_config(page_title="Record Result \u00b7 COMPASS",
                   layout="wide")

page_header(
    "Record an experiment result",
    caption=("Save the PXRD outcome of an experiment."),
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_float(val: Any, default: float) -> float:
    """Convert ``val`` to float, falling back to ``default`` on failure."""
    try:
        if val is None:
            return float(default)
        f = float(val)
        if f != f:   # NaN check without importing math
            return float(default)
        return f
    except (TypeError, ValueError):
        return float(default)


def _coerce_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in ("nan", "none", "na"):
        return default
    return s


@st.cache_data(show_spinner=False)
def _solvent_choices() -> list[str]:
    """All solvent names (uppercase) that have a VT-2005 sigma profile.

    This is the canonical universe the COSMO pipeline can featurize; any
    solvent outside this set would break ``add_solvent_cosmo_features`` on
    save. We also union in any solvents already observed in the experiment
    file so legacy entries never disappear from the dropdown if the index
    is rebuilt.
    """
    names: set[str] = set()
    try:
        from cosmo_features import load_cosmo_index, _DEFAULT_INDEX
        index_map, _, _, _ = load_cosmo_index(_DEFAULT_INDEX)
        names.update(index_map.keys())
    except Exception:
        pass
    try:
        import pandas as pd
        from app.services.status import DATA_FILE
        if os.path.exists(DATA_FILE):
            df = pd.read_excel(DATA_FILE, usecols=lambda c: str(c).startswith("solvent_") and not str(c).endswith(("_volume_ml", "_fraction")))
            for col in df.columns:
                vals = df[col].dropna().astype(str).str.strip().str.upper()
                names.update(v for v in vals if v and v not in ("NAN", "NONE", "NA"))
    except Exception:
        pass
    return sorted(n for n in names if n)


def _solvent_select(label: str, *, key: str, default: str = "", required: bool = False) -> str:
    """Type-to-filter solvent dropdown.

    Returns the uppercase solvent name, or "" for blank. ``default`` is
    matched case-insensitively against the choice list; an empty or unknown
    default leaves the box blank so the chemist can start typing immediately
    instead of seeing an autopopulated solvent or a "— none —" sentinel.
    """
    choices = _solvent_choices()

    default_norm = (default or "").strip().upper()
    if default_norm and default_norm in choices:
        initial_index: int | None = choices.index(default_norm)
    else:
        initial_index = None

    selection = st.selectbox(
        label,
        options=choices,
        index=initial_index,
        key=key,
        help="Start typing to filter — only solvents with a VT-2005 COSMO "
             "profile are listed."
             + ("" if required else ""),
        placeholder="Type to search…",
    )
    return selection or ""


def _build_row(
    *,
    experiment_id: str,
    linker_smiles: str,
    precursor_smiles: str,
    modulator_smiles: str,
    solvent_1: str,
    solvent_2: str,
    solvent_3: str,
    sol1_frac: float,
    sol2_frac: float,
    sol3_frac: float,
    total_volume_ml: float,
    temperature_k: float,
    equivalents: float,
    metal_over_linker: float,
    linker_conc: float,
    reaction_hours: float,
    pxrd_score: int,
    pxrd_comments: str,
) -> dict:
    v1 = round(total_volume_ml * sol1_frac, 4)
    v2 = round(total_volume_ml * sol2_frac, 4)
    v3 = round(total_volume_ml * sol3_frac, 4)

    # Existing rows store solvent names in uppercase (e.g. "DICHLOROMETHANE",
    # "N,N-DIMETHYLFORMAMIDE"). Match that so downstream lookups against the
    # COSMO solvent table key the same way.
    s1 = solvent_1.upper() if solvent_1 else None
    s2 = solvent_2.upper() if solvent_2 else None
    s3 = solvent_3.upper() if solvent_3 else None

    # Derive per-species concentrations and µmol amounts from linker_conc
    # (the direct BO knob), M:L ratio, and modulator equivalents so the
    # appended row matches the historical schema. Invariants:
    #   metal_conc  = M:L  × linker_conc
    #   mod_conc    = equiv × linker_conc
    #   total_conc  = linker_conc × (1 + M:L + equiv)
    #   umol_X      = X_conc × total_solvent_volume_ml
    metal_conc = metal_over_linker * linker_conc
    mod_conc   = equivalents * linker_conc
    total_conc = linker_conc * (1.0 + metal_over_linker + equivalents)
    umol_metal     = round(metal_conc  * total_volume_ml, 4)
    umol_linker    = round(linker_conc * total_volume_ml, 4)
    umol_modulator = round(mod_conc    * total_volume_ml, 4)

    return {
        "experiment_id":           experiment_id,
        "smiles_precursor":        precursor_smiles or None,
        "smiles_linker_1":         linker_smiles or None,
        "smiles_linker_2":         None,
        "smiles_modulator":        modulator_smiles or None,
        "umol_metal_precursor":    umol_metal,
        "umol_linker":             umol_linker,
        "umol_modulator":          umol_modulator,
        "metal_conc":              round(metal_conc,  6),
        "linker_conc":             round(linker_conc, 6),
        "mod_conc":                round(mod_conc,    6),
        "total_conc":              round(total_conc,  6),
        "solvent_1":               s1,
        "solvent_2":               s2,
        "solvent_3":               s3,
        "solvent_1_volume_ml":     v1,
        "solvent_2_volume_ml":     v2,
        "solvent_3_volume_ml":     v3,
        "solvent_1_fraction":      sol1_frac,
        "solvent_2_fraction":      sol2_frac,
        "solvent_3_fraction":      sol3_frac,
        "total_solvent_volume_ml": total_volume_ml,
        "temperature_k":           temperature_k,
        "equivalents":             equivalents,
        "metal_over_linker_ratio": metal_over_linker,
        "reaction_hours":          reaction_hours,
        "pxrd_score":              int(pxrd_score),
        "pxrd_comments":           pxrd_comments or None,
        "source_file":             "gui",
    }


def _try_save(row: dict, on_success_state_key: str) -> None:
    """Wrap the data_writer call with friendly error handling."""
    try:
        result = data_writer.append_row(row)
    except data_writer.ConcurrentEditError as exc:
        st.error(str(exc))
    except data_writer.WriteError as exc:
        st.error(str(exc))
    else:
        st.session_state[on_success_state_key] = {
            "experiment_id": result.experiment_id,
            "n_rows":        result.n_rows_after,
            "backup_path":   result.backup_path,
        }
        st.rerun()


# ── Session-state defaults ───────────────────────────────────────────────────
# Tracks per-pick "saved" outcomes so the card collapses into a green badge
# after the user records that experiment.  Discarded picks are persisted on
# disk via data_writer.discard_recommendation, so they survive page refresh.
st.session_state.setdefault("rr_pick_saved", {})   # {(iter, rank): {...}}
st.session_state.setdefault("rr_manual_saved", None)


# ── Tabs: BO recommendations vs. manual entry ────────────────────────────────
tab_bo, tab_manual = st.tabs([
    "From BO recommendations",
    "Manual entry",
])


# ───────────────────────────────────────────────────────────────────────────────
# Tab 1: From BO recommendations
# ───────────────────────────────────────────────────────────────────────────────
with tab_bo:
    history = iteration_history()

    if not history.recommendations:
        st.info(
            "No BO recommendations are on file yet. Run the **Recommend** page "
            "first, or use the **Manual entry** tab to record an experiment "
            "without a recommendation."
        )
    else:
        # ── Target chemistry (one set of SMILES — applies to every pick) ─────
        st.subheader("Target chemistry")
        st.caption("These SMILES apply to every pick from this batch. Edit any "
                   "value to override.")

        # Auto-prefill from the most recent Recommend run if the chemist
        # arrived from that page.
        last_chem = st.session_state.get("last_recommend_chemistry") or {}
        for ss_key, ck in [
            ("rr_bo_linker",    last_chem.get("linker")),
            ("rr_bo_precursor", last_chem.get("precursor")),
            ("rr_bo_modulator", last_chem.get("modulator")),
        ]:
            if ck and ss_key not in st.session_state:
                st.session_state[ss_key] = ck

        chem_cols = st.columns(3)
        with chem_cols[0]:
            linker_check = smiles_input("Linker SMILES", key="rr_bo_linker",
                                         required=True)
        with chem_cols[1]:
            precursor_check = smiles_input("Metal Precursor SMILES",
                                            key="rr_bo_precursor", required=True)
        with chem_cols[2]:
            modulator_check = smiles_input("Modulator SMILES",
                                            key="rr_bo_modulator", required=False)

        chemistry_ok = (
            linker_check.ok and not linker_check.is_blank
            and precursor_check.ok and not precursor_check.is_blank
            and (modulator_check.is_blank or modulator_check.ok)
        )
        if not chemistry_ok:
            st.warning("Provide valid linker and metal precursor SMILES above before "
                       "saving any pick.")

        st.divider()

        # ── Iteration picker ──────────────────────────────────────────────────
        iter_options = sorted(
            {int(r.get("iteration", 0)) for r in history.recommendations},
            reverse=True,
        )
        iter_col_a, iter_col_b = st.columns([2, 1])
        with iter_col_a:
            selected_iter = st.selectbox(
                "Show picks from iteration",
                options=iter_options,
                index=0,
                format_func=lambda i: f"Iteration {i}",
                key="rr_selected_iter",
            )
        with iter_col_b:
            show_discarded = st.checkbox(
                "Show previously-discarded picks",
                value=False,
                help="Toggle this to bring back picks you dismissed earlier.",
            )

        selected_batch = next(
            (r for r in history.recommendations
             if int(r.get("iteration", -1)) == int(selected_iter)),
            None,
        )
        candidates: list[dict] = list(selected_batch.get("top_candidates", [])) \
            if selected_batch else []

        if not candidates:
            st.info("This iteration has no recorded picks.")
        else:
            st.caption(
                f"Iteration **{selected_iter}** \u00b7 "
                f"{len(candidates)} pick(s) recommended \u00b7 "
                f"dataset size at the time: {selected_batch.get('n_data', '?')}"
            )

        # ── Per-pick cards ────────────────────────────────────────────────────
        for i, cand in enumerate(candidates):
            try:
                rank = int(cand.get("batch_rank", i + 1))
            except (TypeError, ValueError):
                rank = i + 1
            pick_id = (int(selected_iter), rank)
            pick_key = f"iter{pick_id[0]}_pick{pick_id[1]}"

            # Already saved this session?
            saved_info = st.session_state["rr_pick_saved"].get(pick_id)
            if saved_info is not None:
                st.success(
                    f"Pick #{rank} saved as experiment "
                    f"**{saved_info['experiment_id']}** "
                    f"(dataset is now {saved_info['n_rows']} rows)."
                )
                continue

            # Discarded?
            if data_writer.is_discarded(*pick_id):
                if show_discarded:
                    cdc1, cdc2 = st.columns([4, 1])
                    cdc1.caption(f"Pick #{rank} \u2014 discarded")
                    if cdc2.button("Restore", key=f"restore_{pick_key}"):
                        data_writer.undiscard_recommendation(*pick_id)
                        st.rerun()
                continue

            # Build a friendly header
            pred = cand.get("pxrd_predicted")
            unc  = cand.get("uncertainty")
            header = f"Pick #{rank}"
            if pred is not None:
                try:
                    header += f"  \u2014  predicted PXRD {float(pred):.1f}"
                except (TypeError, ValueError):
                    pass
            if unc is not None:
                try:
                    header += f"  \u00b1 {float(unc):.1f}"
                except (TypeError, ValueError):
                    pass

            with st.expander(header, expanded=(i == 0)):
                # ── Identifier ───────────────────────────────────────────────
                default_id = f"GUI-iter{int(selected_iter)}-pick{rank}"
                experiment_id = st.text_input(
                    "Experiment ID",
                    value=default_id,
                    key=f"id_{pick_key}",
                    help="Edit if your lab uses a different numbering scheme.",
                )

                # ── Solvent system ──────────────────────────────────────────
                # Solvent boxes start empty so the chemist can type the
                # solvent they actually used instead of overwriting a
                # pre-filled BO suggestion. Fractions still default from the
                # recommendation since they're a useful starting point.
                st.markdown("**Solvent system**")
                cs1, cs2, cs3, cs4 = st.columns(4)
                with cs1:
                    solvent_1 = _solvent_select(
                        "Solvent 1",
                        key=f"s1_{pick_key}",
                        default="",
                        required=True,
                    )
                with cs2:
                    sol1_frac = st.number_input(
                        "Fraction (v/v)",
                        min_value=0.0, max_value=1.0,
                        value=_coerce_float(cand.get("phi_1"), 1.0),
                        step=0.05, key=f"f1_{pick_key}",
                    )
                with cs3:
                    solvent_2 = _solvent_select(
                        "Solvent 2",
                        key=f"s2_{pick_key}",
                        default="",
                    )
                with cs4:
                    # Default solvent_2 fraction = 1 - phi_1 when the BO pick
                    # suggested a binary mix; 0 otherwise.
                    s2_suggested = _coerce_str(cand.get("solvent_2"), "")
                    sol2_default = max(0.0, 1.0 - sol1_frac) if s2_suggested else 0.0
                    sol2_frac = st.number_input(
                        "Fraction (v/v)",
                        min_value=0.0, max_value=1.0,
                        value=sol2_default,
                        step=0.05, key=f"f2_{pick_key}",
                    )

                cs5, cs6, cs7 = st.columns(3)
                with cs5:
                    solvent_3 = _solvent_select(
                        "Solvent 3",
                        key=f"s3_{pick_key}",
                        default="",
                    )
                with cs6:
                    sol3_frac = st.number_input(
                        "Fraction (v/v)",
                        min_value=0.0, max_value=1.0,
                        value=_coerce_float(cand.get("solvent_3_fraction"), 0.0),
                        step=0.05, key=f"f3_{pick_key}",
                    )
                with cs7:
                    total_volume_ml = st.number_input(
                        "Total solvent volume (mL)",
                        min_value=0.1,
                        value=_coerce_float(cand.get("total_solvent_volume_ml"), 2.0),
                        step=0.1, key=f"vol_{pick_key}",
                    )

                # Solvent fraction sanity check
                frac_sum = sol1_frac + sol2_frac + sol3_frac
                if solvent_1 and abs(frac_sum - 1.0) > 0.02:
                    st.warning(f"Solvent fractions sum to {frac_sum:.2f}; "
                               "should be 1.00.")

                # ── Process conditions ──────────────────────────────────────
                st.markdown("**Process conditions**")
                cp1, cp2, cp3, cp4 = st.columns(4)
                with cp1:
                    # Chemist enters \u00b0C for ease of use; we convert to K on save
                    # so the Excel file keeps its historical Kelvin schema.
                    default_c = _coerce_float(cand.get("temperature_k"), 343.15) - 273.15
                    temperature_c = st.number_input(
                        "Temperature (\u00b0C)",
                        min_value=-73.15, max_value=326.85,
                        value=default_c,
                        step=1.0, key=f"temp_{pick_key}",
                    )
                    temperature_k = temperature_c + 273.15
                    st.caption(f"= {temperature_k:.2f} K")
                with cp2:
                    equivalents = st.number_input(
                        "Modulator equivalents",
                        min_value=0.0,
                        value=_coerce_float(cand.get("equivalents"), 0.0),
                        step=1.0, key=f"eq_{pick_key}",
                    )
                with cp3:
                    metal_over_linker = st.number_input(
                        "Metal:linker ratio",
                        min_value=0.0,
                        value=_coerce_float(cand.get("metal_over_linker_ratio"), 1.0),
                        step=0.1, key=f"mlr_{pick_key}",
                    )
                with cp4:
                    linker_conc = st.number_input(
                        "Linker conc (\u00b5mol/mL)",
                        min_value=0.0,
                        value=_coerce_float(cand.get("linker_conc"), 5.0),
                        step=0.5, key=f"conc_{pick_key}",
                    )

                reaction_hours = st.number_input(
                    "Reaction time (h)",
                    min_value=0.0,
                    value=24.0,
                    step=1.0, key=f"hrs_{pick_key}",
                )

                # ── PXRD outcome ───────────────────────────────────────────
                st.markdown("**PXRD outcome**")
                st.caption(PXRD_KEY)
                cpx1, cpx2 = st.columns([1, 3])
                with cpx1:
                    pxrd_score = st.number_input(
                        "PXRD score (0-9)",
                        min_value=0, max_value=9, value=5, step=1,
                        key=f"pxrd_{pick_key}",
                    )
                with cpx2:
                    pxrd_comments = st.text_area(
                        "Notes (optional)",
                        placeholder="Phase observed, color, anomalies...",
                        height=80, key=f"notes_{pick_key}",
                    )

                # ── Action buttons ──────────────────────────────────────────
                act_save, act_disc = st.columns(2)
                save_disabled = not chemistry_ok or not solvent_1.strip() or not experiment_id.strip()

                if act_save.button(
                    "Save this experiment",
                    key=f"save_{pick_key}",
                    type="primary",
                    disabled=save_disabled,
                    width="stretch",
                ):
                    row = _build_row(
                        experiment_id=experiment_id.strip(),
                        linker_smiles=linker_check.canonical or linker_check.raw,
                        precursor_smiles=precursor_check.canonical or precursor_check.raw,
                        modulator_smiles=(modulator_check.canonical or modulator_check.raw)
                                          if not modulator_check.is_blank else "",
                        solvent_1=solvent_1.strip(),
                        solvent_2=solvent_2.strip(),
                        solvent_3=solvent_3.strip(),
                        sol1_frac=sol1_frac,
                        sol2_frac=sol2_frac,
                        sol3_frac=sol3_frac,
                        total_volume_ml=total_volume_ml,
                        temperature_k=temperature_k,
                        equivalents=equivalents,
                        metal_over_linker=metal_over_linker,
                        linker_conc=linker_conc,
                        reaction_hours=reaction_hours,
                        pxrd_score=pxrd_score,
                        pxrd_comments=pxrd_comments,
                    )
                    try:
                        result = data_writer.append_row(row)
                    except data_writer.ConcurrentEditError as exc:
                        st.error(str(exc))
                    except data_writer.WriteError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["rr_pick_saved"][pick_id] = {
                            "experiment_id": result.experiment_id,
                            "n_rows":        result.n_rows_after,
                            "backup_path":   result.backup_path,
                        }
                        st.rerun()

                if save_disabled and not chemistry_ok:
                    st.caption(":orange[Provide valid linker + metal precursor SMILES "
                               "at the top of the page first.]")

                if act_disc.button(
                    "Discard this pick",
                    key=f"disc_{pick_key}",
                    width="stretch",
                ):
                    data_writer.discard_recommendation(*pick_id)
                    st.rerun()


# ───────────────────────────────────────────────────────────────────────────────
# Tab 2: Manual entry
# ───────────────────────────────────────────────────────────────────────────────
with tab_manual:
    st.caption("Use this tab to record an experiment that wasn't suggested by "
               "the BO loop \u2014 e.g. a one-off control or a hand-designed "
               "synthesis.")

    # Identifier
    st.subheader("Identifier")
    m_experiment_id = st.text_input(
        "Experiment ID",
        value=data_writer.next_experiment_id(),
        key="m_experiment_id",
        help="Auto-suggested as `GUI-N`. Override if you use a different "
             "lab numbering scheme.",
    )

    st.subheader("Chemistry")
    m_linker_check    = smiles_input("Linker SMILES",    key="m_linker",
                                      required=True)
    m_precursor_check = smiles_input("Metal Precursor SMILES", key="m_precursor",
                                      required=True)
    m_modulator_check = smiles_input("Modulator SMILES", key="m_modulator",
                                      required=False)

    st.subheader("Solvent system")
    m_total_volume_ml = st.number_input(
        "Total solvent volume (mL)",
        min_value=0.1, value=2.0, step=0.1, key="m_total_vol",
    )
    mcs1, mcs2, mcs3 = st.columns(3)
    with mcs1:
        m_solvent_1 = _solvent_select("Solvent 1 name", key="m_s1",
                                       required=True)
        m_sol1_frac = st.number_input("Solvent 1 fraction (v/v)",
                                       min_value=0.0, max_value=1.0,
                                       value=1.0, step=0.05, key="m_f1")
    with mcs2:
        m_solvent_2 = _solvent_select("Solvent 2 name", key="m_s2")
        m_sol2_frac = st.number_input("Solvent 2 fraction (v/v)",
                                       min_value=0.0, max_value=1.0,
                                       value=0.0, step=0.05, key="m_f2")
    with mcs3:
        m_solvent_3 = _solvent_select("Solvent 3 name", key="m_s3")
        m_sol3_frac = st.number_input("Solvent 3 fraction (v/v)",
                                       min_value=0.0, max_value=1.0,
                                       value=0.0, step=0.05, key="m_f3")

    m_frac_sum = m_sol1_frac + m_sol2_frac + m_sol3_frac
    if m_solvent_1 and abs(m_frac_sum - 1.0) > 0.02:
        st.warning(f"Solvent fractions sum to {m_frac_sum:.2f}; should be 1.00.")

    st.subheader("Process conditions")
    mcp1, mcp2, mcp3 = st.columns(3)
    with mcp1:
        # Chemist enters \u00b0C for ease of use; we convert to K on save so the
        # Excel file keeps its historical Kelvin schema.
        m_temperature_c = st.number_input(
            "Temperature (\u00b0C)",
            min_value=-73.15, max_value=326.85, value=70.0, step=1.0,
            key="m_temp",
        )
        m_temperature_k = m_temperature_c + 273.15
        st.caption(f"= {m_temperature_k:.2f} K (saved to file)")
    with mcp2:
        m_equivalents = st.number_input(
            "Modulator equivalents",
            min_value=0.0, value=0.0, step=1.0, key="m_eq",
        )
        m_metal_over_linker = st.number_input(
            "Metal:linker ratio",
            min_value=0.0, value=1.0, step=0.1, key="m_mlr",
        )
    with mcp3:
        m_linker_conc = st.number_input(
            "Linker concentration (\u00b5mol/mL)",
            min_value=0.0, value=5.0, step=0.5, key="m_conc",
        )
        m_reaction_hours = st.number_input(
            "Reaction time (h)",
            min_value=0.0, value=24.0, step=1.0, key="m_hours",
        )

    st.subheader("PXRD outcome")
    st.caption(PXRD_KEY)
    m_pxrd_score = st.number_input(
        "PXRD score (0-9)",
        min_value=0, max_value=9, value=5, step=1, key="m_pxrd",
    )
    m_pxrd_comments = st.text_area(
        "Notes (optional)",
        placeholder="Phase observed, color, anomalies...",
        height=80, key="m_notes",
    )

    m_ready = (
        m_linker_check.ok and not m_linker_check.is_blank
        and m_precursor_check.ok and not m_precursor_check.is_blank
        and (m_modulator_check.is_blank or m_modulator_check.ok)
        and bool(m_solvent_1)
        and bool(m_experiment_id.strip())
    )
    if not m_ready:
        st.caption(":orange[Required: experiment ID, linker, metal precursor, and "
                   "at least one solvent.]")

    if st.button(
        "Save manual experiment",
        type="primary",
        disabled=not m_ready,
        width="stretch",
    ):
        row = _build_row(
            experiment_id=m_experiment_id.strip(),
            linker_smiles=m_linker_check.canonical or m_linker_check.raw,
            precursor_smiles=m_precursor_check.canonical or m_precursor_check.raw,
            modulator_smiles=(m_modulator_check.canonical or m_modulator_check.raw)
                              if not m_modulator_check.is_blank else "",
            solvent_1=m_solvent_1.strip(),
            solvent_2=m_solvent_2.strip(),
            solvent_3=m_solvent_3.strip(),
            sol1_frac=m_sol1_frac,
            sol2_frac=m_sol2_frac,
            sol3_frac=m_sol3_frac,
            total_volume_ml=m_total_volume_ml,
            temperature_k=m_temperature_k,
            equivalents=m_equivalents,
            metal_over_linker=m_metal_over_linker,
            linker_conc=m_linker_conc,
            reaction_hours=m_reaction_hours,
            pxrd_score=m_pxrd_score,
            pxrd_comments=m_pxrd_comments,
        )
        try:
            result = data_writer.append_row(row)
        except data_writer.ConcurrentEditError as exc:
            st.error(str(exc))
        except data_writer.WriteError as exc:
            st.error(str(exc))
        else:
            st.session_state["rr_manual_saved"] = {
                "experiment_id": result.experiment_id,
                "n_rows":        result.n_rows_after,
                "backup_path":   result.backup_path,
            }
            st.rerun()

    if st.session_state.get("rr_manual_saved"):
        info = st.session_state["rr_manual_saved"]
        st.success(
            f"Saved experiment **{info['experiment_id']}**. "
            f"Dataset is now {info['n_rows']} rows."
        )
        st.caption(f"Backup written to "
                   f"`{os.path.relpath(info['backup_path'], _PROJECT_ROOT)}`")
