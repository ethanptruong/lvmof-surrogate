"""Manage Data page — edit or remove existing experiment records.

Lets a chemist correct a mistyped value or delete a bad row without opening
the Excel file by hand.  Both actions are guarded:

  - Every write makes a timestamped backup first (handled in ``data_writer``).
  - A confirmation step ("are you sure?") is required before anything is
    edited or removed, so a stray click can't mutate the dataset.
  - A concurrent-edit guard aborts the write if the file changed underneath
    the page (e.g. another chemist saved a row in the meantime).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

# Path bootstrap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.services import data_writer  # noqa: E402
from app.ui.components import page_header  # noqa: E402
from app.ui.theme import PXRD_KEY, friendly, friendly_columns  # noqa: E402

st.set_page_config(page_title="Manage Data · COMPASS", layout="wide")

page_header(
    "Manage Data",
    caption="Edit or remove an experiment record from the dataset.",
)

# Columns a chemist may sensibly correct by hand.  Derived columns (umol_*,
# *_conc, the Mix_* COSMO features) are deliberately omitted: the Mix_* block is
# recomputed automatically on save, and the rest are best fixed through the
# Record-result form's derivations.  Order here drives the editor's row order.
EDITABLE_COLS: tuple[str, ...] = (
    "experiment_id",
    "smiles_precursor",
    "smiles_linker_1",
    "smiles_linker_2",
    "smiles_modulator",
    "solvent_1", "solvent_2", "solvent_3",
    "solvent_1_fraction", "solvent_2_fraction", "solvent_3_fraction",
    "solvent_1_volume_ml", "solvent_2_volume_ml", "solvent_3_volume_ml",
    "total_solvent_volume_ml",
    "temperature_k",
    "equivalents",
    "metal_over_linker_ratio",
    "linker_conc",
    "reaction_hours",
    "pxrd_score",
    "pxrd_comments",
)


def _disp(val: Any) -> str:
    """Render a cell value as an editable string ('' for missing)."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _smart_cast(s: str) -> Any:
    """Best-effort convert an edited string back to a number when it looks like
    one, so we don't silently turn ``343.15`` into the string ``"343.15"``.
    SMILES, solvent names, and ids stay as strings; blanks become ``None``."""
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none", "na"):
        return None
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return s


# ── Flash message from a just-completed action ───────────────────────────────
_flash = st.session_state.pop("mng_flash", None)
if _flash:
    kind, msg = _flash
    getattr(st, kind, st.info)(msg)


# ── Load the dataset ─────────────────────────────────────────────────────────
try:
    df = data_writer.read_experiment_df()
except data_writer.WriteError as exc:
    st.error(str(exc))
    st.stop()

if len(df) == 0:
    st.info("The dataset is empty — there are no experiments to edit or remove.")
    st.stop()

st.caption(f"Dataset has **{len(df)}** experiment(s).")

with st.expander("Browse all experiments", expanded=False):
    st.dataframe(
        df.rename(columns=friendly_columns(df.columns)),
        width="stretch",
    )
    st.caption("The left-most number is the row number used in the selector below.")


# ── Row selector ─────────────────────────────────────────────────────────────
def _row_label(i: int) -> str:
    exp = df.iloc[i].get("experiment_id") if "experiment_id" in df.columns else None
    return f"Row {i}" + (f" — {exp}" if exp not in (None, "") else "")


sel_idx = st.selectbox(
    "Select an experiment",
    options=list(range(len(df))),
    format_func=_row_label,
    key="mng_sel_idx",
)
row = df.iloc[sel_idx]
expected_id = row.get("experiment_id") if "experiment_id" in df.columns else None

st.divider()

tab_edit, tab_remove = st.tabs(["Edit entry", "Remove entry"])


# ───────────────────────────────────────────────────────────────────────────────
# Tab 1: Edit
# ───────────────────────────────────────────────────────────────────────────────
with tab_edit:
    editable = [c for c in EDITABLE_COLS if c in df.columns]
    if not editable:
        st.info("None of the editable fields are present in this dataset.")
    else:
        st.caption(
            "Edit any value below and click **Save changes**. Derived feature "
            "columns (solvent COSMO descriptors, µmol amounts) are recomputed "
            "or refreshed on the next **Retrain Model → Re-featurize**, so "
            "you only need to fix the inputs here."
        )
        st.caption(PXRD_KEY)

        base_table = pd.DataFrame({
            "Field": [friendly(c) for c in editable],
            "Value": [_disp(row[c]) for c in editable],
        })
        edited = st.data_editor(
            base_table,
            hide_index=True,
            width="stretch",
            disabled=["Field"],
            column_config={
                "Field": st.column_config.Column("Field", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="large"),
            },
            key=f"mng_edit_editor_{sel_idx}",
        )

        # Diff edited strings against the originals.
        updates: dict[str, Any] = {}
        for i, col in enumerate(editable):
            old_s = base_table.iloc[i]["Value"]
            new_s = str(edited.iloc[i]["Value"])
            if new_s != old_s:
                updates[col] = _smart_cast(new_s)

        pending = st.session_state.get("mng_pending_edit")
        is_pending = bool(pending) and pending.get("sel_idx") == sel_idx

        if not is_pending:
            if st.button("Save changes", type="primary", key="mng_edit_save"):
                if not updates:
                    st.info("No changes to save — every field is unchanged.")
                else:
                    st.session_state["mng_pending_edit"] = {
                        "sel_idx": sel_idx,
                        "expected_id": expected_id,
                        "updates": updates,
                    }
                    st.rerun()
        else:
            ups = pending["updates"]
            change_lines = "\n".join(
                f"- **{friendly(c)}**: `{_disp(row.get(c))}` → `{_disp(v)}`"
                for c, v in ups.items()
            )
            st.warning(
                f"**Confirm edit to experiment "
                f"`{expected_id if expected_id not in (None, '') else f'row {sel_idx}'}`**\n\n"
                f"{change_lines}\n\n"
                f"A timestamped backup is saved first, but this overwrites the "
                f"record in place. Continue?"
            )
            c_ok, c_cancel = st.columns(2)
            if c_ok.button("Confirm save", type="primary", key="mng_edit_confirm",
                           width="stretch"):
                try:
                    result = data_writer.update_row(
                        sel_idx, ups, expected_experiment_id=pending["expected_id"]
                    )
                except data_writer.WriteError as exc:
                    st.session_state.pop("mng_pending_edit", None)
                    st.error(str(exc))
                else:
                    st.session_state.pop("mng_pending_edit", None)
                    st.session_state["mng_flash"] = (
                        "success",
                        f"Updated experiment **{result.experiment_id}** "
                        f"({len(result.changed_columns)} field(s) changed). "
                        f"Backup: `{os.path.relpath(result.backup_path, _PROJECT_ROOT)}`",
                    )
                    st.rerun()
            if c_cancel.button("Cancel", key="mng_edit_cancel", width="stretch"):
                st.session_state.pop("mng_pending_edit", None)
                st.rerun()


# ───────────────────────────────────────────────────────────────────────────────
# Tab 2: Remove
# ───────────────────────────────────────────────────────────────────────────────
with tab_remove:
    st.caption("Removing an entry deletes the whole row from the dataset.")

    preview_cols = [c for c in EDITABLE_COLS if c in df.columns]
    preview = pd.DataFrame({
        "Field": [friendly(c) for c in preview_cols],
        "Value": [_disp(row[c]) for c in preview_cols],
    })
    st.dataframe(preview, hide_index=True, width="stretch")

    pending_del = st.session_state.get("mng_pending_delete")
    is_pending_del = bool(pending_del) and pending_del.get("sel_idx") == sel_idx

    if not is_pending_del:
        if st.button("Remove this entry", key="mng_del_start"):
            st.session_state["mng_pending_delete"] = {
                "sel_idx": sel_idx,
                "expected_id": expected_id,
            }
            st.rerun()
    else:
        label = expected_id if expected_id not in (None, "") else f"row {sel_idx}"
        st.warning(
            f"**Confirm removal of experiment `{label}`**\n\n"
            f"This permanently deletes the row from the dataset. A timestamped "
            f"backup is saved first, but it cannot be undone from this app. "
            f"Are you sure?"
        )
        d_ok, d_cancel = st.columns(2)
        if d_ok.button("Yes, remove it", type="primary", key="mng_del_confirm",
                       width="stretch"):
            try:
                result = data_writer.delete_row(
                    sel_idx, expected_experiment_id=pending_del["expected_id"]
                )
            except data_writer.WriteError as exc:
                st.session_state.pop("mng_pending_delete", None)
                st.error(str(exc))
            else:
                st.session_state.pop("mng_pending_delete", None)
                st.session_state["mng_flash"] = (
                    "success",
                    f"Removed experiment **{result.experiment_id}**. "
                    f"Dataset is now {result.n_rows_after} row(s). "
                    f"Backup: `{os.path.relpath(result.backup_path, _PROJECT_ROOT)}`",
                )
                st.rerun()
        if d_cancel.button("Cancel", key="mng_del_cancel", width="stretch"):
            st.session_state.pop("mng_pending_delete", None)
            st.rerun()
