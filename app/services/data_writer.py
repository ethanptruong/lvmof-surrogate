"""Safe append-row writes to the experiment Excel file.

The Record-result page sends one row per save.  This module handles:

  1. Backing up the Excel file to ``data/backups/`` (timestamped).
  2. Holding a cross-process file lock so two chemists clicking Save at the
     same time can't corrupt the workbook.
  3. Validating that the new row has the right columns.
  4. Re-running ``add_solvent_cosmo_features`` so the Mix_* columns are
     populated for the new row before write-out.
  5. Touching the file mtime so ``main._data_file_fingerprint`` invalidates
     ``checkpoints/features.pkl`` on the next BO run.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from app.services.status import CHECKPOINT_DIR, DATA_FILE, PROJECT_ROOT

# -- Cross-process file lock (optional dep) ---
try:
    from filelock import FileLock, Timeout
    _LOCK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _LOCK_AVAILABLE = False
    FileLock = None  # type: ignore[assignment]
    Timeout = Exception  # type: ignore[assignment,misc]


BACKUP_DIR = os.path.join(PROJECT_ROOT, "data", "backups")
LOCK_PATH  = DATA_FILE + ".lock"
LOCK_TIMEOUT_S = 10.0

# Required columns the chemist must supply for a meaningful row.  Anything
# missing here is a fatal validation error in ``append_row``.
REQUIRED_COLS: tuple[str, ...] = (
    "experiment_id",
    "smiles_precursor",
    "smiles_linker_1",
    "solvent_1",
    "solvent_1_volume_ml",
    "total_solvent_volume_ml",
    "temperature_k",
    "pxrd_score",
)


# -- Public dataclasses ---

@dataclass
class WriteResult:
    """Outcome of a successful append. Returned to the page."""

    experiment_id: Any
    backup_path: str
    n_rows_after: int


@dataclass
class DeleteResult:
    """Outcome of a successful row deletion. Returned to the Manage-data page."""

    experiment_id: Any
    backup_path: str
    n_rows_after: int


@dataclass
class UpdateResult:
    """Outcome of a successful in-place row edit."""

    experiment_id: Any
    backup_path: str
    changed_columns: list[str]
    n_rows_after: int


class WriteError(Exception):
    """Friendly write failure (validation, lock contention, concurrent edit)."""


class ConcurrentEditError(WriteError):
    """Raised when the Excel file changed mtime between read and write."""


# -- Helpers ---

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _validate_row(row: dict[str, Any]) -> None:
    missing = [c for c in REQUIRED_COLS if c not in row or pd.isna(row.get(c))]
    if missing:
        raise WriteError(
            "The following required fields are missing or blank: "
            + ", ".join(missing)
        )

    score = row.get("pxrd_score")
    try:
        score_f = float(score)
    except (TypeError, ValueError) as exc:
        raise WriteError(f"pxrd_score must be a number, got {score!r}") from exc
    if not (0 <= score_f <= 9):
        raise WriteError(f"pxrd_score must be between 0 and 9, got {score_f}")


def _backup(src: str) -> str:
    _ensure_dir(BACKUP_DIR)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name  = f"Experiments_{stamp}.xlsx"
    dst   = os.path.join(BACKUP_DIR, name)
    shutil.copy2(src, dst)
    return dst


def next_experiment_id() -> str:
    """Suggest a fresh experiment_id for a new GUI-recorded row.

    The existing dataset uses opaque alphanumeric codes (e.g. ``RES-3-004a``)
    that don't map cleanly to a numeric counter, so the GUI emits a
    distinct ``GUI-N`` namespace where N is the count of rows already
    flagged with ``source_file == "gui"`` plus one.  The chemist can
    override the suggestion in the form.
    """
    if not os.path.exists(DATA_FILE):
        return "GUI-1"
    try:
        df = pd.read_excel(DATA_FILE)
    except Exception:
        return "GUI-1"
    n_gui = 0
    if "source_file" in df.columns and "experiment_id" in df.columns:
        gui_rows = df[df["source_file"].astype(str).str.lower() == "gui"]
        n_gui = int(gui_rows["experiment_id"].notna().sum())
    return f"GUI-{n_gui + 1}"


# -- Public write API ---

def append_row(row: dict[str, Any]) -> WriteResult:
    """Append a single experiment row, returning a ``WriteResult``.

    Raises ``WriteError`` (or subclass) on validation / contention failure.
    """
    _validate_row(row)

    if _LOCK_AVAILABLE:
        lock = FileLock(LOCK_PATH, timeout=LOCK_TIMEOUT_S)
        try:
            with lock:
                return _do_append(row)
        except Timeout as exc:
            raise WriteError(
                "Another save is in progress. Wait a moment and try again."
            ) from exc
    # No filelock installed - degrade gracefully (single-user mode).
    return _do_append(row)


def _do_append(row: dict[str, Any]) -> WriteResult:
    if not os.path.exists(DATA_FILE):
        raise WriteError(
            f"Dataset file is missing: {DATA_FILE}. "
            f"Ask the lab admin to restore it from a backup."
        )

    mtime_before = os.path.getmtime(DATA_FILE)
    try:
        df = pd.read_excel(DATA_FILE)
    except PermissionError as exc:
        raise WriteError(
            "The data file is open in Excel. Close it and try again."
        ) from exc

    # Drop the artifact "Unnamed" columns the same way ``load_data`` does so
    # we don't accidentally re-write them with our new row.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")

    # Build the new row aligned to the existing schema.  Unknown extra
    # columns from ``row`` are added (e.g. notes); columns missing from
    # ``row`` get NaN.
    new_row = {col: row.get(col) for col in df.columns}
    for k, v in row.items():
        if k not in new_row:
            new_row[k] = v
    df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Populate Mix_* features for the appended row using the canonical wrapper.
    from data_processing import add_solvent_cosmo_features
    df_new = add_solvent_cosmo_features(df_new)

    # Backup the original file before overwriting.
    backup = _backup(DATA_FILE)

    # Concurrent-edit guard: re-check mtime under the lock.
    mtime_after_read = os.path.getmtime(DATA_FILE)
    if mtime_after_read != mtime_before:
        raise ConcurrentEditError(
            "Another user just saved an experiment. Refresh the page and "
            "try again so you don't overwrite their work."
        )

    try:
        df_new.to_excel(DATA_FILE, index=False)
    except PermissionError as exc:
        raise WriteError(
            "The data file is locked by another program (likely Excel). "
            "Close it and try again."
        ) from exc

    # Touch mtime so the BO featurization cache invalidates on next run.
    now = datetime.utcnow().timestamp()
    os.utime(DATA_FILE, (now, now))

    return WriteResult(
        experiment_id=row.get("experiment_id"),
        backup_path=backup,
        n_rows_after=len(df_new),
    )


# -- Read / edit / delete API ---
# The Manage-data page lets a chemist correct or remove an existing row.  Both
# operations re-use the same safety machinery as ``append_row``: a cross-process
# file lock, a timestamped backup before every write, and a concurrent-edit
# guard so two people editing the workbook at once can't silently clobber each
# other.

def read_experiment_df() -> pd.DataFrame:
    """Return the experiment workbook as a DataFrame.

    Reads it the same way the append path does (dropping the artifact
    ``Unnamed`` columns and resetting the index) so the positional row numbers
    the GUI shows line up exactly with the rows ``delete_row`` / ``update_row``
    act on.
    """
    if not os.path.exists(DATA_FILE):
        raise WriteError(
            f"Dataset file is missing: {DATA_FILE}. "
            f"Ask the lab admin to restore it from a backup."
        )
    try:
        df = pd.read_excel(DATA_FILE)
    except PermissionError as exc:
        raise WriteError(
            "The data file is open in Excel. Close it and try again."
        ) from exc
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=unnamed, errors="ignore").reset_index(drop=True)


def _check_row_identity(
    df: pd.DataFrame, row_index: int, expected_experiment_id: Any
) -> Any:
    """Validate ``row_index`` is in range and (optionally) still points at the
    row the chemist selected. Returns the actual experiment_id at that row.

    The experiment_id check is the concurrent-edit guard: if someone else
    deleted or reordered rows after the page was loaded, the id at this
    position will no longer match and we abort rather than edit the wrong row.
    """
    if row_index < 0 or row_index >= len(df):
        raise WriteError(
            f"Row {row_index} is out of range - the file has {len(df)} row(s). "
            f"Refresh the page and try again."
        )
    actual_id = (
        df.iloc[row_index].get("experiment_id")
        if "experiment_id" in df.columns else None
    )
    if expected_experiment_id is not None and str(actual_id) != str(expected_experiment_id):
        raise ConcurrentEditError(
            "The data file changed since you loaded this page - the selected "
            "row no longer matches. Refresh and try again so you don't edit "
            "the wrong experiment."
        )
    return actual_id


def delete_row(row_index: int, *, expected_experiment_id: Any = None) -> DeleteResult:
    """Delete one row (by position) from the experiment file.

    Backs the file up first and aborts (``ConcurrentEditError``) if the row at
    ``row_index`` no longer carries ``expected_experiment_id``.
    """
    if _LOCK_AVAILABLE:
        lock = FileLock(LOCK_PATH, timeout=LOCK_TIMEOUT_S)
        try:
            with lock:
                return _do_delete(row_index, expected_experiment_id)
        except Timeout as exc:
            raise WriteError(
                "Another save is in progress. Wait a moment and try again."
            ) from exc
    return _do_delete(row_index, expected_experiment_id)


def _do_delete(row_index: int, expected_experiment_id: Any) -> DeleteResult:
    df = read_experiment_df()
    actual_id = _check_row_identity(df, row_index, expected_experiment_id)

    backup = _backup(DATA_FILE)
    df_new = df.drop(index=row_index).reset_index(drop=True)

    try:
        df_new.to_excel(DATA_FILE, index=False)
    except PermissionError as exc:
        raise WriteError(
            "The data file is locked by another program (likely Excel). "
            "Close it and try again."
        ) from exc

    # Touch mtime so the BO featurization cache invalidates on next run.
    now = datetime.utcnow().timestamp()
    os.utime(DATA_FILE, (now, now))

    return DeleteResult(
        experiment_id=actual_id,
        backup_path=backup,
        n_rows_after=len(df_new),
    )


def update_row(
    row_index: int,
    updates: dict[str, Any],
    *,
    expected_experiment_id: Any = None,
) -> UpdateResult:
    """Apply ``updates`` (column → new value) to one row, in place.

    Only the columns present in ``updates`` are touched; unknown columns are
    added to the schema. After applying the edit the Mix_* solvent features are
    recomputed (same as the append path) so they stay consistent with any
    edited solvent. Backs the file up first and guards against concurrent edits.
    """
    if not updates:
        raise WriteError("No changes to save.")

    # Validate the PXRD score up front if the chemist edited it.
    if "pxrd_score" in updates and updates["pxrd_score"] is not None:
        try:
            score_f = float(updates["pxrd_score"])
        except (TypeError, ValueError) as exc:
            raise WriteError(
                f"pxrd_score must be a number, got {updates['pxrd_score']!r}"
            ) from exc
        if not (0 <= score_f <= 9):
            raise WriteError(f"pxrd_score must be between 0 and 9, got {score_f}")

    if _LOCK_AVAILABLE:
        lock = FileLock(LOCK_PATH, timeout=LOCK_TIMEOUT_S)
        try:
            with lock:
                return _do_update(row_index, updates, expected_experiment_id)
        except Timeout as exc:
            raise WriteError(
                "Another save is in progress. Wait a moment and try again."
            ) from exc
    return _do_update(row_index, updates, expected_experiment_id)


def _do_update(
    row_index: int, updates: dict[str, Any], expected_experiment_id: Any
) -> UpdateResult:
    df = read_experiment_df()
    _check_row_identity(df, row_index, expected_experiment_id)

    changed: list[str] = []
    for col, val in updates.items():
        if col not in df.columns:
            df[col] = None
        df.at[row_index, col] = val
        changed.append(col)

    # Recompute Mix_* solvent features so an edited solvent stays consistent
    # with its COSMO descriptors. Done before the backup/write so a featurizer
    # failure (e.g. an unknown solvent) aborts cleanly without touching disk.
    try:
        from data_processing import add_solvent_cosmo_features
        df = add_solvent_cosmo_features(df)
    except Exception as exc:  # noqa: BLE001
        raise WriteError(
            f"Could not recompute solvent features for the edited row: {exc}. "
            f"Check that any solvent you entered has a COSMO profile."
        ) from exc

    backup = _backup(DATA_FILE)

    try:
        df.to_excel(DATA_FILE, index=False)
    except PermissionError as exc:
        raise WriteError(
            "The data file is locked by another program (likely Excel). "
            "Close it and try again."
        ) from exc

    now = datetime.utcnow().timestamp()
    os.utime(DATA_FILE, (now, now))

    actual_id = (
        df.iloc[row_index].get("experiment_id")
        if "experiment_id" in df.columns else None
    )
    return UpdateResult(
        experiment_id=actual_id,
        backup_path=backup,
        changed_columns=changed,
        n_rows_after=len(df),
    )


# -- Discarded-recommendation persistence ---
# A small JSON sidecar file under ``checkpoints/`` records which BO picks the
# chemist has explicitly dismissed.  Discarded picks are hidden from the
# Record-result page even after a refresh / restart, but the underlying
# ``recommend_state.pkl`` is left untouched (the BO loop never re-suggests
# already-tried conditions anyway, so this is purely a UI filter).

DISCARDED_PATH = os.path.join(CHECKPOINT_DIR, "discarded_recommendations.json")


def _load_discarded() -> set[tuple[int, int]]:
    if not os.path.exists(DISCARDED_PATH):
        return set()
    try:
        with open(DISCARDED_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return set()
    out: set[tuple[int, int]] = set()
    for entry in data:
        try:
            out.add((int(entry["iteration"]), int(entry["rank"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _save_discarded(items: set[tuple[int, int]]) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    payload = [{"iteration": i, "rank": r} for (i, r) in sorted(items)]
    with open(DISCARDED_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def is_discarded(iteration: Any, rank: Any) -> bool:
    try:
        key = (int(iteration), int(rank))
    except (TypeError, ValueError):
        return False
    return key in _load_discarded()


def discard_recommendation(iteration: Any, rank: Any) -> None:
    """Mark a (iteration, rank) recommendation as dismissed."""
    items = _load_discarded()
    items.add((int(iteration), int(rank)))
    _save_discarded(items)


def undiscard_recommendation(iteration: Any, rank: Any) -> None:
    """Restore a previously-discarded recommendation."""
    items = _load_discarded()
    items.discard((int(iteration), int(rank)))
    _save_discarded(items)


# -- Recommendation batch editing ---

def update_recommendation_batch(
    iteration: Any,
    candidates: list[dict],
) -> None:
    """Replace the ``top_candidates`` of one recommendation batch in
    ``recommend_state.pkl``.

    The Recommend page uses this so chemist edits made in the inline
    data-editor are visible on the Record-result page (which reads its
    per-pick cards from the same joblib state).  Atomic via a temp file +
    ``os.replace`` so a crash mid-write can't leave a half-baked checkpoint.
    No filelock here - the recommend state is only written by the BO loop
    and the GUI, and the BO loop is single-threaded per process.
    """
    import joblib
    from app.services.status import RECOMMEND_STATE
    if not os.path.exists(RECOMMEND_STATE):
        raise WriteError("No recommendation state found to update.")
    try:
        iter_no = int(iteration)
    except (TypeError, ValueError) as exc:
        raise WriteError(f"Bad iteration number: {iteration!r}") from exc
    try:
        state = joblib.load(RECOMMEND_STATE)
    except Exception as exc:  # noqa: BLE001
        raise WriteError(f"Could not read recommend state: {exc}") from exc

    history = list(state.get("recommendations", []))
    found = False
    for entry in history:
        try:
            if int(entry.get("iteration", -1)) == iter_no:
                entry["top_candidates"] = list(candidates)
                found = True
                break
        except (TypeError, ValueError):
            continue
    if not found:
        raise WriteError(f"Iteration {iter_no} not in recommendation history.")

    state["recommendations"] = history
    tmp = RECOMMEND_STATE + ".tmp"
    joblib.dump(state, tmp)
    os.replace(tmp, RECOMMEND_STATE)


# -- Pending recommendation lookup ---

def pending_recommendations() -> list[dict]:
    """Return BO recommendations from ``recommend_state.pkl`` history that
    don't yet have a corresponding row in the experiment file.

    A recommendation is considered "recorded" once a chemist has saved a row
    whose synthesis conditions match - but matching synthesis conditions
    against floating-point inputs is unreliable, so for v1 we simply return
    the latest batch and let the chemist pick by index.
    """
    from app.services.status import iteration_history

    history = iteration_history()
    out: list[dict] = []
    for entry in history.recommendations:
        iter_no = entry.get("iteration", "?")
        for rank, cand in enumerate(entry.get("top_candidates", []), start=1):
            out.append({
                "label":     f"Iter {iter_no} - pick #{rank}",
                "iteration": iter_no,
                "rank":      rank,
                "candidate": cand,
            })
    # Newest iteration first.
    out.sort(key=lambda r: (r["iteration"], -r["rank"]), reverse=True)
    return out
