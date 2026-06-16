"""Read-only status helpers - dataset stats, BO history, model freshness.

Backs the Home page and the Model-Confidence page.  All paths are resolved
relative to the project root, which is the parent of the ``app/`` directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import joblib
import pandas as pd

from config import BO_CHECKPOINT_DIR, DATA_FILE_PATH

# -- Path constants ---
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR    = os.path.join(PROJECT_ROOT, "checkpoints")
# The BO loop's BOCheckpointer writes its state under ``checkpoints/bo/`` (see
# ``config.BO_CHECKPOINT_DIR``), not directly under ``checkpoints/``. Source
# the path from config so the GUI and the BO loop stay in lock-step if anyone
# ever moves the directory.
RECOMMEND_STATE   = os.path.join(PROJECT_ROOT, BO_CHECKPOINT_DIR, "recommend_state.pkl")
BEST_PARAMS       = os.path.join(CHECKPOINT_DIR, "best_params.pkl")
FEATURES_CKPT     = os.path.join(CHECKPOINT_DIR, "features.pkl")
DATA_CKPT         = os.path.join(CHECKPOINT_DIR, "data.pkl")
DATA_FILE         = os.path.join(PROJECT_ROOT, DATA_FILE_PATH)
DOCS_DIR          = os.path.join(PROJECT_ROOT, "docs")


@dataclass
class DatasetSummary:
    """Aggregate statistics derived from the source Excel file."""

    n_experiments: int
    best_score: Optional[float]
    best_experiment_id: Optional[Any]
    last_modified: Optional[datetime]
    df_path: str

    @property
    def best_score_label(self) -> str:
        if self.best_score is None:
            return "n/a"
        s = int(self.best_score)
        if s >= 7:
            return f"{s} (crystalline)"
        if s >= 3:
            return f"{s} (partial)"
        return f"{s} (amorphous)"


@dataclass
class IterationHistory:
    """Summary of the persistent BO recommend state."""

    iteration: int
    n_data_at_each_iter: list[int] = field(default_factory=list)
    surrogate_name: Optional[str] = None
    acquisition_name: Optional[str] = None
    last_run_mtime: Optional[datetime] = None
    recommendations: list[dict] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.recommendations)


@dataclass
class ModelStatus:
    """Whether the surrogate has been tuned and when."""

    tuned: bool
    tuned_at: Optional[datetime]


# -- Loaders ---

def dataset_summary() -> DatasetSummary:
    """Read the experiment Excel file and return basic stats.

    Re-reads on every call so the GUI always reflects the on-disk state
    (after a Record-result write, the chemist sees the updated count
    immediately).
    """
    if not os.path.exists(DATA_FILE):
        return DatasetSummary(
            n_experiments=0, best_score=None, best_experiment_id=None,
            last_modified=None, df_path=DATA_FILE,
        )

    # Use the canonical loader so unnamed/artifact columns are handled
    # the same way as the training pipeline.
    from data_processing import load_data
    df = load_data(DATA_FILE)

    score_col = "pxrd_score" if "pxrd_score" in df.columns else None
    best_score = None
    best_id = None
    if score_col:
        s = pd.to_numeric(df[score_col], errors="coerce")
        if s.notna().any():
            i = int(s.idxmax())
            best_score = float(s.iloc[i])
            id_col = "experiment_id" if "experiment_id" in df.columns else None
            best_id = df.iloc[i][id_col] if id_col else None

    mtime = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
    return DatasetSummary(
        n_experiments=len(df),
        best_score=best_score,
        best_experiment_id=best_id,
        last_modified=mtime,
        df_path=DATA_FILE,
    )


def iteration_history() -> IterationHistory:
    """Load the BO recommend checkpoint, returning an empty history if none."""
    if not os.path.exists(RECOMMEND_STATE):
        return IterationHistory(iteration=0)

    state = joblib.load(RECOMMEND_STATE)
    return IterationHistory(
        iteration=int(state.get("iteration", 0)),
        n_data_at_each_iter=list(state.get("n_data_at_each_iter", [])),
        surrogate_name=state.get("surrogate_name"),
        acquisition_name=state.get("acquisition_name"),
        last_run_mtime=datetime.fromtimestamp(os.path.getmtime(RECOMMEND_STATE)),
        recommendations=list(state.get("recommendations", [])),
    )


def model_status() -> ModelStatus:
    """Whether ``best_params.pkl`` exists (i.e. tuning has been run)."""
    if not os.path.exists(BEST_PARAMS):
        return ModelStatus(tuned=False, tuned_at=None)
    return ModelStatus(
        tuned=True,
        tuned_at=datetime.fromtimestamp(os.path.getmtime(BEST_PARAMS)),
    )


def list_recommendation_csvs() -> list[str]:
    """Return absolute paths of every BO recommendation CSV in ``docs/``,
    newest first."""
    if not os.path.isdir(DOCS_DIR):
        return []
    files = [
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.startswith("bo_recommendations_iter") and f.endswith(".csv")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def latest_recommendation_csv() -> Optional[str]:
    files = list_recommendation_csvs()
    return files[0] if files else None


def list_calibration_plots() -> list[str]:
    """All ``bo_calibration_*.png`` plots in ``docs/``, newest first."""
    if not os.path.isdir(DOCS_DIR):
        return []
    files = [
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.startswith("bo_calibration_") and f.endswith(".png")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def list_shap_csvs() -> list[str]:
    """All ``shap_importance_*.csv`` files at the project root, newest first.

    Legacy helper kept for callers that just want raw paths. New UI code
    should prefer :func:`shap_csv_index`, which deduplicates across the
    ``shap_importance_*`` / ``shap_named_*`` naming conventions and
    returns structured (model, feature-block, path) metadata.
    """
    if not os.path.isdir(PROJECT_ROOT):
        return []
    files = [
        os.path.join(PROJECT_ROOT, f)
        for f in os.listdir(PROJECT_ROOT)
        if f.startswith("shap_importance_") and f.endswith(".csv")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


@dataclass
class ShapCsvEntry:
    """One per-pipeline SHAP importance CSV, with the feature-block decoded."""

    path: str
    model_family: str   # "RF" or "XGB"
    feature_block: str  # "MI only" | "CL + MI" | "CL only (triplet)" | ...
    label: str          # human-readable, e.g. "RF · CL + MI"
    mtime: float


def shap_csv_index() -> list[ShapCsvEntry]:
    """Return one SHAP CSV per (model_family, feature_block) pipeline.

    The training pipeline has historically written SHAP importances under
    two filename conventions:

    - Legacy (``run_shap.py`` / ``evaluation.run_shap_featurized``):
      ``shap_importance_{pipe_slug}_{kind}.csv`` where ``pipe_slug`` is
      ``pipe_label.replace(" ", "_").replace("|", "-")`` and ``kind`` is
      ``aggregated`` / ``threshold_0`` / ``threshold_1``.
    - Current (``evaluation.py`` plotting helper):
      ``shap_named_{safe_label}.csv`` where ``safe_label`` only strips
      shell-unsafe characters from the pipe label (spaces and pipes
      survive as themselves and become ``  _``).

    Both files carry the same per-feature ``mean_abs_shap`` column. This
    function recognises both, dedupes to the **most recent CSV per
    pipeline**, and decodes the model family + feature block so the UI
    can show a friendly label instead of a slug.
    """
    if not os.path.isdir(PROJECT_ROOT):
        return []

    # Map (model_family, feature_block) → newest ShapCsvEntry seen so far.
    bucket: dict[tuple[str, str], ShapCsvEntry] = {}
    for fname in os.listdir(PROJECT_ROOT):
        if not fname.endswith(".csv"):
            continue
        if not (fname.startswith("shap_importance_")
                or fname.startswith("shap_named_")):
            continue

        # Strip prefix + ``.csv`` so we can decode the pipe label.
        if fname.startswith("shap_importance_"):
            stem = fname[len("shap_importance_"):-len(".csv")]
            # Legacy slug: "RF__-_MI_only_aggregated" - chop the kind suffix.
            for kind in ("_aggregated", "_threshold_0", "_threshold_1"):
                if stem.endswith(kind):
                    stem = stem[:-len(kind)]
                    break
            pipe_label = (stem.replace("_-_", " | ")
                              .replace("__", "  ")
                              .replace("_", " "))
        else:
            # shap_named convention: the only filename-unsafe character
            # the saver replaces is the pipe (``|`` → ``_``); spacing
            # around it is whatever the pipe label had. RF labels in
            # run_shap.py column-align with two spaces ("RF  | MI only"
            # → "RF  _ MI only"), while XGB has one ("XGB | MI only" →
            # "XGB _ MI only"). Restore the pipe with a regex that
            # tolerates either spacing.
            import re as _re
            stem = fname[len("shap_named_"):-len(".csv")]
            pipe_label = _re.sub(r"\s+_\s+", " | ", stem, count=1)

        family = "RF" if pipe_label.lstrip().startswith("RF") else (
            "XGB" if pipe_label.lstrip().startswith("XGB") else "?")
        # Everything after the first "|" is the feature-block descriptor.
        if "|" in pipe_label:
            block = pipe_label.split("|", 1)[1].strip()
        else:
            block = pipe_label.strip()

        full = os.path.join(PROJECT_ROOT, fname)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue

        entry = ShapCsvEntry(
            path=full,
            model_family=family,
            feature_block=block,
            label=f"{family} · {block}",
            mtime=mtime,
        )
        key = (family, block)
        existing = bucket.get(key)
        if existing is None or entry.mtime > existing.mtime:
            bucket[key] = entry

    # Stable ordering: by family then by feature block.
    _BLOCK_ORDER = ["MI only", "CL + MI", "CL only (triplet)",
                    "CL only (supcon)", "CL only"]
    def _sort_key(e: ShapCsvEntry) -> tuple[int, int]:
        fam_rank = {"RF": 0, "XGB": 1}.get(e.model_family, 9)
        try:
            blk_rank = _BLOCK_ORDER.index(e.feature_block)
        except ValueError:
            blk_rank = 99
        return (fam_rank, blk_rank)

    return sorted(bucket.values(), key=_sort_key)


def checkpoint_inventory() -> list[dict]:
    """List ``checkpoints/`` files with size + mtime for the Admin page."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    out = []
    for entry in sorted(os.listdir(CHECKPOINT_DIR)):
        full = os.path.join(CHECKPOINT_DIR, entry)
        try:
            st = os.stat(full)
        except OSError:
            continue
        out.append({
            "name":   entry,
            "size":   st.st_size,
            "mtime":  datetime.fromtimestamp(st.st_mtime),
            "is_dir": os.path.isdir(full),
        })
    return out
