"""Background runner for the full hyperparameter retune pipeline.

Wraps ``main.main`` so the Admin page can fire it from a button click.
The retune is hours-long; the Admin page polls ``RunStatus`` for live
phase + log streaming.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Optional

from app.services.bo_runner import RunStatus, _run_in_thread
from app.services.status import (DATA_CKPT, FEATURES_CKPT,
                                  PROJECT_ROOT, RECOMMEND_STATE)


def _persist_artifacts() -> None:
    """Copy root-level evaluation figures/CSVs onto the persistent volume.

    In the deployed container the project root is the ephemeral image layer:
    anything regenerated there vanishes on restart/redeploy. The entrypoint
    overlays ``$COMPASS_DATA_DIR/artifacts`` back onto the root at boot, so
    copying there makes fresh plots survive. Locally (no volume) this is a
    no-op - the project root is already persistent.
    """
    vol = os.environ.get("COMPASS_DATA_DIR", "")
    if not vol or not os.path.isdir(vol):
        return
    dest = os.path.join(vol, "artifacts")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for pattern in ("*.png", "*.csv"):
        for src in glob.glob(os.path.join(PROJECT_ROOT, pattern)):
            shutil.copy2(src, dest)
            n += 1
    print(f"  Persisted {n} figures/CSVs to {dest}")


def start_retune(
    *,
    skip_tuning: bool = False,
    data_path: Optional[str] = None,
) -> RunStatus:
    """Launch ``main.main(...)`` in the background.

    Parameters
    ---
    skip_tuning : if True, only re-run evaluation against the existing
                  ``best_params.pkl``.  Use this to refresh plots after a
                  small data change without paying the multi-hour Optuna cost.
    data_path   : optional override for the data file path.
    """
    status = RunStatus()
    status.phase = "Loading data..." if not skip_tuning else "Re-evaluating..."

    def _target():
        from main import main as _main
        _main(data_path=data_path, skip_tuning=skip_tuning)
        _persist_artifacts()
        return {"skip_tuning": skip_tuning}

    _run_in_thread(_target, status)
    return status


def start_update(*, data_path: Optional[str] = None) -> RunStatus:
    """One-click model update: pick up the latest experiments AND refresh
    every evaluation figure.

    Drops the feature/data caches so ``main.main`` re-featurizes from the
    current Excel, then re-fits with the existing tuned hyperparameters and
    regenerates all plots. This is the action chemists should use after
    recording new experiments - running "retrain" alone would silently reuse
    the stale cached features and never see the new rows.
    """
    status = RunStatus()
    status.phase = "Re-featurizing from the latest data..."

    def _target():
        for path in (FEATURES_CKPT, DATA_CKPT):
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed {path}")
        from main import main as _main
        _main(data_path=data_path, skip_tuning=True)
        _persist_artifacts()
        return {"updated": True}

    _run_in_thread(_target, status)
    return status


def start_refeaturize(*, data_path: Optional[str] = None) -> RunStatus:
    """Clear the feature/data caches and re-featurize from the Excel file.

    Useful after the chemist edits the Excel directly (or after a Record
    operation that bypassed the GUI).
    """
    status = RunStatus()
    status.phase = "Clearing caches..."

    def _target():
        for path in (FEATURES_CKPT, DATA_CKPT):
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed {path}")
        from main import _featurize_fresh
        out = _featurize_fresh(data_path)
        if isinstance(out, tuple) and len(out) >= 1:
            X = out[0]
            print(f"  Featurization done: X.shape = {getattr(X, 'shape', '?')}")
        return {"refeaturized": True}

    _run_in_thread(_target, status)
    return status


def reset_recommend_state() -> bool:
    """Delete the BO recommend checkpoint after the user types-confirms."""
    if os.path.exists(RECOMMEND_STATE):
        os.remove(RECOMMEND_STATE)
        return True
    return False
