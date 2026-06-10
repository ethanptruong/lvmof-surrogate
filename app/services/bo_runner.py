"""Background runner for BO recommend / simulate.

Wraps the existing CLI handlers ``main._run_recommend`` and ``main.run_bo``
by building an ``argparse.Namespace`` with the same field names that
``main.py``'s argparse parser produces.

Long-running calls execute on a daemon thread; the page polls
``RunStatus`` for the live phase / log buffer.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from config import (BO_BATCH_SIZE, BO_DEFAULT_ACQUISITION,
                    BO_DEFAULT_SURROGATE, BO_N_ITERATIONS)


# ── Default arg construction ──────────────────────────────────────────────────
# Field names mirror ``main.py``'s argparse parser exactly so that the same
# Namespace can be passed straight to ``run_bo`` / ``_run_recommend``.
def _default_args(**overrides: Any) -> argparse.Namespace:
    base = dict(
        data=None,
        bo=True,
        bo_mode="recommend",
        bo_surrogate=BO_DEFAULT_SURROGATE,
        bo_acquisition=BO_DEFAULT_ACQUISITION,
        bo_batch_strategy=None,
        bo_diversity_lambda=0.3,
        bo_batch_size=BO_BATCH_SIZE,
        bo_iterations=BO_N_ITERATIONS,
        bo_ablation=False,
        bo_include_mlr=False,
        bo_ranking_target=False,
        bo_feasibility=False,
        bo_eval_seeds=5,
        skip_tuning=False,
        bo_precursor=None,
        bo_linker=None,
        bo_modulator=None,
        bo_observed_pairs=False,
        bo_solvent_filter="permissive",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _resolve_batch_strategy(args: argparse.Namespace) -> None:
    """Apply the same auto-strategy resolution that ``main.py`` does at parse
    time so the wrapper produces identical behaviour to the CLI."""
    from bo_core import resolve_batch_strategy
    args.bo_batch_strategy = resolve_batch_strategy(
        args.bo_acquisition, args.bo_batch_strategy
    )


# ── Phase parser ──────────────────────────────────────────────────────────────
# Map characteristic substrings printed by _run_recommend → friendly phases.
# These are best-effort heuristics; the runner falls back to "Working..."
# when none match.
_PHASE_PATTERNS: list[tuple[str, str]] = [
    ("[features]",                  "Loading features..."),
    ("Featurizing from data file",  "Featurizing dataset..."),
    ("Building feature name catalog","Building feature catalog..."),
    ("Fitting surrogate",           "Fitting surrogate..."),
    ("BO RECOMMENDATION",           "Starting recommendation..."),
    ("Generating",                  "Generating candidate pool..."),
    ("LHS",                         "Sampling candidate pool..."),
    ("Scoring",                     "Scoring candidates..."),
    ("Selecting batch",             "Selecting batch..."),
    ("Top recommendations",         "Finalizing recommendations..."),
    ("Calibration",                 "Computing calibration..."),
    ("Simulation",                  "Running BO simulation..."),
]


def _phase_from_line(line: str) -> Optional[str]:
    for needle, phase in _PHASE_PATTERNS:
        if needle in line:
            return phase
    return None


# ── Background job state ──────────────────────────────────────────────────────

@dataclass
class RunStatus:
    """Live state of a background BO job (read by the page on each rerun)."""

    running: bool = True
    phase: str = "Starting..."
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Any] = None

    def append(self, line: str) -> None:
        self.log.append(line.rstrip("\n"))
        new_phase = _phase_from_line(line)
        if new_phase:
            self.phase = new_phase

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class _StreamingBuffer(io.TextIOBase):
    """A file-like object that pipes every print() into a RunStatus."""

    def __init__(self, status: RunStatus, mirror: Optional[io.TextIOBase] = None):
        super().__init__()
        self._status = status
        self._buf = ""
        self._mirror = mirror

    def write(self, s):  # noqa: D401
        if self._mirror is not None:
            try:
                self._mirror.write(s)
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._status.append(line)
        return len(s)

    def flush(self):  # noqa: D401
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass


def _run_in_thread(target: Callable[[], Any], status: RunStatus) -> None:
    """Run *target* on a daemon thread, capturing stdout/stderr into status."""

    def _runner():
        # Mirror to the original stdout so server logs still see the output.
        original_stdout = sys.__stdout__
        buf = _StreamingBuffer(status, mirror=original_stdout)
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                status.result = target()
        except BaseException as exc:
            status.error = f"{type(exc).__name__}: {exc}"
            status.append("")
            for line in traceback.format_exc().splitlines():
                status.append(line)
        finally:
            buf.flush()
            status.running = False
            status.finished_at = time.time()
            if status.error is None:
                status.phase = "Done"

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


# ── Public API ────────────────────────────────────────────────────────────────

def start_recommend(
    *,
    linker_smiles: Optional[str],
    precursor_smiles: Optional[str],
    modulator_smiles: Optional[str],
    batch_size: int,
    surrogate: str = BO_DEFAULT_SURROGATE,
    acquisition: str = BO_DEFAULT_ACQUISITION,
    batch_strategy: Optional[str] = None,
    feasibility: bool = False,
    observed_pairs: bool = False,
    include_mlr: bool = False,
    data_path: Optional[str] = None,
) -> RunStatus:
    """Launch a Recommend job in the background. Returns the RunStatus handle."""
    args = _default_args(
        data=data_path,
        bo_mode="recommend",
        bo_surrogate=surrogate,
        bo_acquisition=acquisition,
        bo_batch_strategy=batch_strategy,
        bo_batch_size=int(batch_size),
        bo_feasibility=bool(feasibility),
        bo_observed_pairs=bool(observed_pairs),
        bo_include_mlr=bool(include_mlr),
        bo_precursor=precursor_smiles or None,
        bo_linker=linker_smiles or None,
        bo_modulator=modulator_smiles or None,
    )
    _resolve_batch_strategy(args)

    status = RunStatus()
    status.phase = "Loading data..."

    def _target():
        from main import _run_recommend
        _run_recommend(args)
        # The CSV path is deterministic — return it so the page can render it.
        from app.services.status import latest_recommendation_csv
        import json
        csv_path = latest_recommendation_csv()
        meta = None
        if csv_path:
            meta_path = csv_path.replace(".csv", "_meta.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except FileNotFoundError:
                pass
            except Exception:
                meta = None
        return {"csv_path": csv_path, "meta": meta}

    _run_in_thread(_target, status)
    return status


def start_simulate(
    *,
    surrogate: str = BO_DEFAULT_SURROGATE,
    acquisition: str = BO_DEFAULT_ACQUISITION,
    iterations: int = BO_N_ITERATIONS,
    data_path: Optional[str] = None,
) -> RunStatus:
    """Launch a single-seed BO simulation in the background.

    Produces convergence, top-k, simple-regret, and calibration plots for a
    single random seed.  Fast, good for eyeballing a specific configuration.
    """
    args = _default_args(
        data=data_path,
        bo_mode="simulate",
        bo_surrogate=surrogate,
        bo_acquisition=acquisition,
        bo_iterations=int(iterations),
    )
    _resolve_batch_strategy(args)

    status = RunStatus()
    status.phase = "Loading data..."

    def _target():
        from main import run_bo
        run_bo(args)
        return {"label": f"{args.bo_acquisition}_{args.bo_surrogate}"}

    _run_in_thread(_target, status)
    return status


def start_evaluate(
    *,
    surrogate: str = BO_DEFAULT_SURROGATE,
    acquisition: str = BO_DEFAULT_ACQUISITION,
    iterations: int = BO_N_ITERATIONS,
    n_seeds: int = 5,
    data_path: Optional[str] = None,
) -> RunStatus:
    """Launch a multi-seed per-cluster BO evaluation in the background.

    Runs the BO simulation with *n_seeds* different random seeds and reports
    mean ± std of AF / EF / hit-rate both overall and per chemistry cluster.
    Slower than a single simulate (roughly n_seeds× cost) but gives
    statistically meaningful comparisons between configurations.
    """
    args = _default_args(
        data=data_path,
        bo_mode="evaluate",
        bo_surrogate=surrogate,
        bo_acquisition=acquisition,
        bo_iterations=int(iterations),
        bo_eval_seeds=int(n_seeds),
    )
    _resolve_batch_strategy(args)

    status = RunStatus()
    status.phase = "Loading data..."

    def _target():
        from main import run_bo
        run_bo(args)
        return {"label": f"eval_{args.bo_acquisition}_{args.bo_surrogate}",
                "n_seeds": int(n_seeds)}

    _run_in_thread(_target, status)
    return status
