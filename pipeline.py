"""
pipeline.py
Optuna objective functions, progress callbacks, and eval_pipe.

objective_xgb and objective_rf take X, y, cv, groups as explicit arguments
(refactored away from the notebook's closure-based approach).
"""

import numpy as np
import optuna
from sklearn.model_selection import cross_validate, cross_val_score

from config import RANDOM_STATE, CV_NJOBS
from models import (scoring_ordinal, make_xgb_pipe, make_rf_pipe,
                    make_xgb_pipe_cl_only, make_rf_pipe_cl_only)


# ─────────────────────────────────────────────────────────────
# objective_xgb
# ─────────────────────────────────────────────────────────────
def objective_xgb(trial, X, y, cv, groups):
    param = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 15.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 15.0, log=True),
    }
    pipe   = make_xgb_pipe(param, with_cl=False)          # same structure as final pipe
    cv_res = cross_validate(
        pipe, X, y, cv=cv, groups=groups,
        scoring=scoring_ordinal, n_jobs=-1, return_train_score=False,
    )
    fold_qwk = cv_res["test_qwk"]
    fold_mae = cv_res["test_mae"]
    fold_acc = cv_res["test_exact_acc"]
    mean_qwk = np.mean(fold_qwk)

    try:
        is_best = mean_qwk > trial.study.best_value
    except ValueError:
        is_best = True

    if is_best:
        print(f"\n  ↳ Trial {trial.number} per-fold:")
        print(f"    {'Fold':>6} {'QWK':>8} {'MAE':>8} {'ExactAcc':>10} {'Partial_n':>10}")
        for i, (qwk, mae, acc) in enumerate(zip(fold_qwk, fold_mae, fold_acc)):
            val_idx   = list(cv.split(X, y, groups))[i][1]
            n_partial = (y[val_idx] == 1).sum()
            print(f"    {i+1:>6d} {qwk:>8.4f} {mae:>8.4f} {acc:>10.4f} {n_partial:>10d}")
        print(f"    {'Mean':>6} {mean_qwk:>8.4f} {np.mean(fold_mae):>8.4f} "
              f"{np.mean(fold_acc):>10.4f}")
    return mean_qwk


# ─────────────────────────────────────────────────────────────
# objective_rf
# ─────────────────────────────────────────────────────────────
def objective_rf(trial, X, y, cv, groups):
    param = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features",
                                                        ["sqrt", "log2"]),
    }
    pipe   = make_rf_pipe(param, with_cl=False)            # ← same structure as final pipe
    scores = cross_val_score(
        pipe, X, y, cv=cv, groups=groups,
        scoring=scoring_ordinal["qwk"], n_jobs=-1,
    )
    return np.mean(scores)


# ─────────────────────────────────────────────────────────────
# objective_xgb_cl_only
# ─────────────────────────────────────────────────────────────
def objective_xgb_cl_only(trial, X, y, cv, groups):
    param = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 15.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 15.0, log=True),
    }

    pipe = make_xgb_pipe_cl_only(param)  # CL-only feature pipeline

    cv_res = cross_validate(
        pipe,
        X, y,
        cv=cv,
        groups=groups,
        scoring=scoring_ordinal,
        n_jobs=1,                 # CL uses PyTorch -> avoid multiprocessing
        return_train_score=False,
    )

    fold_qwk = cv_res["test_qwk"]
    fold_mae = cv_res["test_mae"]
    fold_acc = cv_res["test_exact_acc"]
    mean_qwk = np.mean(fold_qwk)

    try:
        is_best = mean_qwk > trial.study.best_value
    except ValueError:
        is_best = True

    if is_best:
        print(f"\n  ↳ [CL-only XGB] Trial {trial.number} per-fold:")
        print(f"    {'Fold':>6} {'QWK':>8} {'MAE':>8} {'ExactAcc':>10} {'Partial_n':>10}")
        for i, (qwk, mae, acc) in enumerate(zip(fold_qwk, fold_mae, fold_acc)):
            val_idx   = list(cv.split(X, y, groups))[i][1]
            n_partial = (y[val_idx] == 1).sum()
            print(f"    {i+1:>6d} {qwk:>8.4f} {mae:>8.4f} {acc:>10.4f} {n_partial:>10d}")
        print(f"    {'Mean':>6} {mean_qwk:>8.4f} {np.mean(fold_mae):>8.4f} "
              f"{np.mean(fold_acc):>10.4f}")
    return mean_qwk


# ─────────────────────────────────────────────────────────────
# objective_rf_cl_only
# ─────────────────────────────────────────────────────────────
def objective_rf_cl_only(trial, X, y, cv, groups):
    param = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),
    }

    pipe = make_rf_pipe_cl_only(param)

    scores = cross_val_score(
        pipe,
        X, y,
        cv=cv,
        groups=groups,
        scoring=scoring_ordinal["qwk"],
        n_jobs=1,   # CL -> keep to 1
    )
    return np.mean(scores)


# ─────────────────────────────────────────────────────────────
# progress_callback
# ─────────────────────────────────────────────────────────────
def progress_callback(study, trial):
    marker = " ◄ NEW BEST" if trial.value == study.best_value else ""
    print(f"  Trial {trial.number:3d} | QWK={trial.value:.4f} "
          f"| Best={study.best_value:.4f}{marker}")


def progress_callback_cl_xgb(study, trial):
    marker = " ◄ NEW BEST" if trial.value == study.best_value else ""
    print(f"[CL-only XGB] Trial {trial.number:3d} | QWK={trial.value:.4f} "
          f"| Best={study.best_value:.4f}{marker}")


def progress_callback_cl_rf(study, trial):
    marker = " ◄ NEW BEST" if trial.value == study.best_value else ""
    print(f"[CL-only RF] Trial {trial.number:3d} | QWK={trial.value:.4f} "
          f"| Best={study.best_value:.4f}{marker}")


# ─────────────────────────────────────────────────────────────
# eval_pipe
# ─────────────────────────────────────────────────────────────
def eval_pipe(name: str, pipe, X, y, cv, groups, scoring, n_jobs: int = 1):
    out = cross_validate(
        pipe, X, y, cv=cv, groups=groups,
        scoring=scoring, n_jobs=n_jobs,
    )
    print(f"\n{name}")
    print(f"  QWK      {np.mean(out['test_qwk']):.4f}  ±{np.std(out['test_qwk']):.4f}")
    print(f"  MAE      {-np.mean(out['test_mae']):.4f}  ±{np.std(out['test_mae']):.4f}")
    print(f"  Within-1 {np.mean(out['test_within1']):.4f}  ±{np.std(out['test_within1']):.4f}")
    print(f"  Exact    {np.mean(out['test_exact_acc']):.4f}  ±{np.std(out['test_exact_acc']):.4f}")


# ─────────────────────────────────────────────────────────────
# suggest_rf_params / suggest_xgb_params
# ─────────────────────────────────────────────────────────────
def suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 15.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 15.0, log=True),
    }


# -----------------------------
# Objective factory
# -----------------------------
def make_objective(model_type, with_cl=False):
    def objective(trial):
        if model_type == "rf":
            params = suggest_rf_params(trial)
            pipe = make_rf_pipe(params, with_cl=with_cl)
        elif model_type == "xgb":
            params = suggest_xgb_params(trial)
            pipe = make_xgb_pipe(params, with_cl=with_cl)
        else:
            raise ValueError(f"Unknown model_type={model_type}")

        out = cross_validate(
            pipe,
            X, y,
            cv=cv,
            groups=groups,
            scoring=scoring_ordinal,
            n_jobs=CV_NJOBS,
            return_train_score=False,
        )

        qwk = np.mean(out["test_qwk"])
        mae = np.mean(out["test_mae"])
        acc = np.mean(out["test_exact_acc"])

        trial.set_user_attr("mean_qwk", float(qwk))
        trial.set_user_attr("mean_mae", float(mae))
        trial.set_user_attr("mean_exact_acc", float(acc))
        return qwk

    return objective


# -----------------------------
# Optuna helpers
# -----------------------------
def make_progress_callback(label):
    def progress_callback(study, trial):
        marker = "  ◄ NEW BEST" if trial.number == study.best_trial.number else ""
        print(
            f"[{label}] Trial {trial.number:3d} | "
            f"QWK={trial.value:.4f} | Best={study.best_value:.4f}{marker}"
        )
    return progress_callback


def run_study(label, model_type, with_cl, stage1_trials=15, stage2_trials=35):
    print(f"\n{'='*80}")
    print(f"TUNING: {label}")
    print(f"{'='*80}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=label)

    objective = make_objective(model_type=model_type, with_cl=with_cl)
    callback = make_progress_callback(label)

    if stage1_trials > 0:
        print(f"\nStage 1: {stage1_trials} trials")
        study.optimize(objective, n_trials=stage1_trials, callbacks=[callback])
        print(f"Stage 1 best QWK: {study.best_value:.4f}")

    if stage2_trials > 0:
        print(f"\nStage 2: {stage2_trials} trials")
        study.optimize(objective, n_trials=stage2_trials, callbacks=[callback])

    print(f"\nBest QWK for {label}: {study.best_value:.4f}")
    print(f"Best params for {label}: {study.best_params}")
    return study
