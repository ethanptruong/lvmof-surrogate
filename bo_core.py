"""
bo_core.py — Bayesian Optimization components for the LVMOF-Surrogate pipeline.

Classes:
  SolventMixer           — COSMO property vectors for binary solvent mixtures
  SearchSpace            — LHS candidate generation over continuous + discrete params
  RegressionSurrogate    — wraps RF/XGB regressor, exposes (mu, sigma)
  XGBoostBootstrapEnsemble — M bootstrap XGB regressors for uncertainty
  OrdinalBOObjective     — raw 0-9 pxrd_score objective, BORE label generation
  EIAcquisition          — Expected Improvement
  BOREAcquisition        — dynamic-tau BORE (Tiao et al.)
  LCBAcquisition         — Lower Confidence Bound
  PIordinalAcquisition   — P(Crystalline) from Frank-Hall (static baseline)
  ThompsonSamplingAcquisition — sample from RF tree ensemble
  BatchSelector          — constant_liar and kriging_believer
  CandidateFeaturizer    — fixed chemistry + process params → full feature matrix
  BOLoop                 — simulation, recommendation, and batch modes
  BOCheckpointer         — save/load BO state
"""

import os
import warnings
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

from config import (
    RANDOM_STATE,
    BO_BORE_GAMMA,
    BO_LCB_KAPPA,
    BO_EI_XI,
    BO_N_ITERATIONS,
    BO_BATCH_SIZE,
    BO_INIT_FRACTION,
    BO_CHECKPOINT_DIR,
    BO_EPSILON_GREEDY,
    BO_N_LHS_SAMPLES,
    BO_BOOTSTRAP_M,
    BO_CONTROLLABLE_PARAMS,
    BO_OPTIONAL_PARAMS,
    BO_TOTAL_CONC_CLIP_PERCENTILES,
    BO_LOG_SCALE_PARAMS,
    XGB_FIXED,
)
from cosmo_features import (
    load_cosmo_index,
    load_sigma_profile,
    compute_sigma_moments,
)


# ─────────────────────────────────────────────────────────────
# SolventMixer
# ─────────────────────────────────────────────────────────────
class SolventMixer:
    """Compute COSMO property vectors for any binary solvent mixture.

    Reuses cosmo_features.py functions for sigma profile loading and moment
    computation.  For mixtures, linearly interpolates sigma profiles by
    mole fraction (Vcosmo-based proxy).
    """

    PURE_SOLVENTS = [
        "DICHLOROMETHANE", "TOLUENE", "TETRAHYDROFURAN",
        "N,N-DIMETHYLFORMAMIDE", "CHLOROFORM", "BENZENE",
        "ACETONITRILE", "ETHANOL",
    ]

    RATIOS = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

    def __init__(self, index_path, cosmo_folder):
        self.index_map, self.bp_map, self.vcosmo_map, self.lnpvap_map = (
            load_cosmo_index(index_path)
        )
        self.profiles = {}
        for name in self.PURE_SOLVENTS:
            idx = self.index_map.get(name)
            if idx is not None:
                prof = load_sigma_profile(idx, cosmo_folder)
                if prof is not None:
                    self.profiles[name] = prof

    def get_cosmo_vector(self, solvent_a, solvent_b=None, ratio=(1, 0)):
        """Return full COSMO property dict for a (possibly mixed) solvent."""
        if solvent_b is None or ratio == (1, 0):
            prof = self.profiles[solvent_a]
            moments = compute_sigma_moments(
                prof["sigma"].values, prof["area"].values
            )
            moments["Mix_Vcosmo"] = self.vcosmo_map.get(solvent_a, np.nan)
            moments["Mix_lnPvap"] = self.lnpvap_map.get(solvent_a, np.nan)
            moments["solvent_1"] = solvent_a
            moments["solvent_2"] = None
            moments["ratio"] = ratio
            vol_a, vol_b = ratio
            moments["solvent_1_fraction"] = 1.0
            moments["solvent_2_fraction"] = 0.0
            return moments

        vol_a, vol_b = ratio
        total = vol_a + vol_b

        vc_a = self.vcosmo_map.get(solvent_a, 1.0)
        vc_b = self.vcosmo_map.get(solvent_b, 1.0)
        proxy_a = (vol_a / total) / vc_a
        proxy_b = (vol_b / total) / vc_b
        frac_a = proxy_a / (proxy_a + proxy_b)
        frac_b = 1.0 - frac_a

        mix_area = (
            frac_a * self.profiles[solvent_a]["area"].values
            + frac_b * self.profiles[solvent_b]["area"].values
        )
        sigma_axis = self.profiles[solvent_a]["sigma"].values

        moments = compute_sigma_moments(sigma_axis, mix_area)
        moments["Mix_Vcosmo"] = (
            frac_a * self.vcosmo_map.get(solvent_a, 0.0)
            + frac_b * self.vcosmo_map.get(solvent_b, 0.0)
        )
        moments["Mix_lnPvap"] = (
            frac_a * self.lnpvap_map.get(solvent_a, 0.0)
            + frac_b * self.lnpvap_map.get(solvent_b, 0.0)
        )
        moments["solvent_1"] = solvent_a
        moments["solvent_2"] = solvent_b
        moments["ratio"] = ratio
        moments["solvent_1_fraction"] = vol_a / total
        moments["solvent_2_fraction"] = vol_b / total
        return moments

    def enumerate_all(self):
        """Generate all ~204 solvent compositions with precomputed COSMO vectors."""
        compositions = []
        available = [s for s in self.PURE_SOLVENTS if s in self.profiles]

        # Pure solvents
        for solvent in available:
            compositions.append(self.get_cosmo_vector(solvent))

        # Binary mixtures
        for a, b in combinations(available, 2):
            for ratio in self.RATIOS[1:]:  # skip (1,0) = pure
                compositions.append(self.get_cosmo_vector(a, b, ratio))
                # Reversed order for asymmetric ratios
                if ratio[0] != ratio[1]:
                    compositions.append(self.get_cosmo_vector(b, a, ratio))

        return compositions


# ─────────────────────────────────────────────────────────────
# SearchSpace
# ─────────────────────────────────────────────────────────────
class SearchSpace:
    """Generate candidate parameter sets: LHS over continuous × solvent compositions."""

    def __init__(self, train_df=None, solvent_mixer=None, extra_params=None):
        """
        Parameters
        ----------
        extra_params : dict or None
            Additional {param: (lo, hi)} to include beyond BO_CONTROLLABLE_PARAMS.
            Use this to toggle optional params like metal_over_linker_ratio.
        """
        all_params = dict(BO_CONTROLLABLE_PARAMS)
        if extra_params:
            all_params.update(extra_params)

        self.bounds = {}
        for param, static_bounds in all_params.items():
            if static_bounds is None and train_df is not None:
                lo_pct, hi_pct = BO_TOTAL_CONC_CLIP_PERCENTILES
                vals = train_df[param].dropna()
                self.bounds[param] = (
                    float(np.percentile(vals, lo_pct)),
                    float(np.percentile(vals, hi_pct)),
                )
            elif static_bounds is not None:
                self.bounds[param] = static_bounds
            else:
                self.bounds[param] = (1.0, 100.0)  # fallback

        self.solvent_mixer = solvent_mixer
        self.solvent_compositions = (
            solvent_mixer.enumerate_all() if solvent_mixer is not None else [{}]
        )

    def generate_lhs_candidates(self, n_samples=BO_N_LHS_SAMPLES, seed=RANDOM_STATE):
        """Generate LHS candidates over continuous params × solvent compositions."""
        rng = np.random.RandomState(seed)
        n_params = len(self.bounds)

        # Simple LHS: divide each dim into n_samples intervals, shuffle
        lhs_samples = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            lhs_samples[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples

        # Scale to bounds
        param_names = list(self.bounds.keys())
        candidates = {}
        for j, param in enumerate(param_names):
            lo, hi = self.bounds[param]
            if param in BO_LOG_SCALE_PARAMS and lo > 0 and hi > 0:
                candidates[param] = np.exp(
                    lhs_samples[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo)
                )
            else:
                candidates[param] = lhs_samples[:, j] * (hi - lo) + lo

        lhs_df = pd.DataFrame(candidates)

        # Cross with solvent compositions
        all_candidates = []
        for comp in self.solvent_compositions:
            chunk = lhs_df.copy()
            for key, val in comp.items():
                if key not in ("solvent_1", "solvent_2", "ratio"):
                    chunk[key] = val
            chunk["solvent_1"] = comp.get("solvent_1", "")
            chunk["solvent_2"] = comp.get("solvent_2", "")
            ratio = comp.get("ratio", (1, 0))
            chunk["ratio_str"] = f"{ratio[0]}:{ratio[1]}"
            all_candidates.append(chunk)

        return pd.concat(all_candidates, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# RegressionSurrogate
# ─────────────────────────────────────────────────────────────
class RegressionSurrogate:
    """Wraps a fitted regression pipeline, exposes (mu, sigma) predictions.

    For RF: inter-tree variance (SMAC approach).
    For XGB: uses XGBoostBootstrapEnsemble externally.
    """

    def __init__(self, pipeline, model_type="rf"):
        self.pipeline = pipeline
        self.model_type = model_type
        self.bootstrap_ensemble = None  # set externally for XGB

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        if self.bootstrap_ensemble is not None:
            Xt = self._transform_features(X)
            self.bootstrap_ensemble.fit(Xt, y)
        return self

    def _get_regressor(self):
        """Extract the final regressor from the pipeline."""
        for name, step in reversed(self.pipeline.steps):
            if isinstance(step, (RandomForestRegressor, XGBRegressor)):
                return step
        raise ValueError("No regressor found in pipeline")

    def _transform_features(self, X):
        """Transform X through all pipeline steps except the final regressor."""
        Xt = X
        for name, step in self.pipeline.steps:
            if isinstance(step, (RandomForestRegressor, XGBRegressor)):
                break
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X):
        """Return (mu, sigma) arrays."""
        Xt = self._transform_features(X)
        reg = self._get_regressor()

        if self.model_type == "rf":
            # Inter-tree variance (SMAC, Hutter et al.)
            predictions = np.array(
                [tree.predict(Xt) for tree in reg.estimators_]
            )
            mu = predictions.mean(axis=0)
            sigma = predictions.std(axis=0)
        elif self.model_type == "xgb" and self.bootstrap_ensemble is not None:
            mu, sigma = self.bootstrap_ensemble.predict(Xt)
        else:
            mu = reg.predict(Xt)
            sigma = np.zeros_like(mu)

        return mu, sigma

    def predict_mean(self, X):
        mu, _ = self.predict(X)
        return mu


# ─────────────────────────────────────────────────────────────
# XGBoostBootstrapEnsemble
# ─────────────────────────────────────────────────────────────
class XGBoostBootstrapEnsemble:
    """M bootstrap XGBoost regressors for uncertainty estimation.

    Inter-tree variance is invalid for boosting (trees are correlated),
    so we train M models on bootstrap samples (Hyperboost, Danjou et al.).
    """

    def __init__(self, base_params, M=BO_BOOTSTRAP_M, random_state=RANDOM_STATE):
        self.base_params = base_params
        self.M = M
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        self.models = []
        xgb_reg_fixed = {k: v for k, v in XGB_FIXED.items() if k != "eval_metric"}
        xgb_reg_fixed["eval_metric"] = "rmse"
        for i in range(self.M):
            idx = rng.choice(n, size=n, replace=True)
            model = XGBRegressor(
                **self.base_params,
                **xgb_reg_fixed,
                random_state=self.random_state + i,
            )
            model.fit(X[idx] if isinstance(X, np.ndarray) else X.iloc[idx],
                       y[idx])
            self.models.append(model)
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


# ─────────────────────────────────────────────────────────────
# OrdinalBOObjective
# ─────────────────────────────────────────────────────────────
class OrdinalBOObjective:
    """Raw 0-9 pxrd_score as BO objective + BORE label generation."""

    def __init__(self, gamma=BO_BORE_GAMMA):
        self.gamma = gamma

    def get_bore_labels(self, y_observed):
        """Generate binary labels for BORE: 1 if y >= tau, else 0."""
        tau = np.quantile(y_observed, 1.0 - self.gamma)
        labels = (y_observed >= tau).astype(int)
        return labels, tau

    @staticmethod
    def is_degenerate(labels):
        """Check if all labels are the same class."""
        return labels.sum() == 0 or labels.sum() == len(labels)


# ─────────────────────────────────────────────────────────────
# Acquisition Functions
# ─────────────────────────────────────────────────────────────
class EIAcquisition:
    """Expected Improvement using regression surrogate mean/variance."""

    def __init__(self, xi=BO_EI_XI):
        self.xi = xi

    def score(self, mu, sigma, f_best):
        """Compute EI(x) for arrays of mu, sigma."""
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = mu - f_best - self.xi
            Z = np.where(sigma > 1e-10, improvement / sigma, 0.0)
            ei = np.where(
                sigma > 1e-10,
                improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
                np.maximum(improvement, 0.0),
            )
        return ei


class LCBAcquisition:
    """Lower Confidence Bound (for maximization: UCB = mu + kappa * sigma)."""

    def __init__(self, kappa=BO_LCB_KAPPA):
        self.kappa = kappa

    def score(self, mu, sigma):
        return mu + self.kappa * sigma


class BOREAcquisition:
    """BORE: dynamic-threshold binary classification (Tiao et al.).

    Each iteration retrains a lightweight classifier on {(x_i, z_i)}
    where z_i = I[y_i >= tau] and tau = quantile(y, 1-gamma).
    """

    def __init__(self, gamma=BO_BORE_GAMMA, random_state=RANDOM_STATE):
        self.gamma = gamma
        self.random_state = random_state
        self.objective = OrdinalBOObjective(gamma=gamma)
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fallback_ei = EIAcquisition()

    def score(self, X_observed, y_observed, X_candidates, surrogate=None):
        """Return acquisition values for candidates.

        Falls back to EI if labels are degenerate.
        """
        labels, tau = self.objective.get_bore_labels(y_observed)

        if self.objective.is_degenerate(labels):
            if surrogate is not None:
                mu, sigma = surrogate.predict(X_candidates)
                return self.fallback_ei.score(mu, sigma, y_observed.max())
            return np.random.RandomState(self.random_state).uniform(
                size=len(X_candidates)
            )

        self.clf.fit(X_observed, labels)
        proba = self.clf.predict_proba(X_candidates)
        # Find column for positive class
        pos_idx = list(self.clf.classes_).index(1)
        return proba[:, pos_idx]


class PIordinalAcquisition:
    """P(Crystalline | x) from Frank-Hall classifier — static baseline."""

    def __init__(self, classifier_pipeline):
        self.pipeline = classifier_pipeline

    def score(self, X_candidates):
        """Return P(y == 2) = P(Crystalline) for each candidate."""
        proba = self.pipeline.predict_proba(X_candidates)
        return proba[:, -1]  # last class = Crystalline


class ThompsonSamplingAcquisition:
    """Thompson Sampling: sample one tree from RF regressor ensemble."""

    def __init__(self, random_state=RANDOM_STATE):
        self.rng = np.random.RandomState(random_state)

    def score(self, surrogate, X_candidates):
        """Sample a single tree prediction as the acquisition value."""
        Xt = surrogate._transform_features(X_candidates)
        reg = surrogate._get_regressor()
        if hasattr(reg, "estimators_"):
            tree_idx = self.rng.randint(len(reg.estimators_))
            return reg.estimators_[tree_idx].predict(Xt)
        else:
            return reg.predict(Xt)


# ─────────────────────────────────────────────────────────────
# BatchSelector
# ─────────────────────────────────────────────────────────────
class BatchSelector:
    """Batch selection via Constant Liar or Kriging Believer."""

    @staticmethod
    def constant_liar(
        surrogate, X_train, y_train, X_candidates,
        acquisition_fn, acquisition_name, batch_size,
        f_best, **acq_kwargs
    ):
        """Select batch_size candidates using constant liar (hallucinate f_best)."""
        selected_indices = []
        X_aug = np.array(X_train, copy=True) if isinstance(X_train, np.ndarray) else X_train.copy()
        y_aug = np.array(y_train, copy=True)

        for _ in range(batch_size):
            acq_vals = _compute_acquisition(
                acquisition_name, surrogate, X_aug, y_aug,
                X_candidates, **acq_kwargs
            )
            # Mask already selected
            for idx in selected_indices:
                acq_vals[idx] = -np.inf
            best_idx = np.argmax(acq_vals)
            selected_indices.append(best_idx)

            # Hallucinate with f_best
            if isinstance(X_candidates, np.ndarray):
                new_x = X_candidates[best_idx:best_idx+1]
            else:
                new_x = X_candidates.iloc[best_idx:best_idx+1].values
            X_aug = np.vstack([X_aug, new_x]) if isinstance(X_aug, np.ndarray) else \
                    pd.concat([X_aug, pd.DataFrame(new_x, columns=X_aug.columns)], ignore_index=True)
            y_aug = np.append(y_aug, f_best)

            surrogate.fit(X_aug, y_aug)

        return selected_indices

    @staticmethod
    def kriging_believer(
        surrogate, X_train, y_train, X_candidates,
        acquisition_fn, acquisition_name, batch_size,
        **acq_kwargs
    ):
        """Select batch_size candidates using kriging believer (hallucinate mu(x))."""
        selected_indices = []
        X_aug = np.array(X_train, copy=True) if isinstance(X_train, np.ndarray) else X_train.copy()
        y_aug = np.array(y_train, copy=True)

        for _ in range(batch_size):
            acq_vals = _compute_acquisition(
                acquisition_name, surrogate, X_aug, y_aug,
                X_candidates, **acq_kwargs
            )
            for idx in selected_indices:
                acq_vals[idx] = -np.inf
            best_idx = np.argmax(acq_vals)
            selected_indices.append(best_idx)

            # Hallucinate with surrogate prediction
            if isinstance(X_candidates, np.ndarray):
                new_x = X_candidates[best_idx:best_idx+1]
            else:
                new_x = X_candidates.iloc[best_idx:best_idx+1].values
            mu_hallucinated = surrogate.predict_mean(
                new_x if isinstance(X_candidates, np.ndarray) else
                pd.DataFrame(new_x, columns=X_candidates.columns)
            )[0]
            X_aug = np.vstack([X_aug, new_x]) if isinstance(X_aug, np.ndarray) else \
                    pd.concat([X_aug, pd.DataFrame(new_x, columns=X_aug.columns)], ignore_index=True)
            y_aug = np.append(y_aug, mu_hallucinated)

            surrogate.fit(X_aug, y_aug)

        return selected_indices


def _compute_acquisition(
    name, surrogate, X_train, y_train, X_candidates, **kwargs
):
    """Dispatch acquisition function by name."""
    if name == "ei":
        mu, sigma = surrogate.predict(X_candidates)
        f_best = kwargs.get("f_best", y_train.max())
        return EIAcquisition(xi=kwargs.get("xi", BO_EI_XI)).score(mu, sigma, f_best)
    elif name == "lcb":
        mu, sigma = surrogate.predict(X_candidates)
        return LCBAcquisition(kappa=kwargs.get("kappa", BO_LCB_KAPPA)).score(mu, sigma)
    elif name == "bore":
        bore = BOREAcquisition(
            gamma=kwargs.get("gamma", BO_BORE_GAMMA),
            random_state=kwargs.get("random_state", RANDOM_STATE),
        )
        return bore.score(X_train, y_train, X_candidates, surrogate=surrogate)
    elif name == "thompson":
        ts = ThompsonSamplingAcquisition(
            random_state=kwargs.get("random_state", RANDOM_STATE)
        )
        return ts.score(surrogate, X_candidates)
    elif name == "pi_ordinal":
        pi = kwargs.get("pi_ordinal_pipeline")
        if pi is not None:
            return PIordinalAcquisition(pi).score(X_candidates)
        # fallback to random
        return np.random.RandomState(RANDOM_STATE).uniform(size=len(X_candidates))
    elif name == "random":
        return np.random.RandomState(
            kwargs.get("random_state", RANDOM_STATE)
        ).uniform(size=len(X_candidates))
    else:
        raise ValueError(f"Unknown acquisition: {name}")


# ─────────────────────────────────────────────────────────────
# CandidateFeaturizer
# ─────────────────────────────────────────────────────────────
class CandidateFeaturizer:
    """Convert fixed chemistry + process parameter candidates → full feature matrix.

    Usage:
        cf = CandidateFeaturizer(template_X_row, feature_columns)
        X_full = cf.featurize(candidates_df)
    """

    def __init__(self, template_row, feature_columns, process_param_cols=None):
        self.template_row = np.array(template_row).flatten()
        self.feature_columns = list(feature_columns)
        self.process_param_cols = process_param_cols or list(BO_CONTROLLABLE_PARAMS.keys())

    def featurize(self, candidates_df):
        """Create full feature matrix by tiling template and overriding process params."""
        n = len(candidates_df)
        X = np.tile(self.template_row, (n, 1))
        X_df = pd.DataFrame(X, columns=self.feature_columns)

        for col in self.process_param_cols:
            if col in candidates_df.columns and col in X_df.columns:
                X_df[col] = candidates_df[col].values

        # Override COSMO columns if present
        cosmo_cols = [
            "Mix_M0_Area", "Mix_M1_NetCharge", "Mix_M2_Polarity",
            "Mix_M3_Asymmetry", "Mix_M4_Kurtosis", "Mix_M_HB_Acc",
            "Mix_M_HB_Don", "Mix_f_nonpolar", "Mix_f_acc", "Mix_f_don",
            "Mix_sigma_std", "Mix_Vcosmo", "Mix_lnPvap",
        ]
        for col in cosmo_cols:
            if col in candidates_df.columns and col in X_df.columns:
                X_df[col] = candidates_df[col].values

        # Solvent fractions
        for col in ["solvent_1_fraction", "solvent_2_fraction"]:
            if col in candidates_df.columns and col in X_df.columns:
                X_df[col] = candidates_df[col].values

        return X_df.values


# ─────────────────────────────────────────────────────────────
# BOCheckpointer
# ─────────────────────────────────────────────────────────────
class BOCheckpointer:
    """Save/load BO state (iteration, history, selected indices)."""

    def __init__(self, checkpoint_dir=BO_CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.checkpoint_dir, f"{name}.pkl")

    def save(self, name, state):
        joblib.dump(state, self._path(name))

    def load(self, name):
        path = self._path(name)
        if os.path.exists(path):
            return joblib.load(path)
        return None


# ─────────────────────────────────────────────────────────────
# BOLoop
# ─────────────────────────────────────────────────────────────
class BOLoop:
    """Main BO loop for simulation, recommendation, and batch modes."""

    def __init__(
        self,
        surrogate,
        acquisition_name="bore",
        batch_strategy="constant_liar",
        batch_size=BO_BATCH_SIZE,
        n_iterations=BO_N_ITERATIONS,
        epsilon_greedy=BO_EPSILON_GREEDY,
        classifier_pipeline=None,
        random_state=RANDOM_STATE,
    ):
        self.surrogate = surrogate
        self.acquisition_name = acquisition_name
        self.batch_strategy = batch_strategy
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.epsilon_greedy = epsilon_greedy
        self.classifier_pipeline = classifier_pipeline
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state

    def run_simulation(self, X, y_raw, init_fraction=BO_INIT_FRACTION):
        """Sequential BO with oracle pool.

        Parameters
        ----------
        X : array-like, shape (n, d) — full feature matrix
        y_raw : array-like, shape (n,) — raw 0-9 pxrd_score
        init_fraction : float — fraction for initial training set

        Returns
        -------
        history : dict with keys 'iterations', 'best_so_far', 'selected_indices',
                  'y_selected', 'init_indices', 'pool_indices'
        """
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        n = len(y_raw)

        # Stratified init split
        # Bin y_raw for stratification
        y_binned = np.clip(y_raw.astype(int), 0, 9)
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=init_fraction, random_state=self.random_state
        )
        init_idx, pool_idx = next(sss.split(X, y_binned))

        # Diagnostic
        high_count = (y_raw[init_idx] >= 7).sum()
        if high_count < 3:
            warnings.warn(
                f"Initial set has only {high_count} examples with score >= 7. "
                "Surrogate may lack signal for high-quality outcomes."
            )

        X_train = X[init_idx].copy()
        y_train = y_raw[init_idx].copy()
        pool_mask = np.ones(n, dtype=bool)
        pool_mask[init_idx] = False

        history = {
            "iterations": [],
            "best_so_far": [],
            "selected_indices": [],
            "y_selected": [],
            "init_indices": init_idx.tolist(),
            "pool_indices": pool_idx.tolist(),
        }

        f_best = y_train.max()
        print(f"[BO simulation] init={len(init_idx)}, pool={pool_mask.sum()}, "
              f"f_best_init={f_best:.1f}")

        for it in range(self.n_iterations):
            remaining = np.where(pool_mask)[0]
            if len(remaining) == 0:
                print(f"[BO] Pool exhausted at iteration {it}")
                break

            # Fit surrogate on current training data
            self.surrogate.fit(X_train, y_train)

            X_pool = X[remaining]

            # Epsilon-greedy exploration
            if self.rng.uniform() < self.epsilon_greedy:
                sel_local = self.rng.randint(len(remaining))
            else:
                acq_kwargs = {
                    "f_best": f_best,
                    "gamma": BO_BORE_GAMMA,
                    "random_state": self.random_state + it,
                    "pi_ordinal_pipeline": self.classifier_pipeline,
                }
                acq_vals = _compute_acquisition(
                    self.acquisition_name, self.surrogate,
                    X_train, y_train, X_pool, **acq_kwargs
                )
                sel_local = np.argmax(acq_vals)

            sel_global = remaining[sel_local]
            oracle_y = y_raw[sel_global]

            # Update
            X_train = np.vstack([X_train, X[sel_global:sel_global+1]])
            y_train = np.append(y_train, oracle_y)
            pool_mask[sel_global] = False
            f_best = max(f_best, oracle_y)

            history["iterations"].append(it)
            history["best_so_far"].append(f_best)
            history["selected_indices"].append(int(sel_global))
            history["y_selected"].append(float(oracle_y))

            if (it + 1) % 10 == 0:
                print(f"  iter {it+1:3d}/{self.n_iterations} | "
                      f"selected y={oracle_y:.0f} | f_best={f_best:.0f} | "
                      f"pool={pool_mask.sum()}")

        return history

    def run_batch(self, X, y_raw, init_fraction=BO_INIT_FRACTION):
        """Batch BO simulation using constant_liar or kriging_believer.

        Returns history dict similar to run_simulation but with batch selections.
        """
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        n = len(y_raw)

        y_binned = np.clip(y_raw.astype(int), 0, 9)
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=init_fraction, random_state=self.random_state
        )
        init_idx, pool_idx = next(sss.split(X, y_binned))

        X_train = X[init_idx].copy()
        y_train = y_raw[init_idx].copy()
        pool_mask = np.ones(n, dtype=bool)
        pool_mask[init_idx] = False

        history = {
            "iterations": [],
            "best_so_far": [],
            "selected_indices": [],
            "y_selected": [],
            "init_indices": init_idx.tolist(),
            "pool_indices": pool_idx.tolist(),
        }

        f_best = y_train.max()
        n_batch_iters = self.n_iterations // self.batch_size

        print(f"[BO batch] init={len(init_idx)}, pool={pool_mask.sum()}, "
              f"batch_size={self.batch_size}, batch_iters={n_batch_iters}")

        for batch_it in range(n_batch_iters):
            remaining = np.where(pool_mask)[0]
            if len(remaining) < self.batch_size:
                print(f"[BO] Pool too small at batch iteration {batch_it}")
                break

            self.surrogate.fit(X_train, y_train)
            X_pool = X[remaining]

            acq_kwargs = {
                "f_best": f_best,
                "gamma": BO_BORE_GAMMA,
                "random_state": self.random_state + batch_it,
                "pi_ordinal_pipeline": self.classifier_pipeline,
            }

            if self.batch_strategy == "constant_liar":
                local_indices = BatchSelector.constant_liar(
                    self.surrogate, X_train, y_train, X_pool,
                    None, self.acquisition_name, self.batch_size,
                    f_best, **acq_kwargs,
                )
            else:  # kriging_believer
                local_indices = BatchSelector.kriging_believer(
                    self.surrogate, X_train, y_train, X_pool,
                    None, self.acquisition_name, self.batch_size,
                    **acq_kwargs,
                )

            for sel_local in local_indices:
                sel_global = remaining[sel_local]
                oracle_y = y_raw[sel_global]

                X_train = np.vstack([X_train, X[sel_global:sel_global+1]])
                y_train = np.append(y_train, oracle_y)
                pool_mask[sel_global] = False
                f_best = max(f_best, oracle_y)

                history["iterations"].append(batch_it)
                history["best_so_far"].append(f_best)
                history["selected_indices"].append(int(sel_global))
                history["y_selected"].append(float(oracle_y))

            if (batch_it + 1) % 5 == 0:
                print(f"  batch {batch_it+1:3d}/{n_batch_iters} | "
                      f"f_best={f_best:.0f} | pool={pool_mask.sum()}")

        return history

    def run_recommend(self, X_train, y_train, candidates_df, candidate_features):
        """One-shot ranking for fixed chemistry.

        Parameters
        ----------
        X_train : array — full training feature matrix
        y_train : array — raw 0-9 scores
        candidates_df : DataFrame — candidate process params (from SearchSpace)
        candidate_features : array — featurized candidates (from CandidateFeaturizer)

        Returns
        -------
        results_df : DataFrame with columns for params + predictions + acquisition
        """
        # Fit surrogate
        self.surrogate.fit(X_train, y_train)
        mu, sigma = self.surrogate.predict(candidate_features)

        acq_kwargs = {
            "f_best": y_train.max(),
            "gamma": BO_BORE_GAMMA,
            "random_state": self.random_state,
            "pi_ordinal_pipeline": self.classifier_pipeline,
        }
        acq_vals = _compute_acquisition(
            self.acquisition_name, self.surrogate,
            X_train, y_train, candidate_features, **acq_kwargs
        )

        results = candidates_df.copy()
        results["pxrd_predicted"] = mu
        results["uncertainty"] = sigma
        results["acquisition_value"] = acq_vals

        # Add classifier probabilities if available
        if self.classifier_pipeline is not None:
            try:
                proba = self.classifier_pipeline.predict_proba(candidate_features)
                results["P_amorphous"] = proba[:, 0]
                results["P_partial"] = proba[:, 1] if proba.shape[1] > 1 else 0.0
                results["P_crystalline"] = proba[:, -1]
            except Exception:
                pass

        results = results.sort_values("acquisition_value", ascending=False)
        return results
