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
    TOTAL_VOLUME_ML,
    XGB_FIXED,
    BO_BORE_ADAPTIVE_GAMMA,
    BO_SSL_ALPHA,
    BO_SSL_N_PSEUDO,
)
from cosmo_features import (
    load_cosmo_index,
    load_sigma_profile,
    compute_sigma_moments,
    CosmoMixer,
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
    """Generate candidate parameter sets: LHS over continuous params × solvent pairs.

    phi_1 (solvent_1 volume fraction) is now a continuous BO parameter in the LHS;
    the solvent_mixer is used only to enumerate unique (sol1, sol2) pairs observed
    in the training data.  COSMO features are computed on-the-fly by
    CandidateFeaturizer using CosmoMixer.
    """

    def __init__(self, train_df=None, solvent_mixer=None, extra_params=None):
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
        # Enumerate unique (sol1, sol2) pairs from training data
        self.solvent_pairs = (
            self._enumerate_pairs(train_df, solvent_mixer)
            if solvent_mixer is not None and train_df is not None
            else [{"solvent_1": "", "solvent_2": ""}]
        )

    def _enumerate_pairs(self, train_df, solvent_mixer):
        """Build unique (sol1, sol2) pairs observed in training data."""
        available = set(solvent_mixer.available_solvents_from_df(train_df))
        seen = set()
        pairs = []
        for _, row in train_df.iterrows():
            sol1 = str(row.get("solvent_1", "")).strip().upper()
            sol2 = str(row.get("solvent_2", "")).strip().upper()
            if sol2 in ("", "NAN", "NONE"):
                sol2 = ""
            if not sol1 or sol1 not in available:
                continue
            if sol2 and sol2 not in available:
                sol2 = ""
            key = (sol1, sol2)
            if key not in seen:
                seen.add(key)
                pairs.append({"solvent_1": sol1, "solvent_2": sol2})
        return pairs if pairs else [{"solvent_1": "", "solvent_2": ""}]

    def generate_lhs_candidates(self, n_samples=BO_N_LHS_SAMPLES, seed=RANDOM_STATE,
                                override_bounds=None):
        """Generate LHS candidates over continuous params × solvent pairs.

        phi_1 is sampled as a continuous variable in the LHS alongside other
        process parameters; it is NOT fixed per solvent pair.

        Parameters
        ----------
        override_bounds : dict or None — param → (lo, hi) to use instead of
            self.bounds (e.g. trust region bounds).  Any param not in
            override_bounds falls back to self.bounds.
        """
        rng = np.random.RandomState(seed)
        n_params = len(self.bounds)

        # Resolve effective bounds (trust region overrides where provided)
        effective_bounds = {}
        for param, full_bound in self.bounds.items():
            if override_bounds is not None and param in override_bounds:
                effective_bounds[param] = override_bounds[param]
            else:
                effective_bounds[param] = full_bound

        # Simple LHS
        lhs_samples = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            lhs_samples[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples

        # Scale to effective bounds
        param_names = list(self.bounds.keys())
        candidates = {}
        for j, param in enumerate(param_names):
            lo, hi = effective_bounds[param]
            if param in BO_LOG_SCALE_PARAMS and lo > 0 and hi > 0:
                candidates[param] = np.exp(
                    lhs_samples[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo)
                )
            else:
                candidates[param] = lhs_samples[:, j] * (hi - lo) + lo

        lhs_df = pd.DataFrame(candidates)

        # Cross LHS samples with solvent pairs
        all_candidates = []
        for pair in self.solvent_pairs:
            chunk = lhs_df.copy()
            chunk["solvent_1"] = pair["solvent_1"]
            chunk["solvent_2"] = pair["solvent_2"]
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


class RankingRegressionSurrogate(RegressionSurrogate):
    """RF/XGB surrogate trained on rank-normalized targets.

    For ordinal objectives (0-9 pxrd_score), standard MSE regression treats
    all adjacent score gaps equally.  In reality the difference between
    score=8 and score=9 (rare highly-crystalline materials) matters far more
    than between score=1 and score=2.

    Rank normalization maps all scores to a uniform [0, 1] space so the
    surrogate focuses on *relative ordering* rather than exact magnitude.
    Acquisition functions then compute improvement in rank space, which
    directly reflects "find something better than current best."

    The raw_to_rank() method converts f_best from raw score space to rank
    space for use with EI/LCB acquisition functions.

    Reference: "Ranking over Regression for BO and Molecule Selection,"
               APL Machine Learning 3(3), 2024.
    """

    def __init__(self, pipeline, model_type="rf"):
        super().__init__(pipeline, model_type)
        self._y_train_raw = None

    def fit(self, X, y):
        from scipy.stats import rankdata
        self._y_train_raw = np.array(y, dtype=float)
        n = len(y)
        # Rank-normalize to [0, 1]: lowest rank → 0, highest rank → 1.
        y_ranked = (rankdata(y, method="average") - 1.0) / max(n - 1, 1)
        return super().fit(X, y_ranked)

    def raw_to_rank(self, raw_score):
        """Convert a raw pxrd_score to its rank-normalised equivalent.

        Used to transform f_best into rank space before computing EI.
        """
        if self._y_train_raw is None:
            return float(raw_score)
        frac = float((self._y_train_raw <= raw_score).sum() - 1) / max(
            len(self._y_train_raw) - 1, 1
        )
        return float(np.clip(frac, 0.0, 1.0))


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
        xgb_reg_fixed = {k: v for k, v in XGB_FIXED.items() if k not in ("eval_metric", "random_state")}
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

    def get_bore_labels(self, y_observed, mode="bore"):
        """Generate binary labels and sample weights for BORE / LFBO.

        mode="bore"             : binary z_i = I[y >= tau], uniform weights.
                                  Recovers Probability of Improvement (PI).
        mode="lfbo"/"lfbo_ssl"  : binary z_i = I[y >= tau], LFBO weights
                                  = max(y_i - tau, eps) for positive class.
                                  Recovers Expected Improvement (EI) via the
                                  density-ratio framework (Song et al., ICML 2022).

        Returns (labels, tau, sample_weight).
        """
        tau = np.quantile(y_observed, 1.0 - self.gamma)
        labels = (y_observed >= tau).astype(int)

        if mode in ("lfbo", "lfbo_ssl"):
            # LFBO-EI: weight each positive by actual improvement above tau.
            # Negative examples get uniform weight = 1.0.
            pos_weights = np.maximum(y_observed - tau, 1e-6)
            sample_weight = np.where(labels == 1, pos_weights, 1.0)
            # Normalise positive weights so scale doesn't dominate negatives.
            if labels.sum() > 0:
                pos_mean = sample_weight[labels == 1].mean()
                sample_weight = np.where(
                    labels == 1, sample_weight / pos_mean, 1.0
                )
        else:
            sample_weight = np.ones(len(labels))

        return labels, tau, sample_weight

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
    """BORE / LFBO / LFBO-SSL acquisition functions.

    mode="bore"
        Original BORE (Tiao et al., ICML 2021).  Trains an RF classifier on
        z_i = I[y_i >= tau].  Song et al. (ICML 2022) prove this recovers PI,
        not EI — it ignores *how much* better a candidate is above tau.

    mode="lfbo"
        LFBO-EI (Song et al., "A General Recipe for Likelihood-Free BO",
        ICML 2022).  Replaces uniform positive weights with improvement weights
        max(y_i - tau, eps), recovering EI from the density-ratio framework.
        No extra computation vs. BORE.

    mode="lfbo_ssl"
        LFBO-EI + semi-supervised pseudo-labeling of unlabeled pool candidates
        (DRE-BO-SSL, arXiv 2023).  After the initial classifier fit, the most
        confident positive and negative candidates from the pool are pseudo-
        labeled and added (down-weighted by ssl_alpha) to prevent the classifier
        from over-exploiting a narrow region as observations accumulate.

    adaptive_gamma
        When True, gamma anneals from gamma_init toward 0.10 as observations
        accumulate, focusing the elite threshold progressively on the top
        candidates rather than holding it fixed at the initial 25% quantile.
    """

    def __init__(
        self,
        gamma=BO_BORE_GAMMA,
        random_state=RANDOM_STATE,
        mode="bore",
        adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
        ssl_alpha=BO_SSL_ALPHA,
        ssl_n_pseudo=BO_SSL_N_PSEUDO,
    ):
        self.gamma_init     = gamma
        self.random_state   = random_state
        self.mode           = mode
        self.adaptive_gamma = adaptive_gamma
        self.ssl_alpha      = ssl_alpha
        self.ssl_n_pseudo   = ssl_n_pseudo
        self.objective      = OrdinalBOObjective(gamma=gamma)
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fallback_ei = EIAcquisition()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _gamma_t(self, n_observed):
        """Anneal gamma from gamma_init toward 0.10 as observations grow."""
        if not self.adaptive_gamma:
            return self.gamma_init
        # Smooth decay: starts at gamma_init, approaches 0.10 after ~100 obs.
        return max(0.10, self.gamma_init / (1.0 + n_observed / 50.0))

    # ── main ──────────────────────────────────────────────────────────────────

    def score(self, X_observed, y_observed, X_candidates, surrogate=None):
        """Return acquisition values for candidates.

        Falls back to EI (if surrogate provided) or random when labels are
        degenerate (all-zero or all-one).
        """
        gamma_t   = self._gamma_t(len(y_observed))
        objective = OrdinalBOObjective(gamma=gamma_t)
        labels, tau, sample_weight = objective.get_bore_labels(
            y_observed, mode=self.mode
        )

        if objective.is_degenerate(labels):
            if surrogate is not None:
                mu, sigma = surrogate.predict(X_candidates)
                return self.fallback_ei.score(mu, sigma, y_observed.max())
            return np.random.RandomState(self.random_state).uniform(
                size=len(X_candidates)
            )

        # Fit BORE / LFBO classifier
        if self.mode in ("lfbo", "lfbo_ssl"):
            self.clf.fit(X_observed, labels, sample_weight=sample_weight)
        else:
            self.clf.fit(X_observed, labels)

        pos_idx = list(self.clf.classes_).index(1)

        if self.mode == "lfbo_ssl":
            return self._score_ssl(
                X_observed, labels, sample_weight, X_candidates, pos_idx
            )

        proba = self.clf.predict_proba(X_candidates)
        return proba[:, pos_idx]

    def _score_ssl(self, X_observed, labels, sample_weight, X_candidates, pos_idx):
        """DRE-BO-SSL: augment training set with pseudo-labeled pool candidates.

        Steps:
          1. Get initial P(positive|x) from the LFBO-fitted classifier.
          2. Select the n_pseudo/2 most confident positives and negatives.
          3. Re-fit the classifier on real observations + down-weighted pseudo-labels.
          4. Return final acquisition scores from the augmented classifier.

        This prevents the classifier from collapsing to a spike around the
        current best region as more observations accumulate.
        """
        # Step 1 — initial probabilities
        proba       = self.clf.predict_proba(X_candidates)
        p_positive  = proba[:, pos_idx]

        # Step 2 — select confident pseudo-labels
        n_cand   = len(X_candidates)
        n_pseudo = min(self.ssl_n_pseudo, max(4, n_cand // 20))
        n_pos    = n_pseudo // 2
        n_neg    = n_pseudo - n_pos

        ranked        = np.argsort(p_positive)
        pseudo_neg_idx = ranked[:n_neg]
        pseudo_pos_idx = ranked[-n_pos:]
        all_pseudo_idx = np.concatenate([pseudo_neg_idx, pseudo_pos_idx])

        X_pseudo = X_candidates[all_pseudo_idx]
        y_pseudo = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        w_pseudo = np.full(len(y_pseudo), self.ssl_alpha)

        # Step 3 — augment and refit
        X_aug = np.vstack([X_observed, X_pseudo])
        y_aug = np.concatenate([labels, y_pseudo])
        w_aug = np.concatenate([sample_weight, w_pseudo])
        self.clf.fit(X_aug, y_aug, sample_weight=w_aug)

        # Step 4 — final scores
        proba_final = self.clf.predict_proba(X_candidates)
        return proba_final[:, pos_idx]


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
    """Dispatch acquisition function by name.

    Supported names: "bore", "lfbo", "lfbo_ssl", "ei", "lcb",
                     "thompson", "pi_ordinal", "random".

    "lfbo" and "lfbo_ssl" are first-class aliases for BOREAcquisition with
    mode="lfbo" and mode="lfbo_ssl" respectively — they require no surrogate.
    """
    if name == "ei":
        mu, sigma = surrogate.predict(X_candidates)
        f_best = kwargs.get("f_best", y_train.max())
        # If surrogate is ranking-based, convert f_best to rank space.
        if isinstance(surrogate, RankingRegressionSurrogate):
            f_best = surrogate.raw_to_rank(f_best)
        return EIAcquisition(xi=kwargs.get("xi", BO_EI_XI)).score(mu, sigma, f_best)

    elif name == "lcb":
        mu, sigma = surrogate.predict(X_candidates)
        return LCBAcquisition(kappa=kwargs.get("kappa", BO_LCB_KAPPA)).score(mu, sigma)

    elif name in ("bore", "lfbo", "lfbo_ssl"):
        # "lfbo" / "lfbo_ssl" are first-class names that map directly to modes.
        mode = name
        bore = BOREAcquisition(
            gamma=kwargs.get("gamma", BO_BORE_GAMMA),
            random_state=kwargs.get("random_state", RANDOM_STATE),
            mode=mode,
            adaptive_gamma=kwargs.get(
                "bore_adaptive_gamma", BO_BORE_ADAPTIVE_GAMMA
            ),
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
    """Convert BO knobs → full X_cv-compatible feature matrix.

    Uses X_names from the feature catalog to locate columns by name, so
    overrides are positionally correct regardless of which features survived
    the variance threshold.

    Handles four categories of derived features:
      1. Process knobs  — proc_raw:X (raw) + proc:X (MinMax-normalised)
      2. Solvent fractions — proc_raw/proc solvent_1/2_fraction from phi_1
      3. Concentration cols — metal_conc / linker_conc / umol_* from total_conc + ratio
      4. COSMO features   — computed on-the-fly via CosmoMixer from (sol1, sol2, phi_1)
      5. Interaction terms — recomputed from updated normalised process values
    """

    def __init__(
        self,
        template_row,
        X_names,
        X_cv,
        process_cols_present,
        cosmo_mixer=None,
        total_volume_ml=TOTAL_VOLUME_ML,
    ):
        self.template_row = np.array(template_row).flatten()
        self.X_names = list(X_names)
        self.process_cols_present = list(process_cols_present)
        self.cosmo_mixer = cosmo_mixer
        self.total_volume_ml = float(total_volume_ml)

        # name → column index lookup
        self._idx = {name: i for i, name in enumerate(X_names)}

        # Infer MinMax scale for each process column from training X_cv.
        # proc_raw:X columns hold the raw values → use their min/max to normalise
        # the corresponding proc:X (normalised) columns.
        self._proc_scale: dict = {}   # bare_col → (vmin, vmax)
        for col in process_cols_present:
            raw_key = f"proc_raw:{col}"
            i = self._idx.get(raw_key)
            if i is not None:
                col_vals = X_cv[:, i]
                self._proc_scale[col] = (
                    float(np.nanmin(col_vals)),
                    float(np.nanmax(col_vals)),
                )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _norm(self, col: str, raw_val) -> float:
        """MinMax-normalise raw_val using training scale for this column."""
        scale = self._proc_scale.get(col)
        if scale is None:
            return float(raw_val)
        vmin, vmax = scale
        if vmax == vmin:
            return 0.0
        return float(np.clip((raw_val - vmin) / (vmax - vmin), 0.0, 1.0))

    def _norm_arr(self, col: str, raw_vals: np.ndarray) -> np.ndarray:
        scale = self._proc_scale.get(col)
        if scale is None:
            return raw_vals.astype(float)
        vmin, vmax = scale
        if vmax == vmin:
            return np.zeros_like(raw_vals, dtype=float)
        return np.clip((raw_vals - vmin) / (vmax - vmin), 0.0, 1.0).astype(float)

    def _set(self, X: np.ndarray, key: str, vals: np.ndarray) -> None:
        """Write vals into column *key* if it exists."""
        idx = self._idx.get(key)
        if idx is not None:
            X[:, idx] = vals

    def _set_proc(self, X: np.ndarray, col: str, raw_vals: np.ndarray) -> None:
        """Set both proc_raw:col and proc:col for a process variable."""
        self._set(X, f"proc_raw:{col}", raw_vals)
        self._set(X, f"proc:{col}", self._norm_arr(col, raw_vals))

    # ── main ──────────────────────────────────────────────────────────────────

    def featurize(self, candidates_df: pd.DataFrame) -> np.ndarray:
        """Return full feature matrix (n_candidates × n_features)."""
        n = len(candidates_df)
        X = np.tile(self.template_row, (n, 1))

        # ── 1. Scalar process knobs ───────────────────────────────────────────
        for col in BO_CONTROLLABLE_PARAMS:
            if col in ("phi_1",):
                continue   # handled separately below
            if col in candidates_df.columns:
                self._set_proc(X, col, candidates_df[col].values.astype(float))

        # ── 1b. Fix total_solvent_volume_ml to the configured synthesis volume ──
        self._set_proc(X, "total_solvent_volume_ml",
                       np.full(n, self.total_volume_ml))

        # ── 2. phi_1 → solvent fractions ─────────────────────────────────────
        phi_1 = np.clip(
            candidates_df.get("phi_1", pd.Series(np.ones(n))).values.astype(float),
            0.0, 1.0,
        )
        self._set_proc(X, "solvent_1_fraction", phi_1)
        self._set_proc(X, "solvent_2_fraction", 1.0 - phi_1)

        # ── 3. Derive concentration cols from total_conc + ratio + volume ─────
        if "total_conc" in candidates_df.columns:
            total_conc = candidates_df["total_conc"].values.astype(float)
            ratio = candidates_df.get(
                "metal_over_linker_ratio", pd.Series(np.ones(n))
            ).values.astype(float)
            ratio = np.where(ratio > 0, ratio, 1.0)
            equiv = candidates_df.get(
                "equivalents", pd.Series(np.ones(n))
            ).values.astype(float)

            metal_conc = total_conc * ratio / (1.0 + ratio)
            linker_conc = total_conc / (1.0 + ratio)
            umol_metal = metal_conc * self.total_volume_ml
            umol_linker = linker_conc * self.total_volume_ml
            # mod: equivalents = umol_mod / umol_metal (stoichiometric, unitless)
            umol_mod  = equiv * umol_metal
            mod_conc  = umol_mod / self.total_volume_ml

            for col, vals in [
                ("total_conc",           total_conc),
                ("metal_conc",           metal_conc),
                ("linker_conc",          linker_conc),
                ("umol_metal_precursor", umol_metal),
                ("umol_linker",          umol_linker),
                ("umol_modulator",       umol_mod),
                ("mod_conc",             mod_conc),
            ]:
                self._set_proc(X, col, vals)

        # ── 4. COSMO features from (sol1, sol2, phi_1) ───────────────────────
        if self.cosmo_mixer is not None and "solvent_1" in candidates_df.columns:
            sol1_arr = candidates_df["solvent_1"].values
            sol2_arr = candidates_df.get(
                "solvent_2", pd.Series([""] * n)
            ).values

            for i in range(n):
                cosmo = self.cosmo_mixer.compute(sol1_arr[i], sol2_arr[i], phi_1[i])
                for col, val in cosmo.items():
                    if np.isfinite(val):
                        self._set_proc(X[i:i+1], col, np.array([val]))

        # ── 5. Recompute interaction features ─────────────────────────────────
        # Gather normalised values for the three terms involved
        def _get_norm(col):
            if col in candidates_df.columns:
                return self._norm_arr(col, candidates_df[col].values.astype(float))
            # Fall back to current (possibly updated) proc: column in X
            proc_idx = self._idx.get(f"proc:{col}")
            return X[:, proc_idx] if proc_idx is not None else np.zeros(n)

        t  = _get_norm("temperature_k")
        mr = _get_norm("metal_over_linker_ratio")
        rh = _get_norm("reaction_hours")

        self._set(X, "proc_int:temp_x_metal_ratio",      t  * mr)
        self._set(X, "proc_int:temp_x_rxn_hours",         t  * rh)
        self._set(X, "proc_int:metal_ratio_x_rxn_hours",  mr * rh)
        self._set(X, "proc_int:temp_sq",                  t  ** 2)
        self._set(X, "proc_int:metal_ratio_sq",           mr ** 2)
        self._set(X, "proc_int:hightemp_flag",            (t > 0.85).astype(float))

        return X


class FeasibilityScorer:
    """Soft feasibility priors based on MOF synthesis domain knowledge.

    Returns a score in [0, 1] for each candidate:
      1.0 = fully feasible
      0.0 = known infeasible (e.g., temperature above solvent boiling point)

    Encodes:
      - Temperature below solvent boiling point (with soft safety margin).
      Concentration bounds are already enforced by SearchSpace LHS bounds.

    Reference: Griffiths et al., "Bayesian Optimization with Known
    Experimental and Design Constraints for Chemistry,"
    Digital Discovery, 2022.
    """

    # Approximate boiling points (K) for solvents used in the candidate pool.
    SOLVENT_BP_K = {
        "DICHLOROMETHANE":       313.0,
        "TETRAHYDROFURAN":       339.0,
        "CHLOROFORM":            334.0,
        "ACETONITRILE":          355.0,
        "ETHANOL":               351.0,
        "BENZENE":               353.0,
        "TOLUENE":               384.0,
        "N,N-DIMETHYLFORMAMIDE": 426.0,
    }

    def __init__(self, temperature_margin_k=15.0):
        """
        temperature_margin_k : safety margin (K) below solvent boiling point.
            Candidates at T > BP - margin receive an exponential soft penalty.
        """
        self.temperature_margin_k = float(temperature_margin_k)

    def score(self, candidates_df):
        """Return per-candidate feasibility score in [0, 1].

        Parameters
        ----------
        candidates_df : DataFrame — must have "temperature_k" and "solvent_1" columns.

        Returns
        -------
        feasibility : ndarray shape (n,) — 1.0 = feasible, 0.0 = infeasible.
        """
        n = len(candidates_df)
        feasibility = np.ones(n)

        if ("temperature_k" not in candidates_df.columns or
                "solvent_1" not in candidates_df.columns):
            return feasibility

        temps = candidates_df["temperature_k"].values.astype(float)
        sol1  = candidates_df["solvent_1"].values

        for i in range(n):
            bp         = self.SOLVENT_BP_K.get(str(sol1[i]).upper().strip(), 500.0)
            safe_limit = bp - self.temperature_margin_k
            if temps[i] > safe_limit:
                # Exponential decay: drops to ~0.05 at T = bp + 45 K
                excess = temps[i] - safe_limit
                feasibility[i] *= float(np.exp(-excess / 15.0))

        return np.clip(feasibility, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# NeighborhoodTemplateSelector
# ─────────────────────────────────────────────────────────────
class NeighborhoodTemplateSelector:
    """Find structurally similar past experiments to seed a trust region center.

    Two-stage similarity pipeline
    ------------------------------
    Stage 1 — Morgan fingerprint (ECFP4) Tanimoto on linker + precursor SMILES.
      Fast initial lookup.  Used to find the single nearest-neighbor experiment
      whose full feature row becomes the chemistry reference vector, and to
      provide a fallback similarity score when the chemistry feature matrix
      is not available.

      Limitation: ECFP4 treats hub-element identity (C vs Si vs Sn) as a hard
      difference, so a C-triphos and Si-triphos will score low even though they
      are topologically analogous.

    Stage 2 — Chemistry feature cosine similarity on X_cv (surrogate features).
      Restricts X_cv to chemistry-only columns (excludes Process / Interaction /
      Unknown groups) and computes L2-normalised cosine similarity between the
      nearest-neighbor row from Stage 1 and every other experiment.

      Why this is better for linker comparison:
        • TTP features encode hub element, arm type, arm length, topicity — so a
          Si-triphos and C-triphos with the same arm structure score high.
        • G14 features explicitly encode Si/Ge/Sn hub presence and geometry.
        • ChemBERTa embeddings capture learned chemical context (element-aware).
        • DRFP encodes reaction-level structural similarity.
      Together these capture the "same topology, different hub" relationship that
      ECFP4 misses.

    The final neighbor weight combines both stages with a configurable blend:
      weight = (α × fp_sim + (1-α) × feat_sim) × (score + 1)

    References: Schmid et al. (2025) arXiv:2502.18966 (Curried BO);
                Felton et al. (2023) ACS Central Science (MTBO for reactions).
    """

    # Maps BO controllable param names → df_merged column names
    _PARAM_TO_COL = {
        "equivalents":             "equivalents",
        "temperature_k":           "temperature_k",
        "total_conc":              "total_conc",
        "phi_1":                   "solvent_1_fraction",
        "metal_over_linker_ratio": "metal_over_linker_ratio",
    }

    # Feature groups that represent process / non-chemistry information
    _PROCESS_GROUPS = {"Process", "Process_Interaction", "Unknown"}

    def __init__(
        self,
        df_train,
        X_cv,
        X_groups,
        linker_col="smiles_linker_1",
        precursor_col="smiles_precursor",
        score_col="pxrd_score",
        fp_blend=0.3,
        linker_weight=0.65,
        top_k=15,
        min_similarity=0.05,
        fp_radius=2,
        fp_nbits=2048,
    ):
        """
        Parameters
        ----------
        df_train      : DataFrame — training experiments (rows match X_cv)
        X_cv          : ndarray (n, d) — full feature matrix from checkpoint
        X_groups      : list[str] len d — feature group labels per column
        linker_col    : str — SMILES column for the linker in df_train
        precursor_col : str — SMILES column for the precursor in df_train
        score_col     : str — outcome column
        fp_blend      : float in [0,1] — weight given to Morgan FP similarity
                        vs chemistry feature cosine similarity (default 0.3).
                        Lower = trust the surrogate features more.
        linker_weight : float — weight for linker vs precursor in FP similarity
        top_k         : int — number of neighbors to return
        min_similarity: float — minimum combined similarity to include
        fp_radius     : int — Morgan FP radius (2 = ECFP4)
        fp_nbits      : int — Morgan FP bit vector length
        """
        self.df            = df_train.copy().reset_index(drop=True)
        self.linker_col    = linker_col
        self.precursor_col = precursor_col
        self.score_col     = score_col
        self.fp_blend      = float(fp_blend)
        self.linker_weight    = linker_weight
        self.precursor_weight = 1.0 - linker_weight
        self.top_k          = top_k
        self.min_similarity = min_similarity
        self.fp_radius = fp_radius
        self.fp_nbits  = fp_nbits

        # ── Stage 1: pre-compute Morgan fingerprints ──────────────────────────
        self._linker_fps    = [self._to_fp(s) for s in
                               df_train[linker_col].fillna("")]
        self._precursor_fps = [self._to_fp(s) for s in
                               df_train[precursor_col].fillna("")]

        # ── Stage 2: build L2-normalised chemistry feature matrix ─────────────
        # Exclude process / interaction / unknown columns so similarity is
        # purely about molecular chemistry, not about what conditions were used.
        chem_mask = np.array([g not in self._PROCESS_GROUPS
                              for g in X_groups], dtype=bool)
        n_chem = int(chem_mask.sum())
        n_proc = len(X_groups) - n_chem
        print(f"[NeighborhoodTemplate] Chemistry feature matrix: "
              f"{n_chem} chemistry cols, {n_proc} process cols excluded")

        X_chem = np.array(X_cv[:, chem_mask], dtype=float)
        X_chem = np.nan_to_num(X_chem, nan=0.0)
        norms  = np.linalg.norm(X_chem, axis=1, keepdims=True)
        norms  = np.where(norms < 1e-10, 1.0, norms)
        self._X_chem_normed = X_chem / norms   # shape (n, n_chem)
        self._chem_mask = chem_mask

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_fp(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_nbits
            )
        except Exception:
            return None

    def _tanimoto(self, fp1, fp2):
        if fp1 is None or fp2 is None:
            return 0.0
        from rdkit.DataStructs import TanimotoSimilarity
        return float(TanimotoSimilarity(fp1, fp2))

    def _fp_similarity(self, tgt_linker_fp, tgt_precursor_fp):
        """Return per-experiment combined Morgan FP similarity array."""
        sims = np.zeros(len(self.df))
        for i in range(len(self.df)):
            l = self._tanimoto(tgt_linker_fp,    self._linker_fps[i])
            p = self._tanimoto(tgt_precursor_fp, self._precursor_fps[i])
            sims[i] = self.linker_weight * l + self.precursor_weight * p
        return sims

    def _feat_similarity(self, ref_idx):
        """Cosine similarity between experiment ref_idx and all others."""
        ref = self._X_chem_normed[ref_idx]          # shape (n_chem,)
        return self._X_chem_normed @ ref             # shape (n,)

    # ── main ──────────────────────────────────────────────────────────────────

    def select(self, target_linker_smiles, target_precursor_smiles,
               search_bounds):
        """Find similar experiments and return weighted process center + spread.

        Parameters
        ----------
        target_linker_smiles    : str — SMILES of the target linker
        target_precursor_smiles : str — SMILES of the target precursor
        search_bounds           : dict — param → (lo, hi) from SearchSpace.bounds

        Returns
        -------
        center      : dict — param → weighted-mean process condition value
        spread      : dict — param → weighted std (seeds trust region radius)
        neighbors   : DataFrame — top-k neighbors with similarity scores
        ref_idx     : int — dataset index of nearest-neighbor (chemistry template)
        """
        tgt_linker_fp    = self._to_fp(target_linker_smiles)
        tgt_precursor_fp = self._to_fp(target_precursor_smiles)

        # Stage 1: Morgan FP similarity for all experiments
        fp_sims = self._fp_similarity(tgt_linker_fp, tgt_precursor_fp)

        # Stage 1b: find the single nearest neighbor by FP similarity.
        # Its chemistry feature row becomes the reference for Stage 2.
        ref_idx  = int(np.argmax(fp_sims))
        ref_fp   = float(fp_sims[ref_idx])
        print(f"[NeighborhoodTemplate] Stage-1 nearest neighbor idx={ref_idx} "
              f"(fp_sim={ref_fp:.3f})")

        # Stage 2: chemistry feature cosine similarity from the NN reference
        feat_sims = self._feat_similarity(ref_idx)

        # Blend the two similarity signals
        combined_sims = (self.fp_blend * fp_sims +
                         (1.0 - self.fp_blend) * np.clip(feat_sims, 0.0, 1.0))

        # Build neighbor DataFrame
        scores = pd.to_numeric(
            self.df[self.score_col], errors="coerce").fillna(0.0).values

        rows = []
        for i in range(len(self.df)):
            entry = {
                "fp_sim":      float(fp_sims[i]),
                "feat_sim":    float(feat_sims[i]),
                "combined_sim": float(combined_sims[i]),
                "score":       float(scores[i]),
            }
            for param, (lo, hi) in search_bounds.items():
                col = self._PARAM_TO_COL.get(param, param)
                val = pd.to_numeric(
                    self.df.iloc[i].get(col, np.nan), errors="coerce")
                entry[param] = float(
                    np.clip(val if np.isfinite(val) else (lo + hi) / 2, lo, hi))
            rows.append(entry)

        sim_df = pd.DataFrame(rows)
        sim_df = sim_df[sim_df["combined_sim"] >= self.min_similarity]

        if len(sim_df) == 0:
            print(f"[NeighborhoodTemplate] No neighbors above "
                  f"min_similarity={self.min_similarity}. "
                  "Falling back to global search.")
            return None, None, None, ref_idx

        sim_df = sim_df.nlargest(self.top_k, "combined_sim").copy()

        # Weight = combined_sim × (score + 1)
        sim_df["weight"] = sim_df["combined_sim"] * (sim_df["score"] + 1.0)
        if sim_df["weight"].sum() <= 0:
            sim_df["weight"] = 1.0

        center = {}
        spread = {}
        for param, (lo, hi) in search_bounds.items():
            vals  = sim_df[param].values.astype(float)
            w     = sim_df["weight"].values
            wmean = float(np.average(vals, weights=w))
            wstd  = float(np.sqrt(np.average((vals - wmean) ** 2, weights=w)))
            center[param] = float(np.clip(wmean, lo, hi))
            spread[param] = wstd

        print(
            f"[NeighborhoodTemplate] {len(sim_df)} neighbors | "
            f"fp_sim {sim_df['fp_sim'].min():.2f}–{sim_df['fp_sim'].max():.2f} | "
            f"feat_sim {sim_df['feat_sim'].min():.2f}–"
            f"{sim_df['feat_sim'].max():.2f}"
        )
        print("[NeighborhoodTemplate] Process center: "
              + "  ".join(f"{p}={v:.3g}" for p, v in center.items()))

        return center, spread, sim_df, ref_idx


# ─────────────────────────────────────────────────────────────
# TrustRegion
# ─────────────────────────────────────────────────────────────
class TrustRegion:
    """TuRBO-style adaptive trust region for recommend mode.

    Maintains a hyperrectangle in process-parameter space, centered on the
    best observed conditions.  The region expands after consecutive successes
    (new best found) and shrinks after consecutive failures.

    Designed for non-GP surrogates (RF/XGB): the region geometry is defined
    as a symmetric fraction of the full parameter range, not using GP
    lengthscales.  The success/failure counter rule follows TuRBO directly.

    Reference: Eriksson et al. (2019) NeurIPS — "Scalable Global Optimization
    via Local Bayesian Optimization."  arXiv:1910.01739.
    """

    def __init__(
        self,
        center,
        full_bounds,
        length=0.8,
        length_min=0.1,
        length_max=1.6,
        success_tol=3,
        failure_tol=3,
        param_scales=None,
    ):
        """
        Parameters
        ----------
        center      : dict — param → initial center value
        full_bounds : dict — param → (lo, hi) global bounds from SearchSpace
        length      : float — initial TR length as fraction of full range [0,1]
        length_min  : float — restart threshold
        length_max  : float — maximum allowed length
        success_tol : int   — consecutive successes before expanding
        failure_tol : int   — consecutive failures before shrinking
        """
        self.center      = dict(center)
        self.full_bounds = dict(full_bounds)
        self.length      = float(length)
        self.length_min  = float(length_min)
        self.length_max  = float(length_max)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self._successes  = 0
        self._failures   = 0
        self._best_score = -np.inf
        # Per-parameter scale factors: param → float in [0.5, 2.0].
        # Narrow-spread parameters (tightly clustered neighbours) get scale < 1;
        # wide-spread parameters get scale > 1 to explore further.
        self.param_scales = dict(param_scales) if param_scales is not None else {}

    def update(self, new_best_score):
        """Call at the start of each recommend iteration with the current f_best.

        Updates success/failure counters and adjusts the trust region length.
        Prints a message when the region expands or shrinks.
        """
        if new_best_score > self._best_score + 1e-6:
            self._successes += 1
            self._failures   = 0
            self._best_score = float(new_best_score)
        else:
            self._failures  += 1
            self._successes  = 0

        if self._successes >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self._successes = 0
            print(f"[TrustRegion] Expanded  → length={self.length:.3f}  "
                  f"(best={self._best_score:.1f})")

        if self._failures >= self.failure_tol:
            self.length = self.length / 2.0
            self._failures = 0
            print(f"[TrustRegion] Shrunk    → length={self.length:.3f}  "
                  f"(best={self._best_score:.1f})")
            if self.length < self.length_min:
                print("[TrustRegion] Below minimum — resetting length to 0.8.")
                self.length = 0.8

    def recenter(self, new_center):
        """Move the trust region to track a new best point."""
        for param in self.center:
            if param in new_center:
                self.center[param] = float(new_center[param])

    def get_bounds(self):
        """Return the current trust region as clipped (lo, hi) per parameter.

        Each parameter's half-width is scaled by its param_scale factor:
          scale < 1 → narrower search (tightly clustered neighbours)
          scale > 1 → wider search  (neighbours spread across the range)
        """
        tr_bounds = {}
        for param, (lo, hi) in self.full_bounds.items():
            scale = self.param_scales.get(param, 1.0)
            half  = self.length * scale * (hi - lo) / 2.0
            ctr   = self.center.get(param, (lo + hi) / 2.0)
            tr_lo = float(np.clip(ctr - half, lo, hi))
            tr_hi = float(np.clip(ctr + half, lo, hi))
            if tr_lo >= tr_hi:
                tr_lo, tr_hi = lo, hi
            tr_bounds[param] = (tr_lo, tr_hi)
        return tr_bounds

    def to_dict(self):
        return {
            "center":       self.center,
            "full_bounds":  self.full_bounds,
            "length":       self.length,
            "length_min":   self.length_min,
            "length_max":   self.length_max,
            "success_tol":  self.success_tol,
            "failure_tol":  self.failure_tol,
            "_successes":   self._successes,
            "_failures":    self._failures,
            "_best_score":  self._best_score,
            "param_scales": self.param_scales,
        }

    @classmethod
    def from_dict(cls, d):
        tr = cls(
            center=d["center"],
            full_bounds=d["full_bounds"],
            length=d["length"],
            length_min=d["length_min"],
            length_max=d["length_max"],
            success_tol=d["success_tol"],
            failure_tol=d["failure_tol"],
            param_scales=d.get("param_scales", {}),
        )
        tr._successes  = d["_successes"]
        tr._failures   = d["_failures"]
        tr._best_score = d["_best_score"]
        return tr


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
        bore_adaptive_gamma=BO_BORE_ADAPTIVE_GAMMA,
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
        self.bore_adaptive_gamma = bore_adaptive_gamma

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
                    "bore_adaptive_gamma": self.bore_adaptive_gamma,
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
                "bore_adaptive_gamma": self.bore_adaptive_gamma,
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
