"""
bo_core.py — Bayesian Optimization components for the LVMOF-Surrogate pipeline.

Classes:
  SolventMixer           — COSMO property vectors for binary solvent mixtures
  SearchSpace            — LHS candidate generation over continuous + discrete params
  RegressionSurrogate    — wraps RF/XGB regressor, exposes (mu, sigma)
  XGBoostBootstrapEnsemble — M bootstrap XGB regressors for uncertainty
  OrdinalBOObjective     — raw 0-9 pxrd_score objective, LFBO label generation
  EIAcquisition          — Expected Improvement
  LFBOAcquisition        — LFBO-EI classifier (Song et al.)
  _consensus_acquisition — EI ∩ LFBO rank-intersection; falls back to LFBO
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
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

from config import (
    RANDOM_STATE,
    BO_LFBO_GAMMA,
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
    BO_LFBO_ADAPTIVE_GAMMA,
    BO_CLUSTER_DIV_LAMBDA,
)
from cosmo_features import (
    load_cosmo_index,
    load_sigma_profile,
    compute_sigma_moments,
    CosmoMixer,
)


# ─────────────────────────────────────────────────────────────
# Phosphine-based stoichiometric ratio
# ─────────────────────────────────────────────────────────────
def count_phosphines(smiles):
    """Count phosphorus atoms in a SMILES string.

    Uses sanitize=False as fallback so organometallic SMILES
    (Rh, Ir, Pd complexes) that fail RDKit valence checks can
    still be atom-counted.
    """
    from rdkit import Chem
    if not smiles or (isinstance(smiles, float) and np.isnan(smiles)):
        return 0
    smi = str(smiles).strip()
    if not smi:
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)


def compute_stoichiometric_ratio(precursor_smi, linker_smi):
    """Compute metal/linker molar ratio from phosphine counts.

    ratio = P_count(linker) / P_count(precursor)

    If the linker has 4 phosphines and the precursor has 2, the ratio
    is 2 — meaning 2 mol metal per 1 mol linker are needed to satisfy
    all phosphine binding sites.

    Returns the ratio (float), or None if either molecule has 0
    phosphines (e.g. metal salts without phosphine ligands).
    """
    p_precursor = count_phosphines(precursor_smi)
    p_linker = count_phosphines(linker_smi)
    if p_precursor > 0 and p_linker > 0:
        return p_linker / p_precursor
    return None


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
# Leakage-free chemistry groups for BO evaluation
# ─────────────────────────────────────────────────────────────

def compute_chemistry_groups(df, linker_col="smiles_linker_1", min_group_size=20):
    """Assign each experiment a group label based on linker identity.

    Unlike the KMeans groups (which are fit on UMAP of the entire dataset
    and leak manifold structure), these groups are intrinsic to the
    chemistry — they are deterministic properties of the molecule and
    require zero fitting.  This makes them leakage-free for BO init/pool
    splitting and evaluation.

    Linkers with fewer than ``min_group_size`` experiments are merged into
    a single 'rare' group so that every group is large enough for a
    meaningful 30/70 split.

    Parameters
    ----------
    df : DataFrame — the masked experiment dataframe (rows match X_cv)
    linker_col : str — column containing linker SMILES
    min_group_size : int — groups smaller than this are merged

    Returns
    -------
    groups : np.ndarray int (n,) — group labels starting from 0
    group_names : list[str] — human-readable label for each group id
    """
    linker_ids, uniques = pd.factorize(df[linker_col].fillna("unknown"))

    # Count per linker
    counts = np.bincount(linker_ids)
    # Merge small groups
    rare_mask = counts < min_group_size
    merged = linker_ids.copy()
    rare_group = len(uniques)  # new id for merged group
    for lid in np.where(rare_mask)[0]:
        merged[linker_ids == lid] = rare_group

    # Re-number to 0..n_groups-1
    final_ids, final_uniques = pd.factorize(merged)

    # Build readable names
    group_names = []
    for uid in final_uniques:
        if uid == rare_group:
            n_rare = int(rare_mask.sum())
            group_names.append(f"rare ({n_rare} linkers)")
        else:
            smiles = str(uniques[uid])
            short = smiles[:30] + "…" if len(smiles) > 30 else smiles
            group_names.append(short)

    groups = np.asarray(final_ids, dtype=int)
    n_groups = int(groups.max()) + 1
    print(f"[BO groups] Chemistry-based grouping: {n_groups} groups "
          f"(from {len(uniques)} linkers, {int(rare_mask.sum())} merged as rare)")
    for gid in range(n_groups):
        print(f"    group {gid}: n={int((groups == gid).sum()):>4d}  "
              f"{group_names[gid]}")

    return groups, group_names


# ─────────────────────────────────────────────────────────────
# SearchSpace
# ─────────────────────────────────────────────────────────────
class SearchSpace:
    """Generate candidate parameter sets: LHS over continuous params × solvent pairs.

    phi_1 (solvent_1 volume fraction) is now a continuous BO parameter in the LHS;
    the solvent_mixer is used only to enumerate (sol1, sol2) pairs whose COSMO
    features are computed on-the-fly by CandidateFeaturizer.

    By default (observed_pairs_only=False) the search space includes every
    single-solvent option and every binary combination that can be formed from
    the individual solvents the lab has ever used, as long as both solvents have
    COSMO profiles.  Pass observed_pairs_only=True to restrict candidates to
    exactly the (sol1, sol2) co-occurrences seen in the training data.
    """

    def __init__(self, train_df=None, solvent_mixer=None, extra_params=None,
                 observed_pairs_only=False):
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
        if solvent_mixer is not None and train_df is not None:
            if observed_pairs_only:
                self.solvent_pairs = self._enumerate_observed_pairs(train_df, solvent_mixer)
            else:
                self.solvent_pairs = self._enumerate_all_pairs(train_df, solvent_mixer)
        else:
            self.solvent_pairs = [{"solvent_1": "", "solvent_2": ""}]

    def _enumerate_all_pairs(self, train_df, solvent_mixer):
        """All single-solvent + binary combinations from individually-used solvents.

        Any solvent seen in any column of train_df that also has a COSMO profile
        is included.  Binary pairs are formed as all unordered combinations of
        those solvents (sol1 < sol2 alphabetically to avoid duplicates).
        """
        import itertools
        solvents = sorted(solvent_mixer.available_solvents_from_df(train_df))
        if not solvents:
            return [{"solvent_1": "", "solvent_2": ""}]
        pairs = [{"solvent_1": s, "solvent_2": ""} for s in solvents]
        for s1, s2 in itertools.combinations(solvents, 2):
            pairs.append({"solvent_1": s1, "solvent_2": s2})
        print(f"  [SearchSpace] {len(solvents)} lab solvents → "
              f"{len(solvents)} single + {len(pairs) - len(solvents)} binary = "
              f"{len(pairs)} solvent combinations")
        return pairs

    def _enumerate_observed_pairs(self, train_df, solvent_mixer):
        """Build unique (sol1, sol2) pairs observed together in training data."""
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

    Sigma calibration
    -----------------
    Raw inter-tree / bootstrap variance systematically underestimates true
    predictive uncertainty (captures epistemic but not aleatoric variance).
    On first fit, K-fold CV estimates a multiplicative scaling factor so that
    z = (y - mu) / (sigma * scale) ~ N(0, 1), which EI assumes.
    """

    def __init__(self, pipeline, model_type="rf"):
        self.pipeline = pipeline
        self.model_type = model_type
        self.bootstrap_ensemble = None  # set externally for XGB
        self.sigma_scale_ = None        # computed on first fit

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        if self.bootstrap_ensemble is not None:
            Xt = self._transform_features(X)
            self.bootstrap_ensemble.fit(Xt, y)

        # One-time sigma calibration via K-fold CV
        if self.sigma_scale_ is None:
            self._calibrate_sigma(X, y)

        return self

    # ── sigma calibration ────────────────────────────────────────────────────

    def _calibrate_sigma(self, X, y, n_splits=5):
        """Compute sigma scaling factor via K-fold cross-validation.

        RF inter-tree std and XGB bootstrap std underestimate the true
        predictive uncertainty.  This estimates a multiplicative factor so
        that calibrated z-scores z = (y - mu) / (sigma * scale) have unit
        variance, which is what EI's Gaussian assumption requires.

        Called once on the first fit(); the factor is reused for subsequent
        fits because it reflects the model architecture's inherent bias, not
        the specific training-set size.
        """
        from sklearn.model_selection import KFold

        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 2 * n_splits:
            self.sigma_scale_ = 1.0
            return

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        z_all = []

        for train_idx, val_idx in kf.split(X):
            try:
                fold_pipe = clone(self.pipeline)
                fold_pipe.fit(X[train_idx], y[train_idx])

                # Transform validation data through preprocessing steps
                Xv = np.asarray(X[val_idx])
                reg = None
                for name, step in fold_pipe.steps:
                    if isinstance(step, (RandomForestRegressor, XGBRegressor)):
                        reg = step
                        break
                    Xv = step.transform(Xv)

                if reg is None:
                    continue

                # Compute raw mu and sigma on validation fold
                if self.model_type == "rf" and hasattr(reg, "estimators_"):
                    preds = np.array(
                        [t.predict(Xv) for t in reg.estimators_]
                    )
                    mu = preds.mean(axis=0)
                    sigma = preds.std(axis=0)
                elif (self.model_type == "xgb"
                      and self.bootstrap_ensemble is not None):
                    # Fit a small bootstrap ensemble on the fold
                    Xt_train = np.asarray(X[train_idx])
                    for name2, step2 in fold_pipe.steps:
                        if isinstance(
                            step2, (RandomForestRegressor, XGBRegressor)
                        ):
                            break
                        Xt_train = step2.transform(Xt_train)
                    fold_boot = XGBoostBootstrapEnsemble(
                        self.bootstrap_ensemble.base_params,
                        M=min(self.bootstrap_ensemble.M, 10),
                        random_state=self.bootstrap_ensemble.random_state,
                    )
                    fold_boot.fit(Xt_train, y[train_idx])
                    mu, sigma = fold_boot.predict(Xv)
                else:
                    continue

                valid = sigma > 1e-10
                if valid.sum() < 2:
                    continue
                z = (y[val_idx][valid] - mu[valid]) / sigma[valid]
                z_all.extend(z.tolist())
            except Exception:
                continue

        if len(z_all) < 10:
            self.sigma_scale_ = 1.0
            return

        z_arr = np.array(z_all)
        scale = float(np.std(z_arr))
        self.sigma_scale_ = max(scale, 0.1)
        print(
            f"[Surrogate] sigma calibrated: scale={self.sigma_scale_:.3f} "
            f"(raw z-std={scale:.3f}, mean_z={float(np.mean(z_arr)):+.3f}, "
            f"n={len(z_all)})"
        )

    # ── core methods ─────────────────────────────────────────────────────────

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
        """Return (mu, sigma) with calibrated sigma."""
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

        # Apply calibration scaling
        scale = self.sigma_scale_ if self.sigma_scale_ is not None else 1.0
        sigma = sigma * scale

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
    space for use with the EI acquisition function.

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

    def _calibrate_sigma(self, X, y_ranked, n_splits=5):
        """Override: calibrate with per-fold rank normalization.

        The parent receives globally-ranked y, but each CV fold must
        independently rank-normalize its training portion so the fold's
        model sees the same [0, 1] target distribution it would in
        production.  Validation targets are mapped to rank space via the
        fold's training CDF.
        """
        from sklearn.model_selection import KFold
        from scipy.stats import rankdata

        y_raw = (self._y_train_raw
                 if self._y_train_raw is not None
                 else np.asarray(y_ranked, dtype=float))

        X = np.asarray(X)
        n = len(y_raw)
        if n < 2 * n_splits:
            self.sigma_scale_ = 1.0
            return

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        z_all = []

        for train_idx, val_idx in kf.split(X):
            try:
                # Per-fold rank normalization
                y_fold_raw = y_raw[train_idx]
                n_fold = len(y_fold_raw)
                y_fold_ranked = (
                    (rankdata(y_fold_raw, method="average") - 1.0)
                    / max(n_fold - 1, 1)
                )

                # Map validation targets into the fold's rank space
                y_val_ranked = np.array([
                    float(np.clip(
                        ((y_fold_raw <= v).sum() - 1) / max(n_fold - 1, 1),
                        0.0, 1.0,
                    ))
                    for v in y_raw[val_idx]
                ])

                fold_pipe = clone(self.pipeline)
                fold_pipe.fit(X[train_idx], y_fold_ranked)

                Xv = np.asarray(X[val_idx])
                reg = None
                for name, step in fold_pipe.steps:
                    if isinstance(
                        step, (RandomForestRegressor, XGBRegressor)
                    ):
                        reg = step
                        break
                    Xv = step.transform(Xv)

                if reg is None:
                    continue

                if self.model_type == "rf" and hasattr(reg, "estimators_"):
                    preds = np.array(
                        [t.predict(Xv) for t in reg.estimators_]
                    )
                    mu = preds.mean(axis=0)
                    sigma = preds.std(axis=0)
                elif (self.model_type == "xgb"
                      and self.bootstrap_ensemble is not None):
                    Xt_train = np.asarray(X[train_idx])
                    for name2, step2 in fold_pipe.steps:
                        if isinstance(
                            step2, (RandomForestRegressor, XGBRegressor)
                        ):
                            break
                        Xt_train = step2.transform(Xt_train)
                    fold_boot = XGBoostBootstrapEnsemble(
                        self.bootstrap_ensemble.base_params,
                        M=min(self.bootstrap_ensemble.M, 10),
                        random_state=self.bootstrap_ensemble.random_state,
                    )
                    fold_boot.fit(Xt_train, y_fold_ranked)
                    mu, sigma = fold_boot.predict(Xv)
                else:
                    continue

                valid = sigma > 1e-10
                if valid.sum() < 2:
                    continue
                z = (y_val_ranked[valid] - mu[valid]) / sigma[valid]
                z_all.extend(z.tolist())
            except Exception:
                continue

        if len(z_all) < 10:
            self.sigma_scale_ = 1.0
            return

        z_arr = np.array(z_all)
        scale = float(np.std(z_arr))
        self.sigma_scale_ = max(scale, 0.1)
        print(
            f"[Surrogate] sigma calibrated (ranking): "
            f"scale={self.sigma_scale_:.3f} "
            f"(raw z-std={scale:.3f}, mean_z={float(np.mean(z_arr)):+.3f}, "
            f"n={len(z_all)})"
        )

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
    """Raw 0-9 pxrd_score as BO objective + LFBO label generation."""

    def __init__(self, gamma=BO_LFBO_GAMMA):
        self.gamma = gamma

    def get_lfbo_labels(self, y_observed):
        """Generate binary labels and LFBO improvement weights.

        Labels: z_i = I[y >= tau], where tau is the (1-gamma) quantile.
        Weights: max(y_i - tau, eps) for positive class, 1.0 for negative.
        Recovers Expected Improvement (EI) via the density-ratio framework
        (Song et al., ICML 2022).

        Returns (labels, tau, sample_weight).
        """
        tau = np.quantile(y_observed, 1.0 - self.gamma)
        labels = (y_observed >= tau).astype(int)

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


class LFBOAcquisition:
    """LFBO-EI acquisition function (Song et al., ICML 2022).

    Trains an RF classifier on binary labels z_i = I[y_i >= tau] with
    improvement weights max(y_i - tau, eps) for the positive class,
    recovering Expected Improvement from the density-ratio framework.

    adaptive_gamma
        When True, gamma anneals from gamma_init toward 0.10 as observations
        accumulate, focusing the elite threshold progressively on the top
        candidates rather than holding it fixed at the initial 25% quantile.
    """

    def __init__(
        self,
        gamma=BO_LFBO_GAMMA,
        random_state=RANDOM_STATE,
        adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
    ):
        self.gamma_init     = gamma
        self.random_state   = random_state
        self.adaptive_gamma = adaptive_gamma
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
        labels, tau, sample_weight = objective.get_lfbo_labels(y_observed)

        if objective.is_degenerate(labels):
            if surrogate is not None:
                mu, sigma = surrogate.predict(X_candidates)
                return self.fallback_ei.score(mu, sigma, y_observed.max())
            return np.random.RandomState(self.random_state).uniform(
                size=len(X_candidates)
            )

        self.clf.fit(X_observed, labels, sample_weight=sample_weight)

        pos_idx = list(self.clf.classes_).index(1)

        proba = self.clf.predict_proba(X_candidates)
        return proba[:, pos_idx]


class ThompsonSamplingAcquisition:
    """Thompson Sampling: sample one tree from RF regressor ensemble."""

    def __init__(self, random_state=RANDOM_STATE):
        self.rng = np.random.RandomState(random_state)

    def score(self, surrogate, X_candidates):
        """Sample a single tree prediction as the acquisition value.

        When the sampled tree produces a constant prediction for all candidates
        (e.g. fixed-chemistry search where all candidates share the same
        molecular features and fall into one RF leaf), the acquisition is
        uninformative.  In that case, retry up to min(n_trees, 20) times to
        find an informative tree.  If no informative tree is found, fall back
        to the ensemble mean plus independent Gaussian noise scaled by the
        ensemble standard deviation, which is a well-known approximate-TS
        strategy.
        """
        Xt = surrogate._transform_features(X_candidates)
        reg = surrogate._get_regressor()
        if hasattr(reg, "estimators_"):
            n_trees = len(reg.estimators_)
            max_tries = min(n_trees, 20)
            for _ in range(max_tries):
                tree_idx = self.rng.randint(n_trees)
                preds = reg.estimators_[tree_idx].predict(Xt)
                if np.std(preds) > 1e-8:
                    return preds
            # All sampled trees degenerate → approximate TS fallback
            mu, sigma = surrogate.predict(X_candidates)
            return mu + self.rng.randn(len(mu)) * np.maximum(sigma, 1e-6)
        else:
            return reg.predict(Xt)


# ─────────────────────────────────────────────────────────────
# BatchSelector
# ─────────────────────────────────────────────────────────────
class BatchSelector:
    """Batch selection via Constant Liar, Kriging Believer, or Diverse Greedy."""

    # Process-parameter columns used for diversity distance.
    _PROC_COLS = ["temperature_k", "total_conc", "phi_1", "equivalents",
                  "metal_over_linker_ratio"]

    @staticmethod
    def diverse_greedy(
        surrogate, X_train, y_train, X_candidates,
        candidates_df, acquisition_name, batch_size,
        diversity_lambda=0.3,
        **acq_kwargs,
    ):
        """Greedy quality-diversity batch selection.

        Addresses the core failure of kriging_believer + LFBO: with n~750+,
        adding one hallucinated point barely moves the LFBO classifier, so all
        batch members receive identical acquisition values.

        Instead, score ALL candidates once with the acquisition function, then
        select the batch greedily by maximising:

            combined(x) = (1 - lambda) * af_norm(x)
                        + lambda       * min_dist(x, already_selected)

        where min_dist is computed over normalised process-parameter space plus
        a solvent-identity bonus, so the strategy naturally diversifies across
        both conditions and solvents.

        Distance metric (bounded to [0, 1]):
            proc_d  = ||x_proc_i - x_proc_j||_2 / sqrt(n_proc_dims)   (normalised)
            sol_d   = 1 if solvent_1 differs, else 0
            dist    = 0.5 * proc_d + 0.5 * sol_d

        Parameters
        ----------
        diversity_lambda : float in [0, 1]
            Weight on diversity vs. acquisition quality.
            0 = pure acquisition (same as argmax), 1 = pure diversity.
            Default 0.3 gives quality-first with meaningful spread.

        Returns
        -------
        selected_indices : list[int]  length batch_size
        combined_scores  : np.ndarray  length len(candidates_df), combined score
                           for selected members; nan for non-selected candidates.
        """
        n_cand = X_candidates.shape[0]

        # --- Step 1: score all candidates once ---
        acq_vals = _compute_acquisition(
            acquisition_name, surrogate, X_train, y_train,
            X_candidates, **acq_kwargs
        )

        af_min, af_max = acq_vals.min(), acq_vals.max()
        af_norm = (acq_vals - af_min) / (af_max - af_min + 1e-9)

        # --- Step 2: build normalised process-param matrix ---
        # When candidates_df is None (simulation mode), use the full feature
        # matrix for distance computation.
        proc_cols = ([c for c in BatchSelector._PROC_COLS
                      if c in candidates_df.columns]
                     if candidates_df is not None else [])
        if proc_cols:
            proc = candidates_df[proc_cols].values.astype(float)
            p_min = proc.min(axis=0)
            p_rng = proc.max(axis=0) - p_min
            p_rng[p_rng < 1e-9] = 1.0
            proc_norm = (proc - p_min) / p_rng          # each col in [0,1]
            n_proc = len(proc_cols)
        else:
            # Simulation mode or no process columns: use full feature space
            proc_norm = X_candidates.copy()
            p_min = proc_norm.min(axis=0)
            p_rng = proc_norm.max(axis=0) - p_min
            p_rng[p_rng < 1e-9] = 1.0
            proc_norm = (proc_norm - p_min) / p_rng
            n_proc = proc_norm.shape[1]

        solvents = (candidates_df["solvent_1"].values
                    if candidates_df is not None
                    and "solvent_1" in candidates_df.columns else None)

        def _dist(i, j):
            proc_d = np.linalg.norm(proc_norm[i] - proc_norm[j]) / np.sqrt(n_proc)
            sol_d  = 1.0 if (solvents is not None and solvents[i] != solvents[j]) else 0.0
            return 0.5 * proc_d + 0.5 * sol_d

        def _min_dist_to_selected(i, selected):
            return min(_dist(i, s) for s in selected)

        # --- Step 3: greedy selection ---
        selected = []
        masked   = np.zeros(n_cand, dtype=bool)

        for step in range(batch_size):
            if step == 0:
                scores = af_norm.copy()
            else:
                div_scores = np.array([
                    _min_dist_to_selected(i, selected) if not masked[i] else -np.inf
                    for i in range(n_cand)
                ])
                scores = (1.0 - diversity_lambda) * af_norm + diversity_lambda * div_scores

            scores[masked] = -np.inf
            best = int(np.argmax(scores))
            selected.append(best)
            masked[best] = True

        # Build per-candidate combined scores (nan for non-batch rows)
        combined = np.full(n_cand, np.nan)
        for step, idx in enumerate(selected):
            if step == 0:
                combined[idx] = af_norm[idx]
            else:
                combined[idx] = (
                    (1.0 - diversity_lambda) * af_norm[idx]
                    + diversity_lambda * _min_dist_to_selected(idx, selected[:step])
                )

        return selected, combined

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

        for i in range(batch_size):
            _kwargs = dict(acq_kwargs)
            _kwargs["random_state"] = acq_kwargs.get("random_state", RANDOM_STATE) + i
            acq_vals = _compute_acquisition(
                acquisition_name, surrogate, X_aug, y_aug,
                X_candidates, **_kwargs
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

        for i in range(batch_size):
            _kwargs = dict(acq_kwargs)
            _kwargs["random_state"] = acq_kwargs.get("random_state", RANDOM_STATE) + i
            acq_vals = _compute_acquisition(
                acquisition_name, surrogate, X_aug, y_aug,
                X_candidates, **_kwargs
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
            # Round to nearest integer so y_aug stays integer-valued:
            # mutual_info_classif (used in AdaptiveSelectKBest) rejects
            # continuous-typed arrays.
            mu_hallucinated = np.round(mu_hallucinated)
            X_aug = np.vstack([X_aug, new_x]) if isinstance(X_aug, np.ndarray) else \
                    pd.concat([X_aug, pd.DataFrame(new_x, columns=X_aug.columns)], ignore_index=True)
            y_aug = np.append(y_aug, mu_hallucinated)

            surrogate.fit(X_aug, y_aug)

        return selected_indices


def _compute_acquisition(
    name, surrogate, X_train, y_train, X_candidates, **kwargs
):
    """Dispatch acquisition function by name.

    Supported names: "lfbo", "ei", "thompson", "random", "consensus".
    """
    if name == "ei":
        mu, sigma = surrogate.predict(X_candidates)
        f_best = kwargs.get("f_best", y_train.max())
        # If surrogate is ranking-based, convert f_best to rank space.
        if isinstance(surrogate, RankingRegressionSurrogate):
            f_best = surrogate.raw_to_rank(f_best)
        return EIAcquisition(xi=kwargs.get("xi", BO_EI_XI)).score(mu, sigma, f_best)

    elif name == "lfbo":
        lfbo = LFBOAcquisition(
            gamma=kwargs.get("gamma", BO_LFBO_GAMMA),
            random_state=kwargs.get("random_state", RANDOM_STATE),
            adaptive_gamma=kwargs.get(
                "lfbo_adaptive_gamma", BO_LFBO_ADAPTIVE_GAMMA
            ),
        )
        return lfbo.score(X_train, y_train, X_candidates, surrogate=surrogate)

    elif name == "consensus":
        return _consensus_acquisition(
            surrogate, X_train, y_train, X_candidates, **kwargs
        )

    elif name == "thompson":
        ts = ThompsonSamplingAcquisition(
            random_state=kwargs.get("random_state", RANDOM_STATE)
        )
        return ts.score(surrogate, X_candidates)

    elif name == "random":
        return np.random.RandomState(
            kwargs.get("random_state", RANDOM_STATE)
        ).uniform(size=len(X_candidates))

    else:
        raise ValueError(f"Unknown acquisition: {name}")


def _consensus_acquisition(
    surrogate, X_train, y_train, X_candidates, *,
    top_k_frac=0.10, **kwargs
):
    """Consensus acquisition: intersect top-K picks from EI and LFBO.

    1. Score all candidates with both EI and LFBO.
    2. Percentile-rank each into [0, 1].
    3. Candidates in the top ``top_k_frac`` of *both* methods get a boosted
       combined score (mean of the two ranks + 1.0 bonus).
    4. All other candidates are scored by their LFBO rank alone, so when the
       intersection is empty the result is equivalent to pure LFBO.

    Returns an acquisition-value array (higher is better) of length
    ``len(X_candidates)``.
    """
    n = len(X_candidates)
    top_k = max(1, int(n * top_k_frac))

    # --- score with both methods ---
    ei_vals = _compute_acquisition(
        "ei", surrogate, X_train, y_train, X_candidates, **kwargs
    )
    lfbo_vals = _compute_acquisition(
        "lfbo", surrogate, X_train, y_train, X_candidates, **kwargs
    )

    # --- percentile-rank into [0, 1] ---
    from scipy.stats import rankdata
    ei_rank = rankdata(ei_vals, method="average") / n
    lfbo_rank = rankdata(lfbo_vals, method="average") / n

    # --- identify top-K sets ---
    ei_topk = set(np.argsort(ei_vals)[-top_k:])
    lfbo_topk = set(np.argsort(lfbo_vals)[-top_k:])
    overlap = ei_topk & lfbo_topk

    # --- build combined score ---
    # Base: LFBO rank (so fallback = pure LFBO ordering)
    combined = lfbo_rank.copy()
    # Consensus candidates: mean of both ranks + bonus so they always rank above
    for idx in overlap:
        combined[idx] = (ei_rank[idx] + lfbo_rank[idx]) / 2.0 + 1.0

    return combined


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

            # total_conc = (linker + metal + modulator) / volume
            # metal      = ratio * linker
            # modulator  = equiv * linker
            denom       = 1.0 + ratio + equiv
            linker_conc = total_conc / denom
            metal_conc  = ratio * linker_conc
            umol_linker = linker_conc * self.total_volume_ml
            umol_metal  = metal_conc  * self.total_volume_ml
            umol_mod    = equiv * umol_linker
            mod_conc    = umol_mod / self.total_volume_ml

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
        modulator_col="smiles_modulator",
        score_col="pxrd_score",
        fp_blend=0.3,
        linker_weight=0.60,
        precursor_weight=0.30,
        modulator_weight=0.10,
        top_k=15,
        min_similarity=0.05,
        fp_radius=2,
        fp_nbits=2048,
        success_threshold=5,
        hub_strat_threshold=0.85,
    ):
        """
        Parameters
        ----------
        df_train             : DataFrame — training experiments (rows match X_cv)
        X_cv                 : ndarray (n, d) — full feature matrix from checkpoint
        X_groups             : list[str] len d — feature group labels per column
        linker_col           : str — SMILES column for the linker in df_train
        precursor_col        : str — SMILES column for the precursor in df_train
        modulator_col        : str — SMILES column for the modulator in df_train
        score_col            : str — outcome column
        fp_blend             : float in [0,1] — weight given to Morgan FP similarity
                               vs chemistry feature cosine similarity (default 0.3).
                               Lower = trust the surrogate features more.
        linker_weight        : float — weight for linker in FP similarity (default 0.60)
        precursor_weight     : float — weight for precursor in FP similarity (default 0.30)
        modulator_weight     : float — weight for modulator in FP similarity (default 0.10).
                               Only applied when a target modulator SMILES is provided.
                               Weights are renormalized when modulator is absent.
        top_k                : int — number of neighbors to return
        min_similarity       : float — minimum combined similarity to include
        fp_radius            : int — Morgan FP radius (2 = ECFP4)
        fp_nbits             : int — Morgan FP bit vector length
        success_threshold    : int — minimum pxrd_score to count as a "success"
                               for center computation (default 5). Only successes
                               anchor the trust region center; all hub-matched
                               experiments inform the spread.
        hub_strat_threshold  : float — mean pairwise FP similarity threshold above
                               which hub-element stratification is triggered
                               (default 0.85). When neighbors are nearly identical
                               by fingerprint, hub atom becomes the discriminator.
                               Disabled automatically when FP diversity is high
                               (e.g. a new linker class with no prior history).
        """
        self.df            = df_train.copy().reset_index(drop=True)
        self.linker_col    = linker_col
        self.precursor_col = precursor_col
        self.modulator_col = modulator_col
        self.score_col     = score_col
        self.fp_blend      = float(fp_blend)
        self.linker_weight    = linker_weight
        self.precursor_weight = precursor_weight
        self.modulator_weight = modulator_weight
        self.top_k          = top_k
        self.min_similarity = min_similarity
        self.fp_radius = fp_radius
        self.fp_nbits  = fp_nbits
        self.success_threshold   = success_threshold
        self.hub_strat_threshold = hub_strat_threshold

        # ── Stage 1: pre-compute Morgan fingerprints ──────────────────────────
        self._linker_fps    = [self._to_fp(s) for s in
                               df_train[linker_col].fillna("")]
        self._precursor_fps = [self._to_fp(s) for s in
                               df_train[precursor_col].fillna("")]
        mod_col_data = (df_train[modulator_col].fillna("")
                        if modulator_col in df_train.columns
                        else pd.Series([""] * len(df_train)))
        self._modulator_fps = [self._to_fp(s) for s in mod_col_data]

        # ── Hub element pre-computation ────────────────────────────────────────
        self._hub_elements = [
            self._detect_hub(s) for s in df_train[linker_col].fillna("")
        ]
        hub_counts = {}
        for h in self._hub_elements:
            hub_counts[h] = hub_counts.get(h, 0) + 1
        print(f"[NeighborhoodTemplate] Hub distribution in training data: "
              + ", ".join(f"{k}={v}" for k, v in sorted(hub_counts.items())))

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

    def _fp_similarity(self, tgt_linker_fp, tgt_precursor_fp, tgt_modulator_fp=None):
        """Return per-experiment combined Morgan FP similarity array.

        When tgt_modulator_fp is None the linker/precursor weights are
        renormalized to sum to 1 so the absence of a modulator target
        doesn't artificially deflate all similarity scores.
        """
        use_mod = tgt_modulator_fp is not None
        if use_mod:
            lw = self.linker_weight
            pw = self.precursor_weight
            mw = self.modulator_weight
        else:
            total = self.linker_weight + self.precursor_weight
            lw = self.linker_weight / total
            pw = self.precursor_weight / total
            mw = 0.0

        sims = np.zeros(len(self.df))
        for i in range(len(self.df)):
            l = self._tanimoto(tgt_linker_fp,    self._linker_fps[i])
            p = self._tanimoto(tgt_precursor_fp, self._precursor_fps[i])
            m = self._tanimoto(tgt_modulator_fp, self._modulator_fps[i]) if use_mod else 0.0
            sims[i] = lw * l + pw * p + mw * m
        return sims

    def _feat_similarity(self, ref_idx):
        """Cosine similarity between experiment ref_idx and all others."""
        ref = self._X_chem_normed[ref_idx]          # shape (n_chem,)
        return self._X_chem_normed @ ref             # shape (n,)

    @staticmethod
    def _detect_hub(smiles):
        """Detect the hub atom/group from a linker SMILES.

        Checks for Group 14 heteroatoms first (unambiguous), then the
        adamantane cage (tricyclic carbon scaffold), then defaults to 'C'.

        Returns one of: 'Sn', 'Ge', 'Si', 'adamantane', 'C', 'unknown'
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return "unknown"
            atomic_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
            if 50 in atomic_nums:   # Sn
                return "Sn"
            if 32 in atomic_nums:   # Ge
                return "Ge"
            if 14 in atomic_nums:   # Si
                return "Si"
            # Adamantane cage: tricyclic bridged carbon scaffold
            adm = Chem.MolFromSmarts("C1C2CC3CC1CC(C2)C3")
            if adm and mol.HasSubstructMatch(adm):
                return "adamantane"
            return "C"
        except Exception:
            return "unknown"

    def _mean_pairwise_fp_sim(self, indices):
        """Mean pairwise Tanimoto FP similarity among a set of linker indices."""
        from rdkit.DataStructs import TanimotoSimilarity
        fps = [self._linker_fps[i] for i in indices
               if self._linker_fps[i] is not None]
        if len(fps) < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                total += TanimotoSimilarity(fps[i], fps[j])
                count += 1
        return total / count if count > 0 else 0.0

    # ── main ──────────────────────────────────────────────────────────────────

    def select(self, target_linker_smiles, target_precursor_smiles,
               search_bounds, target_modulator_smiles=None):
        """Find similar experiments and return weighted process center + spread.

        Parameters
        ----------
        target_linker_smiles    : str — SMILES of the target linker
        target_precursor_smiles : str — SMILES of the target precursor
        search_bounds           : dict — param → (lo, hi) from SearchSpace.bounds
        target_modulator_smiles : str or None — SMILES of the modulator (optional).
                                  When provided, modulator Tanimoto similarity
                                  is included in the neighbor search with weight
                                  self.modulator_weight.

        Returns
        -------
        center      : dict — param → weighted-mean process condition value
        spread      : dict — param → weighted std (seeds trust region radius)
        neighbors   : DataFrame — top-k neighbors with similarity scores
        ref_idx     : int — dataset index of nearest-neighbor (chemistry template)
        """
        tgt_linker_fp    = self._to_fp(target_linker_smiles)
        tgt_precursor_fp = self._to_fp(target_precursor_smiles)
        tgt_modulator_fp = (self._to_fp(target_modulator_smiles)
                            if target_modulator_smiles else None)

        if tgt_modulator_fp is not None:
            print(f"[NeighborhoodTemplate] Modulator included in similarity "
                  f"(weight={self.modulator_weight:.2f})")

        # Stage 1: Morgan FP similarity for all experiments
        fp_sims = self._fp_similarity(tgt_linker_fp, tgt_precursor_fp,
                                      tgt_modulator_fp)

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
        sim_df["hub_elem"] = [self._hub_elements[i] for i in sim_df.index]

        # ── Hub element stratification (auto-triggered) ────────────────────────
        # When all top-k neighbors look nearly identical by Morgan FP (typical
        # for phosphine linker datasets where only the hub atom changes), FP
        # similarity can no longer discriminate hub types.  We detect this by
        # measuring mean pairwise FP similarity among the neighbors: if it
        # exceeds hub_strat_threshold, stratify by hub element so that Sn
        # experiments inform the Sn center, Si inform the Si center, etc.
        target_hub    = self._detect_hub(target_linker_smiles)
        mean_pair_sim = self._mean_pairwise_fp_sim(list(sim_df.index))
        stratify      = mean_pair_sim >= self.hub_strat_threshold

        if stratify:
            hub_matched = sim_df[sim_df["hub_elem"] == target_hub]
            if len(hub_matched) >= 2:
                print(f"[NeighborhoodTemplate] Hub stratification ACTIVE "
                      f"(mean_pair_fp={mean_pair_sim:.3f} ≥ {self.hub_strat_threshold}) | "
                      f"target_hub={target_hub} | "
                      f"{len(hub_matched)}/{len(sim_df)} hub-matched neighbors")
                spread_pool = hub_matched          # spread = uncertainty within hub class
            else:
                print(f"[NeighborhoodTemplate] Hub stratification triggered "
                      f"(mean_pair_fp={mean_pair_sim:.3f}) but only "
                      f"{len(hub_matched)} {target_hub}-hub experiments found — "
                      "falling back to all neighbors")
                hub_matched = sim_df
                spread_pool = sim_df
        else:
            print(f"[NeighborhoodTemplate] Hub stratification OFF "
                  f"(mean_pair_fp={mean_pair_sim:.3f} < {self.hub_strat_threshold})")
            hub_matched = sim_df
            spread_pool = sim_df

        # ── Success-only center ────────────────────────────────────────────────
        # Anchor the trust region center only on successful experiments
        # (score >= success_threshold) to avoid pulling the starting point
        # toward failure conditions.  All hub-matched experiments inform the
        # spread (wider spread = more uncertainty = larger trust region).
        success_pool = hub_matched[
            hub_matched["score"] >= self.success_threshold
        ]
        if len(success_pool) >= 2:
            center_pool = success_pool
            print(f"[NeighborhoodTemplate] Center anchored on "
                  f"{len(success_pool)} successes (score≥{self.success_threshold})")
        else:
            center_pool = hub_matched
            print(f"[NeighborhoodTemplate] Fewer than 2 successes found — "
                  "using all hub-matched neighbors for center")

        # ── Weighted center (success pool) and spread (full hub pool) ──────────
        center_pool = center_pool.copy()
        center_pool["weight"] = (
            center_pool["combined_sim"] * (center_pool["score"] + 1.0)
        )
        if center_pool["weight"].sum() <= 0:
            center_pool["weight"] = 1.0

        center = {}
        spread = {}
        for param, (lo, hi) in search_bounds.items():
            # center: weighted mean over successes
            vals_c = center_pool[param].values.astype(float)
            w_c    = center_pool["weight"].values
            wmean  = float(np.average(vals_c, weights=w_c))
            center[param] = float(np.clip(wmean, lo, hi))

            # spread: std over the full hub-matched pool (includes failures)
            vals_s = spread_pool[param].values.astype(float)
            wstd   = float(np.std(vals_s)) if len(vals_s) > 1 else 0.0
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
# Acquisition → batch strategy compatibility
# ─────────────────────────────────────────────────────────────
#
# Constant Liar (CL) and Kriging Believer (KB) work by hallucinating an
# observation at each selected point and refitting the regression surrogate.
# The updated (mu, sigma) naturally deflates acquisition scores near
# already-selected points.
#
#   CL  hallucinates f_best (pessimistic) → more diverse/exploratory batches.
#   KB  hallucinates mu(x)  (optimistic)  → more exploitative batches
#       concentrated near the predicted optimum.
#
# This mechanism only works for acquisition functions that consume the
# regression surrogate's (mu, sigma).  Classifier-based (LFBO) and
# surrogate-independent (random) acquisitions ignore the refitted surrogate,
# so CL/KB produce near-identical batch members.  Thompson Sampling has a
# natural parallel mechanism (independent tree draws).
#
# Use CL when you want the batch to spread across the search space
# (early exploration, high uncertainty).
# Use KB when you trust the surrogate and want to exploit the predicted
# optimum region more aggressively (later in the campaign).
#
# References:
#   Ginsbourger et al. (2010) — CL/KB for parallel BO
#   Oliveira, Tiao & Ramos (NeurIPS 2022) — CL/KB failure with BORE
#   Kandasamy et al. (AISTATS 2018) — parallel Thompson Sampling

# Valid batch strategies per acquisition function.
VALID_BATCH_STRATEGIES = {
    "ei":        ["constant_liar", "kriging_believer"],
    "lfbo":      ["diverse_greedy"],
    "consensus": ["diverse_greedy"],
    "thompson":  ["diverse_greedy"],
    "random":    ["diverse_greedy"],
}

# Default batch strategy when none is explicitly requested.
DEFAULT_BATCH_STRATEGY = {
    "ei":        "kriging_believer",
    "lfbo":      "diverse_greedy",
    "consensus": "diverse_greedy",
    "thompson":  "diverse_greedy",
    "random":    "diverse_greedy",
}


def resolve_batch_strategy(acquisition_name, requested_strategy=None):
    """Return a valid batch strategy for the given acquisition function.

    If *requested_strategy* is valid for *acquisition_name*, it is returned
    unchanged.  Otherwise a warning is printed and the default for that
    acquisition is used.
    """
    valid = VALID_BATCH_STRATEGIES.get(acquisition_name)
    default = DEFAULT_BATCH_STRATEGY.get(acquisition_name, "diverse_greedy")

    if valid is None:
        return requested_strategy or default

    if requested_strategy is None or requested_strategy not in valid:
        if requested_strategy is not None:
            print(f"[BO] WARNING: batch_strategy='{requested_strategy}' is "
                  f"incompatible with acquisition='{acquisition_name}'. "
                  f"Valid options: {valid}. "
                  f"Auto-selecting '{default}'.")
        return default

    return requested_strategy


class BOLoop:
    """Main BO loop for simulation, recommendation, and batch modes."""

    def __init__(
        self,
        surrogate,
        acquisition_name="lfbo",
        batch_strategy=None,
        batch_size=BO_BATCH_SIZE,
        n_iterations=BO_N_ITERATIONS,
        epsilon_greedy=BO_EPSILON_GREEDY,
        random_state=RANDOM_STATE,
        lfbo_adaptive_gamma=BO_LFBO_ADAPTIVE_GAMMA,
    ):
        self.surrogate = surrogate
        self.acquisition_name = acquisition_name
        self.batch_strategy = resolve_batch_strategy(
            acquisition_name, batch_strategy
        )
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.epsilon_greedy = epsilon_greedy
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state
        self.lfbo_adaptive_gamma = lfbo_adaptive_gamma

    def run_simulation(self, X, y_raw, init_fraction=BO_INIT_FRACTION,
                       groups=None,
                       cluster_div_lambda=BO_CLUSTER_DIV_LAMBDA):
        """Sequential BO with oracle pool.

        Parameters
        ----------
        X : array-like, shape (n, d) — full feature matrix
        y_raw : array-like, shape (n,) — raw 0-9 pxrd_score
        init_fraction : float — fraction for initial training set
        groups : array-like int (n,), optional — KMeans chemistry cluster labels.
            When provided, two things happen:
            1. Init split is stratified by (score, cluster) jointly, guaranteeing
               every chemistry cluster appears in the initial training set.  This
               prevents the surrogate from starting blind to entire chemical families
               (a common source of local-minima trapping on poor random inits).
            2. A cluster diversity penalty is applied to acquisition scores at each
               iteration, discouraging consecutive selections from the same cluster.
               Penalty: acq *= 1 / (1 + lambda * excess_selections / expected)
               where excess = max(0, actual - expected_per_cluster).
        cluster_div_lambda : float — strength of diversity penalty (default 2.0).
            0 disables it; higher values push more aggressively toward unexplored clusters.

        Returns
        -------
        history : dict with keys 'iterations', 'best_so_far', 'selected_indices',
                  'y_selected', 'init_indices', 'pool_indices'
        """
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        n = len(y_raw)

        y_binned = np.clip(y_raw.astype(int), 0, 9)

        # ── Init split ────────────────────────────────────────────────────────
        # When chemistry groups are available, take init_fraction of EACH cluster
        # (score-stratified within the cluster where possible).  This guarantees
        # every chemistry family is represented in the initial training set while
        # holding back 70 % of each cluster as the oracle pool.
        #
        # Why not GroupShuffleSplit (entire clusters in/out)?  Too extreme — the
        # surrogate has zero signal from pool clusters, making generalisation
        # unrealistically hard.  The per-cluster split mirrors the real lab
        # situation: you have done some experiments with each linker type and
        # want to know which remaining experiments to run next.
        #
        # Why not plain StratifiedShuffleSplit (score only)?  Both init and pool
        # contain the same chemistry families, so the surrogate can trivially
        # interpolate within a known cluster → inflated AF/EF.
        if groups is not None:
            groups = np.asarray(groups, dtype=int)
            n_clusters = int(groups.max()) + 1

            init_list, pool_list = [], []
            for cid in range(n_clusters):
                c_idx = np.where(groups == cid)[0]
                if len(c_idx) < 2:
                    # Cluster too small to split — put entirely in init
                    init_list.extend(c_idx.tolist())
                    continue
                y_c = y_binned[c_idx]
                try:
                    sss_c = StratifiedShuffleSplit(
                        n_splits=1, train_size=init_fraction,
                        random_state=self.random_state
                    )
                    i_init, i_pool = next(sss_c.split(c_idx, y_c))
                except ValueError:
                    # Score stratification failed — random split within cluster
                    rng_c = np.random.RandomState(self.random_state + cid)
                    perm = rng_c.permutation(len(c_idx))
                    n_init = max(1, int(len(c_idx) * init_fraction))
                    i_init, i_pool = perm[:n_init], perm[n_init:]
                init_list.extend(c_idx[i_init].tolist())
                pool_list.extend(c_idx[i_pool].tolist())

            init_idx = np.array(init_list, dtype=int)
            pool_idx = np.array(pool_list, dtype=int)

            # Diagnostics per cluster
            print(f"[BO simulation] Per-cluster stratified split "
                  f"({init_fraction:.0%} init / {1 - init_fraction:.0%} pool "
                  f"from each of {n_clusters} clusters):")
            for cid in range(n_clusters):
                n_init_c = int((groups[init_idx] == cid).sum())
                n_pool_c = int((groups[pool_idx] == cid).sum())
                print(f"    cluster {cid}: init={n_init_c}  pool={n_pool_c}")
            print(f"  Total: init={len(init_idx)}  pool={len(pool_idx)}")
        else:
            groups     = None
            n_clusters = None
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=init_fraction,
                random_state=self.random_state
            )
            init_idx, pool_idx = next(sss.split(X, y_binned))
            print("[BO simulation] No chemistry groups — using score-stratified split.")

        return self._run_loop(X, y_raw, init_idx, pool_idx,
                              groups=groups, n_clusters=n_clusters,
                              cluster_div_lambda=cluster_div_lambda)

    # ── Core BO loop (shared by run_simulation and run_simulation_loco) ────

    def _run_loop(self, X, y_raw, init_idx, pool_idx, *,
                  groups=None, n_clusters=None, cluster_div_lambda=0.0,
                  quiet=False):
        """Execute the sequential BO loop given pre-computed init/pool splits.

        Parameters
        ----------
        X, y_raw          : full feature matrix and raw scores
        init_idx, pool_idx: integer arrays of global indices
        groups             : optional cluster labels (int array, len n)
        n_clusters         : number of distinct clusters
        cluster_div_lambda : diversity penalty strength (0 = disabled)
        quiet              : suppress per-iteration printing
        """
        n = len(y_raw)

        # Diagnostic
        high_count = (y_raw[init_idx] >= 7).sum()
        if high_count < 3:
            warnings.warn(
                f"Initial set has only {high_count} examples with score >= 7. "
                "Surrogate may lack signal for high-quality outcomes."
            )

        X_train = X[init_idx].copy()
        y_train = y_raw[init_idx].copy()
        pool_mask = np.zeros(n, dtype=bool)
        pool_mask[pool_idx] = True

        history = {
            "iterations": [],
            "best_so_far": [],
            "selected_indices": [],
            "y_selected": [],
            "init_indices": init_idx.tolist() if hasattr(init_idx, 'tolist') else list(init_idx),
            "pool_indices": pool_idx.tolist() if hasattr(pool_idx, 'tolist') else list(pool_idx),
        }

        f_best = float(y_train.max()) if len(y_train) > 0 else 0.0
        n_iters = min(self.n_iterations, int(pool_mask.sum()))
        if not quiet:
            print(f"[BO simulation] init={len(init_idx)}, pool={pool_mask.sum()}, "
                  f"f_best_init={f_best:.1f}")

        from collections import Counter
        cluster_counts = Counter()

        for it in range(n_iters):
            remaining = np.where(pool_mask)[0]
            if len(remaining) == 0:
                if not quiet:
                    print(f"[BO] Pool exhausted at iteration {it}")
                break

            self.surrogate.fit(X_train, y_train)
            X_pool = X[remaining]

            # Epsilon-greedy exploration
            if self.rng.uniform() < self.epsilon_greedy:
                sel_local = self.rng.randint(len(remaining))
            else:
                acq_kwargs = {
                    "f_best": f_best,
                    "gamma": BO_LFBO_GAMMA,
                    "random_state": self.random_state + it,

                    "lfbo_adaptive_gamma": self.lfbo_adaptive_gamma,
                }
                acq_vals = _compute_acquisition(
                    self.acquisition_name, self.surrogate,
                    X_train, y_train, X_pool, **acq_kwargs
                )

                # Cluster diversity penalty
                if groups is not None and cluster_div_lambda > 0 and n_clusters:
                    total_selected = max(len(history["selected_indices"]), 1)
                    expected = total_selected / n_clusters
                    penalties = np.ones(len(remaining))
                    for j, idx in enumerate(remaining):
                        k = int(groups[idx])
                        excess = max(0.0, cluster_counts[k] - expected)
                        penalties[j] = 1.0 / (
                            1.0 + cluster_div_lambda * excess / (expected + 1e-9)
                        )
                    acq_vals = acq_vals * penalties

                sel_local = np.argmax(acq_vals)

            sel_global = remaining[sel_local]
            oracle_y = y_raw[sel_global]

            X_train = np.vstack([X_train, X[sel_global:sel_global+1]])
            y_train = np.append(y_train, oracle_y)
            pool_mask[sel_global] = False
            f_best = max(f_best, oracle_y)

            if groups is not None:
                cluster_counts[int(groups[sel_global])] += 1

            history["iterations"].append(it)
            history["best_so_far"].append(f_best)
            history["selected_indices"].append(int(sel_global))
            history["y_selected"].append(float(oracle_y))

            if not quiet and (it + 1) % 10 == 0:
                print(f"  iter {it+1:3d}/{n_iters} | "
                      f"selected y={oracle_y:.0f} | f_best={f_best:.0f} | "
                      f"pool={pool_mask.sum()}")

        return history

    def run_simulation_loco(self, X, y_raw, groups, held_out_cluster,
                            max_pool_frac=0.30):
        """Leave-one-cluster-out BO simulation.

        Init = all experiments from clusters != held_out_cluster.
        Pool = all experiments from the held-out cluster.
        Cluster diversity penalty is disabled (irrelevant when pool is
        a single cluster).

        Parameters
        ----------
        max_pool_frac : float — cap iterations at this fraction of the pool
            size so that small clusters are evaluated under the same budget
            pressure as large ones (default 0.30 = match the 30/70 split).
            Without this, a cluster of 25 experiments tested over 50 iterations
            exhausts the entire pool, inflating Top-5% and AF.

        Returns the same history dict as run_simulation.
        """
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        groups = np.asarray(groups, dtype=int)

        init_idx = np.where(groups != held_out_cluster)[0]
        pool_idx = np.where(groups == held_out_cluster)[0]

        # Cap iterations to avoid exhausting small pools
        budget = max(1, int(len(pool_idx) * max_pool_frac))
        orig_iters = self.n_iterations
        self.n_iterations = min(self.n_iterations, budget)

        print(f"[LOCO] Held-out cluster {held_out_cluster}: "
              f"init={len(init_idx)} (other clusters), "
              f"pool={len(pool_idx)} (cluster {held_out_cluster}), "
              f"budget={self.n_iterations} iters "
              f"({max_pool_frac:.0%} of pool)")

        history = self._run_loop(X, y_raw, init_idx, pool_idx,
                                 groups=None, n_clusters=None,
                                 cluster_div_lambda=0.0)
        self.n_iterations = orig_iters
        return history

    def run_batch(self, X, y_raw, init_fraction=BO_INIT_FRACTION, groups=None):
        """Batch BO simulation using constant_liar or kriging_believer.

        Returns history dict similar to run_simulation but with batch selections.
        """
        X = np.asarray(X, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        n = len(y_raw)

        y_binned = np.clip(y_raw.astype(int), 0, 9)
        if groups is not None:
            groups = np.asarray(groups, dtype=int)
            n_clusters = int(groups.max()) + 1
            init_list, pool_list = [], []
            for cid in range(n_clusters):
                c_idx = np.where(groups == cid)[0]
                if len(c_idx) < 2:
                    init_list.extend(c_idx.tolist())
                    continue
                y_c = y_binned[c_idx]
                try:
                    sss_c = StratifiedShuffleSplit(
                        n_splits=1, train_size=init_fraction,
                        random_state=self.random_state
                    )
                    i_init, i_pool = next(sss_c.split(c_idx, y_c))
                except ValueError:
                    rng_c = np.random.RandomState(self.random_state + cid)
                    perm = rng_c.permutation(len(c_idx))
                    n_init = max(1, int(len(c_idx) * init_fraction))
                    i_init, i_pool = perm[:n_init], perm[n_init:]
                init_list.extend(c_idx[i_init].tolist())
                pool_list.extend(c_idx[i_pool].tolist())
            init_idx = np.array(init_list, dtype=int)
            pool_idx = np.array(pool_list, dtype=int)
            print(f"[BO batch] Per-cluster stratified split: "
                  f"init={len(init_idx)} pool={len(pool_idx)} "
                  f"({n_clusters} clusters)")
        else:
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
                "gamma": BO_LFBO_GAMMA,
                "random_state": self.random_state + batch_it,
                "lfbo_adaptive_gamma": self.lfbo_adaptive_gamma,
            }

            if self.batch_strategy == "diverse_greedy":
                local_indices, _ = BatchSelector.diverse_greedy(
                    self.surrogate, X_train, y_train, X_pool,
                    None, self.acquisition_name, self.batch_size,
                    **acq_kwargs,
                )
            elif self.batch_strategy == "constant_liar":
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
            "gamma": BO_LFBO_GAMMA,
            "random_state": self.random_state,
        }
        acq_vals = _compute_acquisition(
            self.acquisition_name, self.surrogate,
            X_train, y_train, candidate_features, **acq_kwargs
        )

        results = candidates_df.copy()
        results["pxrd_predicted"] = mu
        results["uncertainty"] = sigma
        results["acquisition_value"] = acq_vals

        results = results.sort_values("acquisition_value", ascending=False)
        return results
