"""
models.py
Model classes, metric functions, scoring dict, and pipeline factory functions.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (RANDOM_STATE, MI_K, CL_EMB_DIM, CL_MARGIN, CL_NEGATIVE_CLASS,
                    SMOTE_STRATEGY, XGB_FIXED, XGB_TUNED_KEYS)

warnings.filterwarnings(
    "ignore",
    message="Clustering metrics expects discrete values",
    category=UserWarning,
)

ORIGINAL_DISCRETE_MASK = None


# ─────────────────────────────────────────────────────────────
# 1.  Ordinal Classifier
# ─────────────────────────────────────────────────────────────
class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_     = np.sort(np.unique(y))
        self.classifiers_ = {}
        base = self.base_estimator if self.base_estimator is not None \
               else XGBClassifier()
        for k in self.classes_[:-1]:
            y_binary = (y > k).astype(int)
            sample_weights = compute_sample_weight(class_weight="balanced",
                                                   y=y_binary)
            clf = clone(base)
            clf.fit(X, y_binary, sample_weight=sample_weights)
            self.classifiers_[k] = clf
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["classifiers_", "classes_"])
        X = check_array(X)
        probas_gt = {-1: np.ones(X.shape[0])}
        for k in self.classifiers_:
            probas_gt[k] = self.classifiers_[k].predict_proba(X)[:, 1]
        probas_gt[self.classes_[-1]] = np.zeros(X.shape[0])
        probs = []
        for i, k in enumerate(self.classes_):
            prev_prob = probas_gt[self.classes_[i - 1] if i > 0 else -1]
            p = np.maximum(0, prev_prob - probas_gt[k])
            probs.append(p)
        probs_matrix = np.column_stack(probs)
        return probs_matrix / probs_matrix.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self, ["classifiers_", "classes_"])
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba_per_threshold(self, X):
        """Return dict {k: P(y > k | x)} for each threshold k."""
        check_is_fitted(self, ["classifiers_", "classes_"])
        X = check_array(X)
        return {k: clf.predict_proba(X)[:, 1]
                for k, clf in self.classifiers_.items()}

    def predict_proba_with_uncertainty(self, X, n_samples=200):
        """
        Monte Carlo uncertainty from per-threshold classifiers.

        For RF base estimators, samples individual trees to get variance
        on class probabilities.  Returns (mean_proba, std_proba) each
        shaped (n_samples_X, n_classes).
        """
        check_is_fitted(self, ["classifiers_", "classes_"])
        X = check_array(X)
        n = X.shape[0]
        K = len(self.classes_)

        # Collect per-tree predictions for each threshold classifier
        all_proba_samples = []
        for _ in range(n_samples):
            probas_gt = {-1: np.ones(n)}
            for k, clf in self.classifiers_.items():
                if hasattr(clf, 'estimators_'):
                    # RF: sample a random tree
                    tree_idx = np.random.randint(len(clf.estimators_))
                    tree = clf.estimators_[tree_idx]
                    p = tree.predict_proba(X)
                    # Handle case where tree didn't see both classes
                    if p.shape[1] == 1:
                        probas_gt[k] = np.zeros(n)
                    else:
                        probas_gt[k] = p[:, 1]
                else:
                    probas_gt[k] = clf.predict_proba(X)[:, 1]
            probas_gt[self.classes_[-1]] = np.zeros(n)

            probs = []
            for i_cls, c in enumerate(self.classes_):
                prev = probas_gt[self.classes_[i_cls - 1] if i_cls > 0 else -1]
                probs.append(np.maximum(0, prev - probas_gt[c]))
            mat = np.column_stack(probs)
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            all_proba_samples.append(mat / row_sums)

        stacked = np.stack(all_proba_samples, axis=0)  # (n_samples, n_X, K)
        return stacked.mean(axis=0), stacked.std(axis=0)


# ─────────────────────────────────────────────────────────────
# 1b.  Custom Stacking Classifier
# ─────────────────────────────────────────────────────────────
class OrdinalStackingClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, base_estimators, meta_learner, inner_cv=5):
        self.base_estimators = base_estimators
        self.meta_learner    = meta_learner
        self.inner_cv        = inner_cv

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.sort(np.unique(y))

        oof_preds = []
        for name, est in self.base_estimators:
            oof = cross_val_predict(
                est, X, y,
                cv=self.inner_cv,
                method="predict_proba",
                n_jobs=1,
            )
            oof_preds.append(oof)

        X_meta = np.hstack(oof_preds)
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(X_meta, y)

        self.fitted_bases_ = []
        for name, est in self.base_estimators:
            fitted = clone(est)
            fitted.fit(X, y)
            self.fitted_bases_.append((name, fitted))

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["meta_learner_", "fitted_bases_", "classes_"])
        X = check_array(X)
        test_preds = [est.predict_proba(X) for _, est in self.fitted_bases_]
        return self.meta_learner_.predict_proba(np.hstack(test_preds))

    def predict(self, X):
        check_is_fitted(self, ["classes_"])
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ─────────────────────────────────────────────────────────────
# 2.  Scoring Metrics  (3-CLASS)
# ─────────────────────────────────────────────────────────────
def qwk_0_9(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(np.unique(y_true)) < 2:
        return 0.0
    n = 3
    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0
    act  = O.sum(axis=1)
    pred = O.sum(axis=0)
    E    = np.outer(act, pred) / max(O.sum(), 1.0)
    W    = np.array([[(i - j) ** 2 / (n - 1) ** 2
                      for j in range(n)] for i in range(n)])
    num  = (W * O).sum()
    den  = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def mae_0_9(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)))


def within1(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) <= 1.0)


def exact_acc(y_true, y_pred):
    return np.mean(np.asarray(y_true, int) == np.asarray(y_pred, int))


scoring_ordinal = {
    "qwk":       make_scorer(qwk_0_9,   greater_is_better=True),
    "mae":       make_scorer(mae_0_9,   greater_is_better=False),
    "within1":   make_scorer(within1,   greater_is_better=True),
    "exact_acc": make_scorer(exact_acc, greater_is_better=True),
}


# ─────────────────────────────────────────────────────────────
# 3.  Triplet Contrastive Learning
# ─────────────────────────────────────────────────────────────
class _TripletEncoder(nn.Module):
    """MLP: input_dim → hidden_dim (BatchNorm + ReLU + Dropout(0.2)) → embedding_dim."""

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class TripletTrainer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible triplet contrastive feature augmenter.

    fit(X, y)    — trains _TripletEncoder via triplet margin loss.
                   Anchors are crystalline samples (label=2).
                   Positives are other crystalline samples (label=2).
                   Negatives are sampled from negative_class
                   (1=partial/hard, 0=amorphous/easy).
    transform(X) — returns [X | ℓ2-normalized embedding] when concat_original=True,
                   or just the embedding when concat_original=False.

    Parameters
    ----------
    embedding_dim    : int    output embedding size (default 128)
    hidden_dim       : int    MLP hidden layer width (default 256)
    margin           : float  TripletMarginLoss margin (default 1.0)
    negative_class   : int    class used as negatives (1=hard, 0=easy; default 1)
    epochs           : int    training epochs (default 15)
    batch_size       : int    DataLoader batch size (default 128)
    lr               : float  Adam learning rate (default 1e-3)
    weight_decay     : float  Adam weight decay (default 1e-4)
    balanced_batches : bool   use WeightedRandomSampler for class balance
    random_state     : int    seeds numpy, torch, and cuda (default 42)
    device           : str|None  "cpu", "cuda", or None (auto-detect)
    concat_original  : bool   True → [X | emb], False → emb only
    verbose          : bool   print per-epoch loss
    """

    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        margin=1.0,
        negative_class=1,
        epochs=15,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        balanced_batches=True,
        random_state=42,
        device=None,
        concat_original=True,
        verbose=False,
    ):
        self.embedding_dim    = embedding_dim
        self.hidden_dim       = hidden_dim
        self.margin           = margin
        self.negative_class   = negative_class
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.weight_decay     = weight_decay
        self.balanced_batches = balanced_batches
        self.random_state     = random_state
        self.device           = device
        self.concat_original  = concat_original
        self.verbose          = verbose

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _seed(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _triplet_loss(self, z, y):
        """
        Triplet margin loss for MOF crystallinity task.

        Anchor:   label == 2 (crystalline)
        Positive: another label == 2 sample (randomly sampled from the batch)
        Negative: label == self.negative_class
                  (1 = partial/hard negatives, 0 = amorphous/easy negatives)

        Anchors with no valid positive or negative in the batch are skipped.
        Returns None if no valid triplets can be formed.
        """
        z = F.normalize(z, dim=1)
        criterion = nn.TripletMarginLoss(margin=self.margin, reduction="mean")

        anchor_idx   = (y == 2).nonzero(as_tuple=True)[0]
        negative_idx = (y == self.negative_class).nonzero(as_tuple=True)[0]

        if len(anchor_idx) < 2 or len(negative_idx) == 0:
            return None

        anchors, positives, negatives = [], [], []
        for a_i in anchor_idx:
            pos_pool = anchor_idx[anchor_idx != a_i]
            if len(pos_pool) == 0:
                continue
            p_i = pos_pool[torch.randint(len(pos_pool), (1,), device=z.device)].item()
            n_i = negative_idx[torch.randint(len(negative_idx), (1,), device=z.device)].item()
            anchors.append(z[a_i])
            positives.append(z[p_i])
            negatives.append(z[n_i])

        if len(anchors) == 0:
            return None

        return criterion(
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )

    def _make_loader(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)

        gen = torch.Generator()
        gen.manual_seed(self.random_state)

        if not self.balanced_batches:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                generator=gen,
            )

        classes, counts = np.unique(y, return_counts=True)
        class_w  = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_w = np.array([class_w[yi] for yi in y], dtype=np.float32)
        sampler  = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float32),
            num_samples=len(sample_w),
            replacement=True,
            generator=gen,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False,
            generator=gen,
        )

    def fit(self, X, y):
        self._seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]

        # Scale features for CL training
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

        loader = self._make_loader(X_scaled, y)

        # Model, loss, optimizer, scheduler
        device = self._get_device()
        self.encoder_ = _TripletEncoder(
            input_dim=X_scaled.shape[1],
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        ).to(device)

        opt   = optim.Adam(
            self.encoder_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        self.loss_history_ = []

        for epoch in range(self.epochs):
            self.encoder_.train()
            epoch_losses = []

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                z    = self.encoder_(xb)
                loss = self._triplet_loss(z, yb)

                if loss is None or not torch.isfinite(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))

            sched.step()   # once per epoch, after all opt.step() calls

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            self.loss_history_.append(epoch_loss)

            if self.verbose:
                print(f"[TripletTrainer] epoch {epoch+1:02d}/{self.epochs}  "
                      f"loss={epoch_loss:.4f}")

        self.encoder_.cpu()
        self.encoder_.eval()
        return self

    def _embed(self, X):
        X_scaled = self.scaler_.transform(
            np.asarray(X, dtype=np.float32)
        ).astype(np.float32)

        self.encoder_.eval()
        Xt  = torch.tensor(X_scaled, dtype=torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 256):
                z = self.encoder_(Xt[i:i + 256])
                z = F.normalize(z, dim=1)
                out.append(z.cpu().numpy())
        return np.vstack(out)

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "n_features_in_"])
        X   = np.asarray(X, dtype=np.float32)
        emb = self._embed(X)
        if self.concat_original:
            return np.hstack([X, emb])
        return emb


# ─────────────────────────────────────────────────────────────
# 4.  AdaptiveSelectKBest
# ─────────────────────────────────────────────────────────────
class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    """
    SelectKBest that:
      - caps k at the number of available columns
      - marks CL embedding columns as continuous (discrete_features=False)
        so MI scores them correctly
    """

    def __init__(
        self,
        k=5500,
        with_cl=False,
        embedding_dim=128,
        base_discrete_mask=None,
        random_state=42,
    ):
        self.k = k
        self.with_cl = with_cl
        self.embedding_dim = embedding_dim
        self.base_discrete_mask = base_discrete_mask
        self.random_state = random_state

    def _disc_mask(self, n_features):
        if not self.with_cl:
            if self.base_discrete_mask is None:
                return True
            mask = np.asarray(self.base_discrete_mask, dtype=bool)
            if len(mask) != n_features:
                raise ValueError(
                    f"base_discrete_mask has length {len(mask)} "
                    f"but X has {n_features} features."
                )
            return mask

        n_orig = n_features - self.embedding_dim
        if n_orig <= 0:
            raise ValueError(
                f"Expected at least {self.embedding_dim + 1} features, "
                f"got {n_features}."
            )

        if self.base_discrete_mask is None:
            base_mask = np.ones(n_orig, dtype=bool)
        else:
            base_mask = np.asarray(self.base_discrete_mask, dtype=bool)
            if len(base_mask) != n_orig:
                raise ValueError(
                    f"base_discrete_mask has length {len(base_mask)} "
                    f"but original block has {n_orig}."
                )

        # CL embedding columns are continuous
        return np.r_[base_mask, np.zeros(self.embedding_dim, dtype=bool)]

    def fit(self, X, y):
        X = np.asarray(X)
        k_use = min(self.k, X.shape[1])
        disc  = self._disc_mask(X.shape[1])

        self.selector_ = SelectKBest(
            score_func=lambda Xin, yin: mutual_info_classif(
                Xin,
                yin,
                discrete_features=disc,
                random_state=self.random_state,
            ),
            k=k_use,
        )
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        check_is_fitted(self, "selector_")
        return self.selector_.get_support(indices=indices)


# ─────────────────────────────────────────────────────────────
# 5.  Pipeline step builders
# ─────────────────────────────────────────────────────────────
def _base_steps(with_cl: bool) -> list:
    """
    Common feature prefix:
        impute -> vt -> [Triplet CL] -> mi -> pca -> smote

    When with_cl=True the TripletTrainer appends a 128-d embedding to the
    VT-filtered features before MI selection.
    """
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("vt",     VarianceThreshold(threshold=0.0)),
    ]

    if with_cl:
        steps.append((
            "cl",
            TripletTrainer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                margin=CL_MARGIN,
                negative_class=CL_NEGATIVE_CLASS,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                balanced_batches=True,
                random_state=42,
                concat_original=True,   # [original features | embedding]
                verbose=False,
            )
        ))

    steps += [
        (
            "mi",
            AdaptiveSelectKBest(
                k=MI_K,
                with_cl=with_cl,
                embedding_dim=CL_EMB_DIM,
                base_discrete_mask=ORIGINAL_DISCRETE_MASK,
                random_state=42,
            ),
        ),
        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_STRATEGY,
                k_neighbors=5,
                random_state=42,
            ),
        ),
    ]
    return steps


def _cl_only_steps() -> list:
    """
    CL-only feature pipeline:
        impute -> vt -> Triplet CL (embedding only) -> smote

    No MI selection: the downstream classifier operates purely on the
    128-d triplet embedding space.
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("vt",     VarianceThreshold(threshold=0.0)),
        (
            "cl",
            TripletTrainer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                margin=CL_MARGIN,
                negative_class=CL_NEGATIVE_CLASS,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                balanced_batches=True,
                random_state=42,
                concat_original=False,   # embedding only
                verbose=False,
            ),
        ),
        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_STRATEGY,
                k_neighbors=5,
                random_state=42,
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────
# 6.  Pipeline factories
# ─────────────────────────────────────────────────────────────
def make_rf_pipe(rf_params, with_cl=False):
    return ImbPipeline(
        steps=_base_steps(with_cl=with_cl) + [
            (
                "ordinal_rf",
                FrankHallOrdinalClassifier(
                    base_estimator=RandomForestClassifier(
                        **rf_params,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )
                ),
            )
        ]
    )


def make_xgb_pipe(xgb_params, with_cl=False):
    return ImbPipeline(
        steps=_base_steps(with_cl=with_cl) + [
            (
                "ordinal_xgb",
                FrankHallOrdinalClassifier(
                    base_estimator=XGBClassifier(
                        **xgb_params,
                        **XGB_FIXED,
                    )
                ),
            )
        ]
    )


def make_rf_pipe_cl_only(rf_params):
    return ImbPipeline(
        steps=_cl_only_steps() + [
            (
                "ordinal_rf",
                FrankHallOrdinalClassifier(
                    base_estimator=RandomForestClassifier(
                        **rf_params,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )
                ),
            )
        ]
    )


def make_xgb_pipe_cl_only(xgb_params):
    return ImbPipeline(
        steps=_cl_only_steps() + [
            (
                "ordinal_xgb",
                FrankHallOrdinalClassifier(
                    base_estimator=XGBClassifier(
                        **xgb_params,
                        **XGB_FIXED,
                    )
                ),
            )
        ]
    )


# ─────────────────────────────────────────────────────────────
# 7.  Regression pipeline factories (for BO surrogate)
# ─────────────────────────────────────────────────────────────
def _base_steps_regression(with_cl: bool) -> list:
    """Feature prefix for regression: impute → vt → [cl] → mi.  No SMOTE."""
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("vt",     VarianceThreshold(threshold=0.0)),
    ]
    if with_cl:
        steps.append((
            "cl",
            TripletTrainer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                margin=CL_MARGIN,
                negative_class=CL_NEGATIVE_CLASS,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                balanced_batches=True,
                random_state=42,
                concat_original=True,
                verbose=False,
            )
        ))
    steps.append((
        "mi",
        AdaptiveSelectKBest(
            k=MI_K,
            with_cl=with_cl,
            embedding_dim=CL_EMB_DIM,
            base_discrete_mask=ORIGINAL_DISCRETE_MASK,
            random_state=42,
        ),
    ))
    return steps


def make_rf_regressor_pipe(rf_params, with_cl=False):
    """RF regressor pipeline for BO surrogate (no SMOTE, no ordinal wrapper)."""
    return ImbPipeline(
        steps=_base_steps_regression(with_cl=with_cl) + [
            (
                "rf_reg",
                RandomForestRegressor(
                    **rf_params,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def make_xgb_regressor_pipe(xgb_params, with_cl=False):
    """XGBoost regressor pipeline for BO surrogate (no SMOTE, no ordinal wrapper)."""
    xgb_reg_fixed = {k: v for k, v in XGB_FIXED.items() if k != "eval_metric"}
    xgb_reg_fixed["eval_metric"] = "rmse"
    return ImbPipeline(
        steps=_base_steps_regression(with_cl=with_cl) + [
            (
                "xgb_reg",
                XGBRegressor(
                    **xgb_params,
                    **xgb_reg_fixed,
                ),
            )
        ]
    )


def _cl_only_steps_regression() -> list:
    """CL-only feature pipeline for regression: impute → vt → CL (embedding only). No MI, no SMOTE."""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("vt",     VarianceThreshold(threshold=0.0)),
        (
            "cl",
            TripletTrainer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                margin=CL_MARGIN,
                negative_class=CL_NEGATIVE_CLASS,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                balanced_batches=True,
                random_state=42,
                concat_original=False,   # embedding only
                verbose=False,
            ),
        ),
    ]


def make_rf_regressor_pipe_cl_only(rf_params):
    """RF regressor on CL-only embedding (no MI, no SMOTE)."""
    return ImbPipeline(
        steps=_cl_only_steps_regression() + [
            (
                "rf_reg",
                RandomForestRegressor(
                    **rf_params,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def make_xgb_regressor_pipe_cl_only(xgb_params):
    """XGBoost regressor on CL-only embedding (no MI, no SMOTE)."""
    xgb_reg_fixed = {k: v for k, v in XGB_FIXED.items() if k != "eval_metric"}
    xgb_reg_fixed["eval_metric"] = "rmse"
    return ImbPipeline(
        steps=_cl_only_steps_regression() + [
            (
                "xgb_reg",
                XGBRegressor(
                    **xgb_params,
                    **xgb_reg_fixed,
                ),
            )
        ]
    )
