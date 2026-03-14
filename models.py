"""
models.py
Model classes, metric functions, scoring dict, and pipeline factory functions.
All class bodies and function bodies copied EXACTLY from the source notebook.
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (RANDOM_STATE, MI_K, CL_EMB_DIM, SMOTE_STRATEGY,
                    XGB_FIXED, XGB_TUNED_KEYS)

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
# SafeMISelectKBest
# ─────────────────────────────────────────────────────────────
class SafeMISelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=5500, with_cl=False, embedding_dim=128, random_state=42):
        self.k = k
        self.with_cl = with_cl
        self.embedding_dim = embedding_dim
        self.random_state = random_state

    def _make_discrete_mask(self, n_features):
      if not self.with_cl:
          return True

      n_original = n_features - self.embedding_dim
      if n_original <= 0:
          raise ValueError(
              f"Expected appended embedding_dim={self.embedding_dim}, got n_features={n_features}."
          )

      return np.r_[
          np.ones(n_original, dtype=bool),      # original VT features
          np.zeros(self.embedding_dim, dtype=bool)  # CL embedding
      ]


    def fit(self, X, y):
        n_features = X.shape[1]
        discrete_mask = self._make_discrete_mask(n_features)

        self.scores_ = mutual_info_classif(
            X, y,
            discrete_features=discrete_mask,
            random_state=self.random_state
        )

        if self.k == "all":
            self.k_ = n_features
        else:
            self.k_ = min(int(self.k), n_features)

        ranked = np.argsort(self.scores_)[::-1]
        keep = ranked[:self.k_]
        self.support_idx_ = np.sort(keep)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, "support_idx_")
        return X[:, self.support_idx_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_idx_")
        if indices:
            return self.support_idx_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_idx_] = True
        return mask


# ─────────────────────────────────────────────────────────────
# ContrastiveMITransformer inner classes
# ─────────────────────────────────────────────────────────────
class _TripletDataset(Dataset):
    def __init__(self, X, triplets):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        a, p, n = self.triplets[i]
        return self.X[a], self.X[p], self.X[n]



class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, embedding_dim=128, dropout=0.2):
        super().__init__()
        h1 = hidden_dim
        h2 = max(hidden_dim // 2, embedding_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),          
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),          
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(h2, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveMITransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible contrastive feature augmenter.

    Input to fit/transform:
        X = pipeline features AFTER impute -> vt

    Output from transform:
        [X, embedding] if concat_original=True
        embedding only if concat_original=False

    Modes:
        - loss_mode="supcon"  (recommended default)
        - loss_mode="triplet" (exercise-style)
    """

    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        loss_mode="supcon",              # "supcon" or "triplet"
        negative_class="partial",       # for triplet mode: "partial", "amorphous", or 1/0
        temperature=0.07,               # supcon
        margin=0.5,                     # triplet
        epochs=15,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        n_triplets=4000,
        balanced_batches=True,
        scale_for_cl=True,
        concat_original=True,
        device=None,
        random_state=42,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.loss_mode = loss_mode
        self.negative_class = negative_class
        self.temperature = temperature
        self.margin = margin
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_triplets = n_triplets
        self.balanced_batches = balanced_batches
        self.scale_for_cl = scale_for_cl
        self.concat_original = concat_original
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_encoder(self, input_dim):
        return MLPEncoder(
            in_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            dropout=0.2,
        )

    def _supcon_loss(self, z, y):
        z = F.normalize(z, dim=1)
        logits = torch.matmul(z, z.T) / self.temperature

        n = z.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=z.device)

        logits = logits.masked_fill(eye, -1e9)
        pos_mask = (y.view(-1, 1) == y.view(1, -1)) & (~eye)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return None

        log_prob = F.log_softmax(logits, dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss

    def _make_supcon_loader(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)

        # Seeded generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        if not self.balanced_batches:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                generator=generator,   # ← add here too
            )

        classes, counts = np.unique(y, return_counts=True)
        class_w = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_w = np.array([class_w[yi] for yi in y], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float32),
            num_samples=len(sample_w),
            replacement=True,
            generator=generator,      
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False,
            generator=generator,       
        )

    def _build_triplets(self, y):
        y = np.asarray(y).astype(int)

        idx_pos = np.where(y == 2)[0]      # crystalline anchors/positives

        if len(idx_pos) < 2:
            raise ValueError("Triplet mode needs at least 2 class-2 samples.")

        if self.negative_class in ("partial", 1):
            idx_neg = np.where(y == 1)[0]
        elif self.negative_class in ("amorphous", 0):
            idx_neg = np.where(y == 0)[0]
        elif self.negative_class in ("rest", "non_crystalline"):
            idx_neg = np.where(y != 2)[0]
        elif self.negative_class == "mixed":
            # Half easy (amorphous, class 0) + half hard (partial, class 1)
            idx_easy = np.where(y == 0)[0]
            idx_hard = np.where(y == 1)[0]
            if len(idx_easy) == 0 or len(idx_hard) == 0:
                raise ValueError("mixed negative_class requires samples from both class 0 and class 1.")
            rng = np.random.default_rng(self.random_state)
            triplets = []
            n_easy = self.n_triplets // 2
            n_hard = self.n_triplets - n_easy
            for _ in range(n_easy):
                a_idx, p_idx = rng.choice(idx_pos, size=2, replace=False)
                n_idx = rng.choice(idx_easy)
                triplets.append((int(a_idx), int(p_idx), int(n_idx)))
            for _ in range(n_hard):
                a_idx, p_idx = rng.choice(idx_pos, size=2, replace=False)
                n_idx = rng.choice(idx_hard)
                triplets.append((int(a_idx), int(p_idx), int(n_idx)))
            rng.shuffle(triplets)
            return triplets
        else:
            raise ValueError(
                "negative_class must be one of: 'partial', 'amorphous', 'rest', 'mixed', 0, 1."
            )

        if len(idx_neg) == 0:
            raise ValueError(f"No negatives available for negative_class={self.negative_class!r}.")

        rng = np.random.default_rng(self.random_state)
        triplets = []
        for _ in range(self.n_triplets):
            a_idx, p_idx = rng.choice(idx_pos, size=2, replace=False)
            n_idx = rng.choice(idx_neg)
            triplets.append((int(a_idx), int(p_idx), int(n_idx)))
        return triplets

    def fit(self, X, y):
        self._set_seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]

        if self.scale_for_cl:
            self.scaler_ = StandardScaler()
            X_cl = self.scaler_.fit_transform(X).astype(np.float32)
        else:
            self.scaler_ = None
            X_cl = X

        self.encoder_ = self._build_encoder(X_cl.shape[1])
        device = self._get_device()
        self.encoder_.to(device)

        opt = optim.Adam(self.encoder_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        self.loss_history_ = []

        if self.loss_mode == "supcon":
            loader = self._make_supcon_loader(X_cl, y)

            for epoch in range(self.epochs):
                self.encoder_.train()
                losses = []

                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    z = self.encoder_(xb)
                    loss = self._supcon_loss(z, yb)
                    if loss is None or not torch.isfinite(loss):
                        continue

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))

                sched.step()
                epoch_loss = float(np.mean(losses)) if losses else np.nan
                self.loss_history_.append(epoch_loss)

                if self.verbose:
                    print(f"[SupCon] epoch {epoch+1:02d}/{self.epochs}  loss={epoch_loss:.4f}")

        elif self.loss_mode == "triplet":
            self.triplets_ = self._build_triplets(y)
            _triplet_gen = torch.Generator()
            _triplet_gen.manual_seed(self.random_state)
            loader = DataLoader(
                _TripletDataset(X_cl, self.triplets_),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                generator=_triplet_gen,
            )
            loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

            for epoch in range(self.epochs):
                self.encoder_.train()
                losses = []

                for a, p, n in loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)

                    za = F.normalize(self.encoder_(a), dim=1)
                    zp = F.normalize(self.encoder_(p), dim=1)
                    zn = F.normalize(self.encoder_(n), dim=1)

                    loss = loss_fn(za, zp, zn)
                    if not torch.isfinite(loss):
                        continue

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))

                sched.step()
                epoch_loss = float(np.mean(losses)) if losses else np.nan
                self.loss_history_.append(epoch_loss)

                if self.verbose:
                    print(f"[Triplet] epoch {epoch+1:02d}/{self.epochs}  loss={epoch_loss:.4f}")

        else:
            raise ValueError("loss_mode must be 'supcon' or 'triplet'.")

        self.encoder_.cpu()
        self.encoder_.eval()
        return self

    def _embed(self, X):
        X = np.asarray(X, dtype=np.float32)

        if self.scaler_ is not None:
            X_cl = self.scaler_.transform(X).astype(np.float32)
        else:
            X_cl = X

        self.encoder_.eval()
        Xt = torch.tensor(X_cl, dtype=torch.float32)

        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 256):
                z = self.encoder_(Xt[i:i+256])
                z = F.normalize(z, dim=1)
                out.append(z.cpu().numpy())

        return np.vstack(out)

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "n_features_in_"])
        X = np.asarray(X, dtype=np.float32)
        emb = self._embed(X)

        # IMPORTANT:
        # return the ORIGINAL incoming pipeline X plus learned embedding,
        # not the internally scaled CL matrix.
        if self.concat_original:
            return np.hstack([X, emb])
        return emb


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    """
    SelectKBest that:
      - caps k at the number of available columns
      - optionally appends continuous CL embedding columns to the mask
      - can accept a mixed discrete/continuous mask for the original features
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
                    f"base_discrete_mask has length {len(mask)} but X has {n_features} features."
                )
            return mask

        n_orig = n_features - self.embedding_dim
        if n_orig <= 0:
            raise ValueError(
                f"Expected at least {self.embedding_dim + 1} features, got {n_features}."
            )

        if self.base_discrete_mask is None:
            base_mask = np.ones(n_orig, dtype=bool)
        else:
            base_mask = np.asarray(self.base_discrete_mask, dtype=bool)
            if len(base_mask) != n_orig:
                raise ValueError(
                    f"base_discrete_mask has length {len(base_mask)} but original block has {n_orig}."
                )

        return np.r_[base_mask, np.zeros(self.embedding_dim, dtype=bool)]

    def fit(self, X, y):
        X = np.asarray(X)
        k_use = min(self.k, X.shape[1])
        disc = self._disc_mask(X.shape[1])

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


def balanced_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[yi] for yi in y])


def make_cl_transformer():
    return ContrastiveMITransformer(
        embedding_dim=128,
        hidden_dim=512,
        n_epochs=15,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=128,
        margin=0.5,
        n_triplets=4000,
        negative_class="partial",   # hard negatives
        concat_original=True,
        scale_for_cl=True,
        random_state=42,
        verbose=False,
    )


# -----------------------------
# Shared feature pipeline
# Order:
# impute -> vt -> [cl] -> mi -> smote -> ordinal model
# -----------------------------
def make_feature_steps(with_cl=False):
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=0.0)),
    ]

    if with_cl:
        steps.append((
            "cl",
            ContrastiveMITransformer(
                embedding_dim=128,
                hidden_dim=512,
                epochs=15,
                lr=1e-4,
                weight_decay=1e-4,
                batch_size=128,
                margin=0.5,
                n_triplets=4000,
                negative_class="partial",   # hard negatives
                concat_original=True,
                scale_for_cl=True,
                random_state=42,
                verbose=False,
            )
        ))

    steps.append((
        "mi",
        SafeMISelectKBest(
            k=5500,
            with_cl=with_cl,
            embedding_dim=128,
            random_state=42,
        )
    ))

    steps.append((
        "smote",
        SMOTE(sampling_strategy={1: 180, 2: 250}, k_neighbors=5, random_state=42)
    ))

    return steps


# ─────────────────────────────────────────────────────────────
# Pipeline step builders
# ─────────────────────────────────────────────────────────────
def _base_steps(with_cl: bool) -> list:
    """
    Common prefix:
        impute -> vt -> [cl] -> mi -> smote

    Default CL = supervised contrastive over all 3 classes.
    To match the exercise exactly, switch to:
        loss_mode="triplet", negative_class="amorphous"  # easy
    or:
        loss_mode="triplet", negative_class="partial"    # hard
    """
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=0.0)),
    ]

    if with_cl:
        steps.append((
            "cl",
            ContrastiveMITransformer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                loss_mode="supcon",          # recommended default
                negative_class="partial",    # used only in triplet mode
                temperature=0.07,
                margin=0.5,
                epochs=15,                   # keep modest: this runs inside CV
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                n_triplets=4000,
                balanced_batches=True,
                scale_for_cl=True,
                concat_original=True,
                device=None,
                random_state=42,
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
    CL-only pipeline:
        impute -> vt -> cl(embedding only) -> smote
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=0.0)),
        (
            "cl",
            ContrastiveMITransformer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                loss_mode="supcon",
                negative_class="partial",   # used only in triplet mode
                temperature=0.07,
                margin=0.5,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                n_triplets=4000,
                balanced_batches=True,
                scale_for_cl=True,
                concat_original=False,      # <<< CL-only: no raw features
                device=None,
                random_state=42,
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


def _cl_only_steps_triplet() -> list:
    """
    Triplet CL-only pipeline:
        impute -> vt -> cl(triplet, mixed negatives, embedding only) -> smote

    anchor   = crystalline (class 2)
    positive = crystalline (class 2)
    negative = 50 % amorphous (easy, class 0) + 50 % partial (hard, class 1)
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=0.0)),
        (
            "cl",
            ContrastiveMITransformer(
                embedding_dim=CL_EMB_DIM,
                hidden_dim=256,
                loss_mode="triplet",
                negative_class="mixed",      # easy (amorphous) + hard (partial)
                margin=0.5,
                epochs=15,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                n_triplets=4000,
                scale_for_cl=True,
                concat_original=False,       # CL-only: no raw features
                device=None,
                random_state=42,
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
# Pipeline factories
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


def make_rf_pipe_cl_only_triplet(rf_params):
    return ImbPipeline(
        steps=_cl_only_steps_triplet() + [
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


def make_xgb_pipe_cl_only_triplet(xgb_params):
    return ImbPipeline(
        steps=_cl_only_steps_triplet() + [
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
