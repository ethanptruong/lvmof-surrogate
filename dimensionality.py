"""
dimensionality.py
Dimensionality reduction, KMeans grouping, MI diagnostic,
process variable interaction building, and CV matrix assembly.

Group assignment uses PCA (50 components) + KMeans instead of UMAP for
cross-platform reproducibility. UMAP's approximate nearest-neighbor graph
gives different embeddings on Windows vs Linux even with the same
random_state, shifting CV groups and QWK by ±10-15 points.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import silhouette_score

from config import N_CLUSTERS, RANDOM_STATE, MI_K, PROCESS_COLS

warnings.filterwarnings(
    "ignore",
    message="Clustering metrics expects discrete values",
    category=UserWarning,
)


# ─────────────────────────────────────────────────────────────
# prepare_labels
# ─────────────────────────────────────────────────────────────
def prepare_labels(df_merged, X_raw, mask=None):
    """
    Extract y from df_merged["pxrd_score"], apply finite mask to X and y.

    Returns
    -------
    X : np.ndarray  (masked)
    y : np.ndarray  (masked, int)
    mask : np.ndarray bool
    """
    X = np.asarray(X_raw, dtype=float)
    y = pd.to_numeric(df_merged["pxrd_score"], errors="coerce").to_numpy()

    mask = np.isfinite(y)
    X, y = X[mask], y[mask].astype(int)

    y_high = (y >= 7).astype(int)
    print("High (>=7) positives:", int(y_high.sum()), "/", len(y_high),
          "rate=", y_high.mean())
    print("X:", X.shape, "y:", y.shape, "unique y:", np.unique(y))
    print("Any non-finite in X?", (~np.isfinite(X)).any())

    return X, y, mask


# ─────────────────────────────────────────────────────────────
# remap_score
# ─────────────────────────────────────────────────────────────
def remap_score(s) -> int:
    """Map raw pxrd_score (0–9) to 3-class label: 0=Amorphous, 1=Partial, 2=Crystalline."""
    if s <= 2:
        return 0
    if s <= 5:
        return 1
    return 2


# ─────────────────────────────────────────────────────────────
# apply_variance_threshold
# ─────────────────────────────────────────────────────────────
def apply_variance_threshold(X) -> tuple:
    """
    Prepend KMeans OHE features then apply VarianceThreshold(0.0).

    Returns
    -------
    X_vt   : np.ndarray  post-VT features (includes cluster OHE)
    vt_pre : fitted VarianceThreshold transformer
    """
    print(f"[KMeans] Scaling X ({X.shape}) before clustering…")
    _scaler_km = StandardScaler()
    _X_scaled = _scaler_km.fit_transform(X)

    print(f"[KMeans] Fitting k={N_CLUSTERS} clusters…")
    _km_pre = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels_raw = _km_pre.fit_predict(_X_scaled)

    _ohe_km = OneHotEncoder(sparse_output=False, dtype=float)
    _cluster_ohe = _ohe_km.fit_transform(cluster_labels_raw.reshape(-1, 1))

    X = np.hstack([X, _cluster_ohe])
    print(f"[KMeans] X shape after appending cluster OHE features: {X.shape}")
    print(f"[KMeans] Cluster distribution: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(cluster_labels_raw, return_counts=True))} }")

    vt_pre = VarianceThreshold(threshold=0.0)
    X_vt = vt_pre.fit_transform(X)
    print(f"After VarianceThreshold: {X_vt.shape}")

    return X_vt, vt_pre


# ─────────────────────────────────────────────────────────────
# run_mi_diagnostic
# ─────────────────────────────────────────────────────────────
def run_mi_diagnostic(X_vt, y):
    """
    Fit SelectKBest(MI) on X_vt for diagnostic purposes only.
    Does NOT transform X (no leakage).

    Returns
    -------
    mi_pre : fitted SelectKBest transformer
    """
    mi_pre = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X, y, discrete_features=True, random_state=RANDOM_STATE
        ),
        k=min(MI_K, X_vt.shape[1]),
    )
    mi_pre.fit(X_vt, y)

    print(f"After VarianceThreshold   : {X_vt.shape}")
    print(f"MI will select top-{MI_K} INSIDE each CV fold (no leakage)")
    print(f"Diagnostic MI fit complete — cliff plot cell still works unchanged")

    return mi_pre


# ─────────────────────────────────────────────────────────────
# build_process_interactions
# ─────────────────────────────────────────────────────────────
def build_process_interactions(df_merged, mask, process_cols_present) -> tuple:
    """
    Build normalized process variable matrix and pairwise interactions.

    Returns
    -------
    Xprocnorm    : np.ndarray  MinMaxScaled process vars (masked)
    interactions : np.ndarray  6-column interaction/polynomial features
    Xprocess_raw : np.ndarray  raw process vars (masked, for diagnostics)
    """
    Xprocessraw = df_merged[process_cols_present].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0.0).to_numpy()[mask]

    Xprocnorm = MinMaxScaler().fit_transform(Xprocessraw)

    ti  = list(process_cols_present).index('temperature_k')
    rhi = list(process_cols_present).index('reaction_hours')
    mri = list(process_cols_present).index('metal_over_linker_ratio')

    interactions = np.column_stack([
        Xprocnorm[:, ti]  * Xprocnorm[:, mri],   # temp × metal_ratio
        Xprocnorm[:, ti]  * Xprocnorm[:, rhi],   # temp × rxn_hours
        Xprocnorm[:, mri] * Xprocnorm[:, rhi],   # metal_ratio × rxn_hours
        Xprocnorm[:, ti]  ** 2,
        Xprocnorm[:, mri] ** 2,
        (Xprocnorm[:, ti] > 0.85).astype(float), # high-temp flag ≥ 350 K
    ])

    return Xprocnorm, interactions, Xprocessraw


# ─────────────────────────────────────────────────────────────
# assemble_cv_matrix
# ─────────────────────────────────────────────────────────────
def assemble_cv_matrix(X_vt, Xprocnorm, interactions) -> np.ndarray:
    """
    Concatenate VT-filtered features + process vars + interaction features.

    Returns
    -------
    X_cv : np.ndarray  ready for cross-validation
    """
    X = np.hstack([X_vt, Xprocnorm, interactions])
    print(f"X for CV (VT-only + proc/interactions, no MI): {X.shape}")
    return X


# ─────────────────────────────────────────────────────────────
# build_pca_embedding
# ─────────────────────────────────────────────────────────────
def build_pca_embedding(X_for_embedding, n_components=50) -> np.ndarray:
    """
    Fit a PCA embedding on X_for_embedding (MI-filtered view).

    Uses 50 components (vs 2 for UMAP) to retain more chemical variance
    for KMeans clustering, while remaining fully deterministic across
    Windows and Linux.

    Returns
    -------
    X_pca : np.ndarray  shape (n, n_components)
    """
    n_components = min(n_components, X_for_embedding.shape[0] - 1,
                       X_for_embedding.shape[1])
    print(f"PCA input: {X_for_embedding.shape}  →  {n_components} components")
    reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = reducer.fit_transform(X_for_embedding)
    var_explained = reducer.explained_variance_ratio_.sum()
    print(f"PCA done: {X_pca.shape}  (cumulative variance explained: {var_explained:.3f})")
    return X_pca


# ─────────────────────────────────────────────────────────────
# select_kmeans_groups
# ─────────────────────────────────────────────────────────────
def select_kmeans_groups(X_2d, y, n_splits=3) -> tuple:
    """
    Sweep KMeans k values to find the best silhouette score while ensuring
    >= 5 crystalline samples in every CV validation fold.

    Returns
    -------
    groups  : np.ndarray  int cluster labels (n,)
    best_k  : int
    cv      : StratifiedGroupKFold
    """
    print("\n─── KMeans cluster sweep ──────────────────────────────")
    print(f"{'k':>5} {'Silhouette':>12} {'Min class-2 in val':>20} {'Status':>12}")

    best_k, best_score = None, -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for k in range(8, 30, 2):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_k = km.fit_predict(X_2d).astype(int)
        sil = silhouette_score(X_2d, labels_k)

        cv_tmp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        worst = 999
        try:
            for _, val_idx in cv_tmp.split(X_2d, y, labels_k):
                worst = min(worst, (y[val_idx] == 2).sum())
        except ValueError:
            worst = 0

        status = "good" if worst >= 5 else "too few crystalline"
        print(f"{k:>5d} {sil:>12.4f} {worst:>20d} {status}")
        if worst >= 5 and sil > best_score:
            best_score, best_k = sil, k

    if best_k is None:
        print("\nNo k satisfied >=5 crystalline per fold — falling back to n_splits=3 …")
        for k in range(6, 20, 2):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels_k = km.fit_predict(X_2d).astype(int)
            sil = silhouette_score(X_2d, labels_k)
            worst = 999
            try:
                for _, val_idx in cv.split(X_2d, y, labels_k):
                    worst = min(worst, (y[val_idx] == 2).sum())
            except ValueError:
                worst = 0
            status = "good" if worst >= 5 else "bad"
            print(f"  k={k:2d}  sil={sil:.4f}  min_crystalline={worst}  {status}")
            if worst >= 5 and sil > best_score:
                best_score, best_k = sil, k

    print(f"\nSelected k={best_k} (silhouette={best_score:.4f})")
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    groups = km_final.fit_predict(X_2d).astype(int)

    n_groups = len(np.unique(groups))
    print(f"Groups: {n_groups} | Avg per group: {len(y)/n_groups:.1f}")

    return groups, best_k, cv


# ─────────────────────────────────────────────────────────────
# plot_mi_cliff
# ─────────────────────────────────────────────────────────────
def plot_mi_cliff(mi_pre) -> None:
    """Plot and save MI score cliff diagnostic."""
    mi_scores_sorted = np.sort(mi_pre.scores_)[::-1]

    for rank in [300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000,
                 3500, 4000, 4500, 5000, 5500]:
        if rank <= len(mi_scores_sorted):
            print(f"MI at rank {rank:5d}: {mi_scores_sorted[rank-1]:.5f}")

    plt.figure(figsize=(10, 4))
    plt.plot(mi_scores_sorted[:10000])
    plt.axvline(300,  color='red',    linestyle='--', label='k=300')
    plt.axvline(750,  color='orange', linestyle='--', label='k=750')
    plt.axvline(5500, color='green',  linestyle='--', label='k=5500')
    plt.xlabel("Feature rank")
    plt.ylabel("MI score")
    plt.title("MI score cliff")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mi_cliff.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# run_process_variable_diagnostics
# ─────────────────────────────────────────────────────────────
def run_process_variable_diagnostics(
    df_merged,
    mask,
    process_cols_present,
    X_linker,
    X_modulator,
    mod_eq,
    X_precursor_perlig,
    Xinventorynumeric,
    X_process,
    y,
    vt_pre,
    mi_pre,
    X_2d=None,
) -> None:
    """
    Diagnostic: check process variable survival through VT and MI,
    compute MI scores for process vars, and generate diagnostic UMAP plots.
    """
    from sklearn.feature_selection import mutual_info_classif as mic
    from sklearn.preprocessing import StandardScaler as _SS

    Xprocess_raw = df_merged[process_cols_present].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).to_numpy()[mask]

    print(f"Process variable matrix: {Xprocess_raw.shape}")
    print(f"\n{'Feature':<35} {'Unique':>7} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("─" * 75)
    for i, col in enumerate(process_cols_present):
        col_data = Xprocess_raw[:, i]
        print(f"{col:<35} {len(np.unique(col_data)):>7d} "
              f"{col_data.min():>10.3f} {col_data.max():>10.3f} "
              f"{col_data.mean():>10.3f}")

    n_proc = len(process_cols_present)
    n_xfeatures = (X_linker.shape[1] + X_modulator.shape[1] + mod_eq.shape[1] +
                   X_precursor_perlig.shape[1] + Xinventorynumeric.shape[1] +
                   X_process.shape[1])

    proc_start_in_X_final = n_xfeatures - n_proc

    vt_support = vt_pre.get_support()
    proc_survived_vt = [
        (col, vt_support[proc_start_in_X_final + i])
        for i, col in enumerate(process_cols_present)
    ]

    names_vt = np.where(vt_support)[0]
    proc_in_vt_space = {}
    for i, col in enumerate(process_cols_present):
        orig_idx = proc_start_in_X_final + i
        if vt_support[orig_idx]:
            vt_col = np.where(names_vt == orig_idx)[0]
            if len(vt_col):
                proc_in_vt_space[col] = int(vt_col[0])

    mi_support = mi_pre.get_support()
    print(f"\n─── Process variable survival through VT + MI ─────────")
    print(f"{'Variable':<35} {'Survived VT':>12} {'Survived MI':>12}")
    print("─" * 62)
    for col in process_cols_present:
        sv = col in proc_in_vt_space
        sm = sv and mi_support[proc_in_vt_space[col]]
        print(f"{col:<35} {'yes' if sv else 'no':>12}   {'yes' if sm else 'no':>10}")

    n_proc_in_X = sum(
        mi_support[proc_in_vt_space[c]]
        for c in process_cols_present if c in proc_in_vt_space
    )
    print(f"\n{'─'*62}")
    print(f"Process variables in final X ({MI_K} features): {n_proc_in_X} / {n_proc}")
    if n_proc_in_X == 0:
        print("NO process variables reached Optuna — forcing inclusion below.")

    mi_proc = mic(
        Xprocess_raw, y,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )
    print(f"\n─── Process variable MI scores (vs 3-class y) ─────────")
    print(f"{'Variable':<35} {'MI Score':>10}  {'Signal?':>10}")
    print("─" * 60)
    for col, score in sorted(zip(process_cols_present, mi_proc),
                              key=lambda x: -x[1]):
        signal = "strong" if score > 0.05 else ("weak" if score > 0.01 else "noise")
        print(f"{col:<35} {score:>10.4f}  {signal}")

    Xproc_scaled = _SS().fit_transform(Xprocess_raw)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    try:
        sc1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=y, cmap="RdYlGn", s=15, alpha=0.7, vmin=0, vmax=2)
        plt.colorbar(sc1, ax=axes[0], label="3-Class Score")
        axes[0].set_title("All-Feature UMAP (2000 MI features)\nFingerprint-dominated")
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
    except (NameError, TypeError):
        axes[0].text(0.5, 0.5, "X_2d not provided",
                     ha="center", va="center", transform=axes[0].transAxes, fontsize=12)

    print("\nFitting process-variable UMAP...")
    reducer_proc = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                              n_neighbors=min(15, len(y) - 1), min_dist=0.1)
    X_proc_2d = reducer_proc.fit_transform(Xproc_scaled)

    sc2 = axes[1].scatter(X_proc_2d[:, 0], X_proc_2d[:, 1],
                           c=y, cmap="RdYlGn", s=15, alpha=0.7, vmin=0, vmax=2)
    plt.colorbar(sc2, ax=axes[1], label="3-Class Score")
    axes[1].set_title(f"Process Variable UMAP\n({len(process_cols_present)} variables only)")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    top2 = sorted(zip(process_cols_present, mi_proc, range(len(process_cols_present))),
                  key=lambda x: -x[1])[:2]
    (c1, s1, i1), (c2, s2, i2) = top2[0], top2[1]
    sc3 = axes[2].scatter(Xprocess_raw[:, i1], Xprocess_raw[:, i2],
                           c=y, cmap="RdYlGn", s=20, alpha=0.7, vmin=0, vmax=2)
    plt.colorbar(sc3, ax=axes[2], label="3-Class Score")
    axes[2].set_xlabel(f"{c1}\n(MI={s1:.4f})")
    axes[2].set_ylabel(f"{c2}\n(MI={s2:.4f})")
    axes[2].set_title("Top-2 Process Variables\nvs. Crystallinity")

    plt.tight_layout()
    plt.savefig("process_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: process_diagnostic.png")
