# LVMOF-Surrogate

A modular Python pipeline for predicting MOF (Metal-Organic Framework) synthesis outcomes using ordinal classification. Builds a comprehensive feature matrix from molecular descriptors and trains Random Forest and XGBoost models with optional contrastive learning augmentation.

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `mordred` requires `numpy < 2.0` for full compatibility. The `requirements.txt` pins `numpy==2.0.2` and applies a compatibility patch at import time in `featurization.py`.

## Usage

```bash
# Run with default data path (data/Experiments_with_Calculated_Properties_no_linker.xlsx)
python main.py

# Run with a custom data file
python main.py --data path/to/data.xlsx
```

## File Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — orchestrates data loading, featurization, dimensionality reduction, model training, and evaluation |
| `config.py` | All constants: `COLMAP`, `TARGET_METALS`, column lists, model hyperparameter keys, embedding dims, SOAP species, SMARTS patterns |
| `data_processing.py` | Data loading, SMILES cleaning, inventory building, merge logic, QA/audit functions |
| `featurization.py` | All featurization functions: metal descriptors, Morgan FP, Mordred RAC, TEP, morfeus sterics, ChemBERTa-2, G14 hub topology, TTP, DRFP, SOAP |
| `feature_assembly.py` | Assembles `X_final` from featurization outputs via `build_*` functions; master `assemble_features()` entry point |
| `dimensionality.py` | `VarianceThreshold`, KMeans OHE, mutual information diagnostic, UMAP embedding, KMeans group selection, process variable interaction building, CV matrix assembly |
| `models.py` | `FrankHallOrdinalClassifier`, `OrdinalStackingClassifier`, `ContrastiveMITransformer`, `AdaptiveSelectKBest`, `SafeMISelectKBest`, scoring metrics (`qwk_0_9`, `mae_0_9`, `within1`, `exact_acc`), `scoring_ordinal` dict, pipeline factory functions |
| `pipeline.py` | Optuna objective functions (`objective_xgb`, `objective_rf`, `objective_xgb_cl_only`, `objective_rf_cl_only`), progress callbacks, `eval_pipe` |
| `evaluation.py` | `plot_roc_prc`, `plot_learning_curves`, `plot_confusion_matrices`, `run_shap_analysis` |
| `requirements.txt` | Pinned Python dependencies |
| `.gitignore` | Ignores `__pycache__`, build artifacts, `.pkl`, `.csv`, `.png`, `data/`, `.env` |

## Pipeline Overview

```
load_data → build_inventory → merge_data
    ↓
assemble_features()   [feature_assembly.py]
    ↓
prepare_labels → remap_score (3-class: 0=Amorphous, 1=Partial, 2=Crystalline)
    ↓
apply_variance_threshold → run_mi_diagnostic
    ↓
build_process_interactions → assemble_cv_matrix
    ↓
build_umap_embedding → select_kmeans_groups   [for CV stratification]
    ↓
Optuna tuning (XGBoost × 100 trials, RandomForest × 50 trials)
    ↓
Build 4 pipelines:  RF|MI-only,  RF|CL+MI,  XGB|MI-only,  XGB|CL+MI
    ↓
eval_pipe → plot_roc_prc → plot_learning_curves → plot_confusion_matrices → run_shap_analysis
```

### Pipeline step order (inside each cross-validation fold)

```
impute (median) → vt (VarianceThreshold) → [cl (ContrastiveMITransformer)] → mi (AdaptiveSelectKBest) → smote (SMOTE) → FrankHallOrdinalClassifier
```

### Cross-validation strategy

`StratifiedGroupKFold(n_splits=3)` with groups assigned by KMeans clustering on a 2D UMAP embedding. Group selection sweeps `k ∈ [8, 30)` and picks the `k` maximizing silhouette score while ensuring ≥ 5 crystalline samples in every validation fold.

### Primary metric

**QWK** (Quadratic Weighted Kappa) — penalizes predictions proportionally to the squared distance from the true ordinal label.

## Featurization Block Summary

The feature matrix is assembled from 12 descriptor blocks:

### 1. Metal Center Features (Block A)
- **Source:** `mendeleev` + heuristic logic
- **Description:** Captures the fundamental atomic properties of the metal node.
- **Key Features:** Atomic number, period, group, electronegativity (Pauling/Allen), atomic/covalent/vdW radii, ionization energies, valence electrons, oxidation state (parsed from IUPAC), d-electron count, and geometry flags (square planar, tetrahedral, octahedral).

### 2. Co-ligand Inventory (Block B)
- **Source:** RDKit + Lookup Tables
- **Description:** Characterizes the simple ligands attached to the metal precursor (e.g., halides, CO, phosphines).
- **Key Features:** Counts and properties of specific ligands (CO, Cl, Br, I, PPh3), electronic parameters (sigma-donor, pi-acceptor strength), and net charge of the co-ligand sphere.

### 3. Complex-level Descriptors (Block C)
- **Source:** RDKit
- **Description:** Global properties of the metal precursor complex.
- **Key Features:** Dimer/cluster flags, total coordination number, homoleptic status, estimated precursor charge, number of unique ligand types, and presence of bridging halides or metal-metal bonds.

### 4. Revised Autocorrelation (RAC) Descriptors
- **Source:** `mordred`
- **Description:** Encodes topological and electronic structure of ligands.
- **Key Features:** Weighted sum of Autocorrelation descriptors (ATS, MATS, GATS) for all precursor ligands and modulators. Includes missingness indicators for unparseable ligands.

### 5. Physicochemical Descriptors
- **Source:** `rdkit.Chem.Descriptors`
- **Description:** General molecular properties for linkers and modulators.
- **Key Features:** Molecular weight, LogP, TPSA, number of rotatable bonds, H-bond donors/acceptors, Hall-Kier alpha, aromatic ring count, and Gasteiger partial charges.

### 6. Tolman Electronic Parameter (TEP)
- **Source:** `LGBMRegressor` (pre-trained model from Daniel Ess lab)
- **Description:** Predicts the electronic donating ability of phosphine and N-heterocyclic carbene ligands.
- **Key Features:** Predicted TEP value (cm⁻¹) for modulators, linkers, and precursor ligands. Includes flags for missing/non-applicable values.

### 7. Steric Descriptors (Cone Angle & Buried Volume)
- **Source:** `morfeus`
- **Description:** Captures the spatial bulk of ligands, critical for surface chemistry and pore formation.
- **Key Features:** Exact Cone Angle and Percent Buried Volume (%V_bur) for phosphine-containing linkers, modulators, and precursor ligands.

### 8. ChemBERTa-2 Embeddings
- **Source:** `DeepChem/ChemBERTa-77M-MTR` (Hugging Face)
- **Description:** Transformer-based embeddings pre-trained on 77 million molecules.
- **Key Features:** 384-dimensional CLS token vector providing a dense, context-aware representation of the linker and modulator molecular structures.

### 9. Extended RDKit Descriptors
- **Source:** `rdkit`
- **Description:** A curated suite of 2D descriptors focusing on complexity, topology, and functional groups.
- **Key Features:** BertzCT, MolMR, LabuteASA, FractionCSP3, Chi/Kappa connectivity indices, ring counts (aliphatic/saturated), partial charge statistics, and counts of specific MOF-relevant SMARTS patterns (e.g., COOH, pyridyl N).

### 10. 3D Shape Descriptors
- **Source:** `rdkit.Chem.rdMolDescriptors` (via ETKDGv3 conformers)
- **Description:** Captures the 3D geometry of the organic linkers/modulators.
- **Key Features:** Principal Moments of Inertia (PMI), Normalized Principal Moments Ratio (NPR), Asphericity, Eccentricity, Spherocity Index, and Radius of Gyration.

### 11. Tetratopic Phosphine (TTP) Topology
- **Source:** Custom graph traversal logic
- **Description:** Specifically designed for Group 14 (Si, Ge, Sn, C) centered tetratopic linkers.
- **Key Features:** Hub element identity, arm length statistics (min/max/mean), arm backbone composition (alkyl/aryl/alkynyl), topicity (number of P-arms), and hub graph eccentricity.

### 12. Reaction Fingerprint (DRFP)
- **Source:** `drfp`
- **Description:** Encodes the reaction as a whole (precursor + linker + modulator).
- **Key Features:** 2048-bit hashed fingerprint capturing the chemical transformation and local environments of all reactants simultaneously.

## Known Bug Fixes

### `featurization.py` — `get_d_electron_count`

**Original (notebook):**
```python
d_count = 11 - int(oxidationstate) if oxidationstate >= 1 else 10
```

**Fixed:**
```python
d_count = 11 - int(oxidation_state) if oxidation_state >= 1 else 10
# BUG FIXED: NameError — 'oxidationstate' should be 'oxidation_state' (missing underscore, mismatches function parameter name)
```

The variable `oxidation_state` (matching the function parameter name) was referenced as `oxidationstate` (no underscore), which would raise a `NameError` at runtime for any metal requiring d-electron count computation.
