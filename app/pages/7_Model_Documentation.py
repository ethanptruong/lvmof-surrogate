"""Chemist-facing model documentation.

A plain-language tour of how COMPASS's surrogate works, written for a
synthetic chemist rather than a software engineer. Three goals:

  1. Demystify the pipeline - what happens between "PXRD score in Excel"
     and "predicted score for a candidate".
  2. Explain why this project prioritizes **precision** over recall when
     comparing surrogates - BO only needs *some* crystalline picks to
     work, so a model that's stingy-but-right is preferable to one that
     casts a wide net.
  3. Teach how to read the per-surrogate learning curves so the user can
     answer "is more data going to help, or are we plateaued?"
"""

from __future__ import annotations

import os
import sys

# Path bootstrap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import streamlit as st  # noqa: E402

from app.ui.components import page_header  # noqa: E402

st.set_page_config(page_title="Model Documentation · COMPASS",
                   layout="wide")

page_header(
    "Model Documentation",
    caption=("A description of COMPASS implementation, and how to determine, "
             "correct use cases."),
)


# -- Section 1: What the surrogate is for ---
st.subheader("Surrogate Summary")

st.markdown(
    """
The surrogate is a predictor of **PXRD** crystallinity, built from previous experiments from the lab.
Given a candidate synthesis recipe (metal precursor, linker, modulator, solvents,
temperature, concentration, time, etc.) it returns:

- A **predicted PXRD class** - Amorphous (0) / Partial (1) / Crystalline (2).
- A **predicted continuous score** in the original 0-9 range, useful for
  ranking candidates that fall in the same class.
- An **uncertainty estimate** High spread => the model hasn't
  seen anything like this candidate before.

The surrogate is **not** a full model. It will not tell
you *why* a candidate is predicted to be crystalline (SHAP panels can be analyzed for similar information),
 and it will not extrapolate confidently to metal
centers, linkers, or solvents that are far outside the training set. Think of it as interpolation vs extrapolation.
It is important that you think about the data the model has been trained on before asking it to recommend novel syntheses. 
It should be noted that the goal of this model is to maximize the crystallinity as indicated by PXRD. This is not necessarily indicative of MOF formation.

"""
)

st.divider()


# -- Scoring guide: how to label your own PXRD patterns ---
st.subheader("How to Score PXRDs")

st.markdown(
    """
The model is trained on a **0-9 ordinal score** of PXRD crystallinity. This allows for comparison between different systems, but also allows for variance in interrater reliability.
If you plan to record your own experiments and feed them back into the
pipeline, scoring consistency is the single biggest lever on model
quality. Two scorers disagreeing by ±2 across the dataset will swamp
any algorithmic improvement.
I 
**Anchors:**

- **0** - *No product formed.* Not enought material for a PXRD pattern. Use 0 whenever the synthesis failed outright.
- **1** - *Complex / non-MOF product.* If you isolated a coordination
  complex or other non-target solid rather than a MOF, label it **0 or
  1** depending on how strongly you want the model to *avoid* this
  region of synthesis space in future recommendations. A 0 is a hard
  "don't go here"; a 1 is a softer "this didn't give the target, but
  the chemistry is closer than nothing".
- **2-8** - *Partial crystallinity.* Increasingly sharp/intense peaks
  matching the target reference, scaling roughly with peak count,
  sharpness, and signal-to-background.
- **9** - *Fully crystalline target MOF.* Sharp peaks at all expected
  reference positions, strong intensity, low background.

Use the reference patterns below to anchor your scoring. When in
doubt, score conservatively (one bin lower).
"""
)

_DEMO_DIR = os.path.join(_PROJECT_ROOT, "app", "assets", "pxrd_demos")

_DEMO_CAPTIONS = {
    0: "**0 - No product formed.** No PXRD pattern collected, or a flat/noise-only "
       "trace. Use this when the synthesis failed outright. (No reference image - "
       "absence of a pattern is the signal.)",
    1: "**1 - Complex or non-MOF product, or barely-detectable order.** Use 0 or 1 "
       "for isolated complexes depending on how strongly you want to deter the "
       "model from this region.",
    2: "**2 - Very weak, broad features.** A hint of order above background but no "
       "clear peaks matching the target reference.",
    3: "**3 - Broad, weak peaks at some expected positions.** Crystallinity is "
       "emerging but the product is mostly disordered.",
    4: "**4 - Several discernible peaks** at expected positions, still broad and "
       "modest in intensity.",
    5: "**5 - Clear partial crystallinity.** Most expected peaks visible but "
       "broader and weaker than a clean reference.",
    6: "**6 - Good crystallinity.** Sharp peaks at expected positions, moderate "
       "intensity, some background remaining.",
    7: "**7 - Strong crystallinity.** Sharp, intense peaks across the expected "
       "pattern; minor background or impurity peaks acceptable.",
    8: "**8 - Excellent crystallinity.** Sharp, intense, well-resolved peaks "
       "matching the reference; very low background.",
    9: "**9 - Fully crystalline target MOF.** Sharp peaks at all expected "
       "reference positions, strong intensity, clean background.",
}

for score in range(0, 10):
    img_path = os.path.join(_DEMO_DIR, f"{score}_demo.png")
    cols = st.columns([1, 2])
    with cols[0]:
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.markdown(
                f"<div style='border:1px dashed #888; padding:1.5rem; "
                f"text-align:center; color:#888;'>No reference image<br>"
                f"(score {score})</div>",
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.markdown(_DEMO_CAPTIONS[score])
    st.markdown("")  # spacing

st.divider()


# -- Section 2: Step-by-step pipeline ---
st.subheader("Implementation")

st.markdown(
    """
Below is the full pipeline. Each step is independent and reproducible - you can
inspect intermediate artifacts under `checkpoints/` and `data/` folders.
"""
)

with st.expander("Turn each experiment into numbers (featurization)",
                  expanded=True):
    st.markdown(
        """
The dataset starts as SMILES strings, solvent names, and process
numbers. A machine-learning model cannot understand these, so the
pipeline builds a wide feature matrix (~10,000 raw columns) by stacking
**12 chemistry-aware descriptor blocks (generated by an LLM)**:

| Block | What it captures (in chemist terms) |
|---|---|
| **Metal center (A)** | Electronegativity, atomic / covalent radii, d-electron count, oxidation state, preferred geometry - basically a periodic-table fingerprint for the metal. |
| **Co-ligand inventory (B)** | Counts of halides, phosphines, carbonyls; net charge on the metal precursor; σ/π donor / acceptor strength from lookup tables. |
| **Complex-level (C)** | Coordination number, dimer flags, ligand-diversity counts, metal precursor charge - the "shape" of the metal complex before linker addition. |
| **Mordred RAC autocorrelation** | How chemical properties are distributed across each ligand's graph - sensitive to substitution patterns. |
| **RDKit physicochemical** | MW, LogP, TPSA, rotatable-bond count, H-bond donor/acceptor counts - the usual medicinal-chemistry knobs, repurposed. |
| **Tolman Electronic Parameter (TEP)** | Predicted phosphine donor strength (from a small LGBM regressor pretrained on literature TEP values). |
| **Steric descriptors (`morfeus`)** | Cone angle and percent buried volume for phosphine-containing ligands. |
| **ChemBERTa-2 embeddings** | A 384-dim "semantic" fingerprint from a chemistry-pretrained transformer - captures structural similarity the hand-crafted descriptors miss. |
| **Extended RDKit (SMARTS counts)** | Counts of MOF-relevant functional groups: -COOH, pyridyl-N, imidazole, sulfonate, etc. |
| **3D shape (RDKit ETKDGv3)** | PMI, asphericity, eccentricity, spherocity, radius of gyration of each component. |
| **TTP / G14 hub topology** | Tetratopic linker arm geometry, hub element identity, P-arm counts - built specifically for the LVMOF chemistry our lab works in. |
| **DRFP reaction fingerprint** | A 2048-bit hashed fingerprint of the full *reaction* (metal precursor + linker + modulator), not just one molecule. |

Process variables - temperature, M:L ratio, concentrations, reaction
time, solvent fractions - are concatenated on the end, along with their
pairwise interactions (e.g. *T × M:L*, *T × time*). **Solvent mixtures
are summarized by their COSMO-RS σ-profile moments** (area, polarity,
asymmetry, kurtosis, H-bond donor / acceptor capacity), so a mixed
solvent system is described by a continuous vector instead of three
one-hot solvent IDs.
"""
    )

with st.expander("Feature Filtering"):
    st.markdown(
        """
~10,000 raw features is far more than our dataset can support
without overfitting. That is, learning a false signal or memorizing data. We implement two filters in sequence:

1. **Variance threshold:** drop any column that is constant  in the training data. (A descriptor that's the same
   for every row carries zero information.)
2. **Mutual-Information (MI) selection** - for each remaining feature,
   measure how much knowing it reduces uncertainty about the PXRD
   class. Keep the top-K MI features. Two separate budgets are used:
   - One for **discrete** features (fingerprints, one-hot columns). These are
     scored with the entropy estimator.
   - One for **continuous** features. These are scored with the KNN estimator.
   Mixing the two on a small dataset would inflate the KNN scores
   spuriously, so they're ranked and selected independently.

Here, we have chosen a budget of 100 and 1000 discrete and continuous features in an effort to prevent overfitting
"""
    )

with st.expander("Balance the rare crystalline class (SMOTE)"):
    st.markdown(
        """
The dataset is heavily imbalanced. Most experiments end up Amorphous
or Partial, with relatively few Crystalline hits. A model trained
naively on this would learn to predict one class. If the model were to choose Amorphous all of the time, it would be right
most of the time, which is useless.

**SMOTE** (Synthetic Minority Oversampling Technique) generates
synthetic examples of the minority classes by interpolating between
existing minority points in the feature space. It's only applied
**inside the training fold** during cross-validation - never on the
validation set - so the reported metrics still reflect real held-out
performance, not memorized synthetic points.
"""
    )

with st.expander("Cross-Validation"):
    st.markdown(
        """
A naive random K-fold split allows two near-duplicate experiments
(same linker, same metal, slightly different temperature) end up in
training and validation respectively, inflating the apparent accuracy.
The model can use the data of one to predict the outcome of the other. 

Instead, the pipeline:

1. Builds a **2D UMAP embedding** of the MI-filtered features so
   chemically-similar experiments cluster together.
2. Runs **KMeans** on that embedding - sweeping k ∈ [8, 30) and picking
   the value that maximizes silhouette score subject to having at least
   5 crystalline samples per validation fold.
3. Uses those clusters as **groups** in `RepeatedStratifiedGroupKFold`,
   so all experiments in a cluster stay on the same side of the split.

"""
    )

with st.expander("Ordinal Classification (Frank-Hall)"):
    st.markdown(
        """
Amorphous → Partial → Crystalline is an **ordered** class structure. Getting Crystalline misclassified as
Amorphous is worse than misclassifying it as Partial. A standard
classifier ignores that ordering.

**Frank-Hall ordinal decomposition** handles this by training K−1 = 2
binary classifiers on cumulative thresholds:

- Model A: **P(score > 0)** - is this at least Partial?
- Model B: **P(score > 1)** - is this Crystalline?

The three class probabilities are recovered from differences:

```
P(Amorphous)   = 1 − P(score > 0)
P(Partial)     = P(score > 0) − P(score > 1)
P(Crystalline) = P(score > 1)
```

Each of the two underlying binary classifiers is a **Random Forest** or
**XGBoost** ensemble. The final predicted PXRD class is the argmax of
those three probabilities; the continuous score is the expected value.
"""
    )

with st.expander("Surrogate Models"):
    st.markdown(
        """
Six pipeline variants are trained side-by-side:

| Pipeline | Features used | Base classifier |
|---|---|---|
| **RF · MI only** | MI-selected features only | Random Forest |
| **RF · CL + MI** | Triplet-contrastive embedding + MI features | Random Forest |
| **RF · CL only** | Triplet-contrastive embedding only | Random Forest |
| **XGB · MI only** | MI-selected features only | XGBoost |
| **XGB · CL + MI** | Triplet-contrastive embedding + MI features | XGBoost |
| **XGB · CL only** | Triplet-contrastive embedding only | XGBoost |

The **CL embedding** is a small MLP encoder pretrained with **triplet
contrastive loss**: it learns to pull Crystalline experiments close
together in embedding space and push Partial/Amorphous experiments
away. Concatenating that embedding next to (or in place of) the MI
features sometimes helps the downstream classifier, sometimes doesn't -
that's why we keep all three flavors and let the CV pick the winner.
"""
    )

with st.expander("Optuna optimization"):
    st.markdown(
        """
Each of the six variants is tuned with **Optuna** (TPE sampler, ~100
trials per variant), optimizing the **cross-validated Quadratic
Weighted Kappa** (QWK). QWK 
**penalizes misclassifications proportionally to the squared ordinal
distance.** Predicting Amorphous when the truth is Crystalline penalizes the model
4× more than predicting Partial.

The final tuned model is the variant with the highest CV-QWK, and its
checkpoint is what BO loads on the Recommend page.
"""
    )

st.divider()


# -- Section 3: Why LFBO-SSL is the default acquisition function ---
st.subheader("Acquisition Function")

st.markdown(
    """
The surrogate scores every candidate in the pool. We still need a
**decision rule** that turns those scores into a ranked list of
experiments that we can then execute at the bench. We use an **acquisition
function** to do this.

COMPASS supports several (EI, LFBO, **LFBO-SSL**, Thompson, consensus,
random (just in case you need it)). **LFBO-SSL** is the project default. For our chemistry,
the others are mostly there for comparison and validation.
"""
)

with st.expander("Why don't just rank by predicted PXRD score?",
                  expanded=False):
    st.markdown(
        """
Ranking by predicted score alone is **pure exploitation.** You only
ever try candidates that look like things that have already worked. You
never explore the edges of the synthesis space where the surrogate is uncertain
but might be highly crystalline. Acquisition functions
explicitly add an **exploration term** so the surrogate gets to probe
its own blind spots.
"""
    )

with st.expander("Expected Improvement is the textbook approach, but a poor fit in this context."):
    st.markdown(
        """
Classical BO assumes the surrogate predicts a **Gaussian** distribution
over each candidate's score (think bell curve), and computes the expected amount by which
a candidate will *exceed the current best*. EI balances the predicted
mean (exploitation) and the predicted standard deviation (exploration)
analytically. It's the standard textbook acquisition.

However, the PXRD score (Crystallinity Index) is not a Gaussian
variable. It's a **bounded 0-9 ordinal**, heavily skewed toward the
Amorphous end. Fitting a Gaussian over a score that's hard-capped at 0
and 9, and concentrated near 0, insures that EI is miscalibrated.

"""
    )

with st.expander("LFBO (Song et al., ICML 2022)",
                  expanded=True):
    st.markdown(
        """
**Likelihood-Free Bayesian Optimization** skips the Gaussian assumption
entirely. Instead of modeling the score distribution, it converts the
BO problem into a **binary classification problem**. Can you predict that an 
experiment will fall into one of two bins.

1. Pick a quantile γ (default **γ = 0.25** in COMPASS) and let τ be the
   (1 − γ) quantile of observed scores - roughly, the top 25% threshold
   in what's been seen so far.
2. Re-label each observed experiment: **z = 1 if y ≥ τ** ("a hit"),
   **z = 0 otherwise**. (The experiment is in the top γ of experiments)
3. Weight each positive example by *how far above τ it landed*
   (`max(y − τ, ε)`), so highly crystalline experiments (a 9 for example) has more influence on predictions 
   than borderline ones (a 7).. Negatives get uniform weight 1.0.
4. Train a Random Forest classifier on these weighted binary labels.
5. **Score each candidate by the classifier's P(z = 1).**

That probability is **mathematically equivalent to Expected
Improvement** under a density-ratio argument, but it makes **no
parametric assumption** about the score distribution. This is exactly
what we want for an ordinal, skewed PXRD target.
"""
    )

with st.expander("LFBO-SSL: an improvement",
                  expanded=True):
    st.markdown(
        """
Standard LFBO has one critical weakness for our use case: **the
classifier is trained only on the experiments that have actually been
observed**. In a fresh BO run on a new target chemistry, the
initialization set is often just **10-30 experiments**. That's far too
few to train a stable Random-Forest classifier as done by Yaghi et al. (https://doi.org/10.1021/acscentsci.3c01087), and the resulting
acquisition rankings are noisy - small changes in a single observed
point can completely reorder the top-K picks.

**LFBO-SSL (Semi-Supervised LFBO)** fixes this by **borrowing
information from the regression surrogate** to pseudo-label the
unevaluated candidate pool:

1. The regression surrogate (the Frank-Hall ordinal model from
   Section 2) predicts (μ, σ) for every candidate in the pool.
2. Each candidate gets a **pseudo-binary-label**: z = 1 if μ ≥ τ,
   z = 0 otherwise.
3. Each pseudo-label is **weighted by the surrogate's confidence**:
   - **Low σ → high weight.** Confident predictions pull the
     classifier hard toward themselves.
   - **High σ → low weight.** Uncertain candidates barely move the
     classifier, leaving room for exploration there.
   - A configurable `pseudo_weight` multiplier (default in
     `config.BO_LFBO_SSL_PSEUDO_WEIGHT`) controls how much overall
     trust we give pseudo-labels vs. real ones.
4. **LFBO-EI improvement weighting** is then applied on the
   pseudo-positives, mirroring the observed-side scheme - so a
   candidate with a *confidently very high* μ has higher pull than a
   *confidently mid-range* μ.
5. The classifier is trained on the **union** of real labels
   (full weight) + pseudo-labels (down-weighted by σ), giving it
   thousands of effective training points instead of ~30.

This means that

- **Cold-start works.** The acquisition function inherits the
  regression surrogate's view of chemistry-space before any BO
  observations are collected, so the *very first* batch out of LFBO-SSL
  is already informed.
- **Uncertainty is still respected.** Wherever the surrogate is unsure,
  the pseudo-labels barely move the classifier, leaving room for the
  loop to actually explore.
- **Rankings are stable.** Small changes in the handful of real
  observations don't reshuffle the top-K. The bulk of the
  classifier's training signal comes from the 
  surrogate-based pseudo-labels.

TLDR: LFBO gives us the right function to analyze a non-Gaussian PXRD
target; **SSL allows us to experiment with novel ideas.**
"""
    )

with st.expander("When to override the default with a different acquisition"):
    st.markdown(
        """
- **EI** - only when the regression surrogate has been verified to be
  well-calibrated and roughly Gaussian over the candidate pool (rare
  for ordinal PXRD; useful as a comparison baseline on BO Tools).
- **Thompson sampling** - gives batches with implicit diversity by
  sampling from the tree ensemble, at the cost of noisier per-pick
  selection. Useful when you want to deliberately broaden the search.
- **Consensus (LFBO ∩ EI)** - only picks candidates that both methods
  agree are worth running. Aggressive and conservative; useful mainly
  to validate that two methods are pointing at the same chemistry.
- **Random** - for sanity-check baselines only; use if you want random
suggestions for some reason.

For real lab use in this project, the default chain is:
**Frank-Hall ordinal surrogate (RF or XGB) → LFBO-SSL → diverse-greedy
batch selection.**
"""
    )

st.divider()


# -- Section 4: How to choose a surrogate (precision-first) ---
st.subheader("How to choose a surrogate")

st.markdown(
    """
A common instinct is to pick whichever model has the highest overall
accuracy or QWK. For BO, that is the wrong train of thought.
"""
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
#### What BO actually needs

We are not trying to classify **every** experiment correctly. We are
trying to rank a candidate pool and implement the **top** few at the bench. 
 The downstream decision is binary: *"is this candidate worth
running?"*  Only a handful of candidates per round can be run.

The asymmetry that follows is:

- **False positives are expensive.** Synthesizing a non-crystalline wastes a significant amount of time and resources. 
- **False negatives are cheap.** Missing one crystalline candidate is
  fine. There are many crystalline candidates in the synthesis-space and the next BO round will find more (probably).

So the right primary criterion is **precision at the top of the
ranking**, not overall recall or accuracy.
"""
    )

with c2:
    st.markdown(
        """
#### What to actually look at

When comparing two candidate surrogates, look at:

1. **Precision-Recall (PR) curve, top-left corner** - at the
   precisions we care about (say, ≥ 0.7 or .8 for the Crystalline class),
   which model maintains the highest recall? That is, which model retains a high prediction accuracy, while still being able to find crystalline candidates. A model that hits 80%
   precision at 30% recall is great for BO; it gives us a small but
   reliable pool of suggestions.
2. **Confusion matrix, "Crystalline" row** - what fraction of true
   Crystalline experiments is the model correctly placing in the
   Crystalline column? (Recall.) And of everything *predicted*
   Crystalline, what fraction is actually Crystalline? (Precision.)
3. **ROC AUC** is useful as a sanity check, but it weights both classes
   equally. This should be disregarded when making a decision.
4. **QWK** - still a good summary metric and the one Optuna tunes
   against, but use it as a **tie-breaker**, not the primary criterion.
   Two models with similar QWK can have very different PR curves.
5. **Calibration** - if the model says "80% confident this is
   Crystalline" but is only right 50% of the time at that probability
   bin, the acquisition function will be misled. Check the calibration
   plots on the *Model Confidence* page.
"""
    )

st.info(
    "**Rule of thumb for COMPASS:** prefer the model whose PR curve "
    "sits highest in the **high-precision (left) region**, even if its "
    "ROC AUC or overall accuracy is slightly lower. We can afford to "
    "miss crystalline candidates; we cannot afford to waste experiments.",
)

# Embed the ROC/PR comparison plot if present.
_roc_path = os.path.join(_PROJECT_ROOT, "roc_prc_comparison.png")
if os.path.exists(_roc_path):
    st.image(_roc_path,
             caption=("ROC and PR curves for all six trained pipelines. "
                      "For our use case, focus on the PR plot - and "
                      "specifically the region where precision ≥ 0.7."),
             width="stretch")

_cm_path = os.path.join(_PROJECT_ROOT, "confusion_matrices_normalized.png")
if os.path.exists(_cm_path):
    st.image(_cm_path,
             caption=("Normalized confusion matrices. Read the bottom-right "
                      "cell (Crystalline → Crystalline) as the model's "
                      "recall on Crystalline; read the rightmost column "
                      "totals (everything predicted Crystalline) as the "
                      "support for its precision."),
             width="stretch")

st.divider()


# -- Section 4: Reading the learning curves ---
st.subheader("Learning Curve")

st.markdown(
    """
A learning curve plots model performance (here, **CV-QWK**) against
**training-set size**. We retrain each pipeline on progressively
larger subsets of the data and re-evaluate. Each curve answers a
different question about *that specific surrogate*.
"""
)

st.markdown(
    """
#### How to Interpret Learning Curves

- If the validation curve (vc) (orange) is still rising at the right edge → the model is **data-limited**.
  More experiments will improve it. This is a good sign for early
  stages of the project - there's room to grow.
- If the vc plateaus at the right edge → the model has **extracted what
  it can** from this feature set. More of the same kind of data will
  not help; only new feature engineering, a different model class, or
  experiments in genuinely new chemistry-space will move the needle.
- If the training curve (blue) is far above the lc → the model is
  **overfitting**. It's memorizing training rows. This means you need to reduce model
  capacity (smaller forests, shallower trees, stronger regularization)
  or tighten the MI budget.
- **Both train and validation curves low and close together** →
  **underfitting**. The model isn't expressive enough for the signal.
  Increase capacity, add a CL embedding, or expand the feature
  budget.
- **Wide error bands on the validation curve** → the **CV split is
  unstable**. Likely caused by very few minority-class samples in some
  folds. Look at whether the cluster-group count is too high.

#### Surrogate-specific things to watch for

- **RF variants** plateau earlier than XGB variants - random forests
  are bias-limited on small datasets but rarely overfit. If RF is
  already flat and XGB is still rising at our current dataset size,
  XGB is the better long-term bet.
- **CL-only variants** (no MI features) are sensitive to how well the
  contrastive encoder generalizes. A CL-only curve that lags the
  CL+MI curve is a sign the encoder isn't carrying enough information
  on its own.
- **MI-only variants** are the most interpretable baseline. If they
  match or beat the CL variants, prefer them - fewer moving parts,
  easier to debug, easier to SHAP.

#### Rule of Thumb
- Look for the plot that has a steady increase in validation (orange) 
  while maintaining a low training curve (blue) from left to right. 
  Low training curve is relative, but the gap between the two curves should be minimized.
"""
)

# Embed the learning curve image if present.
_lc_path = os.path.join(_PROJECT_ROOT, "learning_curves_qwk.png")
if os.path.exists(_lc_path):
    st.image(_lc_path,
             caption=("Learning curves (CV-QWK vs. training-set size) for "
                      "each of the six pipelines. Look for: still-rising "
                      "vs. plateaued; train-vs-validation gap; error-band "
                      "width."),
             width="stretch")
else:
    st.caption("`learning_curves_qwk.png` not found - generate it by "
               "running the training pipeline from the **Retrain Model** "
               "page.")

st.divider()


# -- Section 5: Quick checklist ---
st.subheader("Checklist")

st.markdown(
    """
Run through this list when deciding which surrogate to deploy for BO:

1. **PR curve at precision ≥ 0.7** - which model has the higher recall
   there?
2. **Confusion matrix, Crystalline column** - how clean is it? Few
   Amorphous-mis-labeled-as-Crystalline = trustworthy positive picks.
3. **Calibration plot** Ensure that acquistion functions are well balanced.
4. **Learning curve** - is the model still improving with data, or
   plateaued? Plateau ≠ bad, but it should inform whether to spend a
   retraining round on this surrogate or invest in feature work.
5. **QWK as a tiebreaker** - only when the four criteria above are
   roughly tied.
6. **SHAP sanity check** - do the top features make chemical sense?
   (See the SHAP outputs on the Model Confidence page.) A surrogate
   whose top driver is, say, a near-constant fingerprint bit is a
   surrogate to distrust regardless of its QWK.
"""
)

st.caption(
    "All plots referenced on this page are regenerated when you run the "
    "training pipeline (Retrain Model page → *Retrain model*). For the "
    "full technical writeup - including ordinal Frank-Hall math, SHAP "
    "computation, and BO acquisition functions - see `README.md` in the "
    "repository root."
)