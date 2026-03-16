Bayesian Optimization Integration Plan for LVMOF-Surrogate

 Context

 The LVMOF-Surrogate pipeline predicts MOF synthesis outcomes (pxrd_score 0-9, binned to 3 ordinal classes) using RF and XGBoost  
 wrapped in Frank-Hall ordinal decomposition. The goal: given fixed chemistry (precursor/linker/modulator), recommend optimal     
 synthesis conditions to maximize crystallinity. Simulation mode with Liang et al.'s AF/EF metrics validates the approach before  
 real use.

 Key data characteristics: 746 experiments. 51% score 0 (amorphous), only 8% score >=7. 22 unique solvent combinations from       
 COSMO-characterized solvents.

 ---
 1. Framework & Surrogate Choice

 Primary BO surrogate: RF/XGBoost regressor on raw 0-9 pxrd_score

 The existing 3-class classifier loses within-class granularity. A regression surrogate on the raw 0-9 score provides:
 - Natural mean mu(x) + variance sigma(x) for standard EI and LCB
 - Finer-grained BORE threshold tau on the 0-9 scale (can distinguish score 6 from 9)
 - RF inter-tree variance works directly for uncertainty

 The existing 3-class Frank-Hall classifier is kept for interpretable recommendation output (P(Crystalline), P(Partial),
 P(Amorphous)).

 Acquisition functions to implement and compare

 1. True BORE — dynamic tau, lightweight classifier relabeling each iteration
 2. Standard EI — EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z), using regression surrogate mean/variance
 3. LCB — UCB(x) = mu(x) + kappa * sigma(x), kappa=2.0 per Liang et al.
 4. PI-ordinal — P(Crystalline | x) from Frank-Hall classifier (static baseline)
 5. Thompson Sampling — sample from RF tree ensemble

 Batch strategies to implement and compare

 1. Constant Liar — hallucinate the current best value (f_best) for selected candidates
 2. Kriging Believer — hallucinate the surrogate's predicted value (mu(x)) for selected candidates

 Why not GP-based BO

 - ~5,500 features after MI selection → GP is O(n^3), infeasible
 - Trees handle mixed feature types natively
 - RF inter-tree variance provides uncertainty without kernel specification

 ---
 2. Search Space

 Fixed inputs (user provides): precursor, linker, modulator SMILES → featurized as context (Morgan FP, ChemBERTa, SOAP, etc.)     

 Optimizable parameters (BO searches over):

 ┌─────────────────────────┬────────────┬─────────────────────────────────────────────────┐
 │        Parameter        │    Type    │               Bounds (from data)                │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┤
 │ equivalents             │ Continuous │ [0, 150]                                        │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┤
 │ temperature_k           │ Continuous │ [298, 393]                                      │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┤
 │ metal_over_linker_ratio │ Continuous │ [0, 4]                                          │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┤
 │ total_conc              │ Continuous │ [2.3, 4222]                                     │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┤
 │ Solvent composition     │ Discrete   │ ~204 compositions (expanded from 22 historical) │
 └─────────────────────────┴────────────┴─────────────────────────────────────────────────┘

 Expanded solvent space via SolventMixer

 8 pure solvents in training data: DCM, Toluene, THF, DMF, Chloroform, Benzene, Acetonitrile, Ethanol.

 Instead of restricting to the 22 historically used combinations, enumerate:
 - Pure solvents: 8 (ratio 1:0)
 - Binary mixtures at ratios 1:1, 1:2, 2:1, 1:3, 3:1, 1:4, 4:1
   - 1:1 (symmetric): C(8,2) = 28 unique
   - {1:2, 2:1} (asymmetric pair): 56 unique
   - {1:3, 3:1}: 56 unique
   - {1:4, 4:1}: 56 unique
 - Total: 8 + 28 + 56×3 = 204 unique solvent compositions

 COSMO properties for novel mixtures computed by linear interpolation of pure-component sigma profiles by mole fraction
 (Vcosmo-based proxy), using the existing cosmo_features.py infrastructure.

 SolventMixer class (in bo_core.py):
 class SolventMixer:
     """Compute COSMO property vectors for any binary solvent mixture.

     Reuses cosmo_features.py functions:
       - load_cosmo_index() → Vcosmo, lnPvap per solvent
       - load_sigma_profile() → 51-point sigma profile per solvent
       - compute_sigma_moments() → all COSMO moments from mixed profile
       - _add_mole_fractions() → Vcosmo-based mole fraction proxy

     For a mixture (solvent_A, solvent_B, ratio_A:ratio_B):
       1. Volume fractions from ratio
       2. Mole fractions via Vcosmo proxy
       3. mix_area = frac_A * area_A + frac_B * area_B (linear interpolation)
       4. compute_sigma_moments(sigma_axis, mix_area) → 13 COSMO features
     """

     def __init__(self, index_path, cosmo_folder):
         # Load index + cache all 8 pure solvent profiles at init
         ...

     def get_cosmo_vector(self, solvent_a, solvent_b, ratio) -> dict:
         """Return full COSMO property dict for a binary mixture."""
         ...

     def enumerate_all(self) -> list[dict]:
         """Generate all ~204 compositions with precomputed COSMO vectors."""
         ...

 Each solvent composition determines: Mix_M0_Area, Mix_M2_Polarity, Mix_M3_Asymmetry, Mix_M_HB_Acc, Mix_M_HB_Don,
 Mix_M1_NetCharge, Mix_M4_Kurtosis, Mix_f_nonpolar, Mix_f_acc, Mix_f_don, Mix_sigma_std, Mix_Vcosmo, Mix_lnPvap, plus
 solvent_1_fraction, solvent_2_fraction (from ratio).

 Total search space: 4 continuous + 1 discrete (204 categories) → ~204,000 candidates with 1000 LHS samples per composition.      

 Bound inference

 Scan training data for (min, max) per continuous param. SolventMixer computes COSMO vectors for all 204 compositions at init.    

 ---
 3. Acquisition Function Details

 3.1 Standard EI (Expected Improvement)

 Using the regression surrogate's mean mu(x) and std sigma(x):
 Z = (mu(x) - f_best - xi) / sigma(x)
 EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
 Where f_best = best observed pxrd_score so far, xi = exploration jitter (default 0.01).

 RF uncertainty: inter-tree variance. For B trees predicting y_b(x):
 mu(x) = (1/B) * sum(y_b(x))
 sigma^2(x) = (1/B) * sum((y_b(x) - mu(x))^2)
 Per SMAC (Hutter et al.), this is the standard approach for RF-based BO.

 XGBoost uncertainty: bootstrap ensemble (M=10 models on bootstrap samples). Trees in boosting are correlated, so inter-tree      
 variance is invalid (Hyperboost, Danjou et al.).

 3.2 True BORE (Dynamic-Threshold Classification)

 Each BO iteration:
 1. Compute tau = quantile(y_observed, 1 - gamma), gamma=0.25
 2. Label: z_i = I[y_i >= tau] — "good" if pxrd_score above current threshold
 3. Train lightweight binary RF/XGBoost on {(x_i, z_i)}
 4. Acquisition: alpha(x) = P(z=1 | x) = bore_clf.predict_proba(X)[:, 1]

 Edge cases:
 - If all z_i are same class (all "bad" or all "good"), fall back to EI for that iteration
 - tau naturally rises as better experiments are found, focusing the search

 Key distinction from PI-ordinal: BORE's tau shifts dynamically. Early iterations may set tau at score 2 (any non-amorphous is    
 "good"). Later, tau rises to score 7+ (only the best crystalline is "good"). PI-ordinal uses a fixed class boundary.

 3.3 LCB (Lower Confidence Bound)

 UCB(x) = mu(x) + kappa * sigma(x)    [kappa=2.0 per Liang et al.]
 For maximization. Uses regression surrogate mean/variance directly.

 3.4 PI-Ordinal (Baseline)

 alpha(x) = P(y > 1 | x)   from Frank-Hall threshold-1 classifier
 Static threshold, no retraining. Baseline for ablation.

 3.5 Thompson Sampling

 Sample one tree from RF ensemble, use its prediction as acquisition value.

 ---
 4. Batch Selection Strategies

 Both implemented and compared in simulation:

 Constant Liar

 1. Select best candidate by acquisition
 2. Add to training set with hallucinated outcome y = f_best (current best observed)
 3. Refit surrogate (or update BORE labels)
 4. Select next best, repeat for batch_size

 Kriging Believer

 1. Select best candidate by acquisition
 2. Add to training set with hallucinated outcome y = mu(x) (surrogate prediction)
 3. Refit, repeat for batch_size

 Constant Liar is more pessimistic (assumes new point matches current best → encourages diversity). Kriging Believer is more      
 optimistic (trusts the surrogate prediction).

 ---
 5. Two Operating Modes

 Mode A: Simulation/Benchmarking (build first)

 1. Split data: 40% initial training (stratified by pxrd_score) + 60% oracle pool
 2. Diagnostic: if initial set has <3 examples with score >=7, log warning — may need extra random exploration early on
 3. Sequential BO: select from pool → query oracle (get real 0-9 score) → add to training → refit → repeat
 4. Liang et al. metrics: AF, EF, Top-5% discovery rate
 5. Full ablation: {BORE, EI, LCB, PI-ordinal, Thompson, Random} × {RF_reg, XGB_reg} × {constant_liar, kriging_believer} ×        
 multiple seeds
 6. Configurable surrogate variant for feature pipeline (all 6 existing variants available via CLI)

 Mode B: Recommendation

 1. Fit regression surrogate + Frank-Hall classifier on all training data
 2. User provides fixed chemistry SMILES
 3. Generate candidates: LHS over 4 continuous params × 204 solvent compositions (via SolventMixer)
 4. Score with acquisition function + compute P(Crystalline/Partial/Amorphous)
 5. Output ranked CSV: [equivalents, temperature_k, metal_over_linker_ratio, total_conc, solvent_combo, pxrd_predicted,
 uncertainty, P_crystalline, P_partial, P_amorphous, acquisition_value]

 ---
 6. New Modules

 bo_core.py (~500-600 lines)

 SolventMixer
   - Reuses cosmo_features.py: load_cosmo_index(), load_sigma_profile(), compute_sigma_moments()
   - Caches all 8 pure solvent sigma profiles at init
   - get_cosmo_vector(solvent_a, solvent_b, ratio) → COSMO property dict
   - enumerate_all() → all ~204 compositions with precomputed COSMO vectors

 SearchSpace
   - Infers continuous bounds from training data
   - Uses SolventMixer to enumerate expanded solvent space (~204 compositions)
   - Generates candidates: LHS (4 continuous) × 204 solvent compositions

 OrdinalBOObjective
   - Uses raw 0-9 pxrd_score as objective
   - BORE label generation with dynamic tau

 RegressionSurrogate
   - Wraps RF/XGBoost regressor + existing ImbPipeline transforms
   - fit(X, y_raw_0_9): trains regressor on raw scores
   - predict(X): returns (mu, sigma) via inter-tree variance (RF) or bootstrap (XGB)
   - Reuses pipeline transforms (impute → vt → [cl] → mi) from existing factories

 BOREAcquisition
   - Dynamic tau, lightweight binary classifier, relabeling each iteration
   - Edge-case fallback to EI when all labels are same class

 EIAcquisition
   - Standard Expected Improvement using regression surrogate mean/variance
   - Configurable jitter xi

 LCBAcquisition
   - UCB = mu + kappa * sigma from regression surrogate

 PIordinalAcquisition (baseline)
   - P(Crystalline) from existing Frank-Hall classifier

 ThompsonSamplingAcquisition
   - Random tree from RF regressor ensemble

 XGBoostBootstrapEnsemble
   - M=10 bootstrap XGBoost regressors for uncertainty when XGB is the BO surrogate

 BatchSelector
   - constant_liar(surrogate, candidates, batch_size, f_best)
   - kriging_believer(surrogate, candidates, batch_size)

 CandidateFeaturizer
   - Fixed chemistry SMILES → featurized once via assemble_features()
   - Process param candidates → fill into DataFrame template
   - Solvent combo → COSMO property lookup
   - Produces full feature matrix for scoring

 BOLoop
   - run_simulation(): sequential BO with oracle, 40% stratified init, convergence tracking
   - run_recommend(): one-shot ranking for fixed chemistry
   - run_batch(): batch BO with constant_liar or kriging_believer
   - Configurable surrogate variant, acquisition, batch strategy
   - Loads Optuna-tuned hyperparameters from checkpoints/best_params.pkl

 BOCheckpointer
   - Save/load BO state (iteration, history, selected indices)
   - joblib, matches existing checkpoint pattern

 bo_metrics.py (~300-400 lines)

 SimulationMetrics
   - Acceleration Factor (AF): fraction of top-k found / fraction evaluated
   - Enhancement Factor (EF): quality of found experiments vs random
   - Top%: fraction of top-5% (by pxrd_score) discovered vs iterations
   - Cumulative best: best observed pxrd_score vs iteration number
   - Track: if initial set has <3 examples with score >=7, log warning

 plot_convergence(): cumulative best vs iteration, multiple seeds/methods
 plot_topk_curves(): Top% vs evaluations for multiple acquisitions
 plot_af_ef_comparison(): AF/EF bar charts across all ablation conditions
 plot_batch_comparison(): constant_liar vs kriging_believer performance
 save_simulation_results(): CSV export of full history

 ---
 7. Modifications to Existing Files

 config.py — Add BO section (~25 lines)

 BO_DEFAULT_ACQUISITION = "bore"
 BO_BORE_GAMMA = 0.25
 BO_LCB_KAPPA = 2.0
 BO_EI_XI = 0.01
 BO_N_ITERATIONS = 50
 BO_BATCH_SIZE = 3
 BO_INIT_FRACTION = 0.40
 BO_CHECKPOINT_DIR = "checkpoints/bo"
 BO_EPSILON_GREEDY = 0.1
 BO_N_LHS_SAMPLES = 1000
 BO_DEFAULT_SURROGATE = "rf_cl_mi"
 BO_BOOTSTRAP_M = 10

 # Controllable process parameters for BO
 BO_CONTROLLABLE_PARAMS = {
     "equivalents": (0.0, 150.0),
     "temperature_k": (298.0, 393.0),
     "metal_over_linker_ratio": (0.0, 4.0),
     "total_conc": (2.3, 4222.0),
 }

 models.py — Add regression surrogate + uncertainty methods (~80 lines)

 - predict_proba_per_threshold(X) → dict of P(y>k|x) per threshold
 - predict_proba_with_uncertainty(X, n_samples=200) → (mean, std) via tree sampling
 - make_rf_regressor_pipe(rf_params, with_cl=False) → ImbPipeline with RF regressor (no SMOTE, since regression)
 - make_xgb_regressor_pipe(xgb_params, with_cl=False) → same for XGBoost

 Note: Regression pipelines skip SMOTE (SMOTE is for classification). Steps: impute → vt → [cl] → mi → RF/XGB Regressor.

 main.py — Add BO entry point (~120 lines)

 - CLI: --bo, --bo-mode {simulate,recommend,batch}, --bo-surrogate {rf_reg,xgb_reg,rf,xgb,...}, --bo-acquisition
 {bore,ei,lcb,pi_ordinal,thompson,random}, --bo-batch-strategy {constant_liar,kriging_believer}, --bo-iterations, --bo-precursor, 
  --bo-linker, --bo-modulator
 - run_bo(): load checkpoints, instantiate components, run selected mode
 - run_bo_ablation(): full grid over surrogates × acquisitions × batch strategies × seeds

 No changes to: featurization.py, feature_assembly.py, data_processing.py, dimensionality.py, pipeline.py, evaluation.py

 ---
 8. Implementation Sequence

 Phase 1: Core Infrastructure (bo_core.py + config.py + models.py)

 1. Add BO constants + controllable param bounds to config.py
 2. Add predict_proba_per_threshold(), predict_proba_with_uncertainty() to FrankHallOrdinalClassifier
 3. Add make_rf_regressor_pipe(), make_xgb_regressor_pipe() to models.py (skip SMOTE, use regressors)
 4. Implement SolventMixer — reuses cosmo_features.py functions, caches 8 pure profiles, enumerates ~204 compositions
 5. Implement SearchSpace — bound inference, uses SolventMixer for expanded solvent space, LHS candidate generation
 6. Implement RegressionSurrogate — wraps regressor + pipeline transforms, exposes (mu, sigma)
 7. Implement OrdinalBOObjective — raw 0-9 objective, BORE label generation with dynamic tau
 8. Implement EIAcquisition — standard EI using regression surrogate mean/variance
 9. Implement BOREAcquisition — true dynamic tau, lightweight classifier, edge-case fallback
 10. Implement LCBAcquisition, PIordinalAcquisition, ThompsonSamplingAcquisition
 11. Implement XGBoostBootstrapEnsemble (M=10 models)
 12. Implement BatchSelector — constant_liar + kriging_believer

 Phase 2: Simulation Mode (bo_core.py + bo_metrics.py + main.py)

 12. Implement BOLoop.run_simulation() — 40% stratified init, sequential BO with oracle, <3 high-score diagnostic
 13. Implement SimulationMetrics in bo_metrics.py — AF, EF, Top%, convergence
 14. Implement simulation + ablation entry points in main.py
 15. Run ablation: {BORE, EI, LCB, PI-ordinal, Thompson, Random} × {RF_reg, XGB_reg} × {constant_liar, kriging_believer}

 Phase 3: Recommendation Mode

 16. Implement CandidateFeaturizer — chemistry SMILES + process params → full feature vectors
 17. Implement BOLoop.run_recommend() — one-shot ranking with regression + classification surrogates
 18. Implement recommendation CLI in main.py

 Phase 4: Polish

 19. BOCheckpointer for state persistence
 20. Comparison plots in bo_metrics.py

 ---
 9. Key Implementation Details

 SMOTE bypass for candidate scoring

 Reuse pattern from evaluation.py:397-410:
 def transform_for_prediction(fitted_pipe, X):
     Xt = X
     for name, step in fitted_pipe.steps:
         if name in ("smote", "ordinal_xgb", "ordinal_rf"):
             break
         Xt = step.transform(Xt)
     return Xt

 Regression pipeline (no SMOTE)

 For the BO regression surrogate, the pipeline is:
 impute → vt → [cl] → mi → RF/XGB Regressor
 No SMOTE step. SMOTE is classification-specific. The regressor trains on the raw 0-9 scores.

 RF uncertainty for regression surrogate

 def rf_regression_uncertainty(rf_regressor, X):
     predictions = np.array([tree.predict(X) for tree in rf_regressor.estimators_])
     mu = predictions.mean(axis=0)
     sigma = predictions.std(axis=0)
     return mu, sigma

 SolventMixer using cosmo_features.py

 from cosmo_features import (load_cosmo_index, load_sigma_profile,
                             compute_sigma_moments, _add_mole_fractions)

 class SolventMixer:
     RATIOS = [(1,0), (1,1), (1,2), (2,1), (1,3), (3,1), (1,4), (4,1)]

     def __init__(self, index_path, cosmo_folder):
         self.index_map, self.bp_map, self.vcosmo_map, self.lnpvap_map = load_cosmo_index(index_path)
         self.profiles = {}  # Cache sigma profiles for all 8 solvents
         self.pure_solvents = [...]  # 8 solvents from training data
         for name in self.pure_solvents:
             idx = self.index_map[name]
             self.profiles[name] = load_sigma_profile(idx, cosmo_folder)

     def get_cosmo_vector(self, solvent_a, solvent_b=None, ratio=(1,0)):
         """Compute COSMO features for a mixture via sigma profile interpolation."""
         if solvent_b is None or ratio == (1,0):
             # Pure solvent
             prof = self.profiles[solvent_a]
             moments = compute_sigma_moments(prof['sigma'].values, prof['area'].values)
             moments['Mix_Vcosmo'] = self.vcosmo_map[solvent_a]
             moments['Mix_lnPvap'] = self.lnpvap_map[solvent_a]
             return moments

         # Binary mixture: linear interpolation of sigma profiles by mole fraction
         vol_a, vol_b = ratio
         total = vol_a + vol_b
         # Mole fractions via Vcosmo proxy
         proxy_a = (vol_a / total) / self.vcosmo_map[solvent_a]
         proxy_b = (vol_b / total) / self.vcosmo_map[solvent_b]
         frac_a = proxy_a / (proxy_a + proxy_b)
         frac_b = 1.0 - frac_a

         mix_area = (frac_a * self.profiles[solvent_a]['area'].values
                    + frac_b * self.profiles[solvent_b]['area'].values)
         sigma_axis = self.profiles[solvent_a]['sigma'].values

         moments = compute_sigma_moments(sigma_axis, mix_area)
         moments['Mix_Vcosmo'] = frac_a * self.vcosmo_map[solvent_a] + frac_b * self.vcosmo_map[solvent_b]
         moments['Mix_lnPvap'] = frac_a * self.lnpvap_map[solvent_a] + frac_b * self.lnpvap_map[solvent_b]
         return moments

     def enumerate_all(self):
         """Generate all ~204 compositions with COSMO vectors."""
         compositions = []
         for solvent in self.pure_solvents:
             compositions.append(self.get_cosmo_vector(solvent))
         for a, b in combinations(self.pure_solvents, 2):
             for ratio in self.RATIOS[1:]:  # skip pure
                 compositions.append(self.get_cosmo_vector(a, b, ratio))
         return compositions  # ~204 dicts

 BORE iteration lifecycle

 # Each iteration:
 tau = np.quantile(y_observed, 1 - gamma)  # e.g., gamma=0.25 → top 25%
 z = (y_observed >= tau).astype(int)
 if z.sum() == 0 or z.sum() == len(z):
     # Degenerate case: fall back to EI
     acq_values = ei_acquisition.score(X_candidates)
 else:
     bore_clf.fit(X_observed, z)
     acq_values = bore_clf.predict_proba(X_candidates)[:, 1]

 Simulation diagnostic

 high_score_count = (y_init >= 7).sum()
 if high_score_count < 3:
     warnings.warn(f"Initial set has only {high_score_count} examples with score >= 7. "
                   "Surrogate may lack signal for high-quality outcomes. "
                   "Consider extra random exploration in early iterations.")

 ---
 10. Verification Plan

 Simulation Validation (Phase 2)

 1. Load training data (746 experiments, raw 0-9 pxrd_score)
 2. Stratified 40% init → ~298 training, ~448 oracle pool
 3. Run 50 iterations of sequential BO with each acquisition function
 4. Success criteria:
   - AF > 2.0 (find top experiments 2x faster than random)
   - Top-5% discovery: >50% of top-scoring experiments found within 30 iterations
   - BORE and EI should outperform random baseline
   - At least one method should outperform PI-ordinal baseline
 5. Compare constant_liar vs kriging_believer for batch_size=3

 End-to-End CLI

 # Full simulation ablation
 python main.py --bo --bo-mode simulate --bo-iterations 50

 # Single simulation run
 python main.py --bo --bo-mode simulate --bo-acquisition bore --bo-surrogate rf_reg --bo-iterations 50

 # Batch simulation
 python main.py --bo --bo-mode batch --bo-batch-strategy constant_liar --bo-batch-size 3

 # Recommendation for specific chemistry
 python main.py --bo --bo-mode recommend \
   --bo-precursor "SMILES" --bo-linker "SMILES" --bo-modulator "SMILES" \
   --bo-acquisition ei --bo-surrogate rf_reg

 ---
 11. Critical Files Reference

 ┌─────────────────────┬────────────────────────────────────────────────────────────────────────────┬────────────────────────┐    
 │        File         │                                    Role                                    │       Key Lines        │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ cosmo_features.py   │ SolventMixer reuses: load_cosmo_index(), load_sigma_profile(),             │ 84-103, 106-124,       │    
 │                     │ compute_sigma_moments(), _add_mole_fractions()                             │ 129-180, 218-252       │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ models.py           │ FrankHallOrdinalClassifier, pipeline factories, add regressor pipes        │ 43-81, 587-652         │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ evaluation.py       │ transform_until_before_smote() pattern                                     │ 397-410                │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ config.py           │ PROCESS_COLS, add BO constants + controllable param bounds                 │ 64-90                  │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ main.py             │ Entry point, checkpoint loading                                            │ full file              │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ feature_assembly.py │ assemble_features() for candidate featurization                            │ called from main       │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ dimensionality.py   │ prepare_labels() extracts raw 0-9 pxrd_score; remap_score()                │ 66-100                 │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ pipeline.py         │ Optuna-tuned params to reuse as starting points                            │ 301-369                │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ data_processing.py  │ Solvent columns, COSMO-derived features                                    │ 171-177, 300-310,      │    
 │                     │                                                                            │ 408-413                │    
 └─────────────────────┴────────────────────────────────────────────────────────────────────────────┴────────────────────────┘    

 12. Paper References

 ┌─────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────────┐     
 │    Component    │                   Paper                   │                         Key Detail                         │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ BORE algorithm  │ Tiao et al. (tiao21a.pdf)                 │ Dynamic tau = gamma-quantile; retrain classifier each      │     
 │                 │                                           │ iteration                                                  │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ Standard EI     │ references/oc3c01087_si_001               │ EI implementation details for MOF optimization             │     
 │                 │ (1)-101-102.pdf                           │                                                            │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ RF uncertainty  │ SMAC (10-TR-SMAC.pdf)                     │ Inter-tree variance for RF surrogates                      │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ LCB kappa=2.0   │ Liang et al. (s41524-021-00656-9.pdf)     │ LCB_2 outperforms EI in materials science                  │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ AF/EF metrics   │ Liang et al. (s41524-021-00656-9.pdf)     │ Acceleration factor, enhancement factor, Top%              │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ XGB uncertainty │ Hyperboost (2206.03301v2.pdf)             │ Bootstrap ensemble M=10 (inter-tree variance invalid)      │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ Constant Liar   │ Xie et al. (accelerate-synthesis...pdf)   │ Batch BO for MOF synthesis, batch_size=3                   │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ COSMO           │ Training data                             │ Solvent mixture descriptors from COSMO quantum-mechanical  │     
 │ properties      │                                           │ model                                                      │     
 └─────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────────┘     
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ Bayesian Optimization Integration Plan for LVMOF-Surrogate

 Context

 The LVMOF-Surrogate pipeline predicts MOF synthesis outcomes (pxrd_score 0-9, binned to 3 ordinal classes) using RF and XGBoost  
 wrapped in Frank-Hall ordinal decomposition. The goal: given fixed chemistry (precursor/linker/modulator), recommend optimal     
 synthesis conditions to maximize crystallinity. Simulation mode with Liang et al.'s AF/EF metrics validates the approach before  
 real use.

 Key data characteristics: 746 experiments. 51% score 0 (amorphous), only 8% score >=7. 22 unique solvent combinations from       
 COSMO-characterized solvents.

 ---
 1. Framework & Surrogate Choice

 Primary BO surrogate: RF/XGBoost regressor on raw 0-9 pxrd_score

 The existing 3-class classifier loses within-class granularity. A regression surrogate on the raw 0-9 score provides:
 - Natural mean mu(x) + variance sigma(x) for standard EI and LCB
 - Finer-grained BORE threshold tau on the 0-9 scale (can distinguish score 6 from 9)
 - RF inter-tree variance works directly for uncertainty

 The existing 3-class Frank-Hall classifier is kept for interpretable recommendation output (P(Crystalline), P(Partial),
 P(Amorphous)).

 Acquisition functions to implement and compare

 1. True BORE — dynamic tau, lightweight classifier relabeling each iteration
 2. Standard EI — EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z), using regression surrogate mean/variance
 3. LCB — UCB(x) = mu(x) + kappa * sigma(x), kappa=2.0 per Liang et al.
 4. PI-ordinal — P(Crystalline | x) from Frank-Hall classifier (static baseline)
 5. Thompson Sampling — sample from RF tree ensemble

 Batch strategies to implement and compare

 1. Constant Liar — hallucinate the current best value (f_best) for selected candidates
 2. Kriging Believer — hallucinate the surrogate's predicted value (mu(x)) for selected candidates

 Why not GP-based BO

 - ~5,500 features after MI selection → GP is O(n^3), infeasible
 - Trees handle mixed feature types natively
 - RF inter-tree variance provides uncertainty without kernel specification

 ---
 2. Search Space

 Fixed inputs (user provides): precursor, linker, modulator SMILES → featurized as context (Morgan FP, ChemBERTa, SOAP, etc.)     

 Optimizable parameters (BO searches over):

 ┌─────────────────────────┬────────────┬─────────────────────────────────────────────────┬───────────────┐
 │        Parameter        │    Type    │                     Bounds                      │   Sampling    │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┼───────────────┤
 │ equivalents             │ Continuous │ [0, 150]                                        │ Linear LHS    │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┼───────────────┤
 │ temperature_k           │ Continuous │ [298, 393]                                      │ Linear LHS    │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┼───────────────┤
 │ metal_over_linker_ratio │ Continuous │ [0, 4]                                          │ Linear LHS    │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┼───────────────┤
 │ total_conc              │ Continuous │ [p5, p95] of training data                      │ Log-scale LHS │
 ├─────────────────────────┼────────────┼─────────────────────────────────────────────────┼───────────────┤
 │ Solvent composition     │ Discrete   │ ~204 compositions (expanded from 22 historical) │ Enumerate all │
 └─────────────────────────┴────────────┴─────────────────────────────────────────────────┴───────────────┘

 total_conc bound handling

 The raw range [2.3, 4222] spans 1800x — likely driven by outliers or a units issue. Two mitigations:
 1. Percentile clipping: Bounds set to 5th–95th percentile of training data (computed at runtime by SearchSpace), trimming        
 extreme outliers.
 2. Log-scale LHS: Sample log10(total_conc) uniformly within the clipped bounds, then exponentiate back. This ensures candidates  
 spread evenly across orders of magnitude rather than clustering near the upper bound.

 In config.py:
 BO_TOTAL_CONC_CLIP_PERCENTILES = (5, 95)  # Clip bounds to this percentile range
 BO_LOG_SCALE_PARAMS = ["total_conc"]        # Parameters sampled in log-space

 Expanded solvent space via SolventMixer

 8 pure solvents in training data: DCM, Toluene, THF, DMF, Chloroform, Benzene, Acetonitrile, Ethanol.

 Instead of restricting to the 22 historically used combinations, enumerate:
 - Pure solvents: 8 (ratio 1:0)
 - Binary mixtures at ratios 1:1, 1:2, 2:1, 1:3, 3:1, 1:4, 4:1
   - 1:1 (symmetric): C(8,2) = 28 unique
   - {1:2, 2:1} (asymmetric pair): 56 unique
   - {1:3, 3:1}: 56 unique
   - {1:4, 4:1}: 56 unique
 - Total: 8 + 28 + 56×3 = 204 unique solvent compositions

 COSMO properties for novel mixtures computed by linear interpolation of pure-component sigma profiles by mole fraction
 (Vcosmo-based proxy), using the existing cosmo_features.py infrastructure.

 SolventMixer class (in bo_core.py):
 class SolventMixer:
     """Compute COSMO property vectors for any binary solvent mixture.

     Reuses cosmo_features.py functions:
       - load_cosmo_index() → Vcosmo, lnPvap per solvent
       - load_sigma_profile() → 51-point sigma profile per solvent
       - compute_sigma_moments() → all COSMO moments from mixed profile
       - _add_mole_fractions() → Vcosmo-based mole fraction proxy

     For a mixture (solvent_A, solvent_B, ratio_A:ratio_B):
       1. Volume fractions from ratio
       2. Mole fractions via Vcosmo proxy
       3. mix_area = frac_A * area_A + frac_B * area_B (linear interpolation)
       4. compute_sigma_moments(sigma_axis, mix_area) → 13 COSMO features
     """

     def __init__(self, index_path, cosmo_folder):
         # Load index + cache all 8 pure solvent profiles at init
         ...

     def get_cosmo_vector(self, solvent_a, solvent_b, ratio) -> dict:
         """Return full COSMO property dict for a binary mixture."""
         ...

     def enumerate_all(self) -> list[dict]:
         """Generate all ~204 compositions with precomputed COSMO vectors."""
         ...

 Each solvent composition determines: Mix_M0_Area, Mix_M2_Polarity, Mix_M3_Asymmetry, Mix_M_HB_Acc, Mix_M_HB_Don,
 Mix_M1_NetCharge, Mix_M4_Kurtosis, Mix_f_nonpolar, Mix_f_acc, Mix_f_don, Mix_sigma_std, Mix_Vcosmo, Mix_lnPvap, plus
 solvent_1_fraction, solvent_2_fraction (from ratio).

 Total search space: 4 continuous + 1 discrete (204 categories) → ~204,000 candidates with 1000 LHS samples per composition.      

 Bound inference

 - equivalents, temperature_k, metal_over_linker_ratio: (min, max) from training data
 - total_conc: (5th percentile, 95th percentile) from training data, log-transformed for LHS
 - SolventMixer computes COSMO vectors for all 204 compositions at init

 ---
 3. Acquisition Function Details

 3.1 Standard EI (Expected Improvement)

 Using the regression surrogate's mean mu(x) and std sigma(x):
 Z = (mu(x) - f_best - xi) / sigma(x)
 EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
 Where f_best = best observed pxrd_score so far, xi = exploration jitter (default 0.01).

 RF uncertainty: inter-tree variance. For B trees predicting y_b(x):
 mu(x) = (1/B) * sum(y_b(x))
 sigma^2(x) = (1/B) * sum((y_b(x) - mu(x))^2)
 Per SMAC (Hutter et al.), this is the standard approach for RF-based BO.

 XGBoost uncertainty: bootstrap ensemble (M=10 models on bootstrap samples). Trees in boosting are correlated, so inter-tree      
 variance is invalid (Hyperboost, Danjou et al.).

 3.2 True BORE (Dynamic-Threshold Classification)

 Each BO iteration:
 1. Compute tau = quantile(y_observed, 1 - gamma), gamma=0.25
 2. Label: z_i = I[y_i >= tau] — "good" if pxrd_score above current threshold
 3. Train lightweight binary RF/XGBoost on {(x_i, z_i)}
 4. Acquisition: alpha(x) = P(z=1 | x) = bore_clf.predict_proba(X)[:, 1]

 Edge cases:
 - If all z_i are same class (all "bad" or all "good"), fall back to EI for that iteration
 - tau naturally rises as better experiments are found, focusing the search

 Key distinction from PI-ordinal: BORE's tau shifts dynamically. Early iterations may set tau at score 2 (any non-amorphous is    
 "good"). Later, tau rises to score 7+ (only the best crystalline is "good"). PI-ordinal uses a fixed class boundary.

 3.3 LCB (Lower Confidence Bound)

 UCB(x) = mu(x) + kappa * sigma(x)    [kappa=2.0 per Liang et al.]
 For maximization. Uses regression surrogate mean/variance directly.

 3.4 PI-Ordinal (Baseline)

 alpha(x) = P(y > 1 | x)   from Frank-Hall threshold-1 classifier
 Static threshold, no retraining. Baseline for ablation.

 3.5 Thompson Sampling

 Sample one tree from RF ensemble, use its prediction as acquisition value.

 ---
 4. Batch Selection Strategies

 Both implemented and compared in simulation:

 Constant Liar

 1. Select best candidate by acquisition
 2. Add to training set with hallucinated outcome y = f_best (current best observed)
 3. Refit surrogate (or update BORE labels)
 4. Select next best, repeat for batch_size

 Kriging Believer

 1. Select best candidate by acquisition
 2. Add to training set with hallucinated outcome y = mu(x) (surrogate prediction)
 3. Refit, repeat for batch_size

 Constant Liar is more pessimistic (assumes new point matches current best → encourages diversity). Kriging Believer is more      
 optimistic (trusts the surrogate prediction).

 ---
 5. Two Operating Modes

 Mode A: Simulation/Benchmarking (build first)

 1. Split data: 40% initial training (stratified by pxrd_score) + 60% oracle pool
 2. Diagnostic: if initial set has <3 examples with score >=7, log warning — may need extra random exploration early on
 3. Sequential BO: select from pool → query oracle (get real 0-9 score) → add to training → refit → repeat
 4. Liang et al. metrics: AF, EF, Top-5% discovery rate
 5. Full ablation: {BORE, EI, LCB, PI-ordinal, Thompson, Random} × {RF_reg, XGB_reg} × {constant_liar, kriging_believer} ×        
 multiple seeds
 6. Configurable surrogate variant for feature pipeline (all 6 existing variants available via CLI)

 Mode B: Recommendation

 1. Fit regression surrogate + Frank-Hall classifier on all training data
 2. User provides fixed chemistry SMILES
 3. Generate candidates: LHS over 4 continuous params × 204 solvent compositions (via SolventMixer)
 4. Score with acquisition function + compute P(Crystalline/Partial/Amorphous)
 5. Output ranked CSV: [equivalents, temperature_k, metal_over_linker_ratio, total_conc, solvent_combo, pxrd_predicted,
 uncertainty, P_crystalline, P_partial, P_amorphous, acquisition_value]

 ---
 6. New Modules

 bo_core.py (~500-600 lines)

 SolventMixer
   - Reuses cosmo_features.py: load_cosmo_index(), load_sigma_profile(), compute_sigma_moments()
   - Caches all 8 pure solvent sigma profiles at init
   - get_cosmo_vector(solvent_a, solvent_b, ratio) → COSMO property dict
   - enumerate_all() → all ~204 compositions with precomputed COSMO vectors

 SearchSpace
   - Infers continuous bounds from training data (total_conc: 5th-95th percentile clip)
   - Log-scale LHS for total_conc (uniform in log-space, exponentiate back)
   - Uses SolventMixer to enumerate expanded solvent space (~204 compositions)
   - Generates candidates: LHS (4 continuous) × 204 solvent compositions

 OrdinalBOObjective
   - Uses raw 0-9 pxrd_score as objective
   - BORE label generation with dynamic tau

 RegressionSurrogate
   - Wraps RF/XGBoost regressor + existing ImbPipeline transforms
   - fit(X, y_raw_0_9): trains regressor on raw scores
   - predict(X): returns (mu, sigma) via inter-tree variance (RF) or bootstrap (XGB)
   - Reuses pipeline transforms (impute → vt → [cl] → mi) from existing factories

 BOREAcquisition
   - Dynamic tau, lightweight binary classifier, relabeling each iteration
   - Edge-case fallback to EI when all labels are same class

 EIAcquisition
   - Standard Expected Improvement using regression surrogate mean/variance
   - Configurable jitter xi

 LCBAcquisition
   - UCB = mu + kappa * sigma from regression surrogate

 PIordinalAcquisition (baseline)
   - P(Crystalline) from existing Frank-Hall classifier

 ThompsonSamplingAcquisition
   - Random tree from RF regressor ensemble

 XGBoostBootstrapEnsemble
   - M=10 bootstrap XGBoost regressors for uncertainty when XGB is the BO surrogate

 BatchSelector
   - constant_liar(surrogate, candidates, batch_size, f_best)
   - kriging_believer(surrogate, candidates, batch_size)

 CandidateFeaturizer
   - Fixed chemistry SMILES → featurized once via assemble_features()
   - Process param candidates → fill into DataFrame template
   - Solvent combo → COSMO property lookup
   - Produces full feature matrix for scoring

 BOLoop
   - run_simulation(): sequential BO with oracle, 40% stratified init, convergence tracking
   - run_recommend(): one-shot ranking for fixed chemistry
   - run_batch(): batch BO with constant_liar or kriging_believer
   - Configurable surrogate variant, acquisition, batch strategy
   - Loads Optuna-tuned hyperparameters from checkpoints/best_params.pkl

 BOCheckpointer
   - Save/load BO state (iteration, history, selected indices)
   - joblib, matches existing checkpoint pattern

 bo_metrics.py (~300-400 lines)

 SimulationMetrics
   - Acceleration Factor (AF): fraction of top-k found / fraction evaluated
   - Enhancement Factor (EF): quality of found experiments vs random
   - Top%: fraction of top-5% (by pxrd_score) discovered vs iterations
   - Cumulative best: best observed pxrd_score vs iteration number
   - Track: if initial set has <3 examples with score >=7, log warning

 plot_convergence(): cumulative best vs iteration, multiple seeds/methods
 plot_topk_curves(): Top% vs evaluations for multiple acquisitions
 plot_af_ef_comparison(): AF/EF bar charts across all ablation conditions
 plot_batch_comparison(): constant_liar vs kriging_believer performance
 save_simulation_results(): CSV export of full history

 ---
 7. Modifications to Existing Files

 config.py — Add BO section (~30 lines)

 BO_DEFAULT_ACQUISITION = "bore"
 BO_BORE_GAMMA = 0.25
 BO_LCB_KAPPA = 2.0
 BO_EI_XI = 0.01
 BO_N_ITERATIONS = 50
 BO_BATCH_SIZE = 3
 BO_INIT_FRACTION = 0.40
 BO_CHECKPOINT_DIR = "checkpoints/bo"
 BO_EPSILON_GREEDY = 0.1
 BO_N_LHS_SAMPLES = 1000
 BO_DEFAULT_SURROGATE = "rf_cl_mi"
 BO_BOOTSTRAP_M = 10

 # Controllable process parameters for BO (bounds for total_conc are placeholders —
 # actual bounds computed at runtime from 5th-95th percentile of training data)
 BO_CONTROLLABLE_PARAMS = {
     "equivalents": (0.0, 150.0),
     "temperature_k": (298.0, 393.0),
     "metal_over_linker_ratio": (0.0, 4.0),
     "total_conc": None,  # Computed at runtime via percentile clipping
 }
 BO_TOTAL_CONC_CLIP_PERCENTILES = (5, 95)  # Clip bounds to this percentile range
 BO_LOG_SCALE_PARAMS = ["total_conc"]        # Parameters sampled in log-space for LHS

 models.py — Add regression surrogate + uncertainty methods (~80 lines)

 - predict_proba_per_threshold(X) → dict of P(y>k|x) per threshold
 - predict_proba_with_uncertainty(X, n_samples=200) → (mean, std) via tree sampling
 - make_rf_regressor_pipe(rf_params, with_cl=False) → ImbPipeline with RF regressor (no SMOTE, since regression)
 - make_xgb_regressor_pipe(xgb_params, with_cl=False) → same for XGBoost

 Note: Regression pipelines skip SMOTE (SMOTE is for classification). Steps: impute → vt → [cl] → mi → RF/XGB Regressor.

 main.py — Add BO entry point (~120 lines)

 - CLI: --bo, --bo-mode {simulate,recommend,batch}, --bo-surrogate {rf_reg,xgb_reg,rf,xgb,...}, --bo-acquisition
 {bore,ei,lcb,pi_ordinal,thompson,random}, --bo-batch-strategy {constant_liar,kriging_believer}, --bo-iterations, --bo-precursor, 
  --bo-linker, --bo-modulator
 - run_bo(): load checkpoints, instantiate components, run selected mode
 - run_bo_ablation(): full grid over surrogates × acquisitions × batch strategies × seeds

 No changes to: featurization.py, feature_assembly.py, data_processing.py, dimensionality.py, pipeline.py, evaluation.py

 ---
 8. Implementation Sequence

 Phase 1: Core Infrastructure (bo_core.py + config.py + models.py)

 1. Add BO constants + controllable param bounds to config.py
 2. Add predict_proba_per_threshold(), predict_proba_with_uncertainty() to FrankHallOrdinalClassifier
 3. Add make_rf_regressor_pipe(), make_xgb_regressor_pipe() to models.py (skip SMOTE, use regressors)
 4. Implement SolventMixer — reuses cosmo_features.py functions, caches 8 pure profiles, enumerates ~204 compositions
 5. Implement SearchSpace — bound inference, uses SolventMixer for expanded solvent space, LHS candidate generation
 6. Implement RegressionSurrogate — wraps regressor + pipeline transforms, exposes (mu, sigma)
 7. Implement OrdinalBOObjective — raw 0-9 objective, BORE label generation with dynamic tau
 8. Implement EIAcquisition — standard EI using regression surrogate mean/variance
 9. Implement BOREAcquisition — true dynamic tau, lightweight classifier, edge-case fallback
 10. Implement LCBAcquisition, PIordinalAcquisition, ThompsonSamplingAcquisition
 11. Implement XGBoostBootstrapEnsemble (M=10 models)
 12. Implement BatchSelector — constant_liar + kriging_believer

 Phase 2: Simulation Mode (bo_core.py + bo_metrics.py + main.py)

 12. Implement BOLoop.run_simulation() — 40% stratified init, sequential BO with oracle, <3 high-score diagnostic
 13. Implement SimulationMetrics in bo_metrics.py — AF, EF, Top%, convergence
 14. Implement simulation + ablation entry points in main.py
 15. Run ablation: {BORE, EI, LCB, PI-ordinal, Thompson, Random} × {RF_reg, XGB_reg} × {constant_liar, kriging_believer}

 Phase 3: Recommendation Mode

 16. Implement CandidateFeaturizer — chemistry SMILES + process params → full feature vectors
 17. Implement BOLoop.run_recommend() — one-shot ranking with regression + classification surrogates
 18. Implement recommendation CLI in main.py

 Phase 4: Polish

 19. BOCheckpointer for state persistence
 20. Comparison plots in bo_metrics.py

 ---
 9. Key Implementation Details

 SMOTE bypass for candidate scoring

 Reuse pattern from evaluation.py:397-410:
 def transform_for_prediction(fitted_pipe, X):
     Xt = X
     for name, step in fitted_pipe.steps:
         if name in ("smote", "ordinal_xgb", "ordinal_rf"):
             break
         Xt = step.transform(Xt)
     return Xt

 Regression pipeline (no SMOTE)

 For the BO regression surrogate, the pipeline is:
 impute → vt → [cl] → mi → RF/XGB Regressor
 No SMOTE step. SMOTE is classification-specific. The regressor trains on the raw 0-9 scores.

 RF uncertainty for regression surrogate

 def rf_regression_uncertainty(rf_regressor, X):
     predictions = np.array([tree.predict(X) for tree in rf_regressor.estimators_])
     mu = predictions.mean(axis=0)
     sigma = predictions.std(axis=0)
     return mu, sigma

 SolventMixer using cosmo_features.py

 from cosmo_features import (load_cosmo_index, load_sigma_profile,
                             compute_sigma_moments, _add_mole_fractions)

 class SolventMixer:
     RATIOS = [(1,0), (1,1), (1,2), (2,1), (1,3), (3,1), (1,4), (4,1)]

     def __init__(self, index_path, cosmo_folder):
         self.index_map, self.bp_map, self.vcosmo_map, self.lnpvap_map = load_cosmo_index(index_path)
         self.profiles = {}  # Cache sigma profiles for all 8 solvents
         self.pure_solvents = [...]  # 8 solvents from training data
         for name in self.pure_solvents:
             idx = self.index_map[name]
             self.profiles[name] = load_sigma_profile(idx, cosmo_folder)

     def get_cosmo_vector(self, solvent_a, solvent_b=None, ratio=(1,0)):
         """Compute COSMO features for a mixture via sigma profile interpolation."""
         if solvent_b is None or ratio == (1,0):
             # Pure solvent
             prof = self.profiles[solvent_a]
             moments = compute_sigma_moments(prof['sigma'].values, prof['area'].values)
             moments['Mix_Vcosmo'] = self.vcosmo_map[solvent_a]
             moments['Mix_lnPvap'] = self.lnpvap_map[solvent_a]
             return moments

         # Binary mixture: linear interpolation of sigma profiles by mole fraction
         vol_a, vol_b = ratio
         total = vol_a + vol_b
         # Mole fractions via Vcosmo proxy
         proxy_a = (vol_a / total) / self.vcosmo_map[solvent_a]
         proxy_b = (vol_b / total) / self.vcosmo_map[solvent_b]
         frac_a = proxy_a / (proxy_a + proxy_b)
         frac_b = 1.0 - frac_a

         mix_area = (frac_a * self.profiles[solvent_a]['area'].values
                    + frac_b * self.profiles[solvent_b]['area'].values)
         sigma_axis = self.profiles[solvent_a]['sigma'].values

         moments = compute_sigma_moments(sigma_axis, mix_area)
         moments['Mix_Vcosmo'] = frac_a * self.vcosmo_map[solvent_a] + frac_b * self.vcosmo_map[solvent_b]
         moments['Mix_lnPvap'] = frac_a * self.lnpvap_map[solvent_a] + frac_b * self.lnpvap_map[solvent_b]
         return moments

     def enumerate_all(self):
         """Generate all ~204 compositions with COSMO vectors."""
         compositions = []
         for solvent in self.pure_solvents:
             compositions.append(self.get_cosmo_vector(solvent))
         for a, b in combinations(self.pure_solvents, 2):
             for ratio in self.RATIOS[1:]:  # skip pure
                 compositions.append(self.get_cosmo_vector(a, b, ratio))
         return compositions  # ~204 dicts

 SearchSpace log-scale LHS for total_conc

 def generate_lhs_candidates(self, n_samples, train_df):
     bounds = {}
     for param, static_bounds in BO_CONTROLLABLE_PARAMS.items():
         if static_bounds is None:
             # Runtime percentile clipping (total_conc)
             lo, hi = BO_TOTAL_CONC_CLIP_PERCENTILES
             bounds[param] = (np.percentile(train_df[param], lo),
                              np.percentile(train_df[param], hi))
         else:
             bounds[param] = static_bounds

     # LHS in unit hypercube, then scale
     lhs_samples = lhs(len(bounds), samples=n_samples)
     candidates = {}
     for i, (param, (lo, hi)) in enumerate(bounds.items()):
         if param in BO_LOG_SCALE_PARAMS:
             # Uniform in log-space → exponentiate back
             candidates[param] = np.exp(
                 lhs_samples[:, i] * (np.log(hi) - np.log(lo)) + np.log(lo)
             )
         else:
             candidates[param] = lhs_samples[:, i] * (hi - lo) + lo
     return pd.DataFrame(candidates)

 BORE iteration lifecycle

 # Each iteration:
 tau = np.quantile(y_observed, 1 - gamma)  # e.g., gamma=0.25 → top 25%
 z = (y_observed >= tau).astype(int)
 if z.sum() == 0 or z.sum() == len(z):
     # Degenerate case: fall back to EI
     acq_values = ei_acquisition.score(X_candidates)
 else:
     bore_clf.fit(X_observed, z)
     acq_values = bore_clf.predict_proba(X_candidates)[:, 1]

 Simulation diagnostic

 high_score_count = (y_init >= 7).sum()
 if high_score_count < 3:
     warnings.warn(f"Initial set has only {high_score_count} examples with score >= 7. "
                   "Surrogate may lack signal for high-quality outcomes. "
                   "Consider extra random exploration in early iterations.")

 ---
 10. Verification Plan

 Simulation Validation (Phase 2)

 1. Load training data (746 experiments, raw 0-9 pxrd_score)
 2. Stratified 40% init → ~298 training, ~448 oracle pool
 3. Run 50 iterations of sequential BO with each acquisition function
 4. Success criteria:
   - AF > 2.0 (find top experiments 2x faster than random)
   - Top-5% discovery: >50% of top-scoring experiments found within 30 iterations
   - BORE and EI should outperform random baseline
   - At least one method should outperform PI-ordinal baseline
 5. Compare constant_liar vs kriging_believer for batch_size=3

 End-to-End CLI

 # Full simulation ablation
 python main.py --bo --bo-mode simulate --bo-iterations 50

 # Single simulation run
 python main.py --bo --bo-mode simulate --bo-acquisition bore --bo-surrogate rf_reg --bo-iterations 50

 # Batch simulation
 python main.py --bo --bo-mode batch --bo-batch-strategy constant_liar --bo-batch-size 3

 # Recommendation for specific chemistry
 python main.py --bo --bo-mode recommend \
   --bo-precursor "SMILES" --bo-linker "SMILES" --bo-modulator "SMILES" \
   --bo-acquisition ei --bo-surrogate rf_reg

 ---
 11. Critical Files Reference

 ┌─────────────────────┬────────────────────────────────────────────────────────────────────────────┬────────────────────────┐    
 │        File         │                                    Role                                    │       Key Lines        │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ cosmo_features.py   │ SolventMixer reuses: load_cosmo_index(), load_sigma_profile(),             │ 84-103, 106-124,       │    
 │                     │ compute_sigma_moments(), _add_mole_fractions()                             │ 129-180, 218-252       │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ models.py           │ FrankHallOrdinalClassifier, pipeline factories, add regressor pipes        │ 43-81, 587-652         │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ evaluation.py       │ transform_until_before_smote() pattern                                     │ 397-410                │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ config.py           │ PROCESS_COLS, add BO constants + controllable param bounds                 │ 64-90                  │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ main.py             │ Entry point, checkpoint loading                                            │ full file              │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ feature_assembly.py │ assemble_features() for candidate featurization                            │ called from main       │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ dimensionality.py   │ prepare_labels() extracts raw 0-9 pxrd_score; remap_score()                │ 66-100                 │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ pipeline.py         │ Optuna-tuned params to reuse as starting points                            │ 301-369                │    
 ├─────────────────────┼────────────────────────────────────────────────────────────────────────────┼────────────────────────┤    
 │ data_processing.py  │ Solvent columns, COSMO-derived features                                    │ 171-177, 300-310,      │    
 │                     │                                                                            │ 408-413                │    
 └─────────────────────┴────────────────────────────────────────────────────────────────────────────┴────────────────────────┘    

 12. Paper References

 ┌─────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────────┐     
 │    Component    │                   Paper                   │                         Key Detail                         │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ BORE algorithm  │ Tiao et al. (tiao21a.pdf)                 │ Dynamic tau = gamma-quantile; retrain classifier each      │     
 │                 │                                           │ iteration                                                  │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ Standard EI     │ references/oc3c01087_si_001               │ EI implementation details for MOF optimization             │     
 │                 │ (1)-101-102.pdf                           │                                                            │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ RF uncertainty  │ SMAC (10-TR-SMAC.pdf)                     │ Inter-tree variance for RF surrogates                      │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ LCB kappa=2.0   │ Liang et al. (s41524-021-00656-9.pdf)     │ LCB_2 outperforms EI in materials science                  │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ AF/EF metrics   │ Liang et al. (s41524-021-00656-9.pdf)     │ Acceleration factor, enhancement factor, Top%              │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ XGB uncertainty │ Hyperboost (2206.03301v2.pdf)             │ Bootstrap ensemble M=10 (inter-tree variance invalid)      │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ Constant Liar   │ Xie et al. (accelerate-synthesis...pdf)   │ Batch BO for MOF synthesis, batch_size=3                   │     
 ├─────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────┤     
 │ COSMO           │ Training data                             │ Solvent mixture descriptors from COSMO quantum-mechanical  │     
 │ properties      │                                           │ model                                                      │     
 └─────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────────┘ 