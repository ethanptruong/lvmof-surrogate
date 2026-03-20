"""
config.py — All constants and configuration for the LVMOF-Surrogate pipeline.
"""

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_FILE_PATH = "data/Experiments_with_Calculated_Properties_no_linker.xlsx"
TEP_MODEL_URL  = "https://github.com/DanielEss-lab/TEPid/raw/main/Machine_Learning_Model/LGBMReg_model.pkl"

# ── Column mapping ────────────────────────────────────────────────────────────
COLMAP = {
    "id":        "experiment_id",
    "precursor": "smiles_precursor",
    "linker1":   "smiles_linker_1",
    "modulator": "smiles_modulator",
    "linker2":   "smiles_linker_2",
}

# ── Target metals ─────────────────────────────────────────────────────────────
TARGET_METALS = ['Pd', 'Rh', 'Pt', 'Ag', 'Ir', 'Au', 'Cu', 'Co', 'Ni', 'Fe', 'Ru', 'Os']

# ── Oxidation state parsing ────────────────────────────────────────────────────
ROMAN_TO_INT = {'0': 0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}

# ── CBC lookup ────────────────────────────────────────────────────────────────
GROUP11_METALS = {'Cu', 'Ag', 'Au'}

PRECURSOR_CBC = {
    'tetrakis(triphenylphosphine)palladium(0)':                  (4, 0),
    'tetrakis(triphenylphosphine)platinum(0)':                   (4, 0),
    'Cu4I4(PPh3)4':                                              (1, 1),
    'Di-μ-chloro-tetracarbonyldirhodium(I)':                     (2, 1),
    'carbonylchlorobis(triphenylphosphine)rhodium(I)':            (3, 1),
    'carbonylchlorobis(triphenylphosphine)iridium(I)':            (3, 1),
    'carbonylbromobis(triphenylphosphine)iridium(I)':             (3, 1),
    'bromocarbonylbis(triphenylphosphine)iridium(I)':             (3, 1),
    'carbonylchlorodiiodobis(triphenylphosphine)iridium(I)':       (3, 3),
    'carbonylbis(triphenylphosphine)(maleimidato)iridium(I)':       (3, 1),
    'Chlorotris(triphenylphosphine)rhodium(I)':                   (3, 1),
    'Chlorotris(triphenylphosphine)cobalt(I)':                    (3, 1),
    'chloro(tristriphenylphosphine)cobalt(I)':                    (3, 1),
    'chloro(bistriphenylphosphine)gold(I)':                       (2, 1),
    'acetatochlorobis(triphenylphosphine)rhodium(I)':             (2, 2),
    'carbonylpyridinebis(triphenylphosphine)rhodium(I) tetrafluoroborate':       (4, 1),
    'Rh(py)(CO)(PPh3)2]BF4':                                     (4, 1),
    'Rh(MeCN)(CO)(PPh3)2]BF4':                                   (4, 1),
    'copper(I) iodide':                                           (0, 1),
    'copper(I) bromide':                                          (0, 1),
    'silver(I) iodide':                                           (0, 1),
    'silver(I) bromide':                                          (0, 1),
    'gold(I) iodide':                                             (0, 1),
    'silver(I) bromide cluster':                                  (0, 1),
    'silver(I) bromide cyclohexane':                              (1, 1),
    'CuBr cluster':                                               (1, 1),
    'silver(I) iodide cluster':                                   (1, 1),
    'Cu2I2(P2)3':                                                 (3, 1),
    'digold complex':                                             (2, 0),
}

# ── Geometry labels ───────────────────────────────────────────────────────────
GEOMETRY_LABELS = ['linear', 'trigonal_planar', 'square_planar',
                   'tetrahedral', 'trigonal_bipyramidal', 'octahedral', 'unknown']

# ── Process variable column list ──────────────────────────────────────────────
PROCESS_COLS = [
    'equivalents',
    'total_solvent_volume_ml',
    'solvent_1_fraction',
    'solvent_2_fraction',
    'solvent_3_fraction',
    'Min_Boiling_Point_K',
    'Max_Boiling_Point_K',
    'Weighted_Boiling_Point_K',
    'Weighted_AN_mole',
    'Weighted_DN_mole',
    'Weighted_Dielectric_vol',
    'Weighted_Polarity_vol',
    'Weighted_sig_h_vol',
    'Weighted_sig_d_vol',
    'Weighted_sig_p_vol',
    'Mix_M0_Area',
    'Mix_M2_Polarity',
    'Mix_M3_Asymmetry',
    'Mix_M_HB_Acc',
    'Mix_M_HB_Don',
    'temperature_k',
    'metal_over_linker_ratio',
    'reaction_hours',
    'reaction_hours_missing',
    'temperature_k_missing',
    'metal_conc',
    'linker_conc',
    'mod_conc',
    'total_conc',
]

# ── Model / training constants ────────────────────────────────────────────────
SMOTE_STRATEGY = {1: 180, 2: 250}
XGB_FIXED      = {"tree_method": "hist", "eval_metric": "logloss", "random_state": 42}
XGB_TUNED_KEYS = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
}
MI_K              = 100   # discrete feature budget 
MI_K_CONTINUOUS   = 1000   # continuous feature budget 
CL_EMB_DIM        = 64
CL_MARGIN         = 1.0   # TripletMarginLoss margin
CL_NEGATIVE_CLASS = 1     # 1 = partial/hard negatives, 0 = amorphous/easy negatives
RANDOM_STATE      = 42
CV_NJOBS     = 1   

# ── ChemBERTa ────────────────────────────────────────────────────────────────
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
BERT_DIM        = 384

# ── KMeans ────────────────────────────────────────────────────────────────────
N_CLUSTERS = 10

# ── SOAP elements ─────────────────────────────────────────────────────────────
ELEMENT_TO_Z = {
    'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17,
    'Br':35,'I':53,'Si':14,'Ge':32,'Sn':50,'B':5
}
Z_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_Z.items()}
SOAP_SPECIES = list(ELEMENT_TO_Z.keys())

# ── Ion map (shared across cells 14, 15, 16) ──────────────────────────────────
ION_MAP = {
    "Cl-": "[Cl-]", "Br-": "[Br-]", "I-": "[I-]", "F-": "[F-]",
    "Cl":  "[Cl-]", "Br":  "[Br-]", "I":  "[I-]", "F":  "[F-]",
}

# ── Metal center block dimensions (cell 16) ───────────────────────────────────
_COLIGAND_LOOKUP = {
    # CO: strong π-acceptor, weak trans influence relative to H,
    #     but highest field strength ligand known
    'CO':    [0.99,  0.90, 1.00,  0.0,  0.0, 1.0],
    '[CO]':  [0.99,  0.90, 1.00,  0.0,  0.0, 1.0],

    # Halides: σ-donors, weak π-donors, negative EL
    # Trans influence order: I > Br > Cl > F (Appleton 1973)
    '[Cl-]': [-0.24, 0.35, 0.47, -1.0,  1.0, 0.0],
    'Cl':    [-0.24, 0.35, 0.47, -1.0,  1.0, 0.0],
    '[Br-]': [-0.22, 0.45, 0.44, -1.0,  1.0, 0.0],
    'Br':    [-0.22, 0.45, 0.44, -1.0,  1.0, 0.0],
    '[I-]':  [-0.24, 0.55, 0.40, -1.0,  1.0, 0.0],
    'I':     [-0.24, 0.55, 0.40, -1.0,  1.0, 0.0],
    '[F-]':  [-0.40, 0.15, 0.52, -1.0,  1.0, 0.0],
    'F':     [-0.40, 0.15, 0.52, -1.0,  1.0, 0.0],
}

# Feature names matching the 6-vector above — use these in SHAP
_COLIGAND_FEATURE_NAMES = [
    'EL_lever_V',          # Lever electrochemical parameter (Lever 1990)
    'trans_influence',     # Appleton 1973 normalized scale
    'field_strength_norm', # Spectrochemical series, CO=1.0
    'formal_charge',       # IUPAC, unambiguous
    'is_halide',           # binary
    'is_carbonyl',         # binary
]
_COLIGAND_DIM = 6

_METAL_OX_STATE_FALLBACK = {
    'Pd': 0, 'Pt': 0,           # M(0) phosphine complexes
    'Rh': 1, 'Ir': 1,           # M(I) carbonyl/phosphine complexes
    'Ag': 1, 'Au': 1, 'Cu': 1,  # M(I) halide complexes
    'Co': 2, 'Ni': 2,
}

METAL_BLOCK_DIM   = 32
COLIGAND_BLOCK_DIM = 28
COMPLEX_BLOCK_DIM  = 12

# ── G14 hub topology constants ────────────────────────────────────────────────
GROUP14_SYMBOLS = {'Si', 'Ge', 'Sn', 'Pb'}
COORD_SYMBOLS   = {'P', 'N', 'O'}

G14_ENEG   = {'Si': 1.90, 'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33}
G14_COVRAD = {'Si': 111,  'Ge': 122,  'Sn': 139,  'Pb': 146}
G14_PERIOD = {'Si': 3,    'Ge': 4,    'Sn': 5,    'Pb': 6}

G14_HUB_NAMES = [
    'g14hub_present',
    'g14hub_degree',
    'g14hub_nCoordArms',
    'g14hub_nP_arms',
    'g14hub_nN_arms',
    'g14hub_nO_arms',
    'g14hub_armLen_min',
    'g14hub_armLen_max',
    'g14hub_armLen_mean',
    'g14hub_armLen_std',
    'g14hub_armFrac_alkynyl',
    'g14hub_armFrac_aromatic',
    'g14hub_armFrac_alkyl',
    'g14hub_eccentricity',
    'g14hub_centrality',
    'g14hub_elem_eneg',
    'g14hub_elem_covrad_pm',
    'g14hub_elem_period',
    'g14hub_isSi',
    'g14hub_isGe',
    'g14hub_isSn',
    'g14hub_isPb',
    'g14hub_nG14total',
    'g14hub_fracG14',
    'g14hub_missing',
]

# ── G14 SMARTS patterns ───────────────────────────────────────────────────────
G14_SMARTS_RAW = {
    'g14_Si_X4':          '[Si;X4]',
    'g14_Ge_X4':          '[Ge;X4]',
    'g14_Sn_X4':          '[Sn;X4]',
    'g14_Sn_X6':          '[Sn;X6]',
    'g14_P_dist1':         '[Si,Ge,Sn]~[P]',
    'g14_P_dist2':         '[Si,Ge,Sn]~*~[P]',
    'g14_P_dist3':         '[Si,Ge,Sn]~*~*~[P]',
    'g14_P_dist4':         '[Si,Ge,Sn]~*~*~*~[P]',
    'g14_P_dist5':         '[Si,Ge,Sn]~*~*~*~*~[P]',
    'g14_P_dist6':         '[Si,Ge,Sn]~*~*~*~*~*~[P]',
    'g14_arm_alkynyl':     '[Si,Ge,Sn]~[#6]#[#6]',
    'g14_arm_alkynyl_ext': '[Si,Ge,Sn]~*~[#6]#[#6]',
    'g14_arm_aryl':        '[Si,Ge,Sn]~c1ccccc1',
    'g14_arm_vinyl':       '[Si,Ge,Sn]~[#6]=[#6]',
    'g14_arm_alkyl_CH2':   '[Si,Ge,Sn]~[CH2]',
    'g14_Si_alkynyl_P':    '[Si]~[#6]#[#6]~[P]',
    'g14_Si_alkynyl_P2':   '[Si]~*~[#6]#[#6]~[P]',
    'g14_Si_alkynyl_P3':   '[Si]~[#6]#[#6]~*~[P]',
    'g14_Si_aryl_P':       '[Si]~c1ccc([P])cc1',
    'g14_Si_CH2CH2_P':     '[Si]~[CH2]~[CH2]~[P]',
    'g14_Ge_aryl_P':       '[Ge]~c1ccc([P])cc1',
    'g14_Sn_aryl_P':       '[Sn]~c1ccc([P])cc1',
    'g14_ditopic_PP':      '[P]~*~[Si,Ge,Sn](~*~[P])',
    'g14_tritopic_PPP':    '[P]~*~[Si,Ge,Sn](~*~[P])(~*~[P])',
    'g14_tetratopic_PPPP': '[P]~*~[Si,Ge,Sn](~*~[P])(~*~[P])~*~[P]',
    'g14_N_dist2':         '[Si,Ge,Sn]~*~[n,N]',
    'g14_N_dist3':         '[Si,Ge,Sn]~*~*~[n,N]',
    'g14_tetratopic_NNNN': '[n,N]~*~[Si,Ge,Sn](~*~[n,N])(~*~[n,N])~*~[n,N]',
}

ALL_G14_SMARTS_NAMES = list(G14_SMARTS_RAW.keys()) + ['g14_smarts_missing']

# ── TTP constants ──────────────────────────────────────────────────────────────
TTP_DIM = 52

TTP_FEATURE_NAMES = [
    'ttp_hub_present',
    'ttp_hub_isSi',
    'ttp_hub_isGe',
    'ttp_hub_isSn',
    'ttp_hub_isC',
    'ttp_hub_eneg',
    'ttp_hub_covrad_pm',
    'ttp_hub_period',
    'ttp_hub_degree',
    'ttp_nP_arms',
    'ttp_n_unique_arm_types',
    'ttp_armLen_min',
    'ttp_armLen_max',
    'ttp_armLen_mean',
    'ttp_armLen_std',
    'ttp_frac_aryl',
    'ttp_frac_alkynyl',
    'ttp_frac_alkyl',
    'ttp_frac_vinyl',
    'ttp_topicity_2',
    'ttp_topicity_3',
    'ttp_topicity_4',
    'ttp_is_symmetric',
    'ttp_hub_eccentricity',
    'ttp_hub_centrality',
    'ttp_n_total_P',
    'ttp_n_heavy',
    'ttp_MolWt',
    'ttp_n_arom_rings',
    'ttp_n_rot_bonds',
    'ttp_n_rings',
    'ttp_frac_arom_heavy',
    'ttp_TPSA',
    'ttp_MolLogP',
    'ttp_HallKierAlpha',
    'ttp_has_biphenyl',
    'ttp_has_terphenyl',
    'ttp_has_spirobifluorene',
    'ttp_has_adamantane',
    'ttp_Pdist_1',
    'ttp_Pdist_2',
    'ttp_Pdist_3',
    'ttp_Pdist_4',
    'ttp_Pdist_5',
    'ttp_Pdist_6',
    'ttp_Pdist_7',
    'ttp_Pdist_8',
    'ttp_missing',
    'ttp_n_G14',
    'ttp_frac_P_heavy',
    'ttp_arm_PhP_count',
    'ttp_n_PPh2_groups',
]

# ── Halide features ───────────────────────────────────────────────────────────
HALIDE_FEAT_COLS = ['halide_type', 'halide_count', 'halide_present',
                    'halide_I_count', 'halide_Br_count', 'halide_Cl_count']

# ── Shape / VSA / Composition / MACCS / Fragment names ───────────────────────
SHAPE_3D_NAMES = [
    "NPR1", "NPR2",
    "PMI1", "PMI2", "PMI3",
    "Asphericity", "Eccentricity",
    "RadiusOfGyration", "SpherocityIndex",
    "conformer_failed",
]

VSA_NAMES = (
    [f"PEOE_VSA{i}" for i in range(1, 15)] +
    [f"SlogP_VSA{i}" for i in range(1, 11)] +
    [f"SMR_VSA{i}"  for i in range(1, 11)]
)

COMPOSITION_NAMES = [
    "n_heavy_atoms", "DBE",
    "N_over_C", "O_over_C", "H_over_C",
    "S_present", "halogen_frac", "MW_per_heavy",
]

MACCS_NAMES = [f"MACCS_{i}" for i in range(167)]

FRAGMENT_NAMES = [
    "fr_COO", "fr_pyridine", "fr_benzene", "fr_ether", "fr_ester",
    "fr_amide", "fr_NH0", "fr_NH1", "fr_NH2", "fr_Ar_NH",
    "fr_phenol", "fr_imide", "fr_sulfone", "fr_nitro", "fr_urea",
]

# ── Extended RDKit SMARTS ─────────────────────────────────────────────────────
_SMARTS_RAW = {
    "COOH":         "[CX3](=O)[OX2H1]",
    "COO_neg":      "[CX3](=O)[O-]",
    "py_N":         "[$([nX2](:*):*),$([nX3](:*)(:*):*)]",
    "amine_prim":   "[NH2;!$(N-C=O);!$(N~[#7,#8,F,Cl,Br,I,S])]",
    "amine_sec":    "[NH1;!$(N-C=O);!$(N~[#7,#8,F,Cl,Br,I,S])]",
    "hydroxyl":     "[OX2H;!$(OC=O)]",
    "ether_O":      "[OX2;!$(O=*);!$([OH])]",
    "imidazole":    "c1cnc[nH]1",
    "tetrazole":    "c1nnn[nH]1",
    "catechol":     "c1ccc(O)c(O)c1",
    "phosphonate":  "[PX4](=O)([OX2H])[OX2H]",
    "sulfonate":    "S(=O)(=O)[OX2H1]",
    "aldehyde":     "[CX3H1](=O)[#6]",
    "ketone":       "[CX3](=O)([#6])[#6]",
    "nitro":        "[N+](=O)[O-]",
    "cyano":        "[CX2]#[NX1]",
    "thiol":        "[SX2H]",
    "F":            "[F]",
    "Cl":           "[Cl]",
    "Br":           "[Br]",
    "I":            "[I]",
}

_COORD_KEYS = ["COOH", "COO_neg", "py_N", "amine_prim",
               "amine_sec", "hydroxyl", "imidazole", "tetrazole",
               "phosphonate", "sulfonate"]

# ── Bayesian Optimization ────────────────────────────────────────────────────
BO_DEFAULT_ACQUISITION = "lfbo"
BO_BORE_GAMMA          = 0.25
BO_LCB_KAPPA           = 2.0
BO_EI_XI               = 0.01
BO_N_ITERATIONS        = 50
BO_BATCH_SIZE          = 3
BO_INIT_FRACTION       = 0.30
BO_CHECKPOINT_DIR      = "checkpoints/bo"
BO_EPSILON_GREEDY      = 0.1
BO_N_LHS_SAMPLES       = 1000
BO_DEFAULT_SURROGATE   = "rf_cl_mi"
BO_BOOTSTRAP_M         = 50
BO_BORE_ADAPTIVE_GAMMA = True   # anneal gamma from BO_BORE_GAMMA → 0.10 over the campaign
BO_SSL_ALPHA           = 0.2    # down-weight factor for pseudo-labeled candidates (DRE-BO-SSL)
BO_SSL_N_PSEUDO        = 20     # pseudo-labeled candidates added per BORE/LFBO iteration
BO_CLUSTER_DIV_LAMBDA  = 2.0    # strength of chemistry-cluster diversity penalty (simulation only)
                                 # 0 = disabled; higher = stronger push toward unexplored clusters

# Controllable process parameters for BO
# (total_conc bounds computed at runtime from 5th-95th percentile of training data)
BO_CONTROLLABLE_PARAMS = {
    "equivalents":           (0.0, 150.0),
    "temperature_k":         (298.0, 393.0),
    "total_conc":            None,   # computed at runtime via percentile clipping
    "phi_1":                 (0.0, 1.0),   # solvent_1 volume fraction
}
# Optional param, off by default — enable via --bo-include-mlr
BO_OPTIONAL_PARAMS = {
    "metal_over_linker_ratio": (0.0, 4.0),
}
BO_TOTAL_CONC_CLIP_PERCENTILES = (5, 95)
BO_LOG_SCALE_PARAMS            = ["total_conc"]
TOTAL_VOLUME_ML = 2.0   # fixed synthesis volume for BO candidate featurization
