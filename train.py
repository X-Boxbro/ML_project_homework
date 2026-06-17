"""
Superconductivity critical_temp regression — Stacking Ensemble v10.
Architecture: XGB + LightGBM + ExtraTrees + Cubist + HistGradientBoosting (5-model)
             → XGBoost Meta-learner (HPO-optimized, 25-trial)

Key changes over train_v9.py:
  - 5 base models: +HistGradientBoosting (diverse gradient boosting, no GPU dependency)
  - Meta-learner HPO: 25-trial Optuna search on OOF predictions
  - HPO n_estimators increased: XGB/LGB=1500, ET/Cubist/HGB=1000
  - Stratified KFold (5 Tc bins) for HPO; Repeated 5×2-fold CV for stacking
  - 3 BCS-inspired physics features (ion_polar_proxy, el_phonon_coupling, band_filling)
  - Global renames: v9 → v10 throughout (LOG_FILE, MODEL_PATH, HPO_CACHE_DIR, checkpoint)
"""
import warnings, time, gc, argparse, traceback, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[0]
LOG_FILE = ROOT / "training_log_v10.txt"
MODEL_PATH = "model_v10.pkl"

def _detect_gpu():
    """Detect CUDA GPU availability. Returns (has_gpu, device_str)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            return True, gpu_name
    except Exception:
        pass
    return False, None

_HasGPU, _GPUName = _detect_gpu()
XGB_DEV = "cuda" if _HasGPU else "cpu"
LGB_GPU = _HasGPU   # try GPU; LGB will fall back to CPU if build lacks GPU support

if _HasGPU:
    print(f"[GPU] Detected: {_GPUName}  |  XGB={XGB_DEV}, LGB=gpu(try)")
else:
    print("[GPU] No GPU detected — all models will run on CPU")

SEEDS = [42, 123, 2026]

SMOGN_BOOST       = True
SMOGN_THRESHOLD   = 50.0
SMOGN_BOOST_RATIO = 2.0
SMOGN_NOISE_SCALE = 0.005

HPO_CACHE_DIR_V9 = ROOT / "hpo_cache_v9"
HPO_CACHE_DIR_V10 = ROOT / "hpo_cache_v10"

BASE_NAMES = ["xgb", "lgb", "et", "cubist", "hgb"]
TRANSFORM_OPTIONS = ["log1p", "sqrt", "power"]

ELEMENT_GROUPS = [
    ("atomic_mass",        ["mean_atomic_mass", "wtd_mean_atomic_mass", "gmean_atomic_mass",
                            "wtd_gmean_atomic_mass", "entropy_atomic_mass", "wtd_entropy_atomic_mass",
                            "range_atomic_mass", "wtd_range_atomic_mass", "std_atomic_mass",
                            "wtd_std_atomic_mass"]),
    ("fie",               ["mean_fie", "wtd_mean_fie", "gmean_fie", "wtd_gmean_fie",
                            "entropy_fie", "wtd_entropy_fie", "range_fie", "wtd_range_fie",
                            "std_fie", "wtd_std_fie"]),
    ("atomic_radius",     ["mean_atomic_radius", "wtd_mean_atomic_radius", "gmean_atomic_radius",
                            "wtd_gmean_atomic_radius", "entropy_atomic_radius", "wtd_entropy_atomic_radius",
                            "range_atomic_radius", "wtd_range_atomic_radius", "std_atomic_radius",
                            "wtd_std_atomic_radius"]),
    ("Density",           ["mean_Density", "wtd_mean_Density", "gmean_Density", "wtd_gmean_Density",
                            "entropy_Density", "wtd_entropy_Density", "range_Density",
                            "wtd_range_Density", "std_Density", "wtd_std_Density"]),
    ("ElectronAffinity",  ["mean_ElectronAffinity", "wtd_mean_ElectronAffinity",
                            "gmean_ElectronAffinity", "wtd_gmean_ElectronAffinity",
                            "entropy_ElectronAffinity", "wtd_entropy_ElectronAffinity",
                            "range_ElectronAffinity", "wtd_range_ElectronAffinity",
                            "std_ElectronAffinity", "wtd_std_ElectronAffinity"]),
    ("FusionHeat",        ["mean_FusionHeat", "wtd_mean_FusionHeat", "gmean_FusionHeat",
                            "wtd_gmean_FusionHeat", "entropy_FusionHeat", "wtd_entropy_FusionHeat",
                            "range_FusionHeat", "wtd_range_FusionHeat", "std_FusionHeat",
                            "wtd_std_FusionHeat"]),
    ("ThermalConductivity",["mean_ThermalConductivity", "wtd_mean_ThermalConductivity",
                             "gmean_ThermalConductivity", "wtd_gmean_ThermalConductivity",
                             "entropy_ThermalConductivity", "wtd_entropy_ThermalConductivity",
                             "range_ThermalConductivity", "wtd_range_ThermalConductivity",
                             "std_ThermalConductivity", "wtd_std_ThermalConductivity"]),
    ("Valence",           ["mean_Valence", "wtd_mean_Valence", "gmean_Valence", "wtd_gmean_Valence",
                            "entropy_Valence", "wtd_entropy_Valence", "range_Valence",
                            "wtd_range_Valence", "std_Valence", "wtd_std_Valence"]),
]

TRANSITION_METALS = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]
RARE_EARTH = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    safe = np.nan_to_num(y_pred, nan=float(np.nanmean(y_true)))
    rmse = np.sqrt(mean_squared_error(y_true, safe))
    mae  = mean_absolute_error(y_true, safe)
    r2   = r2_score(y_true, safe)
    return rmse, mae, r2


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["critical_temp"], errors="ignore")
    new_cols = {}

    for group, cols in ELEMENT_GROUPS:
        mean_col  = [c for c in cols if "mean" in c and "gmean" not in c and "wtd_mean" not in c]
        std_col   = [c for c in cols if "std" in c and "wtd_std" not in c]
        range_col = [c for c in cols if "range" in c and "wtd_range" not in c]
        gmean_col = [c for c in cols if "gmean" in c and "wtd_gmean" not in c]
        ent_col   = [c for c in cols if "entropy" in c and "wtd_entropy" not in c]
        wtd_mean  = [c for c in cols if "wtd_mean" in c and "gmean" not in c]

        m    = X[mean_col[0]].values + 1e-9
        s    = X[std_col[0]].values  + 1e-9
        r    = X[range_col[0]].values + 1e-9
        gm   = X[gmean_col[0]].values + 1e-9
        n_el = np.maximum(X["number_of_elements"].values.astype(np.float64), 1.0)

        if std_col:
            new_cols[f"cv_{group}"] = s / m
        if range_col:
            new_cols[f"range_over_mean_{group}"] = r / m
        if gmean_col:
            new_cols[f"gm_over_am_{group}"] = gm / m
        if ent_col:
            denom = np.log(n_el)
            denom = np.where(denom < 1e-8, 1.0, denom)
            new_cols[f"norm_entropy_{group}"] = X[ent_col[0]].values / denom
        if wtd_mean and mean_col:
            wtd_m = X[wtd_mean[0]].values + 1e-9
            new_cols[f"wtd_vs_mean_{group}"] = wtd_m / m

    en_range = X.get("range_ElectronAffinity",   pd.Series(0, index=X.index)).values
    ar_range = X.get("range_atomic_radius",       pd.Series(0, index=X.index)).values
    new_cols["en_range_x_ar_range"] = (en_range + 1e-9) * (ar_range + 1e-9)

    mass_mean = X.get("mean_atomic_mass",          pd.Series(0, index=X.index)).values + 1e-9
    val_mean  = X.get("mean_Valence",             pd.Series(0, index=X.index)).values + 1e-9
    fie_mean  = X.get("mean_fie",                  pd.Series(0, index=X.index)).values
    ea_mean   = X.get("mean_ElectronAffinity",    pd.Series(0, index=X.index)).values
    tc_mean   = X.get("mean_ThermalConductivity", pd.Series(0, index=X.index)).values
    ar_mean   = X.get("mean_atomic_radius",        pd.Series(0, index=X.index)).values
    dens_mean = X.get("mean_Density",              pd.Series(0, index=X.index)).values
    fh_mean   = X.get("mean_FusionHeat",           pd.Series(0, index=X.index)).values
    val_std   = X.get("std_Valence",              pd.Series(0, index=X.index)).values
    fie_std   = X.get("std_fie",                   pd.Series(0, index=X.index)).values
    ea_std    = X.get("std_ElectronAffinity",     pd.Series(0, index=X.index)).values

    new_cols["fie_x_ea"]       = fie_mean * ea_mean
    new_cols["fie_plus_ea"]   = fie_mean + ea_mean
    new_cols["fie_minus_ea"]   = fie_mean - ea_mean
    new_cols["tc_x_dens"]      = tc_mean * dens_mean
    new_cols["ar_x_dens"]      = ar_mean * dens_mean
    new_cols["fh_x_dens"]      = fh_mean * dens_mean
    new_cols["fie_x_val_std"]  = fie_mean * (val_std + 1e-9)
    new_cols["tc_over_dens"]   = tc_mean / (dens_mean + 1e-9)
    new_cols["fh_over_mass"]   = fh_mean / mass_mean
    new_cols["fie_x_ea_x_val"] = fie_mean * ea_mean * val_mean
    new_cols["ar_x_tc"]        = ar_mean * tc_mean
    new_cols["ea_x_val_std"]   = ea_mean * (val_std + 1e-9)
    new_cols["fie_x_ar"]       = fie_mean * ar_mean
    new_cols["ea_over_ar"]     = ea_mean / (ar_mean + 1e-9)
    new_cols["fie_cv"]         = fie_std  / (np.abs(fie_mean) + 1e-9)
    new_cols["ea_cv"]          = ea_std   / (np.abs(ea_mean)  + 1e-9)
    new_cols["log_mass"]       = np.log1p(mass_mean)
    new_cols["log_tc"]        = np.log1p(tc_mean)

    new_cols["mass_over_valence"] = mass_mean / (val_mean + 1e-9)
    new_cols["fie_x_tc"]          = fie_mean * tc_mean
    new_cols["ea_x_tc"]            = ea_mean * tc_mean
    new_cols["fh_x_tc"]            = fh_mean * tc_mean
    new_cols["ar_x_fh"]            = ar_mean * fh_mean
    new_cols["fie_x_fh"]           = fie_mean * fh_mean
    new_cols["tc_x_val"]           = tc_mean * val_mean
    new_cols["dens_x_val"]         = dens_mean * val_mean
    sum_ea_fie = fie_mean + ea_mean
    new_cols["fie_minus_ea_over"] = (fie_mean - ea_mean) / (np.abs(sum_ea_fie) + 1e-9)
    new_cols["log_dens"]           = np.log1p(dens_mean)
    new_cols["sqrt_tc"]            = np.sqrt(np.abs(tc_mean))
    new_cols["sqrt_ar"]            = np.sqrt(np.abs(ar_mean))

    tc_range = X.get("range_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["tc_range_x_mean"] = tc_range * tc_mean
    new_cols["tc_entropy"]      = X.get("entropy_ThermalConductivity",
                                          pd.Series(0, index=X.index)).values

    new_cols["val_electron_density"] = val_mean * dens_mean / (mass_mean + 1e-9)
    ar3 = np.power(ar_mean, 3)
    new_cols["packing_proxy"] = ar3 * dens_mean / (mass_mean + 1e-9)
    fi_safe = np.abs(fie_mean) + 1e-9
    new_cols["mc_llan_proxy"] = tc_mean * np.exp(-1.04 * (1 + fie_mean)) / (1.91 * fi_safe * (1 + 0.62 * fi_safe) + 1e-9)
    new_cols["ea_over_tc"] = ea_mean / (tc_mean + 1e-9)
    new_cols["fh_over_ar"] = fh_mean / (ar_mean + 1e-9)
    ent_tc = X.get("entropy_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["entropy_tc_product"] = ent_tc * tc_mean
    ent_dens = X.get("entropy_Density", pd.Series(0, index=X.index)).values
    new_cols["entropy_dens_product"] = ent_dens * dens_mean
    new_cols["mass_x_fie"] = mass_mean * fie_mean
    new_cols["mass_x_ea"]  = mass_mean * ea_mean
    n_el_arr = X["number_of_elements"].values.astype(np.float64)
    ent_val  = X.get("entropy_Valence", pd.Series(0, index=X.index)).values
    new_cols["n_elements_x_val_entropy"] = n_el_arr * ent_val
    new_cols["fie_val_heterogeneity"] = fie_std * (val_std + 1e-9)
    wtd_fie = X.get("wtd_mean_fie", pd.Series(0, index=X.index)).values
    new_cols["fie_wtd_minus_mean"] = wtd_fie - fie_mean
    tc_std = X.get("std_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["tc_cv"] = tc_std / (tc_mean + 1e-9)
    dens_std = X.get("std_Density", pd.Series(0, index=X.index)).values
    new_cols["dens_cv"] = dens_std / (dens_mean + 1e-9)
    new_cols["val_tc_dens_3way"] = val_mean * tc_mean * dens_mean

    # Debye temperature proxy: theta_D ~ sqrt(T_melt / M), use FusionHeat as T_melt proxy
    new_cols["debye_proxy"] = np.sqrt(fh_mean / (mass_mean + 1e-9))

    # Atomic radius ratio max/min (tolerance factor variant)
    new_cols["radius_ratio_max_min"] = ar_range / (ar_mean + 1e-9)

    # BCS-inspired physics features
    new_cols["ion_polar_proxy"] = ar3 / (np.abs(fie_mean) + 1e-9)
    debye_val = np.sqrt(fh_mean / (mass_mean + 1e-9))
    new_cols["el_phonon_coupling"] = (val_mean * dens_mean / (mass_mean + 1e-9)) / (
        mass_mean * debye_val + 1e-9)
    new_cols["band_filling"] = val_mean / (n_el_arr + 1e-9)

    new_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, new_df], axis=1)


def build_element_features(df_elem: pd.DataFrame):
    """
    Build features from element composition data (unique_m_train.csv).
    Returns (features_df, elem_arr) where elem_arr is the raw 86-column fraction matrix.
    """
    df = df_elem.drop(columns=["critical_temp", "material"], errors="ignore").copy()

    new_cols = {}

    all_elem_cols = [c for c in df.columns if c not in ("critical_temp", "material")]

    elem_arr = df[all_elem_cols].values.astype(np.float64)

    n_el_from_elem = (elem_arr > 0).sum(axis=1)
    new_cols["n_element_types"] = n_el_from_elem

    tm_mask = [c for c in all_elem_cols if c in TRANSITION_METALS]
    re_mask = [c for c in all_elem_cols if c in RARE_EARTH]
    if tm_mask:
        new_cols["transition_metal_frac"] = df[tm_mask].sum(axis=1).values
    if re_mask:
        new_cols["rare_earth_frac"] = df[re_mask].sum(axis=1).values

    new_cols["Cu_present"]  = (df["Cu"].values > 0).astype(float)
    new_cols["O_present"]   = (df["O"].values > 0).astype(float)
    new_cols["Fe_present"]  = (df["Fe"].values > 0).astype(float)
    new_cols["As_present"]  = (df["As"].values > 0).astype(float)
    new_cols["Se_present"]  = (df["Se"].values > 0).astype(float)
    new_cols["Ba_present"]  = (df["Ba"].values > 0).astype(float)
    new_cols["Y_present"]  = (df["Y"].values > 0).astype(float)
    new_cols["La_present"]  = (df["La"].values > 0).astype(float)
    new_cols["Sr_present"]  = (df["Sr"].values > 0).astype(float)
    new_cols["Nb_present"]  = (df["Nb"].values > 0).astype(float)
    new_cols["Bi_present"]  = (df["Bi"].values > 0).astype(float)

    new_cols["is_cuprate"]           = (new_cols["Cu_present"] * new_cols["O_present"]).astype(float)
    new_cols["is_iron_based"]        = (new_cols["Fe_present"] * (new_cols["As_present"] + new_cols["Se_present"])).astype(float)
    new_cols["is_bcuprates"]         = (new_cols["Ba_present"] * new_cols["Cu_present"] * new_cols["O_present"]).astype(float)
    new_cols["is_rare_earth_based"]  = (new_cols["Y_present"] + new_cols["La_present"]).astype(float)

    n_tm = df[tm_mask].values.sum(axis=1) if tm_mask else np.zeros(len(df))
    n_re = df[re_mask].values.sum(axis=1) if re_mask else np.zeros(len(df))
    n_total = np.maximum(elem_arr.sum(axis=1), 1e-9)
    new_cols["tm_electron_proxy"] = n_tm * fie_mean_from_frac(df, all_elem_cols)
    new_cols["re_electron_proxy"] = n_re * fie_mean_from_frac(df, all_elem_cols)

    row_sum = np.maximum(elem_arr.sum(axis=1), 1e-9)
    max_frac = elem_arr.max(axis=1)
    new_cols["max_element_frac"] = max_frac
    new_cols["element_concentration"] = max_frac / row_sum

    entropy_elem = -(elem_arr * np.log(elem_arr + 1e-12)).sum(axis=1)
    new_cols["element_entropy"] = entropy_elem
    new_cols["element_entropy_norm"] = entropy_elem / np.log(np.maximum(n_el_from_elem, 2))

    max_en_elem_val = max_electroneg_from_frac(df, all_elem_cols)
    min_en_elem_val = min_electroneg_from_frac(df, all_elem_cols)
    new_cols["max_electroneg_in_material"] = max_en_elem_val
    new_cols["min_electroneg_in_material"] = min_en_elem_val
    new_cols["electronegativity_spread"] = max_en_elem_val - min_en_elem_val

    # ---- Mendeleev Number (physics-informed periodic table ordering) ----
    MENDELEEV = {
        "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
        "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
        "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
        "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
        "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
        "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
        "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
        "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
        "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
        "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
        "Bi":83,"Th":84,"Pa":85,"U":86,
    }
    mendel_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = MENDELEEV.get(col, 0.0)
        mendel_arr += df[col].values * v
    new_cols["mendeleev_mean"] = mendel_arr
    # variance of Mendeleev number weighted by fraction
    mendel_mean_sq = mendel_arr ** 2
    mendel_sq_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = MENDELEEV.get(col, 0.0)
        mendel_sq_arr += df[col].values * (v ** 2)
    new_cols["mendeleev_var"] = np.maximum(mendel_sq_arr - mendel_mean_sq, 0)

    # ---- Valence Electron Concentration (Matthias rule) ----
    VALENCE = {
        "H":1,"He":0,"Li":1,"Be":2,"B":3,"C":4,"N":5,"O":6,"F":7,"Ne":0,
        "Na":1,"Mg":2,"Al":3,"Si":4,"P":5,"S":6,"Cl":7,"Ar":0,
        "K":1,"Ca":2,"Sc":3,"Ti":4,"V":5,"Cr":6,"Mn":7,"Fe":8,"Co":9,
        "Ni":10,"Cu":11,"Zn":12,"Ga":3,"Ge":4,"As":5,"Se":6,"Br":7,"Kr":0,
        "Rb":1,"Sr":2,"Y":3,"Zr":4,"Nb":5,"Mo":6,"Tc":7,"Ru":8,"Rh":9,
        "Pd":10,"Ag":11,"Cd":12,"In":3,"Sn":4,"Sb":5,"Te":6,"I":7,"Xe":0,
        "Cs":1,"Ba":2,"La":3,"Ce":4,"Pr":5,"Nd":6,"Pm":7,"Sm":8,"Eu":9,
        "Gd":10,"Tb":11,"Dy":12,"Ho":13,"Er":14,"Tm":15,"Yb":16,"Lu":17,
        "Hf":4,"Ta":5,"W":6,"Re":7,"Os":8,"Ir":9,"Pt":10,"Au":11,
        "Hg":12,"Tl":3,"Pb":4,"Bi":5,"Th":4,"Pa":5,"U":6,
    }
    vec_arr = np.zeros(len(df))
    vec_sq_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = VALENCE.get(col, 0.0)
        vec_arr += df[col].values * v
        vec_sq_arr += df[col].values * (v ** 2)
    new_cols["vec_mean"] = vec_arr
    new_cols["vec_var"] = np.maximum(vec_sq_arr - vec_arr ** 2, 0)

    # ---- Periodic Table Group & Period (dominant-fraction element) ----
    PERIODIC_GROUP = {
        "H":(1,1),"He":(18,1),"Li":(1,2),"Be":(2,2),"B":(13,2),"C":(14,2),
        "N":(15,2),"O":(16,2),"F":(17,2),"Ne":(18,2),"Na":(1,3),"Mg":(2,3),
        "Al":(13,3),"Si":(14,3),"P":(15,3),"S":(16,3),"Cl":(17,3),"Ar":(18,3),
        "K":(1,4),"Ca":(2,4),"Sc":(3,4),"Ti":(4,4),"V":(5,4),"Cr":(6,4),
        "Mn":(7,4),"Fe":(8,4),"Co":(9,4),"Ni":(10,4),"Cu":(11,4),"Zn":(12,4),
        "Ga":(13,4),"Ge":(14,4),"As":(15,4),"Se":(16,4),"Br":(17,4),"Kr":(18,4),
        "Rb":(1,5),"Sr":(2,5),"Y":(3,5),"Zr":(4,5),"Nb":(5,5),"Mo":(6,5),
        "Tc":(7,5),"Ru":(8,5),"Rh":(9,5),"Pd":(10,5),"Ag":(11,5),"Cd":(12,5),
        "In":(13,5),"Sn":(14,5),"Sb":(15,5),"Te":(16,5),"I":(17,5),"Xe":(18,5),
        "Cs":(1,6),"Ba":(2,6),"La":(3,6),"Ce":(4,6),"Pr":(5,6),"Nd":(6,6),
        "Pm":(7,6),"Sm":(8,6),"Eu":(9,6),"Gd":(10,6),"Tb":(11,6),"Dy":(12,6),
        "Ho":(13,6),"Er":(14,6),"Tm":(15,6),"Yb":(16,6),"Lu":(3,6),"Hf":(4,6),
        "Ta":(5,6),"W":(6,6),"Re":(7,6),"Os":(8,6),"Ir":(9,6),"Pt":(10,6),
        "Au":(11,6),"Hg":(12,6),"Tl":(13,6),"Pb":(14,6),"Bi":(15,6),"Th":(3,7),
        "Pa":(3,7),"U":(3,7),
    }
    dom_idx = np.argmax(elem_arr, axis=1)
    dom_elem = np.array(all_elem_cols)[dom_idx]
    group_arr  = np.array([PERIODIC_GROUP.get(e, (0,0))[0] for e in dom_elem])
    period_arr = np.array([PERIODIC_GROUP.get(e, (0,0))[1] for e in dom_elem])
    new_cols["dom_group"]   = group_arr
    new_cols["dom_period"]  = period_arr
    new_cols["group_x_period"] = group_arr * period_arr

    elem_sorted   = np.sort(elem_arr, axis=1)
    second_frac   = elem_sorted[:, -2]
    new_cols["second_dom_frac"] = second_frac

    return pd.DataFrame(new_cols, index=df.index), elem_arr


ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.00, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Ne": 0.00, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.00, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78,
    "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20,
    "Gd": 1.20, "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10,
    "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
    "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.00,
    "At": 2.20, "Rn": 2.20,
}


def fie_mean_from_frac(df_elem, all_cols):
    weights = np.zeros(len(df_elem))
    for col in all_cols:
        en = ELECTRONEGATIVITY.get(col, 0.0)
        weights += df_elem[col].values * en
    return weights


def max_electroneg_from_frac(df_elem, all_cols):
    arr = df_elem[all_cols].values
    result = np.zeros(len(df_elem))
    for col in all_cols:
        en = ELECTRONEGATIVITY.get(col, 0.0)
        mask = (df_elem[col].values > 0) & (en > 0)
        result[mask] = np.maximum(result[mask], en)
    return result


def min_electroneg_from_frac(df_elem, all_cols):
    arr = df_elem[all_cols].values
    result = np.full(len(df_elem), 99.0)
    for col in all_cols:
        en = ELECTRONEGATIVITY.get(col, 0.0)
        if en <= 0:
            continue
        mask = df_elem[col].values > 0
        result[mask] = np.minimum(result[mask], en)
    result = np.where(result == 99.0, 0.0, result)
    return result


def select_features(X, y, feature_cols, threshold=0.001, corr_thresh=0.95):
    X_df = pd.DataFrame(X, columns=feature_cols)
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X_df)
    kept_idx = selector.get_support(indices=True)
    kept_names = [feature_cols[i] for i in kept_idx]
    X_var_df = pd.DataFrame(X_var, columns=kept_names, index=X_df.index)

    corr_mat = X_var_df.corr().abs()
    to_drop = set()
    cols = corr_mat.columns.tolist()
    for i, col in enumerate(cols):
        if col in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if corr_mat.iloc[i, j] > corr_thresh:
                to_drop.add(cols[j])
                break
    final_names = [c for c in kept_names if c not in to_drop]
    final_idx = [feature_cols.index(c) for c in final_names]
    log(f"  Feature selection: {X.shape[1]} → {len(final_names)} (var + corr@{corr_thresh})")
    return np.array(X)[:, final_idx], final_names


def apply_transform(y, method, pt=None):
    y = np.clip(y, 0, None).astype(np.float64)
    if method == "log1p":
        return np.log1p(y)
    elif method == "sqrt":
        return np.sqrt(y)
    elif method == "power":
        if pt is None:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            pt.fit(y.reshape(-1, 1))
        return pt.transform(y.reshape(-1, 1)).ravel()
    return y


def inverse_transform(y_t, method, pt=None):
    y_t = np.asarray(y_t, dtype=np.float64)
    y_t = np.clip(y_t, -100, 700)
    if method == "log1p":
        return np.clip(np.expm1(y_t), 0, None)
    elif method == "sqrt":
        return np.clip(np.square(y_t), 0, None)
    elif method == "power":
        if pt is None:
            raise ValueError("PowerTransformer 'pt' must be passed for 'power' transform.")
        return np.clip(pt.inverse_transform(y_t.reshape(-1, 1)).ravel(), 0, None)
    return np.clip(y_t, 0, None)


def _fit_power_transformer(y):
    y_safe = np.clip(y, 1e-8, None).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(y_safe)
    return pt


def smogn_bagging(X, y_t, y_orig, threshold=77.0, boost_ratio=1.5, noise_scale=0.005, rng=None):
    high_mask = y_orig > threshold
    n_high = high_mask.sum()
    if n_high == 0:
        return X, y_t, y_orig

    n_boost = int(n_high * boost_ratio) - n_high
    if n_boost <= 0:
        return X, y_t, y_orig

    if rng is None:
        rng = np.random.default_rng(2026)

    src_indices = rng.choice(np.where(high_mask)[0], size=n_boost, replace=True)
    X_boost = X[src_indices].copy()
    noise = rng.normal(0, noise_scale, size=X_boost.shape) * (np.abs(X_boost) + 1e-9)
    X_boost = np.clip(X_boost + noise, -1e10, 1e10)

    return (np.concatenate([X, X_boost], axis=0),
            np.concatenate([y_t, y_t[src_indices]], axis=0),
            np.concatenate([y_orig, y_orig[src_indices]], axis=0))


def _enqueue_warm_trials(study, warm_params_list):
    """Inject v3 params as first N trials via Optuna's trial enqueue API."""
    if not warm_params_list:
        return
    for params in warm_params_list:
        study.enqueue_trial(params=params, user_attrs={"source": "v3_warm_start"})


def hpo_single_model_cv(model_name, X, y_t, y_orig, n_trials, seed,
                        n_folds=5, transform_method="log1p", pt=None,
                        use_smogn=False, warm_params_list=None):
    import xgboost as xgb
    from lightgbm import LGBMRegressor

    # Stratified KFold based on Tc bins for stable CV
    tc_bins = np.digitize(y_orig, bins=[10, 30, 50, 77])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if model_name == "xgb":
        def obj(trial):
            p = {
                "learning_rate": trial.suggest_float("lr", 0.015, 0.3, log=True),
                "max_depth": trial.suggest_int("depth", 3, 14),
                "min_child_weight": trial.suggest_int("mcw", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.1, 0.9),
                "reg_alpha": trial.suggest_float("alpha", 1e-4, 200.0, log=True),
                "reg_lambda": trial.suggest_float("lambda", 1e-3, 300.0, log=True),
                "gamma": trial.suggest_float("gamma", 0, 4.0),
                "tree_method": "hist",
                "device": XGB_DEV,
                "random_state": seed,
            }
            oof_t = np.zeros(len(y_orig))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, tc_bins)):
                X_tr, y_tr_t = X[tr_idx], y_t[tr_idx]
                if use_smogn:
                    rng = np.random.default_rng(seed + fold_i)
                    X_tr, y_tr_t, _ = smogn_bagging(
                        X_tr, y_tr_t, y_orig[tr_idx],
                        threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                        noise_scale=SMOGN_NOISE_SCALE, rng=rng)
                try:
                    m = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=50, **p)
                    m.fit(X_tr, y_tr_t, eval_set=[(X[va_idx], y_t[va_idx])], verbose=False)
                except Exception:
                    p_xgb_cpu = {k: v for k, v in p.items() if k != "device"}
                    p_xgb_cpu["device"] = "cpu"
                    m = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=50, **p_xgb_cpu)
                    m.fit(X_tr, y_tr_t, eval_set=[(X[va_idx], y_t[va_idx])], verbose=False)
                oof_t[va_idx] = m.predict(X[va_idx])
                del m; gc.collect()
            oof_orig = inverse_transform(oof_t, transform_method, pt)
            return np.sqrt(mean_squared_error(y_orig, oof_orig))

        log("  XGBoost HPO (5-fold CV + SMOGN, n_est=1500, stratified)...")
        sampler = optuna.samplers.TPESampler(seed=seed)

    elif model_name == "lgb":
        def obj(trial):
            p = {
                "learning_rate": trial.suggest_float("lr", 0.015, 0.3, log=True),
                "max_depth": trial.suggest_int("depth", 4, 14),
                "num_leaves": trial.suggest_int("nl", 10, 200),
                "min_child_samples": trial.suggest_int("mcs", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.1, 0.9),
                "reg_alpha": trial.suggest_float("alpha", 1e-4, 200.0, log=True),
                "reg_lambda": trial.suggest_float("lambda", 1e-3, 300.0, log=True),
                "random_state": seed,
                "verbose": -1,
                "n_jobs": -1,
                "device": "gpu" if LGB_GPU else "cpu",
            }
            oof_t = np.zeros(len(y_orig))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, tc_bins)):
                X_tr, y_tr_t = X[tr_idx], y_t[tr_idx]
                if use_smogn:
                    rng = np.random.default_rng(seed + fold_i)
                    X_tr, y_tr_t, _ = smogn_bagging(
                        X_tr, y_tr_t, y_orig[tr_idx],
                        threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                        noise_scale=SMOGN_NOISE_SCALE, rng=rng)
                try:
                    m = LGBMRegressor(n_estimators=1500, early_stopping_rounds=50, **p)
                    m.fit(X_tr, y_tr_t, eval_set=[(X[va_idx], y_t[va_idx])])
                except Exception:
                    if LGB_GPU:
                        p_cpu = {k: v for k, v in p.items() if k != "device"}
                        p_cpu["device"] = "cpu"
                        m = LGBMRegressor(n_estimators=1500, early_stopping_rounds=50, **p_cpu)
                        m.fit(X_tr, y_tr_t, eval_set=[(X[va_idx], y_t[va_idx])])
                    else:
                        raise
                oof_t[va_idx] = m.predict(X[va_idx])
                del m; gc.collect()
            oof_orig = inverse_transform(oof_t, transform_method, pt)
            return np.sqrt(mean_squared_error(y_orig, oof_orig))

        log("  LightGBM HPO (5-fold CV + SMOGN, n_est=1500, stratified)...")
        sampler = optuna.samplers.TPESampler(seed=seed + 1)

    elif model_name == "et":
        def obj(trial):
            p = {
                "n_estimators": 800,
                "max_depth": trial.suggest_int("depth", 5, 40),
                "min_samples_leaf": trial.suggest_int("msl", 1, 20),
                "min_samples_split": trial.suggest_int("mss", 2, 25),
                "max_features": trial.suggest_float("mf", 0.15, 1.0),
                "random_state": seed,
                "n_jobs": -1,
            }
            oof_t = np.zeros(len(y_orig))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, tc_bins)):
                X_tr, y_tr_t = X[tr_idx], y_t[tr_idx]
                if use_smogn:
                    rng = np.random.default_rng(seed + fold_i)
                    X_tr, y_tr_t, _ = smogn_bagging(
                        X_tr, y_tr_t, y_orig[tr_idx],
                        threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                        noise_scale=SMOGN_NOISE_SCALE, rng=rng)
                m = ExtraTreesRegressor(**p)
                m.fit(X_tr, y_tr_t)
                oof_t[va_idx] = m.predict(X[va_idx])
                del m; gc.collect()
            oof_orig = inverse_transform(oof_t, transform_method, pt)
            return np.sqrt(mean_squared_error(y_orig, oof_orig))

        log("  ExtraTrees HPO (5-fold CV + SMOGN, n_est=800, stratified)...")
        sampler = optuna.samplers.TPESampler(seed=seed + 3)

    elif model_name == "cubist":
        def obj(trial):
            et_p = {
                "n_estimators": 600,
                "max_depth": trial.suggest_int("cub_depth", 4, 30),
                "min_samples_leaf": trial.suggest_int("cub_msl", 1, 20),
                "min_samples_split": trial.suggest_int("cub_mss", 2, 25),
                "max_features": trial.suggest_float("cub_mf", 0.1, 1.0),
                "random_state": seed,
                "n_jobs": -1,
            }
            ridge_alpha = trial.suggest_float("cub_ridge_alpha", 0.001, 500.0, log=True)
            oof_t = np.zeros(len(y_orig))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, tc_bins)):
                X_tr, y_tr_t = X[tr_idx], y_t[tr_idx]
                if use_smogn:
                    rng = np.random.default_rng(seed + fold_i)
                    X_tr, y_tr_t, _ = smogn_bagging(
                        X_tr, y_tr_t, y_orig[tr_idx],
                        threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                        noise_scale=SMOGN_NOISE_SCALE, rng=rng)
                et_m = ExtraTreesRegressor(**et_p)
                et_m.fit(X_tr, y_tr_t)
                ridge_m = Ridge(alpha=ridge_alpha, random_state=seed)
                ridge_m.fit(X_tr, y_tr_t - et_m.predict(X_tr))
                oof_t[va_idx] = et_m.predict(X[va_idx]) + ridge_m.predict(X[va_idx])
                del et_m, ridge_m; gc.collect()
            oof_orig = inverse_transform(oof_t, transform_method, pt)
            return np.sqrt(mean_squared_error(y_orig, oof_orig))

        log("  Cubist HPO (ET+Ridge, n_est=600, stratified)...")
        sampler = optuna.samplers.TPESampler(seed=seed + 4)

    elif model_name == "hgb":
        def obj(trial):
            p = {
                "max_iter": trial.suggest_int("max_iter", 300, 1500),
                "max_depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "min_samples_leaf": trial.suggest_int("msl", 5, 50),
                "l2_regularization": trial.suggest_float("l2", 0.0, 10.0),
                "random_state": seed,
            }
            oof_t = np.zeros(len(y_orig))
            for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, tc_bins)):
                X_tr, y_tr_t = X[tr_idx], y_t[tr_idx]
                if use_smogn:
                    rng = np.random.default_rng(seed + fold_i)
                    X_tr, y_tr_t, _ = smogn_bagging(
                        X_tr, y_tr_t, y_orig[tr_idx],
                        threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                        noise_scale=SMOGN_NOISE_SCALE, rng=rng)
                m = HistGradientBoostingRegressor(**p)
                m.fit(X_tr, y_tr_t)
                oof_t[va_idx] = m.predict(X[va_idx])
                del m; gc.collect()
            oof_orig = inverse_transform(oof_t, transform_method, pt)
            return np.sqrt(mean_squared_error(y_orig, oof_orig))

        log("  HistGradientBoosting HPO (5-fold CV + SMOGN, stratified)...")
        sampler = optuna.samplers.TPESampler(seed=seed + 5)

    t = time.time()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    _enqueue_warm_trials(study, warm_params_list)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    log(f"    CV RMSE={study.best_value:.4f}  [{time.time()-t:.0f}s]")
    best_params_out = dict(study.best_params)
    best_cv_rmse = study.best_value
    del study; gc.collect()
    return {"params": best_params_out, "best_cv_rmse": best_cv_rmse}


def hpo_meta_learner(oof_matrix, y_orig, y_t, seed, transform_method, pt):
    """25-trial Optuna search for XGBoost meta-learner on OOF predictions."""
    import xgboost as xgb

    def objective(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("depth", 2, 4),
            "n_estimators": trial.suggest_int("n_est", 50, 300),
            "reg_alpha": trial.suggest_float("alpha", 0.01, 50.0, log=True),
            "reg_lambda": trial.suggest_float("lambda", 0.01, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "tree_method": "hist",
            "device": XGB_DEV,
            "random_state": seed,
        }
        mk = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = []
        for tr_i, va_i in mk.split(oof_matrix):
            m_cv = xgb.XGBRegressor(**p)
            m_cv.fit(oof_matrix[tr_i], y_t[tr_i], verbose=False)
            cv_scores.append(np.sqrt(mean_squared_error(
                y_orig[va_i], inverse_transform(m_cv.predict(oof_matrix[va_i]), transform_method, pt))))
            del m_cv; gc.collect()
        return np.mean(cv_scores)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    bp = study.best_params
    log(f"    Meta HPO: CV RMSE={study.best_value:.4f}  (depth={bp['depth']}, n={bp['n_est']}, lr={bp['lr']:.4f})")

    meta = xgb.XGBRegressor(
        n_estimators=bp["n_est"], max_depth=bp["depth"], learning_rate=bp["lr"],
        reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
        subsample=bp["subsample"], colsample_bytree=bp["colsample"],
        tree_method="hist", device=XGB_DEV, random_state=seed,
    )
    meta.fit(oof_matrix, y_t, verbose=False)
    meta_pred_orig = inverse_transform(meta.predict(oof_matrix), transform_method, pt)
    del study; gc.collect()
    return meta, meta_pred_orig


def run_stacking_cv(X, y_t, y_orig, params, seed=42,
                    transform_method="log1p", pt=None):
    import xgboost as xgb
    from lightgbm import LGBMRegressor

    # Repeated 5×2-fold CV for stable OOF estimates
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
    n = len(y_orig)

    oof_t = {name: np.zeros(n) for name in BASE_NAMES}

    n_iters = {
        "xgb": 1500, "lgb": 1500,
        "et":      params["et"].get("n_estimators", 800),
        "cubist":  600,
        "hgb":     params["hgb"].get("max_iter", 1000),
    }

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
        t0 = time.time()
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_t, y_va_t = y_t[tr_idx], y_t[va_idx]

        if SMOGN_BOOST:
            rng = np.random.default_rng(seed + fold_i)
            X_tr, y_tr_t, _ = smogn_bagging(
                X_tr, y_tr_t, y_orig[tr_idx],
                threshold=SMOGN_THRESHOLD, boost_ratio=SMOGN_BOOST_RATIO,
                noise_scale=SMOGN_NOISE_SCALE, rng=rng)

        p_xgb = {
            "learning_rate": params["xgb"]["lr"], "max_depth": params["xgb"]["depth"],
            "min_child_weight": params["xgb"]["mcw"], "subsample": params["xgb"]["subsample"],
            "colsample_bytree": params["xgb"]["colsample"], "reg_alpha": params["xgb"]["alpha"],
            "reg_lambda": params["xgb"]["lambda"], "gamma": params["xgb"]["gamma"],
        }
        p_lgb = {
            "learning_rate": params["lgb"]["lr"], "max_depth": params["lgb"]["depth"],
            "num_leaves": params["lgb"]["nl"], "min_child_samples": params["lgb"]["mcs"],
            "subsample": params["lgb"]["subsample"], "colsample_bytree": params["lgb"]["colsample"],
            "reg_alpha": params["lgb"]["alpha"], "reg_lambda": params["lgb"]["lambda"],
        }
        p_et = {
            "n_estimators": n_iters["et"], "max_depth": params["et"]["depth"],
            "min_samples_leaf": params["et"]["msl"], "min_samples_split": params["et"]["mss"],
            "max_features": params["et"]["mf"],
        }

        try:
            m = xgb.XGBRegressor(n_estimators=n_iters["xgb"], tree_method="hist",
                                  device=XGB_DEV, random_state=seed,
                                  early_stopping_rounds=50, **p_xgb)
            m.fit(X_tr, y_tr_t, eval_set=[(X_va, y_va_t)], verbose=False)
        except Exception:
            p_xgb_cpu = {k: v for k, v in p_xgb.items() if k != "device"}
            m = xgb.XGBRegressor(n_estimators=n_iters["xgb"], tree_method="hist",
                                  device="cpu", random_state=seed,
                                  early_stopping_rounds=50, **p_xgb_cpu)
            m.fit(X_tr, y_tr_t, eval_set=[(X_va, y_va_t)], verbose=False)
        n_iters["xgb"] = max(n_iters["xgb"], int(m.best_iteration) + 10)
        oof_t["xgb"][va_idx] = m.predict(X_va)
        del m; gc.collect()

        m = LGBMRegressor(n_estimators=n_iters["lgb"], random_state=seed,
                           verbose=-1, n_jobs=-1, early_stopping_rounds=50,
                           device="gpu" if LGB_GPU else "cpu", **p_lgb)
        try:
            m.fit(X_tr, y_tr_t, eval_set=[(X_va, y_va_t)])
        except Exception:
            if LGB_GPU:
                log("  [LGB] GPU unavailable, falling back to CPU")
                m = LGBMRegressor(n_estimators=n_iters["lgb"], random_state=seed,
                                   verbose=-1, n_jobs=-1, early_stopping_rounds=50, **p_lgb)
                m.fit(X_tr, y_tr_t, eval_set=[(X_va, y_va_t)])
            else:
                raise
        best_iter_lgb = getattr(m, "best_iteration_", None)
        n_iters["lgb"] = max(n_iters["lgb"], (int(best_iter_lgb) + 10) if best_iter_lgb else n_iters["lgb"])
        oof_t["lgb"][va_idx] = m.predict(X_va)
        del m; gc.collect()

        m = ExtraTreesRegressor(**p_et, random_state=seed, n_jobs=-1)
        m.fit(X_tr, y_tr_t)
        oof_t["et"][va_idx] = m.predict(X_va)
        del m; gc.collect()

        p_cub_et = {
            "n_estimators": n_iters["cubist"],
            "max_depth": params["cubist"]["cub_depth"],
            "min_samples_leaf": params["cubist"]["cub_msl"],
            "min_samples_split": params["cubist"]["cub_mss"],
            "max_features": params["cubist"]["cub_mf"],
        }
        et_cub = ExtraTreesRegressor(**p_cub_et, random_state=seed, n_jobs=-1)
        et_cub.fit(X_tr, y_tr_t)
        ridge_cub = Ridge(alpha=params["cubist"]["cub_ridge_alpha"], random_state=seed)
        ridge_cub.fit(X_tr, y_tr_t - et_cub.predict(X_tr))
        oof_t["cubist"][va_idx] = et_cub.predict(X_va) + ridge_cub.predict(X_va)
        del et_cub, ridge_cub; gc.collect()

        p_hgb = {
            "max_iter": n_iters["hgb"],
            "max_depth": params["hgb"]["depth"],
            "learning_rate": params["hgb"]["lr"],
            "min_samples_leaf": params["hgb"]["msl"],
            "l2_regularization": params["hgb"]["l2"],
            "random_state": seed,
        }
        m_hgb = HistGradientBoostingRegressor(**p_hgb)
        m_hgb.fit(X_tr, y_tr_t)
        oof_t["hgb"][va_idx] = m_hgb.predict(X_va)
        del m_hgb; gc.collect()

        blend_t = np.mean([oof_t[name][va_idx] for name in BASE_NAMES], axis=0)
        fold_rmse = np.sqrt(mean_squared_error(y_orig[va_idx],
                                               inverse_transform(blend_t, transform_method, pt)))
        log(f"  Fold {fold_i+1}: Blend RMSE={fold_rmse:.4f}  [{time.time()-t0:.1f}s]")

    base_oof_t = np.column_stack([oof_t[name] for name in BASE_NAMES])

    log("\n  Training HPO-optimized XGBoost meta-learner (25-trial)...")
    meta_model, oof_stack_orig = hpo_meta_learner(
        base_oof_t, y_orig, y_t, seed, transform_method, pt)
    rmse, mae, r2 = get_metrics(y_orig, oof_stack_orig)
    log(f"  Stack OOF: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    return {
        "oof_t": {name: oof_t[name] for name in BASE_NAMES},
        "meta_model": meta_model,
        "rmse": rmse, "mae": mae, "r2": r2,
        "n_iters": n_iters,
        "transform_method": transform_method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/ml_train.csv")
    parser.add_argument("--elem_data",  default="data/unique_m_train.csv")
    parser.add_argument("--model_path", default="model_v10.pkl")
    parser.add_argument("--n_trials",   type=int, default=35,
                        help="Default max n_trials; per-model overrides: XGB=25, LGB=25, ET=20, Cubist=10, HGB=10")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Skip HPO cache and re-run all hyperparameter searches")
    args = parser.parse_args()

    t_all = time.time()
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")
    log(f"[{time.strftime('%H:%M:%S')}] Starting train_v10.py (Stacking Ensemble v10 — HPO Meta + HGB + 5-model)")
    log(f"  Multi-seed: {SEEDS}")
    lgb_dev = "gpu(try)" if LGB_GPU else "CPU"
    log(f"  XGB={XGB_DEV}, LGB={lgb_dev}, ET=CPU, Cubist=CPU, HGB=CPU")
    log(f"  Base models: XGB + LGB + ET + Cubist + HGB (5 models, algorithm diversity)")
    log(f"  SMOGN boost: {SMOGN_BOOST} (threshold={SMOGN_THRESHOLD}K, ratio={SMOGN_BOOST_RATIO})")
    log(f"  Corr thresh: 0.95  |  Meta: HPO-optimized XGBoost (25-trial)  |  Calibration: IsotonicRegression")
    log(f"  HPO cache: hpo_cache_v10/  |  Stratified KFold + Repeated 5×2-fold CV")
    log(f"  HPO: XGB=25/LGB=25/ET=20/Cubist=10/HGB=10 trials, n_est=1500/1000")

    HPO_CACHE_DIR_V10.mkdir(exist_ok=True)

    log("\n" + "=" * 55)
    log("STEP 0  | Load data")
    log("=" * 55)
    df = pd.read_csv(ROOT / args.train_data)
    log(f"  Loaded ml_train.csv: {df.shape}")

    df_elem = pd.read_csv(ROOT / args.elem_data)
    log(f"  Loaded unique_m_train.csv: {df_elem.shape}")

    if len(df) != len(df_elem):
        log(f"  WARNING: Row count mismatch ml={len(df)} vs elem={len(df_elem)}, truncating to min")
        min_len = min(len(df), len(df_elem))
        df = df.iloc[:min_len].reset_index(drop=True)
        df_elem = df_elem.iloc[:min_len].reset_index(drop=True)

    log(f"  Combined: {len(df)} samples")

    X_df = build_features(df)
    elem_df, elem_arr_full = build_element_features(df_elem)

    combined_df = pd.concat([X_df, elem_df], axis=1)
    y_orig = df["critical_temp"].values.astype(np.float64)
    y_full = np.clip(y_orig, 0, None)
    X_full = combined_df.values.astype(np.float64)
    feature_cols = combined_df.columns.tolist()
    del df, df_elem, X_df, elem_df, combined_df; gc.collect()

    log(f"  Raw: Samples={len(y_full)}  Features={X_full.shape[1]}")
    log(f"  Target: mean={y_full.mean():.2f}  std={y_full.std():.2f}  skew={skew(y_full):.3f}")

    log("\n" + "=" * 55)
    log("STEP 0.5 | Train/Test Split (90/10, Val removed)")
    log("=" * 55)
    SPLIT_SEED = 42
    all_idx = np.arange(len(X_full))

    idx_train, idx_test = train_test_split(
        all_idx, test_size=0.1, random_state=SPLIT_SEED)

    X_train  = X_full[idx_train]
    X_test   = X_full[idx_test]
    y_train  = y_full[idx_train]
    y_test   = y_full[idx_test]
    elem_train_full = elem_arr_full[idx_train]
    elem_test_full  = elem_arr_full[idx_test]

    log(f"  Train:  {len(y_train):6d} ({len(y_train)/len(y_full)*100:.1f}%)  y_mean={y_train.mean():.2f}")
    log(f"  Test:   {len(y_test):6d} ({len(y_test)/len(y_full)*100:.1f}%)  y_mean={y_test.mean():.2f}")

    log("\n" + "=" * 55)
    log("STEP 0b | Feature selection (fit on Train only)")
    log("=" * 55)
    X_train_sel, final_names = select_features(X_train, y_train, feature_cols, corr_thresh=0.95)
    final_idx = [feature_cols.index(c) for c in final_names]
    log(f"  Feature selection: {X_full.shape[1]} → {len(final_names)}")

    X_train_sel = np.nan_to_num(X_train_sel, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sel  = np.nan_to_num(np.array(X_test)[:, final_idx],  nan=0.0, posinf=0.0, neginf=0.0)

    log("\n" + "=" * 55)
    log("STEP 0b.5 | PCA on sparse element columns")
    log("=" * 55)
    from sklearn.decomposition import PCA
    elem_train_np = np.nan_to_num(elem_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    elem_test_np  = np.nan_to_num(elem_test_full,  nan=0.0, posinf=0.0, neginf=0.0)
    nz_mask = (np.abs(elem_train_np) > 1e-10).sum(axis=0) >= 20
    elem_sparse = elem_train_np[:, nz_mask]
    pca_obj = None
    pca_nz_mask = None
    log(f"  Element cols: {nz_mask.sum()}/{elem_train_np.shape[1]} pass non-zero threshold (>=20 samples)")
    if elem_sparse.shape[1] >= 5:
        n_comp = min(15, elem_sparse.shape[1] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        pca_train = pca.fit_transform(elem_sparse)
        pca_test  = pca.transform(elem_test_np[:, nz_mask])
        pca_obj = pca
        pca_nz_mask = nz_mask
        log(f"  PCA: {elem_sparse.shape[1]} sparse cols → {n_comp} components  (explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
        X_train_sel = np.column_stack([X_train_sel, pca_train])
        X_test_sel  = np.column_stack([X_test_sel,  pca_test])
        log(f"  Features after PCA: {X_train_sel.shape[1]}")
    else:
        log(f"  Skipped (not enough sparse element columns)")

    log("\n" + "=" * 55)
    log("STEP 0c | Adaptive target transform (fit on Train only)")
    log("=" * 55)
    best_transform = "log1p"
    best_transform_rmse = float("inf")
    best_power_pt = None

    kf_quick = KFold(n_splits=5, shuffle=True, random_state=42)
    import xgboost as xgb_quick
    for t_method in TRANSFORM_OPTIONS:
        pt_tmp = _fit_power_transformer(y_train) if t_method == "power" else None
        y_t_tmp = apply_transform(y_train, t_method, pt_tmp)

        oof_t = np.zeros(len(y_train))
        for tr_idx, va_idx in kf_quick.split(X_train_sel):
            m = xgb_quick.XGBRegressor(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                tree_method="hist", device=XGB_DEV, random_state=42)
            m.fit(X_train_sel[tr_idx], y_t_tmp[tr_idx], verbose=False)
            oof_t[va_idx] = m.predict(X_train_sel[va_idx])
            del m; gc.collect()

        oof_orig = inverse_transform(oof_t, t_method, pt_tmp)
        rmse_t = np.sqrt(mean_squared_error(y_train, oof_orig))
        log(f"  {t_method:8s}: OOF RMSE={rmse_t:.4f}  (skew(y_t)={skew(y_t_tmp):.3f})")
        if rmse_t < best_transform_rmse:
            best_transform_rmse = rmse_t
            best_transform = t_method
            best_power_pt = pt_tmp

    log(f"  -> Selected: {best_transform}  (RMSE={best_transform_rmse:.4f})")

    y_train_t = apply_transform(y_train, best_transform, best_power_pt)
    y_test_t  = apply_transform(y_test,  best_transform, best_power_pt)

    feature_cols = final_names

    all_seed_results = []
    all_test_preds   = []
    all_train_preds  = []

    checkpoint_file = ROOT / "_checkpoint_v10.pkl"
    if checkpoint_file.exists():
        try:
            ck = joblib.load(checkpoint_file)
            restored_seeds = set(ck["seeds_completed"])
            if restored_seeds:
                all_seed_results = ck["all_seed_results"]
                all_train_preds  = [sr["train_pred"]  for sr in all_seed_results]
                all_test_preds   = [sr["test_pred"]   for sr in all_seed_results]
                log(f"\n  [RESTORE] Found checkpoint — {len(restored_seeds)} seed(s) already complete: {sorted(restored_seeds)}")
        except Exception as e:
            log(f"\n  [RESTORE] Failed to load checkpoint: {e}")

    for si, seed in enumerate(SEEDS):
        if any(sr["seed"] == seed for sr in all_seed_results):
            log(f"\n{'=' * 55}")
            log(f"SEED {seed}  ({si+1}/{len(SEEDS)}) — SKIPPED (already in checkpoint)")
            log("=" * 55)
            continue

        log(f"\n{'=' * 55}")
        log(f"SEED {seed}  ({si+1}/{len(SEEDS)})")
        log("=" * 55)

        log(f"\n{'=' * 55}")
        log(f"STEP 1  | Optuna HPO (XGB=25, LGB=25, ET=20, Cubist=10, HGB=10 trials, stratified 5-fold)")
        log("=" * 55)
        TRIAL_MAP = {"xgb": 25, "lgb": 25, "et": 20, "cubist": 10, "hgb": 10}
        all_params = {}
        for model_name in BASE_NAMES:
            cache_file = HPO_CACHE_DIR_V10 / f"{best_transform}_seed{seed}_{model_name}.json"
            if (not args.no_cache) and cache_file.exists():
                cached = json.loads(cache_file.read_text())
                all_params[model_name] = cached["params"]
                log(f"  [{model_name}] loaded from cache  (RMSE={cached['best_cv_rmse']:.4f})")
                continue

            # Warm-start from v9 cache, refine in current feature space
            warm_params_list = None
            n_trials_model = TRIAL_MAP[model_name]
            v9_cache = HPO_CACHE_DIR_V9 / f"{best_transform}_seed{seed}_{model_name}.json"
            if model_name != "hgb" and v9_cache.exists():
                v9_data = json.loads(v9_cache.read_text())
                warm_params_list = [v9_data["params"]]
                log(f"  [{model_name}] v9 warm-start loaded (v9 CV RMSE={v9_data['best_cv_rmse']:.4f}), refining {n_trials_model} trials")
            else:
                log(f"  [{model_name}] no v9 cache / new model, running full {n_trials_model} trials")

            try:
                result = hpo_single_model_cv(
                    model_name, X_train_sel, y_train_t, y_train,
                    n_trials=n_trials_model, seed=seed,
                    transform_method=best_transform, pt=best_power_pt,
                    use_smogn=SMOGN_BOOST,
                    warm_params_list=warm_params_list)
                all_params[model_name] = result["params"]
                cache_file.write_text(json.dumps({
                    "params": result["params"],
                    "best_cv_rmse": result["best_cv_rmse"]
                }))
            except Exception as e:
                log(f"  ERROR in HPO for {model_name}: {e}")
                log(traceback.format_exc())
                raise
            gc.collect()

        log(f"\n{'=' * 55}")
        log(f"STEP 2  | Repeated 5×2-fold Stacking CV (seed={seed})")
        log("=" * 55)
        try:
            cv_result = run_stacking_cv(
                X_train_sel, y_train_t, y_train, seed=seed,
                transform_method=best_transform, params=all_params, pt=best_power_pt)
        except Exception as e:
            log(f"  ERROR in stacking CV: {e}")
            log(traceback.format_exc())
            raise
        gc.collect()

        log(f"\n{'=' * 55}")
        log(f"STEP 3  | Retrain base models on Train split (seed={seed})")
        log("=" * 55)
        import xgboost as xgb
        from lightgbm import LGBMRegressor

        n_iters = cv_result["n_iters"]
        final_models = {
            "xgb": xgb.XGBRegressor(
                n_estimators=n_iters["xgb"], tree_method="hist", device=XGB_DEV,
                random_state=seed,
                learning_rate=all_params["xgb"]["lr"],
                max_depth=all_params["xgb"]["depth"],
                min_child_weight=all_params["xgb"]["mcw"],
                subsample=all_params["xgb"]["subsample"],
                colsample_bytree=all_params["xgb"]["colsample"],
                reg_alpha=all_params["xgb"]["alpha"],
                reg_lambda=all_params["xgb"]["lambda"],
                gamma=all_params["xgb"]["gamma"]),
            "lgb": LGBMRegressor(
                n_estimators=n_iters["lgb"], random_state=seed, verbose=-1, n_jobs=-1,
                device="gpu" if LGB_GPU else "cpu",
                learning_rate=all_params["lgb"]["lr"],
                max_depth=all_params["lgb"]["depth"],
                num_leaves=all_params["lgb"]["nl"],
                min_child_samples=all_params["lgb"]["mcs"],
                subsample=all_params["lgb"]["subsample"],
                colsample_bytree=all_params["lgb"]["colsample"],
                reg_alpha=all_params["lgb"]["alpha"],
                reg_lambda=all_params["lgb"]["lambda"]),
            "et": ExtraTreesRegressor(
                n_estimators=n_iters["et"], random_state=seed, n_jobs=-1,
                max_depth=all_params["et"]["depth"],
                min_samples_leaf=all_params["et"]["msl"],
                min_samples_split=all_params["et"]["mss"],
                max_features=all_params["et"]["mf"]),
            "cubist_et": ExtraTreesRegressor(
                n_estimators=n_iters["cubist"], random_state=seed, n_jobs=-1,
                max_depth=all_params["cubist"]["cub_depth"],
                min_samples_leaf=all_params["cubist"]["cub_msl"],
                min_samples_split=all_params["cubist"]["cub_mss"],
                max_features=all_params["cubist"]["cub_mf"]),
            "cubist_ridge": Ridge(
                alpha=all_params["cubist"]["cub_ridge_alpha"], random_state=seed),
            "hgb": HistGradientBoostingRegressor(
                max_iter=n_iters["hgb"],
                max_depth=all_params["hgb"]["depth"],
                learning_rate=all_params["hgb"]["lr"],
                min_samples_leaf=all_params["hgb"]["msl"],
                l2_regularization=all_params["hgb"]["l2"],
                random_state=seed),
        }

        for name in ["xgb", "lgb", "et", "cubist_et", "cubist_ridge", "hgb"]:
            if name == "xgb":
                try:
                    final_models[name].fit(X_train_sel, y_train_t)
                except Exception:
                    log("  [XGB] GPU unavailable, falling back to CPU")
                    p_xgb_cpu = {k: v for k, v in all_params["xgb"].items() if k != "device"}
                    final_models[name] = xgb.XGBRegressor(
                        n_estimators=n_iters["xgb"], tree_method="hist", device="cpu",
                        random_state=seed, **p_xgb_cpu)
                    final_models[name].fit(X_train_sel, y_train_t)
            elif name == "lgb":
                try:
                    final_models[name].fit(X_train_sel, y_train_t)
                except Exception:
                    log("  [LGB] GPU unavailable, falling back to CPU")
                    final_models[name] = LGBMRegressor(
                        n_estimators=n_iters["lgb"], random_state=seed, verbose=-1,
                        n_jobs=-1, learning_rate=all_params["lgb"]["lr"],
                        max_depth=all_params["lgb"]["depth"],
                        num_leaves=all_params["lgb"]["nl"],
                        min_child_samples=all_params["lgb"]["mcs"],
                        subsample=all_params["lgb"]["subsample"],
                        colsample_bytree=all_params["lgb"]["colsample"],
                        reg_alpha=all_params["lgb"]["alpha"],
                        reg_lambda=all_params["lgb"]["lambda"])
                    final_models[name].fit(X_train_sel, y_train_t)
            elif name == "cubist_ridge":
                final_models[name].fit(X_train_sel, y_train_t - final_models["cubist_et"].predict(X_train_sel))
            else:
                final_models[name].fit(X_train_sel, y_train_t)
            log(f"  Trained {name}")

        log("  All base models trained.")

        meta_model = cv_result["meta_model"]

        log(f"\n{'=' * 55}")
        log(f"STEP 4  | Evaluate — Train/Test (seed={seed})")
        log("=" * 55)

        def _predict_split(X_split, y_split, split_label):
            base_preds_t = {}
            for name in BASE_NAMES:
                if name == "cubist":
                    base_preds_t["cubist"] = (final_models["cubist_et"].predict(X_split) +
                                              final_models["cubist_ridge"].predict(X_split))
                elif name == "hgb":
                    base_preds_t["hgb"] = final_models["hgb"].predict(X_split)
                else:
                    base_preds_t[name] = final_models[name].predict(X_split)

            base_full_t = np.column_stack([base_preds_t[name] for name in BASE_NAMES])
            meta_pred_t = meta_model.predict(base_full_t)
            p_stack_orig = inverse_transform(meta_pred_t, best_transform, best_power_pt)
            rmse, mae, r2 = get_metrics(y_split, p_stack_orig)
            log(f"  [{split_label:5s}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
            return p_stack_orig, rmse, mae, r2

        train_pred, train_rmse, train_mae, train_r2 = _predict_split(X_train_sel, y_train, "Train")
        test_pred,  test_rmse,  test_mae,  test_r2  = _predict_split(X_test_sel,  y_test,  "Test")

        log(f"\n  [Gap] Test-Train: {test_rmse - train_rmse:.4f}")

        all_seed_results.append({
            "seed": seed,
            "models": {k: v for k, v in final_models.items()},
            "meta_model": meta_model,
            "all_params": all_params,
            "n_iters": n_iters,
            "cv_rmse": cv_result["rmse"],
            "cv_mae": cv_result["mae"],
            "cv_r2": cv_result["r2"],
            "train_rmse": train_rmse, "train_mae": train_mae, "train_r2": train_r2,
            "test_rmse":  test_rmse,  "test_mae":  test_mae,  "test_r2":  test_r2,
            "train_pred": train_pred,
            "test_pred":  test_pred,
            "transform_method": best_transform,
        })
        all_train_preds.append(train_pred)
        all_test_preds.append(test_pred)

        checkpoint_pkg = {
            "all_seed_results": all_seed_results,
            "feature_cols": feature_cols,
            "seeds_completed": [sr["seed"] for sr in all_seed_results],
            "transform_method": best_transform,
            "power_pt": best_power_pt,
            "final_idx": final_idx,
            "pca": pca_obj,
            "pca_nz_mask": pca_nz_mask,
        }
        try:
            joblib.dump(checkpoint_pkg, checkpoint_file, compress=3)
        except Exception as e:
            log(f"  WARNING: checkpoint save failed: {e}")

        del final_models; gc.collect()

    log(f"\n{'=' * 55}")
    log("MULTI-SEED ENSEMBLE RESULTS (3-seed default, always ensemble)")
    log("=" * 55)

    ens_train_pred = np.mean(all_train_preds, axis=0)
    ens_test_pred  = np.mean(all_test_preds,  axis=0)
    pre_cal_train_rmse, _, _ = get_metrics(y_train, ens_train_pred)
    pre_cal_test_rmse,  _, _ = get_metrics(y_test,  ens_test_pred)

    # IsotonicRegression residual calibration (fit on ensemble Train, apply to both)
    log("\n  IsotonicRegression residual calibration...")
    ir_calibrator = IsotonicRegression(out_of_bounds="clip")
    ir_calibrator.fit(ens_train_pred, y_train)
    ens_train_pred = ir_calibrator.transform(ens_train_pred)
    ens_test_pred  = ir_calibrator.transform(ens_test_pred)

    ens_train_rmse, ens_train_mae, ens_train_r2 = get_metrics(y_train, ens_train_pred)
    ens_test_rmse,  ens_test_mae,  ens_test_r2  = get_metrics(y_test,  ens_test_pred)
    log(f"    Train RMSE: {pre_cal_train_rmse:.4f} → {ens_train_rmse:.4f}")
    log(f"    Test  RMSE: {pre_cal_test_rmse:.4f} → {ens_test_rmse:.4f}")

    for sr in all_seed_results:
        log(f"  Seed {sr['seed']:4d}: CV={sr['cv_rmse']:.4f}  "
            f"Train={sr['train_rmse']:.4f}  "
            f"Test={sr['test_rmse']:.4f}")

    log(f"\n  Ensemble (avg of {len(SEEDS)} seeds, calibrated):")
    log(f"    Train RMSE: {ens_train_rmse:.4f}  MAE: {ens_train_mae:.4f}  R2: {ens_train_r2:.4f}")
    log(f"    Test  RMSE: {ens_test_rmse:.4f}  MAE: {ens_test_mae:.4f}  R2: {ens_test_r2:.4f}")
    log(f"    [Gap] Test-Train: {ens_test_rmse - ens_train_rmse:.4f}")

    best_rmse_train = ens_train_rmse
    best_rmse_test  = ens_test_rmse
    best_mae_test   = ens_test_mae
    best_r2_test    = ens_test_r2
    best_pred_train = ens_train_pred
    best_pred_test  = ens_test_pred
    best_method     = "multi_seed_ensemble"

    log(f"\n  [PRIMARY] Test RMSE: {best_rmse_test:.4f}  (3-seed ensemble + IsotonicRegression)")

    primary_model_pkg = {
        "all_seed_results": all_seed_results,
        "feature_cols": feature_cols,
        "final_idx": final_idx,
        "test_rmse": best_rmse_test,
        "test_mae": best_mae_test,
        "test_r2": best_r2_test,
        "train_rmse": best_rmse_train,
        "best_method": best_method,
        "seeds": SEEDS,
        "method": "multi_seed_stacking_v10",
        "transform_method": best_transform,
        "power_pt": best_power_pt,
        "isotonic_reg": ir_calibrator,
        "pca": pca_obj,
        "pca_nz_mask": pca_nz_mask,
    }
    try:
        joblib.dump(primary_model_pkg, ROOT / args.model_path, compress=3)
        log(f"\n  Model saved: {args.model_path}")
    except Exception as e:
        log(f"\n  ERROR saving model: {e}")
        log(traceback.format_exc())
        raise

    log("\n" + "=" * 55)
    log("FINAL RESULTS (v10 — 5-model + HPO Meta + HGB + Stratified CV)")
    log("=" * 55)
    log(f"  Features:       {len(feature_cols)} (+3 BCS: ion_polar_proxy, el_phonon_coupling, band_filling)")
    log(f"  Element features: merged from unique_m_train.csv + Mendeleev/VEC/PCA")
    log(f"  Base models:   XGB + LGB + ET + Cubist + HGB (5 models, added HGB)")
    log(f"  Meta:          XGBoost HPO-optimized (25-trial search on OOF predictions)")
    log(f"  CV strategy:   Stratified KFold (HPO) + Repeated 5×2-fold (stacking)")
    log(f"  Seeds:         {SEEDS}")
    log(f"  Transform:     {best_transform}")
    log(f"  SMOGN boost:   {SMOGN_BOOST} (threshold={SMOGN_THRESHOLD}K, ratio={SMOGN_BOOST_RATIO})")
    log(f"  Corr thresh:   0.95  |  Calibration: IsotonicRegression")
    log(f"  HPO:           XGB=25/LGB=25/ET=20/Cubist=10/HGB=10, n_est=1500/800/600 (GPU fallback for XGB/LGB)")
    log(f"  Best method:   {best_method}  (3-seed ensemble, always ensemble)")
    log(f"  Train RMSE:    {best_rmse_train:.4f}")
    log(f"  Test  RMSE:    {best_rmse_test:.4f}  MAE: {best_mae_test:.4f}  R2: {best_r2_test:.4f}")
    log(f"  Total time:    {time.time()-t_all:.0f}s")
    log("=" * 55)


if __name__ == "__main__":
    main()
