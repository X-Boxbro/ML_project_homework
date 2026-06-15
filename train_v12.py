"""
Superconductivity critical_temp regression — Stacking Ensemble v12.
Architecture: XGB + LightGBM + ExtraTrees + Cubist (4-model, 5-seed diversity)
             → XGBoost Meta-learner (HPO-optimized) + ET residual learner

Key changes over train_v11.py:
  - 5 seeds [42, 123, 2026, 888, 777] for lower ensemble variance
  - Seeds 123/888 use random 80% feature subset (forced diversity)
  - HPO fully cached: load from hpo_cache_v11/, zero HPO time
  - ET residual learner: PCA(20) + OOF preds → corrects system bias
  - Meta-learner now operates in transformed space (residuals)
  - Global renames: v11 → v12 throughout (LOG_FILE, MODEL_PATH, HPO_CACHE_DIR, checkpoint)
"""
import warnings, time, gc, argparse, traceback, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import skew

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[0]
LOG_FILE = ROOT / "training_log_v12.txt"
MODEL_PATH = "model_v12.pkl"

def _detect_gpu():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return False, None

_HasGPU, _GPUName = _detect_gpu()
XGB_DEV = "cuda" if _HasGPU else "cpu"
LGB_GPU = _HasGPU

if _HasGPU:
    print(f"[GPU] Detected: {_GPUName}  |  XGB={XGB_DEV}, LGB=gpu(try)")
else:
    print("[GPU] No GPU detected — all models will run on CPU")

SEEDS = [42, 123, 2026, 888, 777]
FEATURE_SUBSAMPLE_SEEDS = {123, 888}
FEATURE_SUBSAMPLE_RATIO = 0.8
SUBSAMPLE_RNG_SEED = 999

SMOGN_BOOST       = True
SMOGN_THRESHOLD   = 50.0
SMOGN_BOOST_RATIO = 2.0
SMOGN_NOISE_SCALE = 0.005

HPO_CACHE_DIR_V11 = ROOT / "hpo_cache_v11"
BASE_NAMES = ["xgb", "lgb", "et", "cubist"]
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


def log(msg):
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


def build_features(df):
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
        if std_col: new_cols[f"cv_{group}"] = s / m
        if range_col: new_cols[f"range_over_mean_{group}"] = r / m
        if gmean_col: new_cols[f"gm_over_am_{group}"] = gm / m
        if ent_col:
            denom = np.maximum(np.log(n_el), 1e-8)
            new_cols[f"norm_entropy_{group}"] = X[ent_col[0]].values / denom
        if wtd_mean and mean_col: new_cols[f"wtd_vs_mean_{group}"] = (X[wtd_mean[0]].values + 1e-9) / m
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
    for k, v in [
        ("fie_x_ea", fie_mean * ea_mean), ("fie_plus_ea", fie_mean + ea_mean),
        ("fie_minus_ea", fie_mean - ea_mean), ("tc_x_dens", tc_mean * dens_mean),
        ("ar_x_dens", ar_mean * dens_mean), ("fh_x_dens", fh_mean * dens_mean),
        ("fie_x_val_std", fie_mean * (val_std + 1e-9)),
        ("tc_over_dens", tc_mean / (dens_mean + 1e-9)),
        ("fh_over_mass", fh_mean / mass_mean),
        ("fie_x_ea_x_val", fie_mean * ea_mean * val_mean),
        ("ar_x_tc", ar_mean * tc_mean), ("ea_x_val_std", ea_mean * (val_std + 1e-9)),
        ("fie_x_ar", fie_mean * ar_mean), ("ea_over_ar", ea_mean / (ar_mean + 1e-9)),
        ("fie_cv", fie_std / (np.abs(fie_mean) + 1e-9)),
        ("ea_cv", ea_std / (np.abs(ea_mean) + 1e-9)),
        ("log_mass", np.log1p(mass_mean)), ("log_tc", np.log1p(tc_mean)),
        ("mass_over_valence", mass_mean / (val_mean + 1e-9)),
        ("fie_x_tc", fie_mean * tc_mean), ("ea_x_tc", ea_mean * tc_mean),
        ("fh_x_tc", fh_mean * tc_mean), ("ar_x_fh", ar_mean * fh_mean),
        ("fie_x_fh", fie_mean * fh_mean), ("tc_x_val", tc_mean * val_mean),
        ("dens_x_val", dens_mean * val_mean),
        ("log_dens", np.log1p(dens_mean)),
        ("sqrt_tc", np.sqrt(np.abs(tc_mean))), ("sqrt_ar", np.sqrt(np.abs(ar_mean))),
    ]:
        new_cols[k] = v
    sum_ea_fie = fie_mean + ea_mean
    new_cols["fie_minus_ea_over"] = (fie_mean - ea_mean) / (np.abs(sum_ea_fie) + 1e-9)
    tc_range = X.get("range_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["tc_range_x_mean"] = tc_range * tc_mean
    new_cols["tc_entropy"] = X.get("entropy_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["val_electron_density"] = val_mean * dens_mean / (mass_mean + 1e-9)
    ar3 = np.power(ar_mean, 3)
    new_cols["packing_proxy"] = ar3 * dens_mean / (mass_mean + 1e-9)
    fi_safe = np.abs(fie_mean) + 1e-9
    new_cols["mc_llan_proxy"] = tc_mean * np.exp(-1.04 * (1 + fie_mean)) / (1.91 * fi_safe * (1 + 0.62 * fi_safe) + 1e-9)
    for k, v in [("ea_over_tc", ea_mean / (tc_mean + 1e-9)), ("fh_over_ar", fh_mean / (ar_mean + 1e-9))]:
        new_cols[k] = v
    ent_tc = X.get("entropy_ThermalConductivity", pd.Series(0, index=X.index)).values
    new_cols["entropy_tc_product"] = ent_tc * tc_mean
    ent_dens = X.get("entropy_Density", pd.Series(0, index=X.index)).values
    new_cols["entropy_dens_product"] = ent_dens * dens_mean
    for k, v in [("mass_x_fie", mass_mean * fie_mean), ("mass_x_ea", mass_mean * ea_mean)]:
        new_cols[k] = v
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
    new_cols["debye_proxy"] = np.sqrt(fh_mean / (mass_mean + 1e-9))
    new_cols["radius_ratio_max_min"] = ar_range / (ar_mean + 1e-9)
    new_cols["ion_polar_proxy"] = ar3 / (np.abs(fie_mean) + 1e-9)
    debye_v = np.sqrt(fh_mean / (mass_mean + 1e-9))
    new_cols["el_phonon_coupling"] = (val_mean * dens_mean / (mass_mean + 1e-9)) / (mass_mean * debye_v + 1e-9)
    new_cols["band_filling"] = val_mean / (n_el_arr + 1e-9)
    new_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, new_df], axis=1)


def build_element_features(df_elem):
    df = df_elem.drop(columns=["critical_temp", "material"], errors="ignore").copy()
    new_cols = {}
    all_elem_cols = [c for c in df.columns if c not in ("critical_temp", "material")]
    elem_arr = df[all_elem_cols].values.astype(np.float64)
    n_el_from_elem = (elem_arr > 0).sum(axis=1)
    new_cols["n_element_types"] = n_el_from_elem
    tm_mask = [c for c in all_elem_cols if c in TRANSITION_METALS]
    re_mask = [c for c in all_elem_cols if c in RARE_EARTH]
    if tm_mask: new_cols["transition_metal_frac"] = df[tm_mask].sum(axis=1).values
    if re_mask: new_cols["rare_earth_frac"] = df[re_mask].sum(axis=1).values
    for el in ["Cu", "O", "Fe", "As", "Se", "Ba", "Y", "La", "Sr", "Nb", "Bi"]:
        new_cols[f"{el}_present"] = (df[el].values > 0).astype(float)
    new_cols["is_cuprate"] = (new_cols["Cu_present"] * new_cols["O_present"]).astype(float)
    new_cols["is_iron_based"] = (new_cols["Fe_present"] * (new_cols["As_present"] + new_cols["Se_present"])).astype(float)
    new_cols["is_bcuprates"] = (new_cols["Ba_present"] * new_cols["Cu_present"] * new_cols["O_present"]).astype(float)
    new_cols["is_rare_earth_based"] = (new_cols["Y_present"] + new_cols["La_present"]).astype(float)
    n_tm = df[tm_mask].values.sum(axis=1) if tm_mask else np.zeros(len(df))
    n_re = df[re_mask].values.sum(axis=1) if re_mask else np.zeros(len(df))
    new_cols["tm_electron_proxy"] = n_tm * fie_mean_from_frac(df, all_elem_cols)
    new_cols["re_electron_proxy"] = n_re * fie_mean_from_frac(df, all_elem_cols)
    row_sum = np.maximum(elem_arr.sum(axis=1), 1e-9)
    max_frac = elem_arr.max(axis=1)
    new_cols["max_element_frac"] = max_frac
    new_cols["element_concentration"] = max_frac / row_sum
    entropy_elem = -(elem_arr * np.log(elem_arr + 1e-12)).sum(axis=1)
    new_cols["element_entropy"] = entropy_elem
    new_cols["element_entropy_norm"] = entropy_elem / np.log(np.maximum(n_el_from_elem, 2))
    max_en = max_electroneg_from_frac(df, all_elem_cols)
    min_en = min_electroneg_from_frac(df, all_elem_cols)
    new_cols["max_electroneg_in_material"] = max_en
    new_cols["min_electroneg_in_material"] = min_en
    new_cols["electronegativity_spread"] = max_en - min_en
    MENDELEEV = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,"Bi":83,"Th":84,"Pa":85,"U":86}
    mendel_arr = np.zeros(len(df))
    for col in all_elem_cols: mendel_arr += df[col].values * MENDELEEV.get(col, 0.0)
    new_cols["mendeleev_mean"] = mendel_arr
    mendel_sq_arr = np.zeros(len(df))
    for col in all_elem_cols: mendel_sq_arr += df[col].values * (MENDELEEV.get(col, 0.0) ** 2)
    new_cols["mendeleev_var"] = np.maximum(mendel_sq_arr - mendel_arr ** 2, 0)
    VALENCE = {"H":1,"He":0,"Li":1,"Be":2,"B":3,"C":4,"N":5,"O":6,"F":7,"Ne":0,"Na":1,"Mg":2,"Al":3,"Si":4,"P":5,"S":6,"Cl":7,"Ar":0,"K":1,"Ca":2,"Sc":3,"Ti":4,"V":5,"Cr":6,"Mn":7,"Fe":8,"Co":9,"Ni":10,"Cu":11,"Zn":12,"Ga":3,"Ge":4,"As":5,"Se":6,"Br":7,"Kr":0,"Rb":1,"Sr":2,"Y":3,"Zr":4,"Nb":5,"Mo":6,"Tc":7,"Ru":8,"Rh":9,"Pd":10,"Ag":11,"Cd":12,"In":3,"Sn":4,"Sb":5,"Te":6,"I":7,"Xe":0,"Cs":1,"Ba":2,"La":3,"Ce":4,"Pr":5,"Nd":6,"Pm":7,"Sm":8,"Eu":9,"Gd":10,"Tb":11,"Dy":12,"Ho":13,"Er":14,"Tm":15,"Yb":16,"Lu":17,"Hf":4,"Ta":5,"W":6,"Re":7,"Os":8,"Ir":9,"Pt":10,"Au":11,"Hg":12,"Tl":3,"Pb":4,"Bi":5,"Th":4,"Pa":5,"U":6}
    vec_arr = np.zeros(len(df))
    for col in all_elem_cols: vec_arr += df[col].values * VALENCE.get(col, 0.0)
    new_cols["vec_mean"] = vec_arr
    vec_sq_arr = np.zeros(len(df))
    for col in all_elem_cols: vec_sq_arr += df[col].values * (VALENCE.get(col, 0.0) ** 2)
    new_cols["vec_var"] = np.maximum(vec_sq_arr - vec_arr ** 2, 0)
    PERIODIC_GROUP = {"H":(1,1),"He":(18,1),"Li":(1,2),"Be":(2,2),"B":(13,2),"C":(14,2),"N":(15,2),"O":(16,2),"F":(17,2),"Ne":(18,2),"Na":(1,3),"Mg":(2,3),"Al":(13,3),"Si":(14,3),"P":(15,3),"S":(16,3),"Cl":(17,3),"Ar":(18,3),"K":(1,4),"Ca":(2,4),"Sc":(3,4),"Ti":(4,4),"V":(5,4),"Cr":(6,4),"Mn":(7,4),"Fe":(8,4),"Co":(9,4),"Ni":(10,4),"Cu":(11,4),"Zn":(12,4),"Ga":(13,4),"Ge":(14,4),"As":(15,4),"Se":(16,4),"Br":(17,4),"Kr":(18,4),"Rb":(1,5),"Sr":(2,5),"Y":(3,5),"Zr":(4,5),"Nb":(5,5),"Mo":(6,5),"Tc":(7,5),"Ru":(8,5),"Rh":(9,5),"Pd":(10,5),"Ag":(11,5),"Cd":(12,5),"In":(13,5),"Sn":(14,5),"Sb":(15,5),"Te":(16,5),"I":(17,5),"Xe":(18,5),"Cs":(1,6),"Ba":(2,6),"La":(3,6),"Ce":(4,6),"Pr":(5,6),"Nd":(6,6),"Pm":(7,6),"Sm":(8,6),"Eu":(9,6),"Gd":(10,6),"Tb":(11,6),"Dy":(12,6),"Ho":(13,6),"Er":(14,6),"Tm":(15,6),"Yb":(16,6),"Lu":(3,6),"Hf":(4,6),"Ta":(5,6),"W":(6,6),"Re":(7,6),"Os":(8,6),"Ir":(9,6),"Pt":(10,6),"Au":(11,6),"Hg":(12,6),"Tl":(13,6),"Pb":(14,6),"Bi":(15,6),"Th":(3,7),"Pa":(3,7),"U":(3,7)}
    dom_idx = np.argmax(elem_arr, axis=1)
    dom_elem = np.array(all_elem_cols)[dom_idx]
    group_arr  = np.array([PERIODIC_GROUP.get(e, (0,0))[0] for e in dom_elem])
    period_arr = np.array([PERIODIC_GROUP.get(e, (0,0))[1] for e in dom_elem])
    new_cols["dom_group"] = group_arr; new_cols["dom_period"] = period_arr
    new_cols["group_x_period"] = group_arr * period_arr
    new_cols["second_dom_frac"] = np.sort(elem_arr, axis=1)[:, -2]
    return pd.DataFrame(new_cols, index=df.index), elem_arr


ELECTRONEGATIVITY = {
    "H":2.20,"He":0.00,"Li":0.98,"Be":1.57,"B":2.04,"C":2.55,"N":3.04,"O":3.44,"F":3.98,"Ne":0.00,
    "Na":0.93,"Mg":1.31,"Al":1.61,"Si":1.90,"P":2.19,"S":2.58,"Cl":3.16,"Ar":0.00,"K":0.82,"Ca":1.00,
    "Sc":1.36,"Ti":1.54,"V":1.63,"Cr":1.66,"Mn":1.55,"Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,"Zn":1.65,
    "Ga":1.81,"Ge":2.01,"As":2.18,"Se":2.55,"Br":2.96,"Kr":3.00,"Rb":0.82,"Sr":0.95,"Y":1.22,"Zr":1.33,
    "Nb":1.60,"Mo":2.16,"Tc":1.90,"Ru":2.20,"Rh":2.28,"Pd":2.20,"Ag":1.93,"Cd":1.69,"In":1.78,"Sn":1.96,
    "Sb":2.05,"Te":2.10,"I":2.66,"Xe":2.60,"Cs":0.79,"Ba":0.89,"La":1.10,"Ce":1.12,"Pr":1.13,"Nd":1.14,
    "Pm":1.13,"Sm":1.17,"Eu":1.20,"Gd":1.20,"Tb":1.10,"Dy":1.22,"Ho":1.23,"Er":1.24,"Tm":1.25,"Yb":1.10,
    "Lu":1.27,"Hf":1.30,"Ta":1.50,"W":2.36,"Re":1.90,"Os":2.20,"Ir":2.20,"Pt":2.28,"Au":2.54,"Hg":2.00,
    "Tl":1.62,"Pb":2.33,"Bi":2.02,"Po":2.00,"At":2.20,"Rn":2.20,
}


def fie_mean_from_frac(df_elem, all_cols):
    w = np.zeros(len(df_elem))
    for col in all_cols: w += df_elem[col].values * ELECTRONEGATIVITY.get(col, 0.0)
    return w


def max_electroneg_from_frac(df_elem, all_cols):
    r = np.zeros(len(df_elem))
    for col in all_cols:
        en = ELECTRONEGATIVITY.get(col, 0.0)
        mask = (df_elem[col].values > 0) & (en > 0)
        r[mask] = np.maximum(r[mask], en)
    return r


def min_electroneg_from_frac(df_elem, all_cols):
    r = np.full(len(df_elem), 99.0)
    for col in all_cols:
        en = ELECTRONEGATIVITY.get(col, 0.0)
        if en <= 0: continue
        mask = df_elem[col].values > 0
        r[mask] = np.minimum(r[mask], en)
    return np.where(r == 99.0, 0.0, r)


def select_features(X, y, feature_cols, threshold=0.001, corr_thresh=0.95):
    X_df = pd.DataFrame(X, columns=feature_cols)
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X_df)
    kept_names = [feature_cols[i] for i in selector.get_support(indices=True)]
    X_var_df = pd.DataFrame(X_var, columns=kept_names, index=X_df.index)
    corr_mat = X_var_df.corr().abs()
    to_drop = set()
    cols = corr_mat.columns.tolist()
    for i, col in enumerate(cols):
        if col in to_drop: continue
        for j in range(i + 1, len(cols)):
            if corr_mat.iloc[i, j] > corr_thresh:
                to_drop.add(cols[j]); break
    final_names = [c for c in kept_names if c not in to_drop]
    log(f"  Feature selection: {X.shape[1]} → {len(final_names)} (var + corr@{corr_thresh})")
    return np.array(X)[:, [feature_cols.index(c) for c in final_names]], final_names


def apply_transform(y, method, pt=None):
    y = np.clip(y, 0, None).astype(np.float64)
    if method == "log1p": return np.log1p(y)
    if method == "sqrt": return np.sqrt(y)
    if method == "power":
        if pt is None:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            pt.fit(y.reshape(-1, 1))
        return pt.transform(y.reshape(-1, 1)).ravel()
    return y


def inverse_transform(y_t, method, pt=None):
    y_t = np.asarray(y_t, dtype=np.float64)
    y_t = np.clip(y_t, -100, 700)
    if method == "log1p": return np.clip(np.expm1(y_t), 0, None)
    if method == "sqrt": return np.clip(np.square(y_t), 0, None)
    if method == "power":
        if pt is None: raise ValueError("PowerTransformer 'pt' must be passed.")
        return np.clip(pt.inverse_transform(y_t.reshape(-1, 1)).ravel(), 0, None)
    return np.clip(y_t, 0, None)


def _fit_power_transformer(y):
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(np.clip(y, 1e-8, None).reshape(-1, 1))
    return pt


def smogn_bagging(X, y_t, y_orig, threshold=77.0, boost_ratio=1.5, noise_scale=0.005, rng=None):
    high_mask = y_orig > threshold
    n_high = high_mask.sum()
    if n_high == 0: return X, y_t, y_orig
    n_boost = int(n_high * boost_ratio) - n_high
    if n_boost <= 0: return X, y_t, y_orig
    if rng is None: rng = np.random.default_rng(2026)
    src = rng.choice(np.where(high_mask)[0], size=n_boost, replace=True)
    Xb = X[src].copy()
    Xb = np.clip(Xb + rng.normal(0, noise_scale, Xb.shape) * (np.abs(Xb) + 1e-9), -1e10, 1e10)
    return (np.concatenate([X, Xb], 0), np.concatenate([y_t, y_t[src]], 0), np.concatenate([y_orig, y_orig[src]], 0))


def hpo_meta_learner(oof_matrix, y_orig, y_t, seed, transform_method, pt):
    """25-trial Optuna search for XGBoost meta-learner on OOF predictions."""
    import xgboost as xgb

    def obj(trial):
        p = {
            "lr": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 4),
            "n_est": trial.suggest_int("n_est", 50, 300),
            "alpha": trial.suggest_float("alpha", 0.01, 50.0, log=True),
            "lda": trial.suggest_float("lda", 0.01, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample": trial.suggest_float("colsample", 0.5, 1.0),
        }
        mk = KFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for tt, vt in mk.split(oof_matrix):
            m = xgb.XGBRegressor(n_estimators=p["n_est"], max_depth=p["depth"], learning_rate=p["lr"],
                reg_alpha=p["alpha"], reg_lambda=p["lda"], subsample=p["subsample"],
                colsample_bytree=p["colsample"], tree_method="hist", device=XGB_DEV, random_state=seed)
            m.fit(oof_matrix[tt], y_t[tt], verbose=False)
            scores.append(np.sqrt(mean_squared_error(y_orig[vt], inverse_transform(m.predict(oof_matrix[vt]), transform_method, pt))))
            del m; gc.collect()
        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(obj, n_trials=25, show_progress_bar=False)
    bp = study.best_params
    log(f"    Meta HPO: CV={study.best_value:.4f} depth={bp['depth']} n={bp['n_est']} lr={bp['lr']:.4f}")
    meta = xgb.XGBRegressor(n_estimators=bp["n_est"], max_depth=bp["depth"], learning_rate=bp["lr"],
        reg_alpha=bp["alpha"], reg_lambda=bp["lda"], subsample=bp["subsample"],
        colsample_bytree=bp["colsample"], tree_method="hist", device=XGB_DEV, random_state=seed)
    meta.fit(oof_matrix, y_t, verbose=False)
    r = inverse_transform(meta.predict(oof_matrix), transform_method, pt)
    del study; gc.collect()
    return meta, r


def run_stacking_cv(X, y_t, y_orig, params, n_folds=10, seed=42, transform_method="log1p", pt=None):
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(y_orig)
    oof_t = {name: np.zeros(n) for name in BASE_NAMES}
    ni = {"xgb": 1500, "lgb": 1500, "et": params["et"].get("n_estimators", 800), "cubist": 600}
    for fi, (tr, va) in enumerate(kf.split(X)):
        t0 = time.time()
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y_t[tr], y_t[va]
        if SMOGN_BOOST:
            rng = np.random.default_rng(seed + fi)
            X_tr, y_tr, _ = smogn_bagging(X_tr, y_tr, y_orig[tr], threshold=SMOGN_THRESHOLD,
                boost_ratio=SMOGN_BOOST_RATIO, noise_scale=SMOGN_NOISE_SCALE, rng=rng)
        try:
            m = xgb.XGBRegressor(n_estimators=ni["xgb"], tree_method="hist", device=XGB_DEV,
                random_state=seed, early_stopping_rounds=50,
                learning_rate=params["xgb"]["lr"], max_depth=params["xgb"]["depth"],
                min_child_weight=params["xgb"]["mcw"], subsample=params["xgb"]["subsample"],
                colsample_bytree=params["xgb"]["colsample"], reg_alpha=params["xgb"]["alpha"],
                reg_lambda=params["xgb"]["lambda"], gamma=params["xgb"]["gamma"])
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            p = {k: v for k, v in params["xgb"].items() if k != "device"}
            m = xgb.XGBRegressor(n_estimators=ni["xgb"], tree_method="hist", device="cpu",
                random_state=seed, early_stopping_rounds=50, **p)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        ni["xgb"] = max(ni["xgb"], int(m.best_iteration) + 10)
        oof_t["xgb"][va] = m.predict(X_va); del m; gc.collect()
        m = LGBMRegressor(n_estimators=ni["lgb"], random_state=seed, verbose=-1, n_jobs=-1,
            early_stopping_rounds=50, device="gpu" if LGB_GPU else "cpu",
            learning_rate=params["lgb"]["lr"], max_depth=params["lgb"]["depth"],
            num_leaves=params["lgb"]["nl"], min_child_samples=params["lgb"]["mcs"],
            subsample=params["lgb"]["subsample"], colsample_bytree=params["lgb"]["colsample"],
            reg_alpha=params["lgb"]["alpha"], reg_lambda=params["lgb"]["lambda"])
        try: m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        except Exception:
            if LGB_GPU:
                log("  [LGB] GPU unavailable, falling back to CPU")
                m = LGBMRegressor(n_estimators=ni["lgb"], random_state=seed, verbose=-1, n_jobs=-1,
                    early_stopping_rounds=50, **{k: v for k, v in params["lgb"].items() if k != "device"})
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            else: raise
        bi = getattr(m, "best_iteration_", None)
        ni["lgb"] = max(ni["lgb"], int(bi) + 10 if bi else ni["lgb"])
        oof_t["lgb"][va] = m.predict(X_va); del m; gc.collect()
        m = ExtraTreesRegressor(n_estimators=ni["et"], random_state=seed, n_jobs=-1,
            max_depth=params["et"]["depth"], min_samples_leaf=params["et"]["msl"],
            min_samples_split=params["et"]["mss"], max_features=params["et"]["mf"])
        m.fit(X_tr, y_tr); oof_t["et"][va] = m.predict(X_va); del m; gc.collect()
        ec = ExtraTreesRegressor(n_estimators=ni["cubist"], random_state=seed, n_jobs=-1,
            max_depth=params["cubist"]["cub_depth"], min_samples_leaf=params["cubist"]["cub_msl"],
            min_samples_split=params["cubist"]["cub_mss"], max_features=params["cubist"]["cub_mf"])
        ec.fit(X_tr, y_tr)
        rc = Ridge(alpha=params["cubist"]["cub_ridge_alpha"], random_state=seed)
        rc.fit(X_tr, y_tr - ec.predict(X_tr))
        oof_t["cubist"][va] = ec.predict(X_va) + rc.predict(X_va)
        del ec, rc; gc.collect()
        fold_rmse = np.sqrt(mean_squared_error(y_orig[va],
            inverse_transform(np.mean([oof_t[n][va] for n in BASE_NAMES], 0), transform_method, pt)))
        log(f"  Fold {fi+1}: Blend RMSE={fold_rmse:.4f}  [{time.time()-t0:.1f}s]")
    base_t = np.column_stack([oof_t[n] for n in BASE_NAMES])
    log("\n  Training HPO-optimized XGBoost meta-learner (25-trial)...")
    mm, pred = hpo_meta_learner(base_t, y_orig, y_t, seed, transform_method, pt)
    r, ma, r2 = get_metrics(y_orig, pred)
    log(f"  Stack OOF: RMSE={r:.4f}  MAE={ma:.4f}  R2={r2:.4f}")
    return {"meta_model": mm, "rmse": r, "mae": ma, "r2": r2, "n_iters": ni}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/ml_train.csv")
    parser.add_argument("--elem_data",  default="data/unique_m_train.csv")
    parser.add_argument("--model_path", default="model_v12.pkl")
    args = parser.parse_args()

    t_all = time.time()
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write("")
    log(f"[{time.strftime('%H:%M:%S')}] Starting train_v12.py (Stacking Ensemble v12 — 5-seed + Residual)")
    log(f"  Multi-seed: {SEEDS}  (seeds {FEATURE_SUBSAMPLE_SEEDS} use {FEATURE_SUBSAMPLE_RATIO*100:.0f}% features)")
    log(f"  XGB={XGB_DEV}, LGB={'gpu(try)' if LGB_GPU else 'CPU'}, ET=CPU, Cubist=CPU")
    log(f"  HPO: fully cached from hpo_cache_v11/  |  Residual: ET(n=300,depth=10) on PCA(20)+4 OOF")

    log("\nSTEP 0 | Load data & build features")
    df = pd.read_csv(ROOT / args.train_data); df_elem = pd.read_csv(ROOT / args.elem_data)
    ml = min(len(df), len(df_elem)); df, df_elem = df.iloc[:ml], df_elem.iloc[:ml]
    X_df = build_features(df); ef, elem_arr = build_element_features(df_elem)
    cf = pd.concat([X_df, ef], axis=1)
    y_orig = df["critical_temp"].values.astype(np.float64); y_full = np.clip(y_orig, 0, None)
    X_full = cf.values.astype(np.float64); cols = cf.columns.tolist()
    del df, df_elem, X_df, ef, cf; gc.collect()
    log(f"  Samples={len(y_full)}  Features={X_full.shape[1]}")

    log("\nSTEP 0.5 | Train/Test split (90/10)")
    tr, te = train_test_split(np.arange(len(X_full)), test_size=0.1, random_state=42)
    X_tr, X_te = X_full[tr], X_full[te]; y_tr, y_te = y_full[tr], y_full[te]
    etr, ete = elem_arr[tr], elem_arr[te]

    log("\nSTEP 0b | Feature selection + PCA")
    X_tr_s, sel = select_features(X_tr, y_tr, cols)
    fi2 = [cols.index(c) for c in sel]; X_te_s = np.nan_to_num(np.array(X_te)[:, fi2])
    X_tr_s = np.nan_to_num(X_tr_s)
    e_np = np.nan_to_num(etr); nz = (np.abs(e_np) > 1e-10).sum(0) >= 20
    if nz.sum() >= 5:
        nc = min(15, nz.sum() - 1)
        p = PCA(n_components=nc, random_state=42)
        X_tr_s = np.column_stack([X_tr_s, p.fit_transform(e_np[:, nz])])
        X_te_s = np.column_stack([X_te_s, p.transform(np.nan_to_num(ete)[:, nz])])

    log("\nSTEP 0c | Target transform (must match v11 cache: sqrt)")
    # Force sqrt to match hpo_cache_v11/ transform prefix
    best_t, best_pt = "sqrt", None
    y_tr_t = apply_transform(y_tr, "sqrt")
    log(f"  Using sqrt (v11 cache transform)")

    all_tr_p, all_te_p = [], []
    for si, seed in enumerate(SEEDS):
        log(f"\n{'='*55}\nSEED {seed}  ({si+1}/{len(SEEDS)})\n{'='*55}")
        log("STEP 1 | Load HPO params from v11 cache")
        ap = {}
        for mn in BASE_NAMES:
            cf = HPO_CACHE_DIR_V11 / f"{best_t}_seed{seed}_{mn}.json"
            if not cf.exists():
                fallback = HPO_CACHE_DIR_V11 / f"{best_t}_seed42_{mn}.json"
                d = json.loads(fallback.read_text()); ap[mn] = d["params"]
                log(f"  [{mn}] seed {seed} cache missing, using seed 42 (CV={d['best_cv_rmse']:.4f})")
            else:
                d = json.loads(cf.read_text()); ap[mn] = d["params"]
                log(f"  [{mn}] loaded (CV={d['best_cv_rmse']:.4f})")

        if seed in FEATURE_SUBSAMPLE_SEEDS:
            rng_s = np.random.default_rng(seed + SUBSAMPLE_RNG_SEED)
            nf = X_tr_s.shape[1]; keep = int(nf * FEATURE_SUBSAMPLE_RATIO)
            fi = rng_s.choice(nf, keep, replace=False)
            X_tr2, X_te2 = X_tr_s[:, fi], X_te_s[:, fi]
            log(f"  Feature subsample: {nf}→{keep}")
        else:
            X_tr2, X_te2 = X_tr_s, X_te_s

        log("STEP 2 | 10-fold stacking CV + meta HPO")
        cv = run_stacking_cv(X_tr2, y_tr_t, y_tr, n_folds=10, seed=seed,
            transform_method=best_t, params=ap, pt=best_pt)
        meta = cv["meta_model"]; ni = cv["n_iters"]

        log("STEP 3 | Retrain + predict")
        import xgboost as xgb; from lightgbm import LGBMRegressor
        p_xgb, p_lgb, p_et = ap["xgb"], ap["lgb"], ap["et"]
        fm = {
            "xgb": xgb.XGBRegressor(n_estimators=ni["xgb"], tree_method="hist", device=XGB_DEV,
                random_state=seed, learning_rate=p_xgb["lr"], max_depth=p_xgb["depth"],
                min_child_weight=p_xgb["mcw"], subsample=p_xgb["subsample"],
                colsample_bytree=p_xgb["colsample"], reg_alpha=p_xgb["alpha"],
                reg_lambda=p_xgb["lambda"], gamma=p_xgb["gamma"]),
            "lgb": LGBMRegressor(n_estimators=ni["lgb"], random_state=seed, verbose=-1, n_jobs=-1,
                device="gpu" if LGB_GPU else "cpu", learning_rate=p_lgb["lr"],
                max_depth=p_lgb["depth"], num_leaves=p_lgb["nl"],
                min_child_samples=p_lgb["mcs"], subsample=p_lgb["subsample"],
                colsample_bytree=p_lgb["colsample"], reg_alpha=p_lgb["alpha"],
                reg_lambda=p_lgb["lambda"]),
            "et": ExtraTreesRegressor(n_estimators=ni["et"], random_state=seed, n_jobs=-1,
                max_depth=p_et["depth"], min_samples_leaf=p_et["msl"],
                min_samples_split=p_et["mss"], max_features=p_et["mf"]),
            "cubist_et": ExtraTreesRegressor(n_estimators=ni["cubist"], random_state=seed, n_jobs=-1, max_depth=ap["cubist"]["cub_depth"], min_samples_leaf=ap["cubist"]["cub_msl"], min_samples_split=ap["cubist"]["cub_mss"], max_features=ap["cubist"]["cub_mf"]),
            "cubist_ridge": Ridge(alpha=ap["cubist"]["cub_ridge_alpha"], random_state=seed),
        }
        for n in ["xgb", "lgb", "et", "cubist_et", "cubist_ridge"]:
            if n == "lgb":
                try: fm[n].fit(X_tr2, y_tr_t)
                except:
                    log("  [LGB] GPU unavailable, falling back to CPU")
                    fm[n] = LGBMRegressor(n_estimators=ni["lgb"], random_state=seed, verbose=-1, n_jobs=-1,
                        device="cpu", learning_rate=p_lgb["lr"], max_depth=p_lgb["depth"],
                        num_leaves=p_lgb["nl"], min_child_samples=p_lgb["mcs"],
                        subsample=p_lgb["subsample"], colsample_bytree=p_lgb["colsample"],
                        reg_alpha=p_lgb["alpha"], reg_lambda=p_lgb["lambda"])
                    fm[n].fit(X_tr2, y_tr_t)
            elif n == "cubist_ridge":
                fm[n].fit(X_tr2, y_tr_t - fm["cubist_et"].predict(X_tr2))
            elif n == "xgb":
                try: fm[n].fit(X_tr2, y_tr_t)
                except:
                    log("  [XGB] GPU unavailable, falling back to CPU")
                    fm[n] = xgb.XGBRegressor(n_estimators=ni["xgb"], tree_method="hist", device="cpu",
                        random_state=seed, learning_rate=p_xgb["lr"], max_depth=p_xgb["depth"],
                        min_child_weight=p_xgb["mcw"], subsample=p_xgb["subsample"],
                        colsample_bytree=p_xgb["colsample"], reg_alpha=p_xgb["alpha"],
                        reg_lambda=p_xgb["lambda"], gamma=p_xgb["gamma"])
                    fm[n].fit(X_tr2, y_tr_t)
            else: fm[n].fit(X_tr2, y_tr_t)
            log(f"  Trained {n}")

        def _pred_t(X):
            b = {}
            for n in BASE_NAMES:
                b[n] = fm["cubist_et"].predict(X) + fm["cubist_ridge"].predict(X) if n == "cubist" else fm[n].predict(X)
            return np.column_stack([b[n] for n in BASE_NAMES])

        trb, teb = _pred_t(X_tr2), _pred_t(X_te2)
        tr_m, te_m = meta.predict(trb), meta.predict(teb)

        log("STEP 4 | ET residual learner")
        pca_r = PCA(n_components=20, random_state=seed)
        trp = pca_r.fit_transform(X_tr2); tep = pca_r.transform(X_te2)
        rx_tr = np.column_stack([trp, trb]); rx_te = np.column_stack([tep, teb])
        res = ExtraTreesRegressor(300, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=-1)
        res.fit(rx_tr, y_tr_t - tr_m)
        trf = inverse_transform(tr_m + res.predict(rx_tr), best_t, best_pt)
        tef = inverse_transform(te_m + res.predict(rx_te), best_t, best_pt)
        b4 = np.sqrt(mean_squared_error(y_tr, inverse_transform(tr_m, best_t, best_pt)))
        log(f"  Train: {np.sqrt(mean_squared_error(y_tr, trf)):.4f}  (before: {b4:.4f})")
        b4 = np.sqrt(mean_squared_error(y_te, inverse_transform(te_m, best_t, best_pt)))
        log(f"  Test:  {np.sqrt(mean_squared_error(y_te, tef)):.4f}  (before: {b4:.4f})")
        all_tr_p.append(trf); all_te_p.append(tef)

    log(f"\n{'='*55}\nFINAL 5-SEED ENSEMBLE RESULTS\n{'='*55}")
    etr, ete = np.mean(all_tr_p, 0), np.mean(all_te_p, 0)
    trr, trm, tr2 = get_metrics(y_tr, etr)
    ter, tem, te2 = get_metrics(y_te, ete)
    log(f"  Train RMSE: {trr:.4f}  MAE: {trm:.4f}  R2: {tr2:.4f}")
    log(f"  Test  RMSE: {ter:.4f}  MAE: {tem:.4f}  R2: {te2:.4f}")
    log(f"  [PRIMARY] Test RMSE: {ter:.4f}  (5-seed ensemble + residual)")

    joblib.dump({"test_rmse": ter, "test_mae": tem, "test_r2": te2,
        "method": "multi_seed_stacking_v12", "transform_method": best_t},
        ROOT / args.model_path, compress=3)
    log(f"\n  Model saved: {args.model_path}")
    log(f"\n{'='*55}\nFINAL RESULTS (v12 — 5-seed + Residual)\n{'='*55}")
    log(f"  Features: {X_full.shape[1]}  |  Seeds: {len(SEEDS)} {SEEDS}")
    log(f"  HPO: cached  |  Residual: ET(300, depth=10) on PCA(20)+4 OOF  |  5-seed ∫")
    log(f"  Train RMSE: {trr:.4f}  |  Test RMSE: {ter:.4f}  MAE: {tem:.4f}  R2: {te2:.4f}")
    log(f"  Total time: {time.time()-t_all:.0f}s")
    log("="*55)


if __name__ == "__main__":
    main()