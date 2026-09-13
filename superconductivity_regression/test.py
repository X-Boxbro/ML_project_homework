"""
Superconductivity critical_temp inference script for Stacking Ensemble the best verion.

Usage:
  python test.py --test_data <eval.csv> --elem_data <elem.csv> --model_path <model.pkl>

The model was trained on TWO datasets (ml_train.csv + unique_m_train.csv).
Both --test_data and --elem_data are REQUIRED for correct results.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants — must match train.py EXACTLY
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helper functions — exact copies from train.py
# ---------------------------------------------------------------------------
def fie_mean_from_frac(df_elem, all_cols):
    weights = np.zeros(len(df_elem))
    for col in all_cols:
        weights += df_elem[col].values * ELECTRONEGATIVITY.get(col, 0.0)
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
    return np.where(result == 99.0, 0.0, result)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match train.py build_features EXACTLY."""
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
            new_cols[f"wtd_vs_mean_{group}"] = (X[wtd_mean[0]].values + 1e-9) / m

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

    new_cols["debye_proxy"] = np.sqrt(fh_mean / (mass_mean + 1e-9))
    new_cols["radius_ratio_max_min"] = ar_range / (ar_mean + 1e-9)

    # BCS-inspired physics features
    new_cols["ion_polar_proxy"] = ar3 / (np.abs(fie_mean) + 1e-9)
    debye_val = np.sqrt(fh_mean / (mass_mean + 1e-9))
    new_cols["el_phonon_coupling"] = (val_mean * dens_mean / (mass_mean + 1e-9)) / (
        mass_mean * debye_val + 1e-9)
    new_cols["band_filling"] = val_mean / (n_el_arr + 1e-9)

    new_df = pd.DataFrame(new_cols, index=X.index)
    # CRITICAL: X first, then new features — matching train.py line 260
    return pd.concat([X, new_df], axis=1)


def build_element_features(df_elem: pd.DataFrame):
    """Must match train.py build_element_features EXACTLY."""
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

    mendel_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = MENDELEEV.get(col, 0.0)
        mendel_arr += df[col].values * v
    new_cols["mendeleev_mean"] = mendel_arr
    mendel_mean_sq = mendel_arr ** 2
    mendel_sq_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = MENDELEEV.get(col, 0.0)
        mendel_sq_arr += df[col].values * (v ** 2)
    new_cols["mendeleev_var"] = np.maximum(mendel_sq_arr - mendel_mean_sq, 0)

    vec_arr = np.zeros(len(df))
    vec_sq_arr = np.zeros(len(df))
    for col in all_elem_cols:
        v = VALENCE.get(col, 0.0)
        vec_arr += df[col].values * v
        vec_sq_arr += df[col].values * (v ** 2)
    new_cols["vec_mean"] = vec_arr
    new_cols["vec_var"] = np.maximum(vec_sq_arr - vec_arr ** 2, 0)

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


def inverse_transform(y_t, method, power_pt=None):
    """Must match train.py inverse_transform EXACTLY."""
    y_t = np.asarray(y_t, dtype=np.float64)
    y_t = np.clip(y_t, -100, 700)
    if method == "log1p":
        return np.clip(np.expm1(y_t), 0, None)
    elif method == "sqrt":
        return np.clip(np.square(y_t), 0, None)
    elif method == "power":
        if power_pt is None:
            return np.clip(y_t, 0, None)
        return np.clip(power_pt.inverse_transform(y_t.reshape(-1, 1)).ravel(), 0, None)
    return np.clip(y_t, 0, None)


def get_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2 with NaN handling."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    safe = np.nan_to_num(y_pred, nan=float(np.nanmean(y_true)))
    rmse = np.sqrt(mean_squared_error(y_true, safe))
    mae  = mean_absolute_error(y_true, safe)
    r2   = r2_score(y_true, safe)
    return rmse, mae, r2


def predict_stack(models, meta_model, X, BASE_NAMES):
    """Predict using one seed's base models + meta model (in transformed space)."""
    bp_t = {}
    for name in BASE_NAMES:
        if name == "cubist":
            bp_t[name] = models["cubist_et"].predict(X) + models["cubist_ridge"].predict(X)
        elif name == "hgb":
            bp_t[name] = models["hgb"].predict(X)
        else:
            bp_t[name] = models[name].predict(X)
    base = np.column_stack([bp_t[n] for n in BASE_NAMES])
    return meta_model.predict(base)


def main():
    parser = argparse.ArgumentParser(
        description="Superconductivity critical_temp inference (Stacking Ensemble v10)")
    parser.add_argument("--test_data", required=True,
                        help="Path to test features CSV (same format as ml_train.csv)")
    parser.add_argument("--elem_data", required=True,
                        help="Path to element composition CSV (same format as unique_m_train.csv)")
    parser.add_argument("--model_path", default="model.pkl",
                        help="Path to trained model .pkl file (default: model.pkl)")
    parser.add_argument("--pred_output", default="predictions.csv",
                        help="Path to save predictions CSV (default: predictions.csv)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load model package
    # -----------------------------------------------------------------------
    model_path = ROOT / args.model_path
    print(f"[1/5] Loading model: {model_path}")
    pkg = joblib.load(str(model_path))

    all_seed_results = pkg["all_seed_results"]
    feature_cols     = pkg["feature_cols"]       # post-selection feature names
    final_idx        = pkg.get("final_idx", None)
    transform_method = pkg.get("transform_method", "log1p")
    power_pt         = pkg.get("power_pt", None)
    isotonic_reg     = pkg.get("isotonic_reg", None)
    pca_model        = pkg.get("pca", None)       # saved PCA transformer
    pca_nz_mask      = pkg.get("pca_nz_mask", None)  # PCA column mask

    # Derive BASE_NAMES from the first seed's model dict
    model_keys = list(all_seed_results[0]["models"].keys())
    BASE_NAMES = []
    for k in model_keys:
        if k == "cubist_ridge":
            continue
        BASE_NAMES.append("cubist" if k == "cubist_et" else k)

    n_seeds = len(all_seed_results)
    n_features_expected = int(all_seed_results[0]["models"]["xgb"].n_features_in_)

    print(f"  Model method:    {pkg.get('method', 'unknown')}")
    print(f"  Seeds:           {pkg.get('seeds', n_seeds)}")
    print(f"  Transform:       {transform_method}")
    print(f"  Feature cols:    {len(feature_cols)} (selected)")
    print(f"  Expected input:  {n_features_expected} features")
    print(f"  Training RMSE:   {pkg.get('train_rmse', 'N/A')}")
    print(f"  Training Test RMSE: {pkg.get('test_rmse', 'N/A')}")

    # -----------------------------------------------------------------------
    # 2. Load and prepare test data
    # -----------------------------------------------------------------------
    test_data_path = ROOT / args.test_data
    elem_data_path = ROOT / args.elem_data

    print(f"\n[2/5] Loading test data:")
    print(f"  Features CSV:  {test_data_path}")
    print(f"  Elements CSV:  {elem_data_path}")

    df_test = pd.read_csv(str(test_data_path))
    df_elem = pd.read_csv(str(elem_data_path))

    # Align row counts
    if len(df_test) != len(df_elem):
        min_len = min(len(df_test), len(df_elem))
        print(f"  WARNING: Row mismatch ({len(df_test)} vs {len(df_elem)}), truncating to {min_len}")
        df_test = df_test.iloc[:min_len].reset_index(drop=True)
        df_elem = df_elem.iloc[:min_len].reset_index(drop=True)

    n_samples = len(df_test)
    print(f"  Samples: {n_samples}")

    # Check for ground truth
    has_ground_truth = "critical_temp" in df_test.columns
    y_true = df_test["critical_temp"].values.astype(np.float64) if has_ground_truth else None

    # -----------------------------------------------------------------------
    # 3. Build features (exact same pipeline as train.py)
    # -----------------------------------------------------------------------
    print(f"\n[3/5] Building features...")
    X_main = build_features(df_test)
    X_elem, elem_arr = build_element_features(df_elem)

    # Concatenate (exactly as train.py does: combined_df = pd.concat([X_df, elem_df], axis=1))
    X_combined = pd.concat([X_main, X_elem], axis=1)
    print(f"  Combined raw features: {X_combined.shape[1]}")

    # Select features by name matching feature_cols
    X_arr = np.zeros((n_samples, len(feature_cols)), dtype=np.float64)
    missing_cols = 0
    for i, col in enumerate(feature_cols):
        if col in X_combined.columns:
            X_arr[:, i] = np.nan_to_num(X_combined[col].values.astype(np.float64),
                                        nan=0.0, posinf=0.0, neginf=0.0)
        else:
            missing_cols += 1
    print(f"  After feature selection: {X_arr.shape[1]} cols  (missing: {missing_cols})")

    # PCA on sparse element columns (use saved PCA from training if available)
    elem_np = np.nan_to_num(elem_arr, nan=0.0, posinf=0.0, neginf=0.0)
    pca_added = 0
    if pca_model is not None and pca_nz_mask is not None:
        # Use the saved PCA transformer (matches training transformation exactly)
        elem_sparse_test = elem_np[:, pca_nz_mask]
        if elem_sparse_test.shape[1] >= pca_model.n_components_:
            pca_test = pca_model.transform(elem_sparse_test)
            X_arr = np.column_stack([X_arr, np.nan_to_num(pca_test, nan=0.0, posinf=0.0, neginf=0.0)])
            pca_added = pca_model.n_components_
            print(f"  PCA (saved): {pca_nz_mask.sum()} sparse cols -> {pca_model.n_components_} components")
        else:
            print(f"  WARNING: Test elem cols ({elem_sparse_test.shape[1]}) < PCA components ({pca_model.n_components_}), skipping PCA")
    else:
        # Fallback for models without saved PCA: fit on test data
        nz_mask = (np.abs(elem_np) > 1e-10).sum(axis=0) >= 20
        if nz_mask.sum() >= 5:
            n_comp = min(15, nz_mask.sum() - 1)
            pca_test = PCA(n_components=n_comp, random_state=42).fit_transform(elem_np[:, nz_mask])
            X_arr = np.column_stack([X_arr, np.nan_to_num(pca_test, nan=0.0, posinf=0.0, neginf=0.0)])
            pca_added = n_comp
            print(f"  PCA (fallback): {nz_mask.sum()} sparse cols -> {n_comp} components")

    # Pad with zeros if still short
    if X_arr.shape[1] < n_features_expected:
        pad = n_features_expected - X_arr.shape[1]
        X_arr = np.column_stack([X_arr, np.zeros((n_samples, pad))])
        print(f"  Padded {pad} zero columns")
    elif X_arr.shape[1] > n_features_expected:
        X_arr = X_arr[:, :n_features_expected]
        print(f"  Truncated to {n_features_expected} cols")

    print(f"  Final X shape: {X_arr.shape}  (models expect {n_features_expected})")

    # -----------------------------------------------------------------------
    # 4. Multi-seed ensemble prediction
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Running multi-seed ensemble prediction ({n_seeds} seeds)...")

    all_seed_preds_transformed = []
    for si, sr in enumerate(all_seed_results):
        models     = sr["models"]
        meta_model = sr["meta_model"]
        pred_t = predict_stack(models, meta_model, X_arr, BASE_NAMES)
        all_seed_preds_transformed.append(pred_t)

    # Average in transformed space, then inverse transform
    ens_pred_t = np.mean(all_seed_preds_transformed, axis=0)
    pred_raw = inverse_transform(ens_pred_t, transform_method, power_pt)

    # Apply IsotonicRegression calibration (fit during training on ensemble train preds)
    if isotonic_reg is not None:
        pred_final = isotonic_reg.transform(pred_raw)
        print(f"  IsotonicRegression calibration applied")
    else:
        pred_final = pred_raw
        print(f"  No calibrator found, using raw predictions")

    print(f"  Prediction mean: {pred_final.mean():.2f}  std: {pred_final.std():.2f}")

    # -----------------------------------------------------------------------
    # 5. Metrics & output
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Results")
    print("=" * 50)

    if has_ground_truth:
        rmse, mae, r2 = get_metrics(y_true, pred_final)
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")

        # Per-bin breakdown
        bins = [-1, 10, 30, 77, np.inf]
        labels = ["<10K", "10-30K", "30-77K", ">77K"]
        y_bin = pd.cut(y_true, bins=bins, labels=labels)
        print(f"\n  Per-bin breakdown:")
        for lab in labels:
            mask = y_bin == lab
            if mask.sum() > 0:
                b_rmse = np.sqrt(mean_squared_error(y_true[mask], pred_final[mask]))
                b_mae  = mean_absolute_error(y_true[mask], pred_final[mask])
                print(f"    {lab:8s}: n={mask.sum():5d}  RMSE={b_rmse:.4f}  MAE={b_mae:.4f}")
    else:
        rmse = mae = r2 = None
        print("  No ground truth (critical_temp column) in test data — metrics unavailable.")

    # Save predictions
    output_path = ROOT / args.pred_output
    out_df = pd.DataFrame({"predicted_critical_temp": pred_final})
    if has_ground_truth:
        out_df["actual_critical_temp"] = y_true
        out_df["residual"] = y_true - pred_final
    out_df.to_csv(str(output_path), index=False)
    print(f"\n  Predictions saved to: {args.pred_output}")

    print("=" * 50)
    print("Done.")
    return rmse, mae, r2


if __name__ == "__main__":
    main()
