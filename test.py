"""
Inference script for Stacking Ensemble v3 (multi-seed + adaptive transform + Cubist + nonlinear meta).
Receives: --test_data <eval_data.csv> --model_path <model_v2.pkl>
Prints: RMSE, MAE, R2 + per-bin analysis.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[0]

BASE_NAMES = ["xgb", "lgb", "cat", "et", "cubist"]

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match train_v2.py (v3) exactly."""
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
    new_cols["fie_plus_ea"]    = fie_mean + ea_mean
    new_cols["fie_minus_ea"]   = fie_mean - ea_mean
    new_cols["tc_x_dens"]       = tc_mean * dens_mean
    new_cols["ar_x_dens"]       = ar_mean * dens_mean
    new_cols["fh_x_dens"]       = fh_mean * dens_mean
    new_cols["fie_x_val_std"]  = fie_mean * (val_std + 1e-9)
    new_cols["tc_over_dens"]    = tc_mean / (dens_mean + 1e-9)
    new_cols["fh_over_mass"]    = fh_mean / mass_mean
    new_cols["fie_x_ea_x_val"] = fie_mean * ea_mean * val_mean
    new_cols["ar_x_tc"]         = ar_mean * tc_mean
    new_cols["ea_x_val_std"]    = ea_mean * (val_std + 1e-9)
    new_cols["fie_x_ar"]        = fie_mean * ar_mean
    new_cols["ea_over_ar"]      = ea_mean / (ar_mean + 1e-9)
    new_cols["fie_cv"]          = fie_std  / (np.abs(fie_mean) + 1e-9)
    new_cols["ea_cv"]           = ea_std   / (np.abs(ea_mean)  + 1e-9)
    new_cols["log_mass"]        = np.log1p(mass_mean)
    new_cols["log_tc"]         = np.log1p(tc_mean)

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

    # v3 physics-informed features
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

    new_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, new_df], axis=1)


def inverse_transform(y_t, method, power_pt=None):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",   type=str, required=True)
    parser.add_argument("--model_path",  type=str, default="model_v2.pkl")
    parser.add_argument("--pred_output", type=str, default="predictions_v2.csv")
    args = parser.parse_args()

    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(ROOT / args.test_data)
    has_target = "critical_temp" in test_df.columns
    if has_target:
        y_test = test_df["critical_temp"].values.astype(np.float64)

    pkg = joblib.load(ROOT / args.model_path)
    feature_cols = pkg["feature_cols"]

    test_fe_raw = build_features(test_df)
    for col in feature_cols:
        if col not in test_fe_raw.columns:
            test_fe_raw[col] = 0.0
    X_test = test_fe_raw[feature_cols].values.astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Test shape: {X_test.shape}")

    # Get the transform method from the first seed (all seeds use the same chosen method)
    transform_method = pkg.get("transform_method", "log1p")
    power_pt = pkg.get("power_pt", None)
    seeds = pkg.get("seeds", [42])
    print(f"Predicting with {len(seeds)} seed(s): {seeds}  (transform={transform_method})...")

    all_preds = []
    for sr in pkg["all_seed_results"]:
        models = sr["models"]
        scaler_meta = sr["scaler_meta"]
        meta_candidates = sr["meta_candidates"]
        final_meta_info = sr.get("final_meta", None)
        cubist_models = sr.get("cubist_models", None)

        base_preds_t = {}
        for name in ["xgb", "lgb", "cat", "et"]:
            base_preds_t[name] = models[name].predict(X_test)

        # Cubist in transformed space (no inverse_transform here)
        if cubist_models is not None:
            et_cub = cubist_models["et"]
            ridge_cub = cubist_models["ridge"]
            base_preds_t["cubist"] = et_cub.predict(X_test) + ridge_cub.predict(X_test)
        else:
            base_preds_t["cubist"] = models["et"].predict(X_test)

        # Stack in transformed space
        base_full_t = np.column_stack([base_preds_t[name] for name in BASE_NAMES])
        base_full_s = scaler_meta.transform(base_full_t)

        # Apply final meta-learner (single or blended)
        if final_meta_info is not None:
            best_name, second_name, w1, w2, use_blend = final_meta_info
            if use_blend:
                best_model = meta_candidates[best_name][0]
                second_model = meta_candidates[second_name][0]
                p_log = (best_model.predict(base_full_s) * w1 +
                         second_model.predict(base_full_s) * w2)
            else:
                p_log = meta_candidates[best_name][0].predict(base_full_s)
        else:
            # Legacy fallback: assume meta key exists
            meta = sr.get("meta")
            if meta is not None:
                p_log = meta.predict(base_full_s)
            else:
                # Fallback: simple average in transformed space
                p_log = np.mean([base_preds_t[name] for name in BASE_NAMES], axis=0)

        p_orig = inverse_transform(p_log, transform_method, power_pt)
        all_preds.append(p_orig)

    pred = np.mean(all_preds, axis=0)

    if has_target:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)
        print(f"\n=== Test Results (Multi-seed Stacking v3, {len(seeds)} seeds) ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")

        bins   = [-1, 10, 30, 77, np.inf]
        labels = ["极低温(<10K)", "低温(10-30K)", "中温(30-77K)", "高温(>77K)"]
        y_binned = pd.cut(y_test, bins=bins, labels=labels)
        print("\n=== Per-bin Results ===")
        for label in labels:
            mask = y_binned == label
            if mask.sum() > 0:
                r = np.sqrt(mean_squared_error(y_test[mask], pred[mask]))
                m = mean_absolute_error(y_test[mask], pred[mask])
                print(f"  {label}: n={mask.sum():4d}  RMSE={r:.4f}  MAE={m:.4f}")

    out_df = pd.DataFrame({"predicted_critical_temp": pred})
    if has_target:
        out_df["actual_critical_temp"] = y_test
        out_df["residual"] = y_test - pred
    out_df.to_csv(ROOT / args.pred_output, index=False)
    print(f"\nPredictions saved to {args.pred_output}")


if __name__ == "__main__":
    main()
