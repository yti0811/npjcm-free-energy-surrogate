from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTION_SCRIPT = SCRIPT_DIR / "run_piml_core_pipeline.py"
SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_SYSTEM_RUNS = TABLE_DIR / "stricter_system_holdout_runs.csv"
OUT_SYSTEM_SUMMARY = TABLE_DIR / "stricter_system_holdout_summary.csv"

OUT_BULK2INT_RUNS = TABLE_DIR / "stricter_bulk_to_interface_runs.csv"
OUT_BULK2INT_SUMMARY = TABLE_DIR / "stricter_bulk_to_interface_summary.csv"

OUT_TEMP_RUNS = TABLE_DIR / "stricter_temperature_holdout_runs.csv"
OUT_TEMP_SUMMARY = TABLE_DIR / "stricter_temperature_holdout_summary.csv"

OUT_L2H_RUNS = TABLE_DIR / "stricter_low_to_high_runs.csv"
OUT_L2H_SUMMARY = TABLE_DIR / "stricter_low_to_high_summary.csv"

SEED = 0
TI_DOMAIN_SYSTEMS = ["alpha-TiAl", "beta-TiV", "Ti64"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def import_module_from_file(module_path: Path, module_name: str = "prod_stricter_module"):
    if not module_path.exists():
        raise FileNotFoundError(f"Production script not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def mean_abs_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def ordering_metrics_ties_fail(y_pred, y_ref):
    y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    conc = 0
    disc = 0
    total = 0
    n = len(y_pred)

    for i in range(n):
        for j in range(i + 1, n):
            dp = y_pred[i] - y_pred[j]
            dr = y_ref[i] - y_ref[j]

            if dr == 0:
                continue

            total += 1
            if dp == 0 or dp * dr < 0:
                disc += 1
            else:
                conc += 1

    if total == 0:
        return np.nan, np.nan, 0

    f_viol = disc / total
    tau = (conc - disc) / total
    return float(f_viol), float(tau), int(total)


def reconstruct_fhat(df_sub: pd.DataFrame, colmap: dict, dFres_pred):
    dFraw = dFres_pred + df_sub[colmap["dFraw_T0_mean"]].to_numpy(dtype=float)
    fhat = df_sub[colmap["F0"]].to_numpy(dtype=float) + dFraw
    return fhat


def resolve_temperature_column(df: pd.DataFrame, colmap: dict) -> str:
    """
    Robustly resolve the temperature column name.

    Priority:
      1) explicit keys in colmap
      2) known raw column-name candidates in df
    """
    key_candidates = [
        "temperature",
        "temp",
        "T",
        "temperature_K",
        "Temp",
        "TEMP",
    ]

    # First: try colmap keys
    for key in key_candidates:
        if key in colmap and colmap[key] in df.columns:
            return colmap[key]

    # Second: try raw dataframe columns directly
    col_candidates = [
        "temperature",
        "temp",
        "T",
        "temperature_K",
        "Temp",
        "TEMP",
    ]
    for col in col_candidates:
        if col in df.columns:
            return col

    # Third: heuristic search
    lowered = {c.lower(): c for c in df.columns}
    heuristic_names = [
        "temperature",
        "temp",
        "temperature_k",
        "t",
    ]
    for h in heuristic_names:
        if h in lowered:
            return lowered[h]

    raise KeyError(
        "Could not resolve a temperature column. "
        f"Available colmap keys: {list(colmap.keys())}\n"
        f"Available dataframe columns: {list(df.columns)}"
    )


# -----------------------------------------------------------------------------
# Model fitting / prediction wrappers
# -----------------------------------------------------------------------------
def fit_best_model(best_row, prod, X_train, y_train):
    candidate_models = getattr(prod, "candidate_models")
    models = candidate_models(seed=SEED)

    model_name = best_row["model"]

    if str(model_name).startswith("blend("):
        inside = str(model_name)[len("blend("):-1]
        m1_name, m2_name = [x.strip() for x in inside.split(",")]

        m1 = models[m1_name]
        m2 = models[m2_name]

        m1.fit(X_train, y_train)
        m2.fit(X_train, y_train)

        return {
            "type": "blend",
            "name": model_name,
            "w": float(best_row["w"]),
            "m1_name": m1_name,
            "m2_name": m2_name,
            "m1": m1,
            "m2": m2,
        }

    model = models[model_name]
    model.fit(X_train, y_train)
    return {
        "type": "single",
        "name": model_name,
        "w": np.nan,
        "model": model,
    }


def predict_best_model(fitted, prod, X_test):
    if fitted["type"] == "single":
        if fitted["name"] == "gpr":
            gpr_predict = getattr(prod, "gpr_predict")
            pred, _ = gpr_predict(fitted["model"], X_test)
            return pred
        return fitted["model"].predict(X_test)

    gpr_predict = getattr(prod, "gpr_predict")
    w = fitted["w"]

    if fitted["m1_name"] == "gpr":
        pred1, _ = gpr_predict(fitted["m1"], X_test)
    else:
        pred1 = fitted["m1"].predict(X_test)

    if fitted["m2_name"] == "gpr":
        pred2, _ = gpr_predict(fitted["m2"], X_test)
    else:
        pred2 = fitted["m2"].predict(X_test)

    return w * pred1 + (1.0 - w) * pred2


def select_and_fit_on_train(prod, train_df: pd.DataFrame, colmap: dict):
    make_state_features = getattr(prod, "make_state_features")
    cv_select_ti_model = getattr(prod, "cv_select_ti_model")

    dFres_col = colmap["dFres"]

    X_train, feat_cols = make_state_features(
        train_df,
        colmap,
        include_baseline_terms=True,
    )

    sel_df = cv_select_ti_model(
        df_ti=train_df,
        X_ti=X_train,
        colmap=colmap,
        feat_cols=feat_cols,
        seed=SEED,
    )

    best = sel_df.iloc[0]
    fitted = fit_best_model(
        best,
        prod,
        X_train,
        train_df[dFres_col].to_numpy(dtype=float),
    )
    return best, fitted, feat_cols


def evaluate_split(prod, test_df: pd.DataFrame, feat_cols, fitted, colmap: dict):
    fanc_col = colmap["FANC"]

    X_test = test_df[feat_cols].copy()
    dFres_pred_test = predict_best_model(fitted, prod, X_test)
    fhat_test = reconstruct_fhat(test_df, colmap, dFres_pred_test)
    y_ref_test = test_df[fanc_col].to_numpy(dtype=float)

    mae_holdout = mean_abs_error(y_ref_test, fhat_test)
    fviol_holdout, tau_holdout, npairs = ordering_metrics_ties_fail(fhat_test, y_ref_test)

    return {
        "holdout_MAE_abs": mae_holdout,
        "holdout_fviol": fviol_holdout,
        "holdout_tau": tau_holdout,
        "n_pairs": npairs,
        "n_test": len(test_df),
    }


# -----------------------------------------------------------------------------
# Split runners
# -----------------------------------------------------------------------------
def run_leave_one_system_out(prod, df_ti: pd.DataFrame, colmap: dict):
    system_col = colmap["system"]
    runs = []

    for held_out_system in TI_DOMAIN_SYSTEMS:
        train_df = df_ti[df_ti[system_col] != held_out_system].copy()
        test_df = df_ti[df_ti[system_col] == held_out_system].copy()

        if train_df.empty or test_df.empty:
            continue

        best, fitted, feat_cols = select_and_fit_on_train(prod, train_df, colmap)
        metrics = evaluate_split(prod, test_df, feat_cols, fitted, colmap)

        runs.append({
            "split_type": "leave_one_system_out",
            "held_out_system": held_out_system,
            "n_train": len(train_df),
            "n_test": metrics["n_test"],
            "selected_model": best["model"],
            "blend_weight": float(best["w"]) if "w" in best.index and pd.notna(best["w"]) else np.nan,
            "holdout_MAE_abs": metrics["holdout_MAE_abs"],
            "holdout_fviol": metrics["holdout_fviol"],
            "holdout_tau": metrics["holdout_tau"],
            "n_pairs": metrics["n_pairs"],
        })

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT_SYSTEM_RUNS, index=False)

    summary = pd.DataFrame([{
        "split_type": "leave_one_system_out",
        "n_runs": len(runs_df),
        "mean_holdout_MAE": runs_df["holdout_MAE_abs"].mean() if not runs_df.empty else np.nan,
        "std_holdout_MAE": runs_df["holdout_MAE_abs"].std(ddof=1) if len(runs_df) > 1 else np.nan,
        "mean_holdout_fviol": runs_df["holdout_fviol"].mean() if not runs_df.empty else np.nan,
        "mean_holdout_tau": runs_df["holdout_tau"].mean() if not runs_df.empty else np.nan,
        "max_holdout_MAE": runs_df["holdout_MAE_abs"].max() if not runs_df.empty else np.nan,
        "most_frequent_model": runs_df["selected_model"].mode().iloc[0] if not runs_df.empty else None,
    }])
    summary.to_csv(OUT_SYSTEM_SUMMARY, index=False)

    return runs_df, summary


def run_bulk_to_interface(prod, df_ti: pd.DataFrame, colmap: dict):
    system_col = colmap["system"]

    bulk_systems = ["alpha-TiAl", "beta-TiV"]
    interface_system = "Ti64"

    train_df = df_ti[df_ti[system_col].isin(bulk_systems)].copy()
    test_df = df_ti[df_ti[system_col] == interface_system].copy()

    if train_df.empty:
        raise ValueError("Bulk-only training subset is empty.")
    if test_df.empty:
        raise ValueError("Ti64 interface test subset is empty.")

    best, fitted, feat_cols = select_and_fit_on_train(prod, train_df, colmap)
    metrics = evaluate_split(prod, test_df, feat_cols, fitted, colmap)

    runs_df = pd.DataFrame([{
        "split_type": "bulk_to_interface",
        "train_systems": ",".join(bulk_systems),
        "held_out_system": interface_system,
        "n_train": len(train_df),
        "n_test": metrics["n_test"],
        "selected_model": best["model"],
        "blend_weight": float(best["w"]) if "w" in best.index and pd.notna(best["w"]) else np.nan,
        "holdout_MAE_abs": metrics["holdout_MAE_abs"],
        "holdout_fviol": metrics["holdout_fviol"],
        "holdout_tau": metrics["holdout_tau"],
        "n_pairs": metrics["n_pairs"],
    }])
    runs_df.to_csv(OUT_BULK2INT_RUNS, index=False)

    summary = pd.DataFrame([{
        "split_type": "bulk_to_interface",
        "n_runs": 1,
        "mean_holdout_MAE": metrics["holdout_MAE_abs"],
        "mean_holdout_fviol": metrics["holdout_fviol"],
        "mean_holdout_tau": metrics["holdout_tau"],
        "most_frequent_model": best["model"],
    }])
    summary.to_csv(OUT_BULK2INT_SUMMARY, index=False)

    return runs_df, summary


def run_leave_one_temperature_out(prod, df_ti: pd.DataFrame, colmap: dict):
    temp_col = resolve_temperature_column(df_ti, colmap)
    runs = []

    temps = sorted(pd.Series(df_ti[temp_col]).dropna().unique().tolist())

    for held_out_temp in temps:
        train_df = df_ti[df_ti[temp_col] != held_out_temp].copy()
        test_df = df_ti[df_ti[temp_col] == held_out_temp].copy()

        if train_df.empty or test_df.empty:
            continue

        best, fitted, feat_cols = select_and_fit_on_train(prod, train_df, colmap)
        metrics = evaluate_split(prod, test_df, feat_cols, fitted, colmap)

        runs.append({
            "split_type": "leave_one_temperature_out",
            "held_out_temperature": held_out_temp,
            "n_train": len(train_df),
            "n_test": metrics["n_test"],
            "selected_model": best["model"],
            "blend_weight": float(best["w"]) if "w" in best.index and pd.notna(best["w"]) else np.nan,
            "holdout_MAE_abs": metrics["holdout_MAE_abs"],
            "holdout_fviol": metrics["holdout_fviol"],
            "holdout_tau": metrics["holdout_tau"],
            "n_pairs": metrics["n_pairs"],
        })

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT_TEMP_RUNS, index=False)

    summary = pd.DataFrame([{
        "split_type": "leave_one_temperature_out",
        "n_runs": len(runs_df),
        "mean_holdout_MAE": runs_df["holdout_MAE_abs"].mean() if not runs_df.empty else np.nan,
        "std_holdout_MAE": runs_df["holdout_MAE_abs"].std(ddof=1) if len(runs_df) > 1 else np.nan,
        "mean_holdout_fviol": runs_df["holdout_fviol"].mean() if not runs_df.empty else np.nan,
        "mean_holdout_tau": runs_df["holdout_tau"].mean() if not runs_df.empty else np.nan,
        "max_holdout_MAE": runs_df["holdout_MAE_abs"].max() if not runs_df.empty else np.nan,
        "worst_temperature": (
            runs_df.loc[runs_df["holdout_MAE_abs"].idxmax(), "held_out_temperature"]
            if not runs_df.empty else np.nan
        ),
        "most_frequent_model": runs_df["selected_model"].mode().iloc[0] if not runs_df.empty else None,
    }])
    summary.to_csv(OUT_TEMP_SUMMARY, index=False)

    return runs_df, summary


def run_low_to_high_transfer(
    prod,
    df_ti: pd.DataFrame,
    colmap: dict,
    train_temps=(300, 400, 500),
    test_temps=(600, 700),
):
    temp_col = resolve_temperature_column(df_ti, colmap)

    train_df = df_ti[df_ti[temp_col].isin(train_temps)].copy()
    test_df = df_ti[df_ti[temp_col].isin(test_temps)].copy()

    if train_df.empty:
        raise ValueError(
            f"Low-temperature training subset is empty. "
            f"Resolved temperature column: {temp_col}, available temperatures: {sorted(df_ti[temp_col].unique())}"
        )
    if test_df.empty:
        raise ValueError(
            f"High-temperature test subset is empty. "
            f"Resolved temperature column: {temp_col}, available temperatures: {sorted(df_ti[temp_col].unique())}"
        )

    best, fitted, feat_cols = select_and_fit_on_train(prod, train_df, colmap)
    metrics = evaluate_split(prod, test_df, feat_cols, fitted, colmap)

    runs_df = pd.DataFrame([{
        "split_type": "low_to_high_transfer",
        "train_temps": ",".join(map(str, train_temps)),
        "test_temps": ",".join(map(str, test_temps)),
        "n_train": len(train_df),
        "n_test": metrics["n_test"],
        "selected_model": best["model"],
        "blend_weight": float(best["w"]) if "w" in best.index and pd.notna(best["w"]) else np.nan,
        "holdout_MAE_abs": metrics["holdout_MAE_abs"],
        "holdout_fviol": metrics["holdout_fviol"],
        "holdout_tau": metrics["holdout_tau"],
        "n_pairs": metrics["n_pairs"],
    }])
    runs_df.to_csv(OUT_L2H_RUNS, index=False)

    summary = pd.DataFrame([{
        "split_type": "low_to_high_transfer",
        "n_runs": 1,
        "mean_holdout_MAE": metrics["holdout_MAE_abs"],
        "mean_holdout_fviol": metrics["holdout_fviol"],
        "mean_holdout_tau": metrics["holdout_tau"],
        "most_frequent_model": best["model"],
    }])
    summary.to_csv(OUT_L2H_SUMMARY, index=False)

    return runs_df, summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    prod = import_module_from_file(PRODUCTION_SCRIPT)

    required_funcs = [
        "auto_map_columns",
        "make_state_features",
        "cv_select_ti_model",
        "candidate_models",
        "gpr_predict",
    ]
    for fname in required_funcs:
        if not hasattr(prod, fname):
            raise AttributeError(f"Production script missing required function: {fname}")

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    auto_map_columns = getattr(prod, "auto_map_columns")

    df = pd.read_csv(SNAPSHOT_FILE)
    df, colmap = auto_map_columns(df)

    system_col = colmap["system"]
    df_ti = df[df[system_col].isin(TI_DOMAIN_SYSTEMS)].copy()

    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty.")

    temp_col = resolve_temperature_column(df_ti, colmap)
    print(f"[INFO] Resolved temperature column: {temp_col}")
    print(f"[INFO] Available Ti-domain temperatures: {sorted(pd.Series(df_ti[temp_col]).dropna().unique().tolist())}")

    # 1) Leave-one-system-out
    system_runs, system_summary = run_leave_one_system_out(prod, df_ti, colmap)

    # 2) Bulk-to-interface
    bulk2int_runs, bulk2int_summary = run_bulk_to_interface(prod, df_ti, colmap)

    # 3) Leave-one-temperature-out
    temp_runs, temp_summary = run_leave_one_temperature_out(prod, df_ti, colmap)

    # 4) Low-to-high transfer
    l2h_runs, l2h_summary = run_low_to_high_transfer(
        prod,
        df_ti,
        colmap,
        train_temps=(300, 400, 500),
        test_temps=(600, 700),
    )

    print("\n=== Leave-one-system-out results ===")
    print(system_runs.to_string(index=False))
    print("\n=== Leave-one-system-out summary ===")
    print(system_summary.to_string(index=False))

    print("\n=== Bulk-to-interface result ===")
    print(bulk2int_runs.to_string(index=False))
    print("\n=== Bulk-to-interface summary ===")
    print(bulk2int_summary.to_string(index=False))

    print("\n=== Leave-one-temperature-out results ===")
    print(temp_runs.to_string(index=False))
    print("\n=== Leave-one-temperature-out summary ===")
    print(temp_summary.to_string(index=False))

    print("\n=== Low-to-high transfer result ===")
    print(l2h_runs.to_string(index=False))
    print("\n=== Low-to-high transfer summary ===")
    print(l2h_summary.to_string(index=False))

    print(f"\nSaved: {OUT_SYSTEM_RUNS}")
    print(f"Saved: {OUT_SYSTEM_SUMMARY}")
    print(f"Saved: {OUT_BULK2INT_RUNS}")
    print(f"Saved: {OUT_BULK2INT_SUMMARY}")
    print(f"Saved: {OUT_TEMP_RUNS}")
    print(f"Saved: {OUT_TEMP_SUMMARY}")
    print(f"Saved: {OUT_L2H_RUNS}")
    print(f"Saved: {OUT_L2H_SUMMARY}")


if __name__ == "__main__":
    main()