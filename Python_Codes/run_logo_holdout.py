from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTION_SCRIPT = SCRIPT_DIR / "run_piml_core_pipeline.py"
SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_RUNS = TABLE_DIR / "logo_holdout_runs.csv"
OUT_SUMMARY = TABLE_DIR / "table_s12_logo_holdout.csv"

SEED = 0
TI_DOMAIN_SYSTEMS = ["alpha-TiAl", "beta-TiV", "Ti64"]


def import_module_from_file(module_path: Path, module_name: str = "prod_logo_module"):
    if not module_path.exists():
        raise FileNotFoundError(f"Production script not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_group_labels(df: pd.DataFrame, system_col: str, T_col: str) -> pd.Series:
    return df[system_col].astype(str) + "_" + df[T_col].astype(int).astype(str)


def mean_abs_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


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
            "w": best_row["w"],
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


def reconstruct_fhat(df_sub, colmap, dFres_pred):
    dFraw = dFres_pred + df_sub[colmap["dFraw_T0_mean"]].to_numpy(dtype=float)
    Fhat = df_sub[colmap["F0"]].to_numpy(dtype=float) + dFraw
    return Fhat


def ordering_metrics_ties_fail(y_pred, y_ref):
    y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    conc = disc = total = 0
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
    return f_viol, tau, total


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

    auto_map_columns = getattr(prod, "auto_map_columns")
    make_state_features = getattr(prod, "make_state_features")
    cv_select_ti_model = getattr(prod, "cv_select_ti_model")

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    df = pd.read_csv(SNAPSHOT_FILE)
    df, colmap = auto_map_columns(df)

    system_col = colmap["system"]
    T_col = colmap["T"]
    fanc_col = colmap["FANC"]
    dFres_col = colmap["dFres"]
    f0_col = colmap["F0"]

    df_ti = df[df[system_col].isin(TI_DOMAIN_SYSTEMS)].copy()
    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty.")

    groups = get_group_labels(df_ti, system_col, T_col)
    unique_groups = sorted(groups.unique())

    run_rows = []
    summary_rows = []

    for held_out in unique_groups:
        is_test = groups == held_out
        train_df = df_ti.loc[~is_test].copy().reset_index(drop=True)
        test_df = df_ti.loc[is_test].copy().reset_index(drop=True)

        X_train, feat_cols = make_state_features(train_df, colmap, include_baseline_terms=True)
        X_test, _ = make_state_features(test_df, colmap, include_baseline_terms=True)

        sel_df = cv_select_ti_model(
            df_ti=train_df,
            X_ti=X_train,
            colmap=colmap,
            feat_cols=feat_cols,
            seed=SEED,
        )

        best = sel_df.iloc[0]
        fitted = fit_best_model(best, prod, X_train, train_df[dFres_col].to_numpy(dtype=float))
        dFres_pred_test = predict_best_model(fitted, prod, X_test)

        Fhat_test = reconstruct_fhat(test_df, colmap, dFres_pred_test)
        y_ref_test = test_df[fanc_col].to_numpy(dtype=float)
        y_base_test = test_df[f0_col].to_numpy(dtype=float)

        baseline_mae = mean_abs_error(y_ref_test, y_base_test)
        holdout_mae = mean_abs_error(y_ref_test, Fhat_test)

        run_rows.append({
            "held_out_group": held_out,
            "snapshots": len(test_df),
            "selected_model": best["model"],
            "blend_weight": best["w"] if "w" in best.index else np.nan,
            "baseline_MAE_abs": baseline_mae,
            "holdout_MAE_abs": holdout_mae,
        })

        summary_rows.append({
            "held_out_group": held_out,
            "selected_model": best["model"],
            "blend_weight": best["w"] if "w" in best.index else np.nan,
            "n_snapshots": len(test_df),
            "baseline_MAE_abs": baseline_mae,
            "holdout_MAE_abs": holdout_mae,
        })

    runs = pd.DataFrame(run_rows).sort_values("held_out_group").reset_index(drop=True)
    runs.to_csv(OUT_RUNS, index=False)

    summary = pd.DataFrame(summary_rows).sort_values("held_out_group").reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(runs.to_string(index=False))
    print(f"\nSaved: {OUT_RUNS}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()