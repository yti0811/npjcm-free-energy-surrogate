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

OUT_RUNS = TABLE_DIR / "Table_S_repeated_training_runs.csv"
OUT_SUMMARY = TABLE_DIR / "Table_S_repeated_training_summary.csv"

SEEDS = list(range(10))
TI_DOMAIN_SYSTEMS = ["alpha-TiAl", "beta-TiV", "Ti64"]


def import_module_from_file(module_path: Path, module_name: str = "prod_selection_module"):
    if not module_path.exists():
        raise FileNotFoundError(f"Production script not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sem(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def most_frequent_or_nan(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    mode_vals = s.mode()
    return mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan


def safe_get_weight(best_row: pd.Series) -> float:
    return best_row["w"] if "w" in best_row.index else np.nan


def main():
    prod = import_module_from_file(PRODUCTION_SCRIPT)

    required_funcs = ["auto_map_columns", "make_state_features", "cv_select_ti_model"]
    for fname in required_funcs:
        if not hasattr(prod, fname):
            raise AttributeError(
                f"Production script is missing required function: {fname}\n"
                f"Expected to find {required_funcs} in:\n{PRODUCTION_SCRIPT}"
            )

    auto_map_columns = getattr(prod, "auto_map_columns")
    make_state_features = getattr(prod, "make_state_features")
    cv_select_ti_model = getattr(prod, "cv_select_ti_model")

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    df = pd.read_csv(SNAPSHOT_FILE)
    df, colmap = auto_map_columns(df)

    system_col = colmap["system"]
    df_ti = df[df[system_col].isin(TI_DOMAIN_SYSTEMS)].copy()

    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty after system filtering.")

    X_ti, feat_cols = make_state_features(
        df_ti,
        colmap,
        include_baseline_terms=True
    )

    run_rows = []

    for seed in SEEDS:
        sel_df = cv_select_ti_model(
            df_ti=df_ti,
            X_ti=X_ti,
            colmap=colmap,
            feat_cols=feat_cols,
            seed=seed
        )

        if sel_df is None or len(sel_df) == 0:
            raise RuntimeError(f"cv_select_ti_model returned no rows for seed={seed}")

        best = sel_df.iloc[0]

        run_rows.append({
            "seed": seed,
            "selected_model": best["model"],
            "blend_weight": safe_get_weight(best),
            "MAE_abs": best["MAE_abs(Fhat)"],
            "f_viol": best["f_viol_abs(Fhat)"],
            "tau": best["kendall_tau_abs(Fhat)"],
            "score": best["score"],
        })

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUT_RUNS, index=False)

    summary = pd.DataFrame([{
        "n_repeats": len(runs),
        "most_frequent_model": most_frequent_or_nan(runs["selected_model"]),
        "MAE_mean": runs["MAE_abs"].mean(),
        "MAE_std": runs["MAE_abs"].std(ddof=1),
        "MAE_sem": sem(runs["MAE_abs"]),
        "f_viol_mean": runs["f_viol"].mean(),
        "f_viol_std": runs["f_viol"].std(ddof=1),
        "f_viol_sem": sem(runs["f_viol"]),
        "tau_mean": runs["tau"].mean(),
        "tau_std": runs["tau"].std(ddof=1),
        "tau_sem": sem(runs["tau"]),
        "score_mean": runs["score"].mean(),
        "score_std": runs["score"].std(ddof=1),
        "score_sem": sem(runs["score"]),
        "blend_weight_mean": runs["blend_weight"].dropna().mean() if runs["blend_weight"].notna().any() else np.nan,
        "blend_weight_std": runs["blend_weight"].dropna().std(ddof=1) if runs["blend_weight"].notna().sum() > 1 else np.nan,
        "blend_weight_sem": sem(runs["blend_weight"].dropna()) if runs["blend_weight"].notna().sum() > 1 else np.nan,
    }])

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nRepeated training runs:")
    print(runs.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved runs table:    {OUT_RUNS}")
    print(f"Saved summary table: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()