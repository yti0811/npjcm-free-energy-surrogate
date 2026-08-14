#!/usr/bin/env python3
"""
run_repeated_training.py

Repeated execution audit for the anchor-relative Ridge production selector.

The explicitly named Ridge core is loaded first:
    run_piml_core_pipeline_ridge.py
and run_piml_core_pipeline.py is used only as a fallback.

The script maps the complete Ti-domain table once, verifies the actual generated
feature schema, and repeats the production grouped-CV selector for the requested
seeds. Because fixed Ridge + GroupKFold is deterministic, identical results across
seeds should be interpreted as pipeline reproducibility rather than independent
stochastic-training uncertainty.

Outputs
-------
table/table_S13_repeated_training_runs.csv
table/table_S13_repeated_training_summary.csv
table/repeated_training_audit.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_CORE = SCRIPT_DIR / "run_piml_core_pipeline_backbone.py"
RIDGE_CORE = SCRIPT_DIR / "run_piml_core_pipeline_ridge.py"
LEGACY_NAME_CORE = SCRIPT_DIR / "run_piml_core_pipeline.py"
PRODUCTION_SCRIPT = (
    BACKBONE_CORE if BACKBONE_CORE.exists()
    else RIDGE_CORE if RIDGE_CORE.exists()
    else LEGACY_NAME_CORE
)

SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_RUNS = TABLE_DIR / "table_S13_repeated_training_runs.csv"
OUT_SUMMARY = TABLE_DIR / "table_S13_repeated_training_summary.csv"
OUT_AUDIT = TABLE_DIR / "repeated_training_audit.csv"

SEEDS = list(range(10))
TI_DOMAIN_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
EXPECTED_MODEL = "backbone_ridge"
EXPECTED_REPRESENTATION = "quadratic_thermal_backbone_plus_T0_relative_correction"


def import_module_from_file(
    module_path: Path,
    module_name: str = "prod_repeated_module",
):
    if not module_path.exists():
        raise FileNotFoundError(f"Production script not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_t0(prod: Any) -> float:
    if hasattr(prod, "T0"):
        value = float(getattr(prod, "T0"))
    elif hasattr(prod, "ANCHOR_TEMPERATURE"):
        value = float(getattr(prod, "ANCHOR_TEMPERATURE"))
    elif hasattr(prod, "T_ANCHOR"):
        value = float(getattr(prod, "T_ANCHOR"))
    else:
        value = 300.0
    setattr(prod, "T0", value)
    return value


def require_core(prod: Any) -> float:
    required = (
        "auto_map_columns",
        "make_state_features",
        "candidate_models",
        "cv_select_ti_model",
    )
    missing = [name for name in required if not hasattr(prod, name)]
    if missing:
        raise AttributeError(
            "Production core is missing required functions: "
            + ", ".join(missing)
        )

    models = prod.candidate_models(seed=0)
    if EXPECTED_MODEL not in models:
        raise RuntimeError(
            f"Expected production model {EXPECTED_MODEL!r}; "
            f"available models: {sorted(models)}"
        )
    return resolve_t0(prod)


def validate_feature_schema(feature_columns: list[str]) -> None:
    if not feature_columns:
        raise RuntimeError("No production features were generated.")

    lower = [str(column).lower() for column in feature_columns]
    has_relative_temperature = any(
        column in {"dt", "dt_anchor", "d_t", "delta_t"}
        or column.startswith("dt_")
        for column in lower
    )
    n_relative = sum(
        column.startswith("d_") or "anchor" in column
        for column in lower
    )
    if not has_relative_temperature or n_relative < 3:
        raise RuntimeError(
            "The loaded production core did not generate the expected "
            "anchor-relative feature schema. Generated columns: "
            + ", ".join(map(str, feature_columns))
        )

    forbidden_tokens = (
        "fanc", "dfraw", "dfres", "pred", "fhat",
        "stress", "vmises", "natoms", "snap", "timestep",
    )
    forbidden = [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden feature columns detected: {forbidden}")


def sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) <= 1:
        return np.nan
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def most_frequent_or_nan(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return np.nan
    mode_values = clean.mode()
    return mode_values.iloc[0] if len(mode_values) else np.nan


def safe_get_weight(best_row: pd.Series) -> float:
    if "w" in best_row.index and pd.notna(best_row["w"]):
        return float(best_row["w"])
    return np.nan


def main() -> None:
    prod = import_module_from_file(PRODUCTION_SCRIPT)
    t0 = require_core(prod)

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    raw = pd.read_csv(SNAPSHOT_FILE)
    mapped, colmap = prod.auto_map_columns(raw)

    system_col = str(colmap["system"])
    df_ti = (
        mapped[mapped[system_col].astype(str).isin(TI_DOMAIN_SYSTEMS)]
        .copy()
        .reset_index(drop=True)
    )
    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty after system filtering.")

    x_ti, feature_columns = prod.make_state_features(
        df_ti,
        colmap,
        include_baseline_terms=True,
    )
    validate_feature_schema(feature_columns)

    run_rows = []
    for seed in SEEDS:
        selection = prod.cv_select_ti_model(
            df_ti=df_ti,
            X_ti=x_ti,
            colmap=colmap,
            feat_cols=feature_columns,
            seed=seed,
        )
        if selection is None or selection.empty:
            raise RuntimeError(
                f"cv_select_ti_model returned no rows for seed={seed}."
            )

        best = selection.iloc[0]
        selected_model = str(best["model"])
        if selected_model != EXPECTED_MODEL:
            raise RuntimeError(
                f"Seed {seed} selected {selected_model!r}; "
                f"expected {EXPECTED_MODEL!r}."
            )

        run_rows.append({
            "seed": seed,
            "selected_model": selected_model,
            "blend_weight": safe_get_weight(best),
            "MAE_abs": float(best["MAE_abs(Fhat)"]),
            "f_viol": float(best["f_viol_abs(Fhat)"]),
            "tau": float(best["kendall_tau_abs(Fhat)"]),
            "score": float(best["score"]),
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
        "blend_weight_mean": (
            runs["blend_weight"].dropna().mean()
            if runs["blend_weight"].notna().any() else np.nan
        ),
        "blend_weight_std": (
            runs["blend_weight"].dropna().std(ddof=1)
            if runs["blend_weight"].notna().sum() > 1 else np.nan
        ),
        "blend_weight_sem": (
            sem(runs["blend_weight"].dropna())
            if runs["blend_weight"].notna().sum() > 1 else np.nan
        ),
    }])
    summary.to_csv(OUT_SUMMARY, index=False)

    audit = pd.DataFrame([{
        "production_core_file": str(PRODUCTION_SCRIPT),
        "production_model": EXPECTED_MODEL,
        "feature_representation": EXPECTED_REPRESENTATION,
        "anchor_temperature_K": t0,
        "n_features": len(feature_columns),
        "feature_columns": "|".join(feature_columns),
        "n_seeds": len(SEEDS),
        "split_randomized_by_seed": False,
        "model_stochastic_by_seed": False,
        "interpretation": "deterministic_pipeline_reproducibility",
        "contains_forbidden_feature": False,
    }])
    audit.to_csv(OUT_AUDIT, index=False)

    print("\nProduction core:")
    print(f"  file: {PRODUCTION_SCRIPT}")
    print(f"  model: {EXPECTED_MODEL}")
    print(f"  representation: {EXPECTED_REPRESENTATION}")
    print(f"  anchor temperature: {t0:g} K")
    print(f"  features: {len(feature_columns)}")

    print("\nRepeated training runs:")
    print(runs.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved runs table:    {OUT_RUNS}")
    print(f"Saved summary table: {OUT_SUMMARY}")
    print(f"Saved audit table:   {OUT_AUDIT}")


if __name__ == "__main__":
    main()
