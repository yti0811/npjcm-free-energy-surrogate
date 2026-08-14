#!/usr/bin/env python3
"""
summarize_residual_metrics_revised_v5.py

Compatibility and staged-reconstruction summary for the audited prediction
tables produced by the anchor-relative Backbone Ridge core.

Inputs
------
table/piml_predictions_Ti.csv
table/piml_predictions_Al.csv
table/piml_predictions_Cu.csv

The retained Backbone Ridge prediction is never refitted here. It is read
directly from the production columns and validated against

    Fhat_final
    = F0_eVatom
    + dFraw_T0_mean
    + dFres_pred_backbone
    + dFres_pred_correction

and

    FANC_eVatom
    = F0_eVatom
    + dFraw_T0_mean
    + dFres_eVatom.

Version 5 additions
-------------------
1. Preserves the established residual-summary outputs from v4.
2. Adds absolute free-energy reconstruction stages:
       minimal_baseline : F0
       gauge_aligned    : F0 + C_s
       thermal_backbone : F0 + C_s + g_hat(T)
       backbone_ridge   : F0 + C_s + g_hat(T) + r_hat(delta x)
3. Writes stage-resolved MAE/RMSE/R2 tables globally, by system, and by
   system-temperature partition.
4. Adds all four absolute reconstruction columns to residual_predictions.csv,
   allowing direct construction of the staged-reconstruction summaries reported in Supplementary Table S16 and Fig. 8(c).

Important protocol note
-----------------------
The combined residual-model summary is retained only for backward-compatible
diagnostics. The retained Backbone Ridge model uses Ti-domain training followed
by anchor-aligned Al/Cu no-retraining transfer, whereas the auxiliary
descriptor/control models are fitted here by grouped OOF prediction on the
combined five-system table. Therefore residual_summary_global.csv must not be
used as a fair model leaderboard.

For manuscript-facing retained-model comparisons, use the new
reconstruction_stage_summary_*.csv files together with the dedicated Ti-domain
and transfer summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILENAMES = (
    "piml_predictions_Ti.csv",
    "piml_predictions_Al.csv",
    "piml_predictions_Cu.csv",
)

# Backward-compatible outputs
OUT_PRED = TABLE_DIR / "residual_predictions.csv"
OUT_BY_ST = TABLE_DIR / "residual_summary_by_system_T.csv"
OUT_BY_SYS = TABLE_DIR / "residual_summary_by_system.csv"
OUT_GLOBAL = TABLE_DIR / "residual_summary_global.csv"
OUT_AUDIT = TABLE_DIR / "residual_summary_audit.csv"

# New v5 staged absolute-reconstruction outputs
OUT_STAGE_BY_ST = TABLE_DIR / "reconstruction_stage_summary_by_system_T.csv"
OUT_STAGE_BY_SYS = TABLE_DIR / "reconstruction_stage_summary_by_system.csv"
OUT_STAGE_GLOBAL = TABLE_DIR / "reconstruction_stage_summary_global.csv"

THERMODYNAMIC_REQUIRED = {
    "system",
    "T",
    "snap",
    "FANC_eVatom",
    "F0_eVatom",
    "dFraw_T0_mean",
    "dFres_eVatom",
    "dFres_pred_backbone",
    "dFres_pred_correction",
    "dFres_pred_final",
    "Fhat_final",
}

SLE_FEATURE_CANDIDATES = (
    "SLE_mean", "SLE_std", "SLE_q25",
    "SLE_q50", "SLE_q75", "SLE_iqr",
)
VORO_FEATURE_CANDIDATES = (
    "Voro_mean", "Voro_q25", "Voro_q50",
    "Voro_q75", "Voro_iqr",
)
Q6_FEATURE_CANDIDATES = (
    "q6_mean", "q6_std", "q6_q25",
    "q6_q50", "q6_q75", "q6_iqr",
    "q6knn_mean",
)
RESPONSE_FEATURE_CANDIDATES = (
    "vMises_mean", "vMises_std", "vMises_q25",
    "vMises_q50", "vMises_q75", "vMises_iqr",
)

METHOD_ORDER = (
    "enriched_d2",
    "enriched_d3",
    "temp_only",
    "response_only",
    "piml",
)

STAGE_ORDER = (
    "minimal_baseline",
    "gauge_aligned",
    "thermal_backbone",
    "backbone_ridge",
)

STAGE_LABELS = {
    "minimal_baseline": r"F0",
    "gauge_aligned": r"F0+Cs",
    "thermal_backbone": r"F0+Cs+g(T)",
    "backbone_ridge": r"F0+Cs+g(T)+r(delta_x)",
}

STAGE_PREDICTION_COLUMNS = {
    "minimal_baseline": "Fpred_minimal_baseline",
    "gauge_aligned": "Fpred_gauge_aligned",
    "thermal_backbone": "Fpred_thermal_backbone",
    "backbone_ridge": "Fpred_backbone_ridge",
}


def resolve_input_file(filename: str) -> Path:
    candidates = (
        TABLE_DIR / filename,
        SCRIPT_DIR / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Required input file {filename!r} was not found. Searched:\n"
        f"  - {searched}"
    )


def existing_columns(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> list[str]:
    return [column for column in candidates if column in df.columns]


def load_prediction_tables() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for filename in INPUT_FILENAMES:
        path = resolve_input_file(filename)
        frame = pd.read_csv(path)

        missing = THERMODYNAMIC_REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing)}"
            )

        frame = frame.copy()
        frame["__source_file"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)

    numeric_required = (
        "T",
        "FANC_eVatom",
        "F0_eVatom",
        "dFraw_T0_mean",
        "dFres_eVatom",
        "dFres_pred_backbone",
        "dFres_pred_correction",
        "dFres_pred_final",
        "Fhat_final",
    )
    for column in numeric_required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    bad = df[list(numeric_required)].isna().any(axis=1)
    if bad.any():
        examples = df.loc[
            bad,
            ["__source_file", "system", "T", "snap"],
        ].head(10)
        raise ValueError(
            "Missing/non-numeric required thermodynamic values. Examples:\n"
            + examples.to_string(index=False)
        )

    key_columns = ["system", "T", "snap"]
    duplicate = df.duplicated(key_columns, keep=False)
    if duplicate.any():
        examples = df.loc[
            duplicate,
            ["__source_file", *key_columns],
        ].head(20)
        raise ValueError(
            "Duplicate snapshot keys were found:\n"
            + examples.to_string(index=False)
        )

    # Validate the Backbone Ridge residual decomposition.
    expected_dFres_final = (
        df["dFres_pred_backbone"].to_numpy(float)
        + df["dFres_pred_correction"].to_numpy(float)
    )
    actual_dFres_final = df["dFres_pred_final"].to_numpy(float)
    if not np.allclose(
        expected_dFres_final,
        actual_dFres_final,
        rtol=1e-10,
        atol=1e-10,
    ):
        discrepancy = float(
            np.max(np.abs(expected_dFres_final - actual_dFres_final))
        )
        raise ValueError(
            "dFres_pred_final is inconsistent with backbone + correction. "
            f"Maximum discrepancy: {discrepancy:.6e} eV/atom."
        )

    expected_fhat = (
        df["F0_eVatom"].to_numpy(float)
        + df["dFraw_T0_mean"].to_numpy(float)
        + df["dFres_pred_final"].to_numpy(float)
    )
    actual_fhat = df["Fhat_final"].to_numpy(float)
    if not np.allclose(expected_fhat, actual_fhat, rtol=1e-10, atol=1e-10):
        discrepancy = float(np.max(np.abs(expected_fhat - actual_fhat)))
        raise ValueError(
            "Fhat_final is inconsistent with the retained reconstruction. "
            f"Maximum discrepancy: {discrepancy:.6e} eV/atom."
        )

    expected_reference = (
        df["F0_eVatom"].to_numpy(float)
        + df["dFraw_T0_mean"].to_numpy(float)
        + df["dFres_eVatom"].to_numpy(float)
    )
    actual_reference = df["FANC_eVatom"].to_numpy(float)
    if not np.allclose(
        expected_reference,
        actual_reference,
        rtol=1e-10,
        atol=1e-10,
    ):
        discrepancy = float(
            np.max(np.abs(expected_reference - actual_reference))
        )
        raise ValueError(
            "Reference columns are inconsistent with the anchored "
            "decomposition. "
            f"Maximum discrepancy: {discrepancy:.6e} eV/atom."
        )

    df["cv_group"] = (
        df["system"].astype(str)
        + "_"
        + df["T"].astype(int).astype(str)
    )
    return df


def add_reconstruction_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Add all absolute reconstruction stages without refitting."""
    out = df.copy()

    out["Fpred_minimal_baseline"] = out["F0_eVatom"].to_numpy(float)
    out["Fpred_gauge_aligned"] = (
        out["F0_eVatom"].to_numpy(float)
        + out["dFraw_T0_mean"].to_numpy(float)
    )
    out["Fpred_thermal_backbone"] = (
        out["F0_eVatom"].to_numpy(float)
        + out["dFraw_T0_mean"].to_numpy(float)
        + out["dFres_pred_backbone"].to_numpy(float)
    )
    out["Fpred_backbone_ridge"] = out["Fhat_final"].to_numpy(float)

    return out


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str,
) -> None:
    if not feature_columns:
        raise ValueError(f"No usable features found for {method!r}.")

    forbidden_tokens = (
        "fanc",
        "dfraw",
        "dfres",
        "pred",
        "fhat",
        "natoms",
        "snap",
        "timestep",
    )
    forbidden = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(
            f"Forbidden features detected for {method}: {forbidden}"
        )

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[feature_columns].isna().any(axis=1)
    if invalid.any():
        examples = df.loc[
            invalid,
            ["__source_file", "system", "T", "snap", *feature_columns],
        ].head(10)
        raise ValueError(
            f"Missing/non-numeric features for {method!r}. Examples:\n"
            + examples.to_string(index=False)
        )


def fit_predict_group_cv(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    group_column: str,
    model,
    maximum_splits: int = 5,
) -> np.ndarray:
    x = df.loc[:, feature_columns].to_numpy(float)
    y = df[target_column].to_numpy(float)
    groups = df[group_column].to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(maximum_splits, len(unique_groups))
    if n_splits < 2:
        fitted = clone(model)
        fitted.fit(x, y)
        return np.asarray(fitted.predict(x), dtype=float)

    output = np.full(len(df), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)

    for train_index, test_index in splitter.split(x, y, groups):
        fitted = clone(model)
        fitted.fit(x[train_index], y[train_index])
        output[test_index] = fitted.predict(x[test_index])

    if np.isnan(output).any():
        raise RuntimeError(
            f"Grouped OOF prediction left "
            f"{int(np.isnan(output).sum())} rows empty."
        )
    return output


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    sle = existing_columns(df, SLE_FEATURE_CANDIDATES)
    voro = existing_columns(df, VORO_FEATURE_CANDIDATES)
    q6 = existing_columns(df, Q6_FEATURE_CANDIDATES)
    response = existing_columns(df, RESPONSE_FEATURE_CANDIDATES)

    if not sle:
        raise ValueError("No SLE descriptor columns found.")
    if not voro:
        raise ValueError("No Voronoi descriptor columns found.")
    if not q6:
        raise ValueError("No q6 descriptor columns found.")
    if not response:
        raise ValueError("No response-only von Mises columns found.")

    return {
        "enriched_d2": ["T", *sle, *voro],
        "enriched_d3": ["T", *sle, *voro, *q6],
        "temp_only": ["T"],
        "response_only": ["T", *response],
        "piml": [],
    }


def reconstruct_absolute(
    df: pd.DataFrame,
    residual_prediction: np.ndarray,
) -> np.ndarray:
    return (
        df["F0_eVatom"].to_numpy(float)
        + df["dFraw_T0_mean"].to_numpy(float)
        + np.asarray(residual_prediction, dtype=float)
    )


def build_stage_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build global, by-system, and by-system-temperature stage summaries."""
    rows_by_state: list[dict] = []

    for (system, temperature), subset in df.groupby(
        ["system", "T"],
        sort=True,
    ):
        y_true = subset["FANC_eVatom"].to_numpy(float)
        for stage in STAGE_ORDER:
            pred_column = STAGE_PREDICTION_COLUMNS[stage]
            y_pred = subset[pred_column].to_numpy(float)
            mae, rmse, r2 = regression_metrics(y_true, y_pred)
            rows_by_state.append({
                "system": system,
                "T": int(temperature),
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "n_snap": len(subset),
                "MAE_F": mae,
                "RMSE_F": rmse,
                "R2_F": r2,
            })

    by_state = (
        pd.DataFrame(rows_by_state)
        .sort_values(["system", "T", "stage"])
        .reset_index(drop=True)
    )

    by_system = (
        by_state.groupby(
            ["system", "stage", "stage_label"],
            as_index=False,
        )
        .agg(
            n_T=("T", "count"),
            n_snap=("n_snap", "sum"),
            MAE_mean_of_partitions=("MAE_F", "mean"),
            RMSE_mean_of_partitions=("RMSE_F", "mean"),
            R2_mean_of_partitions=("R2_F", "mean"),
        )
        .sort_values(["system", "stage"])
        .reset_index(drop=True)
    )

    global_rows: list[dict] = []
    y_true_all = df["FANC_eVatom"].to_numpy(float)
    for stage in STAGE_ORDER:
        pred_column = STAGE_PREDICTION_COLUMNS[stage]
        y_pred_all = df[pred_column].to_numpy(float)
        mae, rmse, r2 = regression_metrics(y_true_all, y_pred_all)
        global_rows.append({
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "n_snap": len(df),
            "MAE_F": mae,
            "RMSE_F": rmse,
            "R2_F": r2,
        })

    global_table = (
        pd.DataFrame(global_rows)
        .sort_values("stage")
        .reset_index(drop=True)
    )

    return by_state, by_system, global_table


def main() -> None:
    df = load_prediction_tables()
    df = add_reconstruction_stages(df)
    feature_sets = build_feature_sets(df)

    models = {
        "enriched_d2": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "enriched_d3": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "temp_only": LinearRegression(),
        "response_only": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "piml": None,
    }

    feature_audit_rows = []

    for method in METHOD_ORDER:
        if method == "piml":
            df["dFpred_piml"] = df["dFres_pred_final"].to_numpy(float)
            df["Fpred_piml"] = df["Fhat_final"].to_numpy(float)
            feature_audit_rows.append({
                "method": method,
                "evaluation_protocol":
                    "retained_Ti_trained_model_plus_external_transfer",
                "n_features": np.nan,
                "feature_columns": "",
                "refitted_in_summary": False,
            })
            continue

        features = feature_sets[method]
        validate_feature_columns(df, features, method)

        df[f"dFpred_{method}"] = fit_predict_group_cv(
            df=df,
            feature_columns=features,
            target_column="dFres_eVatom",
            group_column="cv_group",
            model=models[method],
        )
        df[f"Fpred_{method}"] = reconstruct_absolute(
            df,
            df[f"dFpred_{method}"].to_numpy(float),
        )
        feature_audit_rows.append({
            "method": method,
            "evaluation_protocol": "combined_five_system_grouped_OOF",
            "n_features": len(features),
            "feature_columns": "|".join(features),
            "refitted_in_summary": True,
        })

    keep_columns = [
        "system",
        "T",
        "snap",
        "timestep",
        "FANC_eVatom",
        "F0_eVatom",
        "dFraw_T0_mean",
        "dFres_eVatom",
        "dFres_pred_backbone",
        "dFres_pred_correction",
        "dFres_pred_final",
        "Fhat_final",
        "Fpred_minimal_baseline",
        "Fpred_gauge_aligned",
        "Fpred_thermal_backbone",
        "Fpred_backbone_ridge",
        "dFpred_enriched_d2",
        "dFpred_enriched_d3",
        "dFpred_temp_only",
        "dFpred_response_only",
        "dFpred_piml",
        "Fpred_enriched_d2",
        "Fpred_enriched_d3",
        "Fpred_temp_only",
        "Fpred_response_only",
        "Fpred_piml",
    ]
    keep_columns = [column for column in keep_columns if column in df.columns]
    df.loc[:, keep_columns].to_csv(OUT_PRED, index=False)

    # Backward-compatible residual summaries
    rows_by_state: list[dict] = []
    for (system, temperature), subset in df.groupby(
        ["system", "T"],
        sort=True,
    ):
        y_true = subset["dFres_eVatom"].to_numpy(float)
        for method in METHOD_ORDER:
            y_pred = subset[f"dFpred_{method}"].to_numpy(float)
            mae, rmse, r2 = regression_metrics(y_true, y_pred)
            rows_by_state.append({
                "system": system,
                "T": int(temperature),
                "method": method,
                "n_snap": len(subset),
                "MAE_dFres": mae,
                "RMSE_dFres": rmse,
                "R2_dFres": r2,
            })

    by_state = (
        pd.DataFrame(rows_by_state)
        .sort_values(["system", "T", "method"])
        .reset_index(drop=True)
    )
    by_state.to_csv(OUT_BY_ST, index=False)

    by_system = (
        by_state.groupby(["system", "method"], as_index=False)
        .agg(
            n_T=("T", "count"),
            MAE_mean=("MAE_dFres", "mean"),
            RMSE_mean=("RMSE_dFres", "mean"),
            R2_mean=("R2_dFres", "mean"),
        )
        .sort_values(["system", "method"])
        .reset_index(drop=True)
    )
    by_system.to_csv(OUT_BY_SYS, index=False)

    global_rows = []
    y_true_all = df["dFres_eVatom"].to_numpy(float)
    for method in METHOD_ORDER:
        y_pred_all = df[f"dFpred_{method}"].to_numpy(float)
        mae, rmse, r2 = regression_metrics(y_true_all, y_pred_all)
        global_rows.append({
            "method": method,
            "n_snap": len(df),
            "MAE_dFres": mae,
            "RMSE_dFres": rmse,
            "R2_dFres": r2,
            "protocol_comparable_to_piml": method == "piml",
        })

    global_table = (
        pd.DataFrame(global_rows)
        .sort_values("method")
        .reset_index(drop=True)
    )
    global_table.to_csv(OUT_GLOBAL, index=False)

    # New v5 staged absolute-free-energy summaries
    stage_by_state, stage_by_system, stage_global = build_stage_tables(df)
    stage_by_state.to_csv(OUT_STAGE_BY_ST, index=False)
    stage_by_system.to_csv(OUT_STAGE_BY_SYS, index=False)
    stage_global.to_csv(OUT_STAGE_GLOBAL, index=False)

    audit = pd.DataFrame(feature_audit_rows)
    audit["combined_global_table_is_fair_leaderboard"] = False
    audit["recommended_manuscript_summary"] = (
        "reconstruction_stage_summary_*.csv + dedicated Ti/transfer summaries"
    )
    audit["v5_stage_outputs_added"] = True
    audit["stage_definitions"] = (
        "minimal_baseline=F0;"
        "gauge_aligned=F0+Cs;"
        "thermal_backbone=F0+Cs+g_hat(T);"
        "backbone_ridge=F0+Cs+g_hat(T)+r_hat(delta_x)"
    )
    audit.to_csv(OUT_AUDIT, index=False)

    print("Input files:")
    for filename in INPUT_FILENAMES:
        print(" -", resolve_input_file(filename))

    print("\nProtocol warning:")
    print(
        " - residual_summary_global.csv combines different evaluation "
        "protocols and must not be interpreted as a fair model leaderboard."
    )

    print("\nBackward-compatible outputs:")
    for path in (OUT_PRED, OUT_BY_ST, OUT_BY_SYS, OUT_GLOBAL, OUT_AUDIT):
        print(" -", path)

    print("\nNew v5 staged reconstruction outputs:")
    for path in (OUT_STAGE_BY_ST, OUT_STAGE_BY_SYS, OUT_STAGE_GLOBAL):
        print(" -", path)


if __name__ == "__main__":
    main()
