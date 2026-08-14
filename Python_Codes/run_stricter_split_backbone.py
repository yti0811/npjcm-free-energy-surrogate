#!/usr/bin/env python3
"""
run_stricter_split.py

Stricter Ti-domain validation using the audited anchor-relative Ridge production
core in run_piml_core_pipeline.py.

The complete processed Ti-domain table is anchor-aligned once before splitting.
This permits system-specific 300 K descriptor centering without using held-out
free-energy targets for fitting. Absolute reconstruction uses

    F_hat = F0 + C_s + DeltaF_res,pred.

Evaluations
-----------
1. Leave-one-system-out
2. Bulk-to-interface transfer
3. Leave-one-temperature-out
4. Low-to-high-temperature transfer

Interpretation
--------------
These are anchor-aligned residual-transfer tests. A held-out system's 300 K
descriptor ensemble is used only for feature alignment, and its C_s is used only
for absolute reconstruction. No held-out FANC, dFraw, or dFres value enters model
fitting or Ti-only model selection.

Version 5 additionally writes snapshot-level and state-mean prediction tables for
failure diagnosis, especially for leave-one-system-out transfer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

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

OUT_SYSTEM_RUNS = TABLE_DIR / "stricter_system_holdout_runs.csv"
OUT_SYSTEM_SUMMARY = TABLE_DIR / "stricter_system_holdout_summary.csv"
OUT_BULK2INT_RUNS = TABLE_DIR / "stricter_bulk_to_interface_runs.csv"
OUT_BULK2INT_SUMMARY = TABLE_DIR / "stricter_bulk_to_interface_summary.csv"
OUT_TEMP_RUNS = TABLE_DIR / "stricter_temperature_holdout_runs.csv"
OUT_TEMP_SUMMARY = TABLE_DIR / "stricter_temperature_holdout_summary.csv"
OUT_L2H_RUNS = TABLE_DIR / "stricter_low_to_high_runs.csv"
OUT_L2H_SUMMARY = TABLE_DIR / "stricter_low_to_high_summary.csv"
OUT_AUDIT = TABLE_DIR / "stricter_split_audit.csv"
OUT_SYSTEM_PRED = TABLE_DIR / "stricter_system_holdout_predictions.csv"
OUT_STATE_MEANS = TABLE_DIR / "stricter_state_mean_predictions.csv"
OUT_TEMP_PRED = TABLE_DIR / "stricter_temperature_holdout_predictions.csv"
OUT_BULK2INT_PRED = TABLE_DIR / "stricter_bulk_to_interface_predictions.csv"
OUT_L2H_PRED = TABLE_DIR / "stricter_low_to_high_predictions.csv"

SEED = 0
TI_DOMAIN_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
EXPECTED_MODEL = "backbone_ridge"
EXPECTED_REPRESENTATION = "quadratic_thermal_backbone_plus_T0_relative_correction"


def import_module_from_file(path: Path, name: str = "prod_stricter_module"):
    if not path.exists():
        raise FileNotFoundError(f"Production script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_core(prod: Any) -> None:
    required = (
        "auto_map_columns", "make_state_features", "candidate_models",
        "cv_select_ti_model", "fit_final_model", "predict_final",
        "absolute_metrics",
    )
    missing = [name for name in required if not hasattr(prod, name)]
    if missing:
        raise AttributeError("Production core missing: " + ", ".join(missing))

    models = prod.candidate_models(seed=SEED)
    if EXPECTED_MODEL not in models:
        raise RuntimeError(
            f"Expected production model {EXPECTED_MODEL!r}; "
            f"available models: {sorted(models)}"
        )

    if hasattr(prod, "T0"):
        resolved_t0 = float(getattr(prod, "T0"))
    elif hasattr(prod, "ANCHOR_TEMPERATURE"):
        resolved_t0 = float(getattr(prod, "ANCHOR_TEMPERATURE"))
    elif hasattr(prod, "T_ANCHOR"):
        resolved_t0 = float(getattr(prod, "T_ANCHOR"))
    else:
        resolved_t0 = 300.0
    setattr(prod, "T0", resolved_t0)


def selection_and_fit(
    prod: Any,
    train_df: pd.DataFrame,
    x_train: pd.DataFrame,
    colmap: Dict[str, Any],
    feature_columns: Sequence[str],
):
    selection = prod.cv_select_ti_model(
        df_ti=train_df,
        X_ti=x_train,
        colmap=colmap,
        feat_cols=feature_columns,
        seed=SEED,
    )
    if selection.empty:
        raise RuntimeError("Production selector returned no model.")

    best = selection.iloc[0]
    model_name = str(best["model"])
    if model_name != EXPECTED_MODEL:
        raise RuntimeError(
            f"Unexpected selected model {model_name!r}; expected {EXPECTED_MODEL!r}."
        )

    fitted = prod.fit_final_model(
        df_train=train_df,
        X_train=x_train,
        colmap=colmap,
        model_name=model_name,
        sel_df=selection,
        seed=SEED,
    )
    return best, fitted


def evaluate_positions(
    prod: Any,
    df_all: pd.DataFrame,
    x_all: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    colmap: Dict[str, Any],
    feature_columns: Sequence[str],
) -> Tuple[pd.Series, Dict[str, float], pd.DataFrame]:
    train_df = df_all.iloc[train_positions].copy().reset_index(drop=True)
    test_df = df_all.iloc[test_positions].copy().reset_index(drop=True)
    x_train = x_all.iloc[train_positions].copy().reset_index(drop=True)
    x_test = x_all.iloc[test_positions].copy().reset_index(drop=True)

    if list(x_train.columns) != list(feature_columns):
        raise RuntimeError("Training feature schema changed after splitting.")
    if list(x_test.columns) != list(feature_columns):
        raise RuntimeError("Test feature schema changed after splitting.")

    best, fitted = selection_and_fit(
        prod, train_df, x_train, colmap, feature_columns
    )
    prediction, _ = prod.predict_final(
        fitted, x_test, test_df, colmap, reanchor=True
    )
    prediction = np.asarray(prediction, dtype=float)

    mae, fviol, tau, npairs, _ = prod.absolute_metrics(
        test_df, colmap, prediction
    )
    metrics = {
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
        "holdout_MAE_abs": float(mae),
        "holdout_fviol": float(fviol),
        "holdout_tau": float(tau),
        "n_pairs": float(npairs),
    }

    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    fanc_col = str(colmap["FANC"])
    f0_col = str(colmap["F0"])
    dfres_col = str(colmap["dFres"])

    if "dFraw_T0_mean" in test_df.columns:
        cs_values = test_df["dFraw_T0_mean"].to_numpy(dtype=float)
    elif "Cs" in test_df.columns:
        cs_values = test_df["Cs"].to_numpy(dtype=float)
    else:
        # Derive C_s from the thermodynamic identity rather than silently using 0.
        cs_values = (
            test_df[fanc_col].to_numpy(dtype=float)
            - test_df[f0_col].to_numpy(dtype=float)
            - test_df[dfres_col].to_numpy(dtype=float)
        )

    fanc = test_df[fanc_col].to_numpy(dtype=float)
    f0 = test_df[f0_col].to_numpy(dtype=float)
    dfres_true = test_df[dfres_col].to_numpy(dtype=float)
    fhat = f0 + cs_values + prediction

    pred_df = pd.DataFrame({
        "system": test_df[system_col].astype(str).to_numpy(),
        "T": test_df[temp_col].to_numpy(dtype=float),
        "FANC_eVatom": fanc,
        "F0_eVatom": f0,
        "dFraw_T0_mean": cs_values,
        "dFres_eVatom": dfres_true,
        "dFres_pred": prediction,
        "Fhat_pred": fhat,
        "signed_error_absF": fhat - fanc,
        "absolute_error_absF": np.abs(fhat - fanc),
        "signed_error_dFres": prediction - dfres_true,
        "absolute_error_dFres": np.abs(prediction - dfres_true),
    })

    for optional_col in ("snap", "timestep"):
        if optional_col in test_df.columns:
            pred_df[optional_col] = test_df[optional_col].to_numpy()

    # Preserve original row position for exact traceability to the mapped table.
    pred_df["source_row_position"] = test_positions

    return best, metrics, pred_df


def summarize_runs(
    runs: pd.DataFrame,
    split_type: str,
    include_worst_temperature: bool = False,
) -> pd.DataFrame:
    row: Dict[str, object] = {
        "split_type": split_type,
        "n_runs": len(runs),
        "mean_holdout_MAE": runs["holdout_MAE_abs"].mean() if not runs.empty else np.nan,
        "mean_holdout_fviol": runs["holdout_fviol"].mean() if not runs.empty else np.nan,
        "mean_holdout_tau": runs["holdout_tau"].mean() if not runs.empty else np.nan,
        "most_frequent_model": (
            runs["selected_model"].mode().iloc[0] if not runs.empty else None
        ),
    }
    if len(runs) > 1:
        row["std_holdout_MAE"] = runs["holdout_MAE_abs"].std(ddof=1)
        row["max_holdout_MAE"] = runs["holdout_MAE_abs"].max()
    if include_worst_temperature and not runs.empty:
        idx = runs["holdout_MAE_abs"].idxmax()
        row["worst_temperature"] = runs.loc[idx, "held_out_temperature"]
    return pd.DataFrame([row])



def concatenate_or_empty(
    frames: List[pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_state_mean_table(
    prediction_tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    nonempty = [frame for frame in prediction_tables if not frame.empty]
    if not nonempty:
        return pd.DataFrame()

    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    grouping = [
        "split_type",
        "held_out_system",
        "held_out_temperature",
        "system",
        "T",
        "selected_model",
    ]

    state = (
        combined.groupby(grouping, dropna=False, observed=True)
        .agg(
            n_snapshots=("FANC_eVatom", "size"),
            FANC_mean=("FANC_eVatom", "mean"),
            F0_mean=("F0_eVatom", "mean"),
            Cs_mean=("dFraw_T0_mean", "mean"),
            dFres_true_mean=("dFres_eVatom", "mean"),
            dFres_pred_mean=("dFres_pred", "mean"),
            Fhat_mean=("Fhat_pred", "mean"),
            snapshot_MAE_absF=("absolute_error_absF", "mean"),
            snapshot_RMSE_absF=(
                "signed_error_absF",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
            snapshot_MAE_dFres=("absolute_error_dFres", "mean"),
            prediction_std_dFres=("dFres_pred", "std"),
            reference_std_dFres=("dFres_eVatom", "std"),
        )
        .reset_index()
    )

    state["state_bias_absF"] = state["Fhat_mean"] - state["FANC_mean"]
    state["state_abs_error_absF"] = np.abs(state["state_bias_absF"])
    state["state_bias_dFres"] = (
        state["dFres_pred_mean"] - state["dFres_true_mean"]
    )
    state["state_abs_error_dFres"] = np.abs(state["state_bias_dFres"])

    # T0-referenced state trajectories make slope reversal immediately visible.
    anchor_temp = 300.0
    anchor_keys = [
        "split_type",
        "held_out_system",
        "held_out_temperature",
        "system",
        "selected_model",
    ]
    anchor = (
        state[np.isclose(state["T"].astype(float), anchor_temp)]
        .loc[:, anchor_keys + ["dFres_true_mean", "dFres_pred_mean", "FANC_mean", "Fhat_mean"]]
        .rename(columns={
            "dFres_true_mean": "dFres_true_T0_mean",
            "dFres_pred_mean": "dFres_pred_T0_mean",
            "FANC_mean": "FANC_T0_mean",
            "Fhat_mean": "Fhat_T0_mean",
        })
    )
    state = state.merge(anchor, on=anchor_keys, how="left", validate="many_to_one")

    state["dFres_true_from_T0"] = (
        state["dFres_true_mean"] - state["dFres_true_T0_mean"]
    )
    state["dFres_pred_from_T0"] = (
        state["dFres_pred_mean"] - state["dFres_pred_T0_mean"]
    )
    state["FANC_from_T0"] = state["FANC_mean"] - state["FANC_T0_mean"]
    state["Fhat_from_T0"] = state["Fhat_mean"] - state["Fhat_T0_mean"]

    denominator = state["dFres_true_from_T0"].to_numpy(dtype=float)
    numerator = state["dFres_pred_from_T0"].to_numpy(dtype=float)
    state["thermal_slope_ratio_from_T0"] = np.where(
        np.abs(denominator) > 1e-12,
        numerator / denominator,
        np.nan,
    )

    return state.sort_values(
        ["split_type", "held_out_system", "held_out_temperature", "system", "T"],
        na_position="last",
    ).reset_index(drop=True)

def main() -> None:
    prod = import_module_from_file(PRODUCTION_SCRIPT)
    require_core(prod)

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    raw = pd.read_csv(SNAPSHOT_FILE)

    # Anchor-relative columns are built once on the complete table before any split.
    mapped, colmap = prod.auto_map_columns(raw)
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])

    df_ti = (
        mapped[mapped[system_col].astype(str).isin(TI_DOMAIN_SYSTEMS)]
        .copy()
        .reset_index(drop=True)
    )
    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty.")

    x_all, feature_columns = prod.make_state_features(
        df_ti, colmap, include_baseline_terms=True
    )
    if not feature_columns:
        raise RuntimeError("No production features were generated.")

    lower_features = [str(col).lower() for col in feature_columns]
    has_relative_temperature = any(
        col in {"dt", "dt_anchor", "d_t", "delta_t"} or col.startswith("dt_")
        for col in lower_features
    )
    n_anchor_relative = sum(
        col.startswith("d_") or "anchor" in col
        for col in lower_features
    )
    if not has_relative_temperature or n_anchor_relative < 3:
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
        col for col in feature_columns
        if any(token in str(col).lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden feature columns detected: {forbidden}")

    systems = df_ti[system_col].astype(str).to_numpy()
    temperatures = df_ti[temp_col].to_numpy(dtype=float)

    audit_rows: List[Dict[str, object]] = []
    system_prediction_frames: List[pd.DataFrame] = []
    temperature_prediction_frames: List[pd.DataFrame] = []
    bulk_prediction_frames: List[pd.DataFrame] = []
    low_high_prediction_frames: List[pd.DataFrame] = []

    # 1. Leave-one-system-out
    system_rows: List[Dict[str, object]] = []
    for held_out in TI_DOMAIN_SYSTEMS:
        test_mask = systems == held_out
        train_pos = np.flatnonzero(~test_mask)
        test_pos = np.flatnonzero(test_mask)
        best, metrics, pred_df = evaluate_positions(
            prod, df_ti, x_all, train_pos, test_pos,
            colmap, feature_columns,
        )
        pred_df.insert(0, "split_type", "leave_one_system_out")
        pred_df.insert(1, "held_out_system", held_out)
        pred_df.insert(2, "held_out_temperature", np.nan)
        pred_df["selected_model"] = str(best["model"])
        system_prediction_frames.append(pred_df)

        system_rows.append({
            "split_type": "leave_one_system_out",
            "held_out_system": held_out,
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
            "selected_model": str(best["model"]),
            "blend_weight": np.nan,
            "holdout_MAE_abs": metrics["holdout_MAE_abs"],
            "holdout_fviol": metrics["holdout_fviol"],
            "holdout_tau": metrics["holdout_tau"],
            "n_pairs": int(metrics["n_pairs"]),
        })
        audit_rows.append({
            "split_type": "leave_one_system_out",
            "held_out_value": held_out,
            "feature_representation": EXPECTED_REPRESENTATION,
            "anchor_temperature_K": float(getattr(prod, "T0", 300.0)),
            "anchor_alignment_protocol": "complete_table_before_split",
            "heldout_descriptor_T0_used_for_alignment": True,
            "heldout_targets_used_for_fit": False,
            "known_Cs_used_for_absolute_reconstruction": True,
            "n_features": len(feature_columns),
            "feature_columns": "|".join(feature_columns),
        })

    system_runs = pd.DataFrame(system_rows)
    system_runs.to_csv(OUT_SYSTEM_RUNS, index=False)
    system_summary = summarize_runs(system_runs, "leave_one_system_out")
    system_summary.to_csv(OUT_SYSTEM_SUMMARY, index=False)

    # 2. Bulk-to-interface
    train_mask = np.isin(systems, ("alpha-TiAl", "beta-TiV"))
    test_mask = systems == "Ti64"
    best, metrics, pred_df = evaluate_positions(
        prod, df_ti, x_all,
        np.flatnonzero(train_mask), np.flatnonzero(test_mask),
        colmap, feature_columns,
    )
    pred_df.insert(0, "split_type", "bulk_to_interface")
    pred_df.insert(1, "held_out_system", "Ti64")
    pred_df.insert(2, "held_out_temperature", np.nan)
    pred_df["selected_model"] = str(best["model"])
    bulk_prediction_frames.append(pred_df)

    bulk_runs = pd.DataFrame([{
        "split_type": "bulk_to_interface",
        "train_systems": "alpha-TiAl,beta-TiV",
        "held_out_system": "Ti64",
        "n_train": int(metrics["n_train"]),
        "n_test": int(metrics["n_test"]),
        "selected_model": str(best["model"]),
        "blend_weight": np.nan,
        "holdout_MAE_abs": metrics["holdout_MAE_abs"],
        "holdout_fviol": metrics["holdout_fviol"],
        "holdout_tau": metrics["holdout_tau"],
        "n_pairs": int(metrics["n_pairs"]),
    }])
    bulk_runs.to_csv(OUT_BULK2INT_RUNS, index=False)
    bulk_summary = summarize_runs(bulk_runs, "bulk_to_interface")
    bulk_summary.to_csv(OUT_BULK2INT_SUMMARY, index=False)
    audit_rows.append({
        "split_type": "bulk_to_interface",
        "held_out_value": "Ti64",
        "feature_representation": EXPECTED_REPRESENTATION,
        "anchor_temperature_K": float(getattr(prod, "T0", 300.0)),
        "anchor_alignment_protocol": "complete_table_before_split",
        "heldout_descriptor_T0_used_for_alignment": True,
        "heldout_targets_used_for_fit": False,
        "known_Cs_used_for_absolute_reconstruction": True,
        "n_features": len(feature_columns),
        "feature_columns": "|".join(feature_columns),
    })

    # 3. Leave-one-temperature-out
    temp_rows: List[Dict[str, object]] = []
    for held_out_temp in sorted(np.unique(temperatures)):
        test_mask = np.isclose(temperatures, held_out_temp)
        best, metrics, pred_df = evaluate_positions(
            prod, df_ti, x_all,
            np.flatnonzero(~test_mask), np.flatnonzero(test_mask),
            colmap, feature_columns,
        )
        pred_df.insert(0, "split_type", "leave_one_temperature_out")
        pred_df.insert(1, "held_out_system", "")
        pred_df.insert(2, "held_out_temperature", int(held_out_temp))
        pred_df["selected_model"] = str(best["model"])
        temperature_prediction_frames.append(pred_df)

        temp_rows.append({
            "split_type": "leave_one_temperature_out",
            "held_out_temperature": int(held_out_temp),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
            "selected_model": str(best["model"]),
            "blend_weight": np.nan,
            "holdout_MAE_abs": metrics["holdout_MAE_abs"],
            "holdout_fviol": metrics["holdout_fviol"],
            "holdout_tau": metrics["holdout_tau"],
            "n_pairs": int(metrics["n_pairs"]),
        })
        audit_rows.append({
            "split_type": "leave_one_temperature_out",
            "held_out_value": int(held_out_temp),
            "feature_representation": EXPECTED_REPRESENTATION,
            "anchor_temperature_K": float(getattr(prod, "T0", 300.0)),
            "anchor_alignment_protocol": "complete_table_before_split",
            "heldout_descriptor_T0_used_for_alignment": bool(
                np.isclose(held_out_temp, getattr(prod, "T0", 300.0))
            ),
            "heldout_targets_used_for_fit": False,
            "known_Cs_used_for_absolute_reconstruction": True,
            "n_features": len(feature_columns),
            "feature_columns": "|".join(feature_columns),
        })

    temp_runs = pd.DataFrame(temp_rows)
    temp_runs.to_csv(OUT_TEMP_RUNS, index=False)
    temp_summary = summarize_runs(
        temp_runs, "leave_one_temperature_out", include_worst_temperature=True
    )
    temp_summary.to_csv(OUT_TEMP_SUMMARY, index=False)

    # 4. Low-to-high transfer
    train_temps = (300, 400, 500)
    test_temps = (600, 700)
    train_mask = np.isin(temperatures, train_temps)
    test_mask = np.isin(temperatures, test_temps)
    best, metrics, pred_df = evaluate_positions(
        prod, df_ti, x_all,
        np.flatnonzero(train_mask), np.flatnonzero(test_mask),
        colmap, feature_columns,
    )
    pred_df.insert(0, "split_type", "low_to_high_transfer")
    pred_df.insert(1, "held_out_system", "")
    pred_df.insert(2, "held_out_temperature", np.nan)
    pred_df["selected_model"] = str(best["model"])
    low_high_prediction_frames.append(pred_df)

    l2h_runs = pd.DataFrame([{
        "split_type": "low_to_high_transfer",
        "train_temps": ",".join(map(str, train_temps)),
        "test_temps": ",".join(map(str, test_temps)),
        "n_train": int(metrics["n_train"]),
        "n_test": int(metrics["n_test"]),
        "selected_model": str(best["model"]),
        "blend_weight": np.nan,
        "holdout_MAE_abs": metrics["holdout_MAE_abs"],
        "holdout_fviol": metrics["holdout_fviol"],
        "holdout_tau": metrics["holdout_tau"],
        "n_pairs": int(metrics["n_pairs"]),
    }])
    l2h_runs.to_csv(OUT_L2H_RUNS, index=False)
    l2h_summary = summarize_runs(l2h_runs, "low_to_high_transfer")
    l2h_summary.to_csv(OUT_L2H_SUMMARY, index=False)
    audit_rows.append({
        "split_type": "low_to_high_transfer",
        "held_out_value": "600,700",
        "feature_representation": EXPECTED_REPRESENTATION,
        "anchor_temperature_K": float(getattr(prod, "T0", 300.0)),
        "anchor_alignment_protocol": "complete_table_before_split",
        "heldout_descriptor_T0_used_for_alignment": False,
        "heldout_targets_used_for_fit": False,
        "known_Cs_used_for_absolute_reconstruction": True,
        "n_features": len(feature_columns),
        "feature_columns": "|".join(feature_columns),
    })

    system_predictions = concatenate_or_empty(system_prediction_frames)
    temperature_predictions = concatenate_or_empty(temperature_prediction_frames)
    bulk_predictions = concatenate_or_empty(bulk_prediction_frames)
    low_high_predictions = concatenate_or_empty(low_high_prediction_frames)

    system_predictions.to_csv(OUT_SYSTEM_PRED, index=False)
    temperature_predictions.to_csv(OUT_TEMP_PRED, index=False)
    bulk_predictions.to_csv(OUT_BULK2INT_PRED, index=False)
    low_high_predictions.to_csv(OUT_L2H_PRED, index=False)

    state_means = build_state_mean_table([
        system_predictions,
        temperature_predictions,
        bulk_predictions,
        low_high_predictions,
    ])
    state_means.to_csv(OUT_STATE_MEANS, index=False)

    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)

    print("\n=== Leave-one-system-out results ===")
    print(system_runs.to_string(index=False))
    print("\n=== Leave-one-system-out summary ===")
    print(system_summary.to_string(index=False))
    print("\n=== Bulk-to-interface result ===")
    print(bulk_runs.to_string(index=False))
    print("\n=== Bulk-to-interface summary ===")
    print(bulk_summary.to_string(index=False))
    print("\n=== Leave-one-temperature-out results ===")
    print(temp_runs.to_string(index=False))
    print("\n=== Leave-one-temperature-out summary ===")
    print(temp_summary.to_string(index=False))
    print("\n=== Low-to-high transfer result ===")
    print(l2h_runs.to_string(index=False))
    print("\n=== Low-to-high transfer summary ===")
    print(l2h_summary.to_string(index=False))
    print(f"\nSaved audit: {OUT_AUDIT}")
    print(f"Saved system-holdout predictions: {OUT_SYSTEM_PRED}")
    print(f"Saved temperature-holdout predictions: {OUT_TEMP_PRED}")
    print(f"Saved bulk-to-interface predictions: {OUT_BULK2INT_PRED}")
    print(f"Saved low-to-high predictions: {OUT_L2H_PRED}")
    print(f"Saved state-mean diagnostics: {OUT_STATE_MEANS}")


if __name__ == "__main__":
    main()
