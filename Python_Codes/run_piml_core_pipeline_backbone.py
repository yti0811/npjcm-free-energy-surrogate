#!/usr/bin/env python3
"""
run_piml_core_pipeline_backbone.py

Production implementation of the thermodynamically anchored thermal-backbone
surrogate.

For system s and snapshot x at temperature T,

    F_ANC = F0 + C_s + DeltaF_res,

and the gauge-fixed residual is decomposed as

    DeltaF_res(x,T) = g(T) + r(x,T).

The common thermal backbone is a system-balanced quadratic state-mean model,

    g(T) = b0 + b1 (T-T0) + b2 (T-T0)^2,

and the remaining correction r is learned from anchor-relative structural and
baseline changes by standardized Ridge regression. The explicit DeltaT channel is
reserved for the backbone and is not reused by the correction estimator.

Absolute reconstruction is

    F_hat = F0 + C_s + g(T) + r_hat(x,T).

Al and Cu are predicted without retraining or target-based tuning. Their T0
descriptor ensembles are used only for system-specific feature alignment, and
their C_s values are required for absolute reconstruction.

Public API
----------
auto_map_columns
make_state_features
candidate_models
cv_select_ti_model
fit_final_model
predict_final
reconstruct_absolute
absolute_metrics
ordering_metrics_ties_fail
gpr_predict

Outputs
-------
table/piml_model_selection_summary.csv
table/piml_metrics.csv
table/piml_predictions_Ti.csv
table/piml_predictions_Al.csv
table/piml_predictions_Cu.csv
table/piml_feature_audit.csv
table/piml_backbone_coefficients.csv
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"
OUT_MODELSEL = TABLE_DIR / "piml_model_selection_summary.csv"
OUT_METRICS = TABLE_DIR / "piml_metrics.csv"
OUT_PRED_TI = TABLE_DIR / "piml_predictions_Ti.csv"
OUT_PRED_AL = TABLE_DIR / "piml_predictions_Al.csv"
OUT_PRED_CU = TABLE_DIR / "piml_predictions_Cu.csv"
OUT_FEATURE_AUDIT = TABLE_DIR / "piml_feature_audit.csv"
OUT_BACKBONE = TABLE_DIR / "piml_backbone_coefficients.csv"

SEED = 0
T0 = 300.0
RIDGE_ALPHA = 1.0
TI_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
PRODUCTION_MODEL_NAME = "backbone_ridge"
FEATURE_REPRESENTATION = "quadratic_thermal_backbone_plus_T0_relative_correction"

STRUCTURAL_KEYS = (
    "SLE_mean", "SLE_std", "SLE_q25", "SLE_q50", "SLE_q75", "SLE_iqr",
    "Voro_mean", "Voro_q25", "Voro_q50", "Voro_q75", "Voro_iqr",
    "q6_mean", "q6_std", "q6_q25", "q6_q50", "q6_q75", "q6_iqr",
    "q6knn_mean",
)
BASELINE_KEYS = ("U_eVatom", "TSLE_eVatom", "F0")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(
    df: pd.DataFrame,
    aliases: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    norm_map = {normalize_name(column): column for column in df.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in norm_map:
            return norm_map[key]
    for alias in aliases:
        key = normalize_name(alias)
        for norm_name, original in norm_map.items():
            if key in norm_name:
                return original
    if required:
        raise KeyError(f"Could not find a column matching aliases: {aliases}")
    return None


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"Column {column!r} contains {int(values.isna().sum())} invalid values."
        )
    return values.astype(float)


def _validate_eq7_identity(
    df: pd.DataFrame,
    colmap: Mapping[str, Optional[str]],
    atol: float = 5.0e-9,
) -> None:
    reconstructed = (
        _numeric_series(df, str(colmap["F0"]))
        + _numeric_series(df, str(colmap["dFraw_T0_mean"]))
        + _numeric_series(df, str(colmap["dFres"]))
    )
    reference = _numeric_series(df, str(colmap["FANC"]))
    mismatch = float(np.max(np.abs(reconstructed - reference)))
    if not np.isfinite(mismatch) or mismatch > atol:
        raise ValueError(
            "Eq. (7) identity failed: FANC != F0 + C_s + dFres. "
            f"Maximum mismatch={mismatch:.6e} eV/atom."
        )


def _add_anchor_relative_columns(
    df: pd.DataFrame,
    colmap: Dict[str, Any],
    t0: float = T0,
) -> pd.DataFrame:
    out = df.copy()
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    temperatures = _numeric_series(out, temp_col)
    system_values = out[system_col].astype(str)

    for system in system_values.unique():
        mask = system_values.eq(system) & np.isclose(temperatures, t0)
        if not mask.any():
            raise ValueError(
                f"System {system!r} has no T0={t0:g} K rows for feature alignment."
            )

    out["dT_anchor"] = temperatures - t0
    colmap["dT_anchor"] = "dT_anchor"
    relative_map: Dict[str, str] = {}

    for key in (*STRUCTURAL_KEYS, *BASELINE_KEYS):
        source = colmap.get(key)
        if source is None or str(source) not in out.columns:
            continue
        source = str(source)
        values = pd.to_numeric(out[source], errors="coerce")
        anchor_mask = np.isclose(temperatures, t0)
        anchors = (
            pd.DataFrame({
                "system": system_values.loc[anchor_mask],
                "value": values.loc[anchor_mask],
            })
            .groupby("system", observed=True)["value"]
            .mean()
        )
        if anchors.isna().any():
            raise ValueError(f"Invalid T0 anchor means for {source!r}.")
        relative = f"d_{source}_anchor"
        out[relative] = values - system_values.map(anchors)
        relative_map[key] = relative

    colmap["relative_columns"] = relative_map
    return out


def auto_map_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df.copy()
    colmap: Dict[str, Any] = {}

    colmap["system"] = find_col(
        work, ["system", "phase", "label", "group", "material"], True
    )
    colmap["T"] = find_col(work, ["T", "temp", "temperature"], True)
    colmap["F0"] = find_col(work, ["F0_eVatom", "F0", "f0"], True)
    colmap["FANC"] = find_col(
        work, ["FANC_eVatom", "FANC", "F_ANC"], True
    )
    colmap["dFraw"] = find_col(
        work, ["dFraw_eVatom", "dFraw", "deltafraw"], True
    )
    colmap["dFraw_T0_mean"] = find_col(
        work, ["dFraw_T0_mean", "dfrawt0mean", "C_s", "Cs"], True
    )
    colmap["dFres"] = find_col(
        work, ["dFres_eVatom", "dFres", "deltafres"], True
    )

    optional = {
        "SLE_mean": ["SLE_mean", "slemean"],
        "SLE_std": ["SLE_std", "slestd"],
        "SLE_q25": ["SLE_q25", "sleq25"],
        "SLE_q50": ["SLE_q50", "sleq50"],
        "SLE_q75": ["SLE_q75", "sleq75"],
        "SLE_iqr": ["SLE_iqr", "sleiqr"],
        "Voro_mean": ["Voro_mean", "voromean", "voronoi_mean"],
        "Voro_q25": ["Voro_q25", "voroq25"],
        "Voro_q50": ["Voro_q50", "voroq50"],
        "Voro_q75": ["Voro_q75", "voroq75"],
        "Voro_iqr": ["Voro_iqr", "voroiqr"],
        "q6_mean": ["q6_mean", "q6mean"],
        "q6_std": ["q6_std", "q6std"],
        "q6_q25": ["q6_q25", "q6q25"],
        "q6_q50": ["q6_q50", "q6q50"],
        "q6_q75": ["q6_q75", "q6q75"],
        "q6_iqr": ["q6_iqr", "q6iqr"],
        "q6knn_mean": ["q6knn_mean", "q6knnmean"],
        "U_eVatom": ["U_eVatom", "u_evatom", "internal_energy_evatom"],
        "TSLE_eVatom": ["TSLE_eVatom", "tsle_evatom", "tsle"],
        "natoms": ["natoms", "n_atoms", "num_atoms"],
        "interface_distance": [
            "interface_distance", "distance_to_interface", "dist_interface"
        ],
    }
    for key, aliases in optional.items():
        colmap[key] = find_col(work, aliases, required=False)

    _validate_eq7_identity(work, colmap)
    work = _add_anchor_relative_columns(work, colmap, T0)
    return work, colmap


def ordering_metrics_ties_fail(
    y_pred: Sequence[float],
    y_ref: Sequence[float],
    tol: float = 1.0e-12,
) -> Tuple[float, float, int]:
    pred = np.asarray(y_pred, dtype=float)
    ref = np.asarray(y_ref, dtype=float)
    concordant = discordant = total = 0
    for i, j in itertools.combinations(range(len(pred)), 2):
        dp = pred[i] - pred[j]
        dr = ref[i] - ref[j]
        if abs(dr) <= tol:
            continue
        total += 1
        if abs(dp) <= tol or dp * dr < 0.0:
            discordant += 1
        else:
            concordant += 1
    if total == 0:
        return np.nan, np.nan, 0
    return (
        float(discordant / total),
        float((concordant - discordant) / total),
        int(total),
    )


def reconstruct_absolute(
    df_sub: pd.DataFrame,
    colmap: Mapping[str, Any],
    dFres_pred: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(dFres_pred, dtype=float)
    if len(prediction) != len(df_sub):
        raise ValueError("Prediction and dataframe lengths differ.")
    dFraw_pred = (
        prediction
        + df_sub[str(colmap["dFraw_T0_mean"])].to_numpy(dtype=float)
    )
    fhat = df_sub[str(colmap["F0"])].to_numpy(dtype=float) + dFraw_pred
    return dFraw_pred, fhat


def absolute_metrics(
    df_sub: pd.DataFrame,
    colmap: Mapping[str, Any],
    dFres_pred: Sequence[float],
) -> Tuple[float, float, float, int, np.ndarray]:
    _, fhat = reconstruct_absolute(df_sub, colmap, dFres_pred)
    reference = df_sub[str(colmap["FANC"])].to_numpy(dtype=float)
    fviol, tau, npairs = ordering_metrics_ties_fail(fhat, reference)
    return (
        float(mean_absolute_error(reference, fhat)),
        fviol,
        tau,
        npairs,
        fhat,
    )


def safe_fill(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, list(columns)].copy()
    for column in columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
        if x[column].isna().all():
            x[column] = 0.0
        elif x[column].isna().any():
            x[column] = x[column].fillna(x[column].median())
    return x


def make_state_features(
    df: pd.DataFrame,
    colmap: Mapping[str, Any],
    include_baseline_terms: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    relative_map = colmap.get("relative_columns")
    if not isinstance(relative_map, dict):
        raise KeyError("Call auto_map_columns() before make_state_features().")

    columns: List[str] = [str(colmap["dT_anchor"])]
    for key in STRUCTURAL_KEYS:
        relative = relative_map.get(key)
        if relative is not None and relative in df.columns:
            columns.append(relative)
    if include_baseline_terms:
        for key in BASELINE_KEYS:
            relative = relative_map.get(key)
            if relative is not None and relative in df.columns:
                columns.append(relative)

    columns = list(dict.fromkeys(columns))
    forbidden_tokens = (
        "fanc", "dfraw", "dfres", "pred", "fhat", "natoms",
        "snap", "timestep", "vmises", "stress", "response",
    )
    forbidden = [
        column for column in columns
        if any(token in normalize_name(column) for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden production features: {forbidden}")
    return safe_fill(df, columns), columns


def correction_feature_columns(feature_columns: Sequence[str]) -> List[str]:
    correction = [
        column for column in feature_columns
        if normalize_name(column) not in {
            "dt", "dtanchor", "deltat", "deltatanchor"
        }
    ]
    if not correction:
        raise ValueError("No descriptor/baseline correction features remain.")
    return correction


def candidate_models(seed: int = 0) -> Dict[str, Pipeline]:
    correction = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])
    kernel = (
        C(1.0, (1.0e-2, 1.0e2))
        * RBF(1.0, (1.0e-2, 1.0e2))
        + WhiteKernel(1.0e-5, (1.0e-8, 1.0e-1))
    )
    gpr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=1,
        )),
    ])
    return {
        PRODUCTION_MODEL_NAME: correction,
        "ridge": correction,
        "gpr": gpr,
    }


def fit_quadratic_backbone(
    df_train: pd.DataFrame,
    colmap: Mapping[str, Any],
) -> np.ndarray:
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    target_col = str(colmap["dFres"])
    state = (
        df_train.groupby([system_col, temp_col], observed=True)[target_col]
        .mean()
        .reset_index()
    )
    dt = state[temp_col].to_numpy(dtype=float) - T0
    design = np.column_stack([np.ones(len(state)), dt, dt ** 2])
    target = state[target_col].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coefficients, dtype=float)


def predict_backbone_from_X(
    X: pd.DataFrame,
    coefficients: Sequence[float],
) -> np.ndarray:
    if "dT_anchor" not in X.columns:
        raise KeyError("Feature matrix does not contain dT_anchor.")
    dt = X["dT_anchor"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X)), dt, dt ** 2])
    return design @ np.asarray(coefficients, dtype=float)


def reanchor_prediction_by_system(
    df_sub: pd.DataFrame,
    colmap: Mapping[str, Any],
    prediction: Sequence[float],
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float).copy()
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    systems = df_sub[system_col].astype(str).to_numpy()
    temperatures = df_sub[temp_col].to_numpy(dtype=float)
    for system in np.unique(systems):
        system_mask = systems == system
        anchor_mask = system_mask & np.isclose(temperatures, T0)
        if anchor_mask.any():
            pred[system_mask] -= float(np.mean(pred[anchor_mask]))
    return pred


def composite_score(
    mae_abs: float,
    f_viol: float,
    tau: float,
    y_ref_scale: float,
) -> float:
    return float(
        tau
        - 1.5 * f_viol
        - 0.25 * mae_abs / (y_ref_scale + 1.0e-12)
    )


def cv_select_ti_model(
    df_ti: pd.DataFrame,
    X_ti: pd.DataFrame,
    colmap: Mapping[str, Any],
    feat_cols: Sequence[str],
    seed: int = 0,
) -> pd.DataFrame:
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    target_col = str(colmap["dFres"])
    groups = (
        df_ti[system_col].astype(str)
        + "_"
        + df_ti[temp_col].astype(int).astype(str)
    )
    n_splits = min(3, int(groups.nunique()))
    if n_splits < 2:
        raise ValueError("At least two system-temperature groups are required.")

    output = np.full(len(df_ti), np.nan)
    target = df_ti[target_col].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=n_splits)

    for train_idx, valid_idx in splitter.split(X_ti, target, groups):
        fitted = fit_final_model(
            df_ti.iloc[train_idx].reset_index(drop=True),
            X_ti.iloc[train_idx].reset_index(drop=True),
            colmap,
            PRODUCTION_MODEL_NAME,
            pd.DataFrame(),
            seed,
        )
        prediction, _ = predict_final(
            fitted,
            X_ti.iloc[valid_idx].reset_index(drop=True),
            df_ti.iloc[valid_idx].reset_index(drop=True),
            colmap,
            reanchor=True,
        )
        output[valid_idx] = prediction

    if np.isnan(output).any():
        raise RuntimeError("Grouped OOF prediction is incomplete.")

    mae, fviol, tau, npairs, _ = absolute_metrics(df_ti, colmap, output)
    scale = float(df_ti[str(colmap["FANC"])].std())
    score = composite_score(mae, fviol, tau, scale)
    return pd.DataFrame([{
        "model": PRODUCTION_MODEL_NAME,
        "MAE_abs(Fhat)": mae,
        "f_viol_abs(Fhat)": fviol,
        "kendall_tau_abs(Fhat)": tau,
        "Npairs": npairs,
        "score": score,
        "features": ";".join(feat_cols),
        "correction_features": ";".join(correction_feature_columns(feat_cols)),
        "w": np.nan,
        "feature_representation": FEATURE_REPRESENTATION,
        "anchor_temperature_K": T0,
        "backbone_order": 2,
    }])


def fit_final_model(
    df_train: pd.DataFrame,
    X_train: pd.DataFrame,
    colmap: Mapping[str, Any],
    model_name: str,
    sel_df: pd.DataFrame,
    seed: int = 0,
) -> Dict[str, Any]:
    if model_name not in {PRODUCTION_MODEL_NAME, "ridge"}:
        raise ValueError(f"Unsupported production model: {model_name!r}")

    coefficients = fit_quadratic_backbone(df_train, colmap)
    backbone = predict_backbone_from_X(X_train, coefficients)
    target = df_train[str(colmap["dFres"])].to_numpy(dtype=float)
    correction_target = target - backbone
    correction_columns = correction_feature_columns(X_train.columns)

    model = clone(candidate_models(seed)[PRODUCTION_MODEL_NAME])
    model.fit(X_train.loc[:, correction_columns], correction_target)
    return {
        "type": "thermal_backbone_correction",
        "name": PRODUCTION_MODEL_NAME,
        "backbone_coefficients": coefficients,
        "correction_columns": correction_columns,
        "correction_model": model,
    }


def predict_final(
    final_model: Mapping[str, Any],
    X: pd.DataFrame,
    df_sub: Optional[pd.DataFrame] = None,
    colmap: Optional[Mapping[str, Any]] = None,
    reanchor: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if final_model.get("type") != "thermal_backbone_correction":
        raise ValueError("Expected thermal-backbone correction model.")
    coefficients = final_model["backbone_coefficients"]
    columns = list(final_model["correction_columns"])
    model = final_model["correction_model"]

    backbone = predict_backbone_from_X(X, coefficients)
    correction = np.asarray(
        model.predict(X.loc[:, columns]), dtype=float
    )
    prediction = backbone + correction

    if reanchor and df_sub is not None and colmap is not None:
        prediction = reanchor_prediction_by_system(
            df_sub, colmap, prediction
        )
    return prediction, None


def gpr_predict(
    pipe: Pipeline,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    x_imp = pipe.named_steps["imputer"].transform(X)
    x_scaled = pipe.named_steps["scaler"].transform(x_imp)
    model = pipe.named_steps["model"]
    mean, std = model.predict(x_scaled, return_std=True)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def _append_prediction_columns(
    df_sub: pd.DataFrame,
    colmap: Mapping[str, Any],
    prediction: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_std: np.ndarray,
    final_model: Mapping[str, Any],
    X: pd.DataFrame,
) -> pd.DataFrame:
    dFraw_pred, fhat = reconstruct_absolute(df_sub, colmap, prediction)
    backbone = predict_backbone_from_X(
        X, final_model["backbone_coefficients"]
    )
    correction = prediction - backbone

    out = df_sub.copy()
    out["dFres_pred_backbone"] = backbone
    out["dFres_pred_correction"] = correction
    out["dFres_pred_final"] = prediction
    out["dFraw_pred_final"] = dFraw_pred
    out["Fhat_final"] = fhat
    out["dFres_pred_gpr_mu"] = auxiliary_mean
    out["dFres_pred_gpr_std"] = auxiliary_std
    out["production_model"] = PRODUCTION_MODEL_NAME
    return out


def _write_feature_audit(
    feature_columns: Sequence[str],
    final_model: Optional[Mapping[str, Any]] = None,
) -> None:
    correction = set(
        final_model["correction_columns"] if final_model is not None
        else correction_feature_columns(feature_columns)
    )
    rows = []
    for column in feature_columns:
        rows.append({
            "feature": column,
            "representation": FEATURE_REPRESENTATION,
            "anchor_temperature_K": T0,
            "production_model": PRODUCTION_MODEL_NAME,
            "used_by_backbone": column == "dT_anchor",
            "used_by_correction": column in correction,
            "contains_target_or_raw_residual": any(
                token in normalize_name(column)
                for token in ("fanc", "dfraw", "dfres", "pred", "fhat")
            ),
        })
    pd.DataFrame(rows).to_csv(OUT_FEATURE_AUDIT, index=False)


def _domain_metric_row(
    domain: str,
    pred_df: pd.DataFrame,
    colmap: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = pred_df[str(colmap["F0"])].to_numpy(dtype=float)
    reference = pred_df[str(colmap["FANC"])].to_numpy(dtype=float)
    prediction = pred_df["Fhat_final"].to_numpy(dtype=float)
    f_b, tau_b, _ = ordering_metrics_ties_fail(baseline, reference)
    f_p, tau_p, _ = ordering_metrics_ties_fail(prediction, reference)
    return {
        "domain": domain,
        "baseline_MAE_abs": mean_absolute_error(reference, baseline),
        "baseline_f_viol": f_b,
        "baseline_tau": tau_b,
        "piml_model": PRODUCTION_MODEL_NAME,
        "feature_representation": FEATURE_REPRESENTATION,
        "piml_MAE_abs": mean_absolute_error(reference, prediction),
        "piml_f_viol": f_p,
        "piml_tau": tau_p,
    }


def main() -> None:
    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {SNAPSHOT_FILE}")

    raw = pd.read_csv(SNAPSHOT_FILE)
    df, colmap = auto_map_columns(raw)
    system_col = str(colmap["system"])
    present = df[system_col].astype(str).unique().tolist()
    ti_present = [system for system in TI_SYSTEMS if system in present]
    if len(ti_present) < 2:
        raise ValueError(f"Insufficient Ti systems: {ti_present}")

    df_ti = (
        df[df[system_col].astype(str).isin(ti_present)]
        .copy()
        .reset_index(drop=True)
    )
    x_ti, feature_columns = make_state_features(
        df_ti, colmap, include_baseline_terms=True
    )

    selection = cv_select_ti_model(
        df_ti, x_ti, colmap, feature_columns, SEED
    )
    selection.to_csv(OUT_MODELSEL, index=False)

    final_model = fit_final_model(
        df_ti,
        x_ti,
        colmap,
        PRODUCTION_MODEL_NAME,
        selection,
        SEED,
    )
    _write_feature_audit(feature_columns, final_model)

    coefficients = np.asarray(
        final_model["backbone_coefficients"], dtype=float
    )
    pd.DataFrame([{
        "b0_eVatom": coefficients[0],
        "b1_eVatom_per_K": coefficients[1],
        "b2_eVatom_per_K2": coefficients[2],
        "anchor_temperature_K": T0,
        "fit_level": "system_temperature_state_means",
        "training_systems": ",".join(ti_present),
    }]).to_csv(OUT_BACKBONE, index=False)

    # Auxiliary nonlinear diagnostic only; not used for production prediction.
    auxiliary = clone(candidate_models(SEED + 1)["gpr"])
    target = df_ti[str(colmap["dFres"])].to_numpy(dtype=float)
    auxiliary.fit(x_ti, target)

    prediction_ti, _ = predict_final(
        final_model, x_ti, df_ti, colmap, reanchor=True
    )
    gpr_mean, gpr_std = gpr_predict(auxiliary, x_ti)
    pred_ti = _append_prediction_columns(
        df_ti, colmap, prediction_ti, gpr_mean, gpr_std,
        final_model, x_ti
    )
    pred_ti.to_csv(OUT_PRED_TI, index=False)

    metrics = [_domain_metric_row("Ti_in_domain", pred_ti, colmap)]

    for system, path, domain in (
        ("Al", OUT_PRED_AL, "Al_transfer"),
        ("Cu", OUT_PRED_CU, "Cu_transfer"),
    ):
        if system not in present:
            continue
        subset = (
            df[df[system_col].astype(str).eq(system)]
            .copy()
            .reset_index(drop=True)
        )
        x_subset, subset_columns = make_state_features(
            subset, colmap, include_baseline_terms=True
        )
        if subset_columns != feature_columns:
            raise RuntimeError(
                f"Feature mismatch for {system}: {subset_columns}"
            )
        prediction, _ = predict_final(
            final_model, x_subset, subset, colmap, reanchor=True
        )
        gpr_mean, gpr_std = gpr_predict(auxiliary, x_subset)
        pred = _append_prediction_columns(
            subset, colmap, prediction, gpr_mean, gpr_std,
            final_model, x_subset
        )
        pred.to_csv(path, index=False)
        metrics.append(_domain_metric_row(domain, pred, colmap))

    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)

    print("\nProduction model:", PRODUCTION_MODEL_NAME)
    print("Representation:", FEATURE_REPRESENTATION)
    print("Backbone coefficients:", coefficients.tolist())
    print("Correction features:", len(final_model["correction_columns"]))
    print("\nSaved:")
    for path in (
        OUT_MODELSEL, OUT_METRICS, OUT_PRED_TI, OUT_PRED_AL,
        OUT_PRED_CU, OUT_FEATURE_AUDIT, OUT_BACKBONE,
    ):
        if path.exists():
            print(" -", path)


if __name__ == "__main__":
    main()
