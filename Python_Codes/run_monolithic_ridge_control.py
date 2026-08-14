#!/usr/bin/env python3
"""
run_monolithic_ridge_control.py

Matched direct-control analysis for the manuscript:

    Backbone Ridge:
        DeltaF_res = g_hat(T) + r_hat(delta x)
        Ridge learns DeltaF_corr = DeltaF_res - g_hat(T)

    Monolithic Ridge:
        Ridge learns the complete DeltaF_res directly using
        the SAME production correction features plus dT and dT^2,
        so that quadratic temperature expressivity is retained.

The script uses the same system-temperature grouped folds supplied in
SI_Table_S6_oof_predictions_existing.csv and the retained Ridge regularization alpha=1.0
unless overridden on the command line.

Required input files (place in one directory):
    F0_dF_by_snapshot.csv
    piml_predictions_Ti.csv
    SI_Table_S6_oof_predictions_existing.csv
    piml_backbone_coefficients.csv

Optional but recommended:
    piml_model_selection_summary.csv

Outputs:
    matched_backbone_oof_predictions.csv
    monolithic_ridge_oof_predictions.csv
    matched_backbone_monolithic_metrics.csv
    matched_control_state_mean.csv
    matched_control_state_mean_metrics.csv
    target_burden_summary.csv
    matched_control_fold_assignment.csv
    matched_control_run_summary.txt

Usage:
    python run_monolithic_ridge_control.py --input-dir ./table --output-dir ./table

Notes:
1. Only Ti-domain systems are used: alpha-TiAl, beta-TiV, Ti64.
2. The thermal backbone is refit INSIDE each grouped OOF training fold
   from training-state means only.
3. Median imputation and standardization are also fitted inside each fold only.
4. The Monolithic Ridge differs from Backbone Ridge only in target organization:
   it learns DeltaF_res directly and receives dT and dT^2 explicitly.
5. Ordering metrics use all snapshot pairs belonging to different
   system-temperature states. This reproduces Npairs = 1,050,000 for 1500
   Ti-domain snapshots in 15 groups of 100 snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TI_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
T0 = 300.0

# Exact retained production correction features from piml_model_selection_summary.csv
# and piml_predictions_Ti.csv.
CORRECTION_FEATURES = [
    "d_SLE_mean_anchor",
    "d_SLE_std_anchor",
    "d_SLE_q25_anchor",
    "d_SLE_q50_anchor",
    "d_SLE_q75_anchor",
    "d_SLE_iqr_anchor",
    "d_Voro_mean_anchor",
    "d_Voro_q25_anchor",
    "d_Voro_q50_anchor",
    "d_Voro_q75_anchor",
    "d_Voro_iqr_anchor",
    "d_q6_mean_anchor",
    "d_q6_std_anchor",
    "d_q6_q25_anchor",
    "d_q6_q50_anchor",
    "d_q6_q75_anchor",
    "d_q6_iqr_anchor",
    "d_q6knn_mean_anchor",
    "d_U_eVatom_anchor",
    "d_TSLE_eVatom_anchor",
    "d_F0_eVatom_anchor",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("table"))
    p.add_argument("--output-dir", type=Path, default=Path("table"))
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularization alpha. Retained production value = 1.0.")
    p.add_argument("--backbone-order", type=int, default=2,
                   choices=(1, 2),
                   help="Thermal backbone polynomial order. Production value = 2.")
    return p.parse_args()


def require_file(folder: Path, name: str) -> Path:
    p = folder / name
    if not p.exists():
        raise FileNotFoundError(f"Required input file not found: {p}")
    return p


def make_group(system: pd.Series, temperature: pd.Series) -> pd.Series:
    return system.astype(str) + "__" + temperature.astype(int).astype(str)


def load_inputs(input_dir: Path):
    f0_path = require_file(input_dir, "F0_dF_by_snapshot.csv")
    pred_path = require_file(input_dir, "piml_predictions_Ti.csv")
    fold_path = require_file(input_dir, "SI_Table_S6_oof_predictions_existing.csv")
    backbone_path = require_file(input_dir, "piml_backbone_coefficients.csv")

    f0 = pd.read_csv(f0_path)
    pred = pd.read_csv(pred_path)
    fold_df = pd.read_csv(fold_path)
    backbone_coeff = pd.read_csv(backbone_path)

    # Optional metadata.
    sel_path = input_dir / "piml_model_selection_summary.csv"
    model_selection = pd.read_csv(sel_path) if sel_path.exists() else None

    return f0, pred, fold_df, backbone_coeff, model_selection


def validate_and_merge(f0, pred):
    f0_ti = f0.loc[f0["system"].isin(TI_SYSTEMS)].copy()
    pred_ti = pred.loc[pred["system"].isin(TI_SYSTEMS)].copy()

    key = ["system", "T", "snap"]
    if f0_ti.duplicated(key).any():
        raise ValueError("F0_dF_by_snapshot.csv has duplicate Ti-domain system/T/snap keys.")
    if pred_ti.duplicated(key).any():
        raise ValueError("piml_predictions_Ti.csv has duplicate system/T/snap keys.")

    # Use thermodynamic quantities from F0 table and production anchor-relative
    # feature columns from prediction table.
    pred_cols = key + ["dT_anchor"] + CORRECTION_FEATURES
    missing = [c for c in pred_cols if c not in pred_ti.columns]
    if missing:
        raise KeyError(f"Missing required production columns in piml_predictions_Ti.csv: {missing}")

    thermo_cols = key + [
        "FANC_eVatom", "F0_eVatom", "dFraw_eVatom",
        "dFraw_T0_mean", "dFres_eVatom"
    ]
    missing = [c for c in thermo_cols if c not in f0_ti.columns]
    if missing:
        raise KeyError(f"Missing required thermodynamic columns in F0_dF_by_snapshot.csv: {missing}")

    df = f0_ti[thermo_cols].merge(
        pred_ti[pred_cols],
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(f0_ti) or len(df) != len(pred_ti):
        raise ValueError(
            f"Merge mismatch: F0 Ti rows={len(f0_ti)}, prediction Ti rows={len(pred_ti)}, merged={len(df)}"
        )

    df["group"] = make_group(df["system"], df["T"])
    df["dT_anchor_sq"] = df["dT_anchor"].astype(float) ** 2

    # Internal consistency checks.
    recon = df["FANC_eVatom"] - df["F0_eVatom"] - df["dFraw_T0_mean"]
    max_err = float(np.max(np.abs(recon - df["dFres_eVatom"])))
    if max_err > 1e-10:
        print(
            f"WARNING: max |FANC - F0 - C_s - dFres| = {max_err:.3e} eV/atom",
            file=sys.stderr,
        )

    return df.sort_values(["system", "T", "snap"]).reset_index(drop=True)


def extract_fold_map(fold_df: pd.DataFrame, data_groups):
    required = {"group", "fold"}
    if not required.issubset(fold_df.columns):
        raise KeyError("SI_Table_S6_oof_predictions_existing.csv must contain 'group' and 'fold' columns.")

    tmp = fold_df[["group", "fold"]].drop_duplicates()
    # Each group must map to one fold only.
    counts = tmp.groupby("group")["fold"].nunique()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"Some groups map to multiple folds: {bad.to_dict()}")

    fmap = tmp.groupby("group")["fold"].first().to_dict()
    missing = sorted(set(data_groups) - set(fmap))
    if missing:
        raise KeyError(
            "Fold mapping is missing Ti-domain groups: " + ", ".join(missing)
        )
    return {g: int(fmap[g]) for g in sorted(set(data_groups))}


def fit_backbone_training_state_means(train_df: pd.DataFrame, order: int = 2):
    """
    Fit common thermal backbone to TRAINING system-temperature state means only.
    This mirrors the stated production fit level:
        system_temperature_state_means
    """
    state = (
        train_df.groupby(["system", "T"], as_index=False)["dFres_eVatom"]
        .mean()
        .rename(columns={"dFres_eVatom": "dFres_state_mean"})
    )
    dt = state["T"].to_numpy(float) - T0

    if order == 2:
        X = np.column_stack([np.ones(len(state)), dt, dt**2])
    elif order == 1:
        X = np.column_stack([np.ones(len(state)), dt])
    else:
        raise ValueError("Only order 1 or 2 supported.")

    y = state["dFres_state_mean"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    if order == 1:
        coef = np.array([coef[0], coef[1], 0.0], dtype=float)

    return coef  # b0, b1, b2


def eval_backbone(T, coef):
    dt = np.asarray(T, dtype=float) - T0
    b0, b1, b2 = coef
    return b0 + b1 * dt + b2 * dt**2


def make_ridge(alpha: float):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
    ])


def cross_state_pair_metrics(ref, pred, groups, tol=1e-14):
    """
    Pairwise ordering over ALL snapshot pairs from DIFFERENT state groups.
    Predicted ties count as failures when the reference is non-tied.

    Kendall-style tau is defined on the same eligible pair set as
        (N_concordant - N_discordant) / N_eligible
    with predicted ties against non-tied references counted as discordant/fail.
    For no reference ties this gives tau = 1 - 2*f_viol.
    """
    ref = np.asarray(ref, float)
    pred = np.asarray(pred, float)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_eligible = 0
    n_viol = 0
    n_concordant = 0
    n_ref_ties = 0
    n_pred_ties = 0

    for ia in range(len(unique_groups)):
        ga = unique_groups[ia]
        a = np.where(groups == ga)[0]
        for ib in range(ia + 1, len(unique_groups)):
            gb = unique_groups[ib]
            b = np.where(groups == gb)[0]

            dr = ref[a][:, None] - ref[b][None, :]
            dp = pred[a][:, None] - pred[b][None, :]

            ref_non_tie = np.abs(dr) > tol
            n_ref_ties += int(np.size(dr) - np.count_nonzero(ref_non_tie))

            # Only non-tied reference pairs are eligible.
            if not np.any(ref_non_tie):
                continue

            sr = np.sign(dr[ref_non_tie])
            pp = dp[ref_non_tie]
            pred_tie = np.abs(pp) <= tol
            sp = np.sign(pp)

            concord = (~pred_tie) & (sr == sp)
            violation = ~concord  # reversal OR predicted tie

            n = int(len(sr))
            n_eligible += n
            n_concordant += int(np.count_nonzero(concord))
            n_viol += int(np.count_nonzero(violation))
            n_pred_ties += int(np.count_nonzero(pred_tie))

    fviol = n_viol / n_eligible if n_eligible else np.nan
    tau = (n_concordant - n_viol) / n_eligible if n_eligible else np.nan

    return {
        "Npairs": int(n_eligible),
        "Nviol": int(n_viol),
        "Nconcordant": int(n_concordant),
        "Nref_ties_excluded": int(n_ref_ties),
        "Npred_ties_as_fail": int(n_pred_ties),
        "f_viol": float(fviol),
        "kendall_tau_pairwise": float(tau),
    }


def continuous_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "MAE_eVatom": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_eVatom": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def target_stats(name, x):
    x = np.asarray(x, float)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "target": name,
        "N": len(x),
        "mean_eVatom": float(np.mean(x)),
        "std_eVatom": float(np.std(x, ddof=1)),
        "rms_eVatom": float(np.sqrt(np.mean(x**2))),
        "mean_abs_eVatom": float(np.mean(np.abs(x))),
        "median_abs_eVatom": float(np.median(np.abs(x))),
        "iqr_eVatom": float(q75 - q25),
        "min_eVatom": float(np.min(x)),
        "max_eVatom": float(np.max(x)),
        "range_eVatom": float(np.max(x) - np.min(x)),
    }


def state_mean_table(oof: pd.DataFrame):
    cols = {
        "FANC_eVatom": "FANC_state_mean",
        "Fhat_backbone_oof": "Fhat_backbone_state_mean",
        "Fhat_monolithic_oof": "Fhat_monolithic_state_mean",
        "dFres_eVatom": "dFres_state_mean",
        "dFres_pred_backbone_oof": "dFres_pred_backbone_state_mean",
        "dFres_pred_monolithic_oof": "dFres_pred_monolithic_state_mean",
    }
    out = (
        oof.groupby(["system", "T", "group"], as_index=False)[list(cols)]
        .mean()
        .rename(columns=cols)
    )
    return out


def state_mean_ordering_metrics(state_df):
    rows = []
    for name, pred_col in [
        ("Backbone Ridge", "Fhat_backbone_state_mean"),
        ("Monolithic Ridge", "Fhat_monolithic_state_mean"),
    ]:
        m = continuous_metrics(state_df["FANC_state_mean"], state_df[pred_col])
        pair = cross_state_pair_metrics(
            state_df["FANC_state_mean"],
            state_df[pred_col],
            state_df["group"],
        )
        rows.append({"model": name, **m, **pair})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    f0, pred, fold_df, backbone_full, model_selection = load_inputs(args.input_dir)
    df = validate_and_merge(f0, pred)
    fold_map = extract_fold_map(fold_df, df["group"].unique())
    df["fold"] = df["group"].map(fold_map).astype(int)

    folds = sorted(df["fold"].unique())
    print(f"Ti-domain rows: {len(df)}")
    print(f"Groups: {df['group'].nunique()}")
    print(f"Folds: {folds}")

    # Save auditable group -> fold mapping.
    fold_assignment = (
        df[["system", "T", "group", "fold"]]
        .drop_duplicates()
        .sort_values(["fold", "system", "T"])
    )
    fold_assignment.to_csv(
        args.output_dir / "matched_control_fold_assignment.csv", index=False
    )

    # Allocate OOF outputs.
    df["g_oof"] = np.nan
    df["dFcorr_oof_target"] = np.nan
    df["dFcorr_pred_backbone_oof"] = np.nan
    df["dFres_pred_backbone_oof"] = np.nan
    df["dFres_pred_monolithic_oof"] = np.nan
    df["Fhat_backbone_oof"] = np.nan
    df["Fhat_monolithic_oof"] = np.nan

    fold_backbone_rows = []

    backbone_features = CORRECTION_FEATURES
    monolithic_features = CORRECTION_FEATURES + ["dT_anchor", "dT_anchor_sq"]

    for fold in folds:
        train = df["fold"] != fold
        test = df["fold"] == fold

        train_df = df.loc[train].copy()
        test_df = df.loc[test].copy()

        # ----- Backbone Ridge -----
        coef = fit_backbone_training_state_means(
            train_df, order=args.backbone_order
        )
        g_train = eval_backbone(train_df["T"], coef)
        g_test = eval_backbone(test_df["T"], coef)

        ycorr_train = train_df["dFres_eVatom"].to_numpy() - g_train
        ycorr_test = test_df["dFres_eVatom"].to_numpy() - g_test

        br = make_ridge(args.alpha)
        br.fit(train_df[backbone_features], ycorr_train)
        corr_pred = br.predict(test_df[backbone_features])
        df.loc[test, "g_oof"] = g_test
        df.loc[test, "dFcorr_oof_target"] = ycorr_test
        df.loc[test, "dFcorr_pred_backbone_oof"] = corr_pred
        df.loc[test, "dFres_pred_backbone_oof"] = g_test + corr_pred

        # ----- Monolithic Ridge -----
        mono = make_ridge(args.alpha)
        mono.fit(
            train_df[monolithic_features],
            train_df["dFres_eVatom"].to_numpy(),
        )
        mono_pred = mono.predict(test_df[monolithic_features])
        df.loc[test, "dFres_pred_monolithic_oof"] = mono_pred

        # Absolute free energies: Fhat = F0 + C_s + predicted gauge-fixed residual.
        df.loc[test, "Fhat_backbone_oof"] = (
            test_df["F0_eVatom"].to_numpy()
            + test_df["dFraw_T0_mean"].to_numpy()
            + (g_test + corr_pred)
        )
        df.loc[test, "Fhat_monolithic_oof"] = (
            test_df["F0_eVatom"].to_numpy()
            + test_df["dFraw_T0_mean"].to_numpy()
            + mono_pred
        )

        fold_backbone_rows.append({
            "fold": int(fold),
            "b0_eVatom": coef[0],
            "b1_eVatom_per_K": coef[1],
            "b2_eVatom_per_K2": coef[2],
            "n_train_rows": int(train.sum()),
            "n_test_rows": int(test.sum()),
            "n_train_groups": int(train_df["group"].nunique()),
            "n_test_groups": int(test_df["group"].nunique()),
        })

    if df[
        [
            "dFres_pred_backbone_oof",
            "dFres_pred_monolithic_oof",
            "Fhat_backbone_oof",
            "Fhat_monolithic_oof",
        ]
    ].isna().any().any():
        raise RuntimeError("OOF predictions contain missing values.")

    pd.DataFrame(fold_backbone_rows).to_csv(
        args.output_dir / "matched_backbone_fold_coefficients.csv", index=False
    )

    # Save separate prediction tables for straightforward SI plotting/auditing.
    common = [
        "system", "T", "snap", "group", "fold",
        "FANC_eVatom", "F0_eVatom", "dFraw_T0_mean", "dFres_eVatom",
    ]

    backbone_out = df[
        common + [
            "g_oof", "dFcorr_oof_target",
            "dFcorr_pred_backbone_oof",
            "dFres_pred_backbone_oof",
            "Fhat_backbone_oof",
        ]
    ].copy()
    backbone_out["error_Fhat_eVatom"] = (
        backbone_out["Fhat_backbone_oof"] - backbone_out["FANC_eVatom"]
    )
    backbone_out["abs_error_Fhat_eVatom"] = np.abs(
        backbone_out["error_Fhat_eVatom"]
    )
    backbone_out.to_csv(
        args.output_dir / "matched_backbone_oof_predictions.csv", index=False
    )

    mono_out = df[
        common + [
            "dT_anchor", "dT_anchor_sq",
            "dFres_pred_monolithic_oof",
            "Fhat_monolithic_oof",
        ]
    ].copy()
    mono_out["error_Fhat_eVatom"] = (
        mono_out["Fhat_monolithic_oof"] - mono_out["FANC_eVatom"]
    )
    mono_out["abs_error_Fhat_eVatom"] = np.abs(
        mono_out["error_Fhat_eVatom"]
    )
    mono_out.to_csv(
        args.output_dir / "monolithic_ridge_oof_predictions.csv", index=False
    )

    # ----- Primary matched metrics: snapshot-level absolute free energy -----
    rows = []
    for name, pred_col in [
        ("Backbone Ridge (recomputed matched OOF)", "Fhat_backbone_oof"),
        ("Monolithic Ridge (matched OOF)", "Fhat_monolithic_oof"),
    ]:
        cont = continuous_metrics(df["FANC_eVatom"], df[pred_col])
        pair = cross_state_pair_metrics(
            df["FANC_eVatom"], df[pred_col], df["group"]
        )
        rows.append({"model": name, **cont, **pair})

    # Add the previously reported production Backbone OOF metric as a reference row,
    # but do not fabricate RMSE/R2 values that are not available in the summary file.
    if model_selection is not None and len(model_selection) > 0:
        r = model_selection.iloc[0]
        rows.insert(0, {
            "model": "Backbone Ridge (reported production OOF reference)",
            "MAE_eVatom": float(r["MAE_abs(Fhat)"]),
            "RMSE_eVatom": np.nan,
            "R2": np.nan,
            "Npairs": int(r["Npairs"]),
            "Nviol": np.nan,
            "Nconcordant": np.nan,
            "Nref_ties_excluded": np.nan,
            "Npred_ties_as_fail": np.nan,
            "f_viol": float(r["f_viol_abs(Fhat)"]),
            "kendall_tau_pairwise": float(r["kendall_tau_abs(Fhat)"]),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        args.output_dir / "matched_backbone_monolithic_metrics.csv", index=False
    )

    # ----- 15-state mean table and metrics -----
    state_df = state_mean_table(df)
    state_df.to_csv(
        args.output_dir / "matched_control_state_mean.csv", index=False
    )
    state_metrics = state_mean_ordering_metrics(state_df)
    state_metrics.to_csv(
        args.output_dir / "matched_control_state_mean_metrics.csv", index=False
    )

    # ----- Learning-target burden -----
    # Use the full-data retained backbone coefficients for a purely descriptive
    # target-width comparison. This is not an OOF performance calculation.
    if len(backbone_full) != 1:
        raise ValueError("piml_backbone_coefficients.csv should contain exactly one row.")
    brc = backbone_full.iloc[0]
    full_coef = np.array(
        [brc["b0_eVatom"], brc["b1_eVatom_per_K"], brc["b2_eVatom_per_K2"]],
        dtype=float,
    )
    g_full = eval_backbone(df["T"], full_coef)
    dfcorr_full = df["dFres_eVatom"].to_numpy() - g_full

    burden = pd.DataFrame([
        target_stats("Gauge-fixed residual, DeltaF_res", df["dFres_eVatom"]),
        target_stats("Backbone-removed correction, DeltaF_corr", dfcorr_full),
    ])
    burden.to_csv(
        args.output_dir / "target_burden_summary.csv", index=False
    )

    # Fold-wise target burden, useful if a reviewer asks whether reduction is fold-dependent.
    fold_burden_rows = []
    for fold in folds:
        m = df["fold"] == fold
        fold_burden_rows.append({
            "fold": int(fold),
            **{
                "dFres_std_eVatom": float(np.std(df.loc[m, "dFres_eVatom"], ddof=1)),
                "dFcorr_oof_std_eVatom": float(np.std(df.loc[m, "dFcorr_oof_target"], ddof=1)),
                "dFres_rms_eVatom": float(np.sqrt(np.mean(df.loc[m, "dFres_eVatom"]**2))),
                "dFcorr_oof_rms_eVatom": float(np.sqrt(np.mean(df.loc[m, "dFcorr_oof_target"]**2))),
            }
        })
    pd.DataFrame(fold_burden_rows).to_csv(
        args.output_dir / "target_burden_by_fold.csv", index=False
    )

    # Human-readable summary.
    summary_lines = []
    summary_lines.append("MATCHED BACKBONE RIDGE VS MONOLITHIC RIDGE CONTROL")
    summary_lines.append("=" * 68)
    summary_lines.append(f"Rows: {len(df)}")
    summary_lines.append(f"State groups: {df['group'].nunique()}")
    summary_lines.append(f"Grouped folds: {len(folds)}")
    summary_lines.append(f"Ridge alpha: {args.alpha}")
    summary_lines.append(f"Backbone order: {args.backbone_order}")
    summary_lines.append("")
    summary_lines.append("Correction features (common to both models):")
    summary_lines.extend([f"  - {x}" for x in CORRECTION_FEATURES])
    summary_lines.append("")
    summary_lines.append("Monolithic-only explicit thermal terms:")
    summary_lines.append("  - dT_anchor")
    summary_lines.append("  - dT_anchor_sq")
    summary_lines.append("")
    summary_lines.append("Primary OOF metrics:")
    summary_lines.append(metrics_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("15-state mean metrics:")
    summary_lines.append(state_metrics.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Target burden:")
    summary_lines.append(burden.to_string(index=False))
    summary_lines.append("")
    summary_lines.append(
        "Interpretation note: the reported production Backbone OOF reference row "
        "is imported from piml_model_selection_summary.csv. The recomputed matched "
        "Backbone row is generated by this script using the fold mapping supplied "
        "in SI_Table_S6_oof_predictions_existing.csv. If the two Backbone rows differ materially, "
        "the supplied fold mapping is not identical to the original production fold "
        "assignment and the direct-control result should not be used until the exact "
        "production fold map is supplied."
    )
    (args.output_dir / "matched_control_run_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\nPrimary OOF metrics")
    print(metrics_df.to_string(index=False))
    print("\n15-state mean metrics")
    print(state_metrics.to_string(index=False))
    print("\nTarget burden")
    print(burden.to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
