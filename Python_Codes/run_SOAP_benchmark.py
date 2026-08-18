#!/usr/bin/env python3
"""
run_soap_benchmark.py

Processed-input SOAP benchmark for reviewer-oriented reproducibility.

This script starts from three precomputed descriptor-level inputs:

    ./table/BL39_existing_structural_features_for_soap.csv
    ./table/soap_anchor_relative_rcut4.5.csv
    ./table/soap_anchor_relative_rcut6.csv

The complete atomic snapshot archive (*.extxyz) is NOT required.

The two SOAP input tables are assumed to have been generated previously
from the atomic snapshots using the upstream SOAP preprocessing workflow.
They contain the system-anchor-relative SOAP representations used directly
in the manuscript benchmark.

Benchmark protocol
------------------
Target:
    dFcorr = dFres - ghat(T)

Ti-domain model development:
    alpha-TiAl
    beta-TiV
    Ti64

External fixed-model evaluation:
    Al
    Cu

Descriptor sets:
    1. Existing structural descriptors
    2. SOAP-only, rcut = 4.5 A
    3. Existing + SOAP, rcut = 4.5 A
    4. SOAP-only, rcut = 6.0 A
    5. Existing + SOAP, rcut = 6.0 A

Statistical protocol:
    - GroupKFold over Ti-domain system-temperature groups
    - median imputation fitted inside each training fold
    - standardization fitted inside each training fold
    - PCA fitted inside each training fold for SOAP and Hybrid models
    - Ridge regression fitted inside each training fold
    - final Ti-domain model transferred to Al/Cu without retraining

PRETEST
-------
All generated files are prefixed with:

    SI_Table_S6_

The prefix should be removed only after the reproduced numerical results
have been verified against the manuscript/Supplementary Information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths and fixed benchmark definitions
# ============================================================

HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE / "table"

OUTPUT_PREFIX = "SI_Table_S6_"

DEFAULT_EXISTING_FILE = (
    TABLE_DIR / "BL39_existing_structural_features_for_soap_clean.csv"
)

DEFAULT_SOAP_45_FILE = (
    TABLE_DIR / "soap_anchor_relative_rcut4.5.csv"
)

# Historical preprocessing code used f"{cutoff:g}", so 6.0 -> "6".
DEFAULT_SOAP_60_FILE = (
    TABLE_DIR / "soap_anchor_relative_rcut6.csv"
)


TI_DOMAIN_SYSTEMS = {
    "alpha-TiAl",
    "beta-TiV",
    "Ti64",
}

EXTERNAL_SYSTEMS = {
    "Al",
    "Cu",
}


# These columns must never enter Existing/Hybrid descriptor learning.
LEAKAGE_OR_RESPONSE_TOKENS = (
    "target",
    "dfraw",
    "dfres",
    "dfcorr",
    "f0",
    "fanc",
    "ffl",
    "ghat",
    "u_evatom",
    "pe_evatom",
    "ke_evatom",
    "potential_energy",
    "kinetic_energy",
    "free_energy",
    "vmises",
    "stress",
    "pressure",
)


META_COLUMNS = {
    "snapshot_path",
    "system",
    "temperature",
    "snapshot_index",
    "target",
    "group",
    "domain",
    "ensemble",
    "split_role",
    "fold",
    "sample_id",
    "id",
    "material",
    "state",
}


# ============================================================
# General utilities
# ============================================================

def output_path(
    output_dir: Path,
    filename: str,
) -> Path:
    """Create a PRETEST-prefixed output path."""
    return output_dir / f"{OUTPUT_PREFIX}{filename}"


def require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required columns {missing}"
        )


def normalize_metadata_columns(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Normalize metadata carried by a precomputed SOAP table.

    The original preprocessing output contains:
        snapshot_path
        system
        temperature
        target
        dsoap__*

    If group is absent, it is reconstructed exactly from
    system and temperature.
    """

    out = df.copy()

    require_columns(
        out,
        [
            "snapshot_path",
            "system",
            "temperature",
            target_col,
        ],
        "precomputed SOAP table",
    )

    out["snapshot_path"] = (
        out["snapshot_path"]
        .astype(str)
        .str.strip()
    )

    out["system"] = (
        out["system"]
        .astype(str)
        .str.strip()
    )

    out["temperature"] = pd.to_numeric(
        out["temperature"],
        errors="raise",
    ).astype(int)

    out[target_col] = pd.to_numeric(
        out[target_col],
        errors="raise",
    )

    if group_col not in out.columns:
        out[group_col] = (
            out["system"].astype(str)
            + "__"
            + out["temperature"].astype(str)
        )

    if "domain" not in out.columns:
        out["domain"] = np.where(
            out["system"].isin(TI_DOMAIN_SYSTEMS),
            "Ti-domain",
            "external",
        )

    if out["snapshot_path"].duplicated().any():
        examples = out.loc[
            out["snapshot_path"].duplicated(
                keep=False
            ),
            [
                "snapshot_path",
                "system",
                "temperature",
            ],
        ].head(20)

        raise ValueError(
            "SOAP table contains duplicate snapshot_path "
            "values. Examples:\n"
            + examples.to_string(index=False)
        )

    return out.reset_index(drop=True)


# ============================================================
# Processed SOAP input loading
# ============================================================

def load_soap_table(
    path: Path,
    target_col: str,
    group_col: str,
    label: str,
) -> tuple[pd.DataFrame, list[str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n  {path}"
        )

    table = pd.read_csv(path)

    table = normalize_metadata_columns(
        table,
        target_col=target_col,
        group_col=group_col,
    )

    soap_columns = [
        column
        for column in table.columns
        if column.startswith("dsoap__")
    ]

    if not soap_columns:
        raise ValueError(
            f"{label}: no columns beginning with "
            "'dsoap__' were found."
        )

    # Force SOAP columns to numeric.
    for column in soap_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    return table, soap_columns


def validate_soap_tables_match(
    soap45: pd.DataFrame,
    soap60: pd.DataFrame,
    target_col: str,
) -> None:
    """
    Ensure that the two precomputed SOAP representations correspond
    to the same snapshot-level thermodynamic dataset.
    """

    keys = [
        "snapshot_path",
        "system",
        "temperature",
    ]

    left = soap45[
        keys + [target_col]
    ].copy()

    right = soap60[
        keys + [target_col]
    ].copy()

    merged = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=("_45", "_60"),
        indicator=True,
        validate="one_to_one",
    )

    unmatched = (
        merged["_merge"] != "both"
    )

    if unmatched.any():
        raise ValueError(
            "The rcut=4.5 and rcut=6 SOAP tables "
            "do not contain identical snapshot sets.\n"
            + merged.loc[
                unmatched,
                keys + ["_merge"],
            ].head(20).to_string(index=False)
        )

    target_a = merged[
        f"{target_col}_45"
    ].to_numpy(dtype=float)

    target_b = merged[
        f"{target_col}_60"
    ].to_numpy(dtype=float)

    if not np.allclose(
        target_a,
        target_b,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError(
            "Target values differ between the two "
            "precomputed SOAP tables."
        )


# ============================================================
# Existing structural baseline
# ============================================================

def allowed_existing_features(
    table: pd.DataFrame,
    merge_key: str,
    explicitly_include: Sequence[str] | None,
    explicitly_exclude: Sequence[str] | None,
) -> list[str]:

    if explicitly_include:

        missing = [
            column
            for column in explicitly_include
            if column not in table.columns
        ]

        if missing:
            raise ValueError(
                "Requested existing features not found: "
                f"{missing}"
            )

        candidates = list(
            explicitly_include
        )

    else:

        candidates = [
            column
            for column in table.columns
            if column not in META_COLUMNS
            and column != merge_key
            and pd.api.types.is_numeric_dtype(
                table[column]
            )
        ]

    excluded_exact = set(
        explicitly_exclude or []
    )

    safe: list[str] = []
    rejected: list[str] = []

    for column in candidates:

        normalized = column.lower()

        unsafe = (
            column in excluded_exact
            or any(
                token in normalized
                for token
                in LEAKAGE_OR_RESPONSE_TOKENS
            )
        )

        if unsafe:
            rejected.append(column)
        else:
            safe.append(column)

    if rejected:
        warnings.warn(
            "Excluded leakage/response-derived columns "
            "from Existing/Hybrid benchmark: "
            + ", ".join(sorted(rejected))
        )

    if not safe:
        raise ValueError(
            "No safe existing structural descriptor "
            "columns remain."
        )

    return safe


def load_existing_features(
    path: Path,
    merge_key: str,
    explicitly_include: Sequence[str] | None,
    explicitly_exclude: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str]]:

    if not path.exists():
        raise FileNotFoundError(
            "Existing structural feature table not found:\n"
            f"  {path}"
        )

    table = pd.read_csv(path)

    require_columns(
        table,
        [merge_key],
        "existing structural feature table",
    )

    if table[merge_key].duplicated().any():
        raise ValueError(
            "existing_structural_features_for_soap.csv "
            f"contains duplicate {merge_key} values."
        )

    feature_columns = (
        allowed_existing_features(
            table,
            merge_key=merge_key,
            explicitly_include=explicitly_include,
            explicitly_exclude=explicitly_exclude,
        )
    )

    for column in feature_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    # Keep only the merge key and safe structural descriptors.
    table = table[
        [merge_key] + feature_columns
    ].copy()

    return table, feature_columns


def merge_existing_into_soap(
    soap_table: pd.DataFrame,
    existing_table: pd.DataFrame,
    merge_key: str,
    existing_columns: list[str],
) -> pd.DataFrame:

    merged = soap_table.merge(
        existing_table,
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    missing_all = (
        merged[existing_columns]
        .isna()
        .all(axis=1)
    )

    if missing_all.any():
        examples = merged.loc[
            missing_all,
            [
                "snapshot_path",
                "system",
                "temperature",
            ],
        ].head(20)

        raise ValueError(
            "Some SOAP snapshots have no matching "
            "existing structural features.\n"
            + examples.to_string(index=False)
        )

    return merged


# ============================================================
# Ordering metrics
# ============================================================

def ordering_on_state_means(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temperatures: np.ndarray,
    systems: np.ndarray,
    tolerance: float,
) -> tuple[float, float, int]:
    """
    Evaluate thermodynamic state ordering across systems
    at fixed temperature.

    This reproduces the state-mean ordering procedure used
    in the original SOAP benchmark implementation.
    """

    frame = pd.DataFrame(
        {
            "target": y_true,
            "prediction": y_pred,
            "temperature": temperatures,
            "system": systems,
        }
    )

    state = (
        frame
        .groupby(
            ["temperature", "system"],
            as_index=False,
        )
        .agg(
            target=("target", "mean"),
            prediction=("prediction", "mean"),
        )
    )

    violations = 0
    total_pairs = 0

    tau_true: list[float] = []
    tau_pred: list[float] = []

    for _, subset in state.groupby(
        "temperature"
    ):

        true_values = (
            subset["target"].to_numpy()
        )

        pred_values = (
            subset["prediction"].to_numpy()
        )

        for i in range(len(subset)):

            for j in range(
                i + 1,
                len(subset),
            ):

                true_difference = (
                    true_values[i]
                    - true_values[j]
                )

                if (
                    abs(true_difference)
                    <= tolerance
                ):
                    continue

                pred_difference = (
                    pred_values[i]
                    - pred_values[j]
                )

                total_pairs += 1

                # This preserves the strict-sign comparison
                # used in the original independent SOAP benchmark.
                if (
                    np.sign(true_difference)
                    != np.sign(pred_difference)
                ):
                    violations += 1

        tau_true.extend(
            true_values.tolist()
        )

        tau_pred.extend(
            pred_values.tolist()
        )

    fviol = (
        violations / total_pairs
        if total_pairs
        else np.nan
    )

    tau_result = kendalltau(
        tau_true,
        tau_pred,
        nan_policy="omit",
    )

    tau = (
        float(tau_result.statistic)
        if tau_result.statistic
        is not None
        else np.nan
    )

    return (
        float(fviol),
        tau,
        int(total_pairs),
    )


# ============================================================
# Pipeline
# ============================================================

def build_pipeline(
    n_features: int,
    pca_components: int,
    alpha: float,
) -> Pipeline:

    steps: list[
        tuple[str, object]
    ] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]

    if pca_components > 0:

        steps.append(
            (
                "pca",
                PCA(
                    n_components=min(
                        pca_components,
                        n_features,
                    ),
                    random_state=0,
                ),
            )
        )

    steps.append(
        (
            "ridge",
            Ridge(
                alpha=alpha,
                fit_intercept=True,
            ),
        )
    )

    return Pipeline(steps)


@dataclass
class Metric:
    descriptor_set: str
    cutoff_A: float | None
    evaluation: str
    system: str
    n_samples: int
    n_features: int
    mae_eVatom: float
    rmse_eVatom: float
    r2: float
    fviol: float
    kendall_tau: float
    n_ordering_pairs: int


def calculate_metric(
    descriptor_set: str,
    cutoff: float | None,
    evaluation: str,
    system_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temperatures: np.ndarray,
    systems: np.ndarray,
    n_features: int,
    tolerance: float,
) -> Metric:

    fviol, tau, n_pairs = (
        ordering_on_state_means(
            y_true=y_true,
            y_pred=y_pred,
            temperatures=temperatures,
            systems=systems,
            tolerance=tolerance,
        )
    )

    return Metric(
        descriptor_set=descriptor_set,
        cutoff_A=cutoff,
        evaluation=evaluation,
        system=system_label,
        n_samples=len(y_true),
        n_features=n_features,
        mae_eVatom=float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        rmse_eVatom=float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        r2=(
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(y_true) > 1
            else np.nan
        ),
        fviol=fviol,
        kendall_tau=tau,
        n_ordering_pairs=n_pairs,
    )


# ============================================================
# Grouped OOF + external transfer
# ============================================================

def evaluate_model(
    table: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    group_col: str,
    descriptor_set: str,
    cutoff: float | None,
    n_splits: int,
    pca_components: int,
    alpha: float,
    tolerance: float,
    output_dir: Path,
) -> list[Metric]:

    training = table[
        table["system"].isin(
            TI_DOMAIN_SYSTEMS
        )
    ].copy()

    external = table[
        table["system"].isin(
            EXTERNAL_SYSTEMS
        )
    ].copy()

    if training.empty:
        raise ValueError(
            "No Ti-domain samples were found."
        )

    X = training[
        feature_columns
    ].to_numpy(dtype=float)

    y = training[
        target_col
    ].to_numpy(dtype=float)

    groups = (
        training[group_col]
        .astype(str)
        .to_numpy()
    )

    temperatures = (
        training["temperature"]
        .to_numpy(dtype=float)
    )

    systems = (
        training["system"]
        .astype(str)
        .to_numpy()
    )

    unique_groups = np.unique(groups)

    effective_splits = min(
        n_splits,
        len(unique_groups),
    )

    if effective_splits < 2:
        raise ValueError(
            "At least two Ti-domain groups "
            "are required."
        )

    prediction = np.full(
        len(training),
        np.nan,
        dtype=float,
    )

    folds = np.full(
        len(training),
        -1,
        dtype=int,
    )

    cv = GroupKFold(
        n_splits=effective_splits
    )

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=groups,
        )
    ):

        model = build_pipeline(
            n_features=X.shape[1],
            pca_components=pca_components,
            alpha=alpha,
        )

        model.fit(
            X[train_index],
            y[train_index],
        )

        prediction[test_index] = (
            model.predict(
                X[test_index]
            )
        )

        folds[test_index] = fold

    if np.isnan(prediction).any():
        raise RuntimeError(
            "Grouped OOF prediction array "
            "contains NaN values."
        )

    slug = (
        descriptor_set
        .lower()
        .replace("+", "_")
        .replace(" ", "_")
    )

    if cutoff is not None:
        slug += f"_rcut{cutoff:g}"

    # --------------------------------------------------------
    # Ti-domain grouped OOF output
    # --------------------------------------------------------

    output_prediction = training[
        [
            "snapshot_path",
            "system",
            "temperature",
            group_col,
            target_col,
        ]
    ].copy()

    output_prediction[
        "prediction"
    ] = prediction

    output_prediction[
        "error"
    ] = prediction - y

    output_prediction[
        "abs_error"
    ] = np.abs(
        prediction - y
    )

    output_prediction[
        "fold"
    ] = folds

    output_prediction.to_csv(
        output_path(
            output_dir,
            f"oof_predictions_{slug}.csv",
        ),
        index=False,
    )

    metrics = [
        calculate_metric(
            descriptor_set=descriptor_set,
            cutoff=cutoff,
            evaluation="Ti_grouped_OOF",
            system_label="Ti-domain",
            y_true=y,
            y_pred=prediction,
            temperatures=temperatures,
            systems=systems,
            n_features=len(
                feature_columns
            ),
            tolerance=tolerance,
        )
    ]

    # --------------------------------------------------------
    # Final Ti-domain model
    # --------------------------------------------------------

    final_model = build_pipeline(
        n_features=X.shape[1],
        pca_components=pca_components,
        alpha=alpha,
    )

    final_model.fit(
        X,
        y,
    )

    joblib.dump(
        final_model,
        output_path(
            output_dir,
            f"final_model_{slug}.joblib",
        ),
        compress=3,
    )

    # --------------------------------------------------------
    # External anchor-aligned no-retraining transfer
    # --------------------------------------------------------

    for external_system in sorted(
        EXTERNAL_SYSTEMS
    ):

        subset = external[
            external["system"]
            == external_system
        ].copy()

        if subset.empty:
            continue

        X_external = subset[
            feature_columns
        ].to_numpy(dtype=float)

        y_external = subset[
            target_col
        ].to_numpy(dtype=float)

        external_prediction = (
            final_model.predict(
                X_external
            )
        )

        transfer_output = subset[
            [
                "snapshot_path",
                "system",
                "temperature",
                group_col,
                target_col,
            ]
        ].copy()

        transfer_output[
            "prediction"
        ] = external_prediction

        transfer_output[
            "error"
        ] = (
            external_prediction
            - y_external
        )

        transfer_output[
            "abs_error"
        ] = np.abs(
            external_prediction
            - y_external
        )

        transfer_output.to_csv(
            output_path(
                output_dir,
                (
                    f"transfer_{slug}_"
                    f"{external_system}.csv"
                ),
            ),
            index=False,
        )

        metrics.append(
            calculate_metric(
                descriptor_set=descriptor_set,
                cutoff=cutoff,
                evaluation=(
                    "anchor_aligned_"
                    "no_retraining_transfer"
                ),
                system_label=external_system,
                y_true=y_external,
                y_pred=external_prediction,
                temperatures=(
                    subset["temperature"]
                    .to_numpy(dtype=float)
                ),
                systems=(
                    subset["system"]
                    .astype(str)
                    .to_numpy()
                ),
                n_features=len(
                    feature_columns
                ),
                tolerance=tolerance,
            )
        )

    return metrics


# ============================================================
# Main benchmark
# ============================================================

def run_benchmark(
    args: argparse.Namespace,
) -> None:

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_path = Path(
        args.existing_features
    ).resolve()

    soap45_path = Path(
        args.soap_45
    ).resolve()

    soap60_path = Path(
        args.soap_60
    ).resolve()

    print("=" * 76)
    print(
        "PROCESSED-INPUT SOAP BENCHMARK "
        "(PRETEST DEBUG)"
    )
    print("=" * 76)
    print(f"Existing : {existing_path}")
    print(f"SOAP 4.5 : {soap45_path}")
    print(f"SOAP 6.0 : {soap60_path}")
    print(f"Output   : {output_dir}")
    print(f"Prefix   : {OUTPUT_PREFIX}")
    print("=" * 76)

    # --------------------------------------------------------
    # Load processed SOAP representations
    # --------------------------------------------------------

    soap45, soap45_columns = (
        load_soap_table(
            soap45_path,
            target_col=args.target_col,
            group_col=args.group_col,
            label="SOAP rcut=4.5 table",
        )
    )

    soap60, soap60_columns = (
        load_soap_table(
            soap60_path,
            target_col=args.target_col,
            group_col=args.group_col,
            label="SOAP rcut=6.0 table",
        )
    )

    validate_soap_tables_match(
        soap45,
        soap60,
        target_col=args.target_col,
    )

    # --------------------------------------------------------
    # Load fixed non-SOAP structural baseline
    # --------------------------------------------------------

    (
        existing,
        existing_columns,
    ) = load_existing_features(
        existing_path,
        merge_key=args.merge_key,
        explicitly_include=(
            args.existing_include_cols
        ),
        explicitly_exclude=(
            args.existing_exclude_cols
        ),
    )

    print(
        f"[INFO] Existing structural features: "
        f"{len(existing_columns)}"
    )

    print(
        f"[INFO] SOAP rcut=4.5 features: "
        f"{len(soap45_columns)}"
    )

    print(
        f"[INFO] SOAP rcut=6.0 features: "
        f"{len(soap60_columns)}"
    )

    # Merge the same existing feature baseline into
    # each SOAP representation.
    table45 = merge_existing_into_soap(
        soap45,
        existing,
        merge_key=args.merge_key,
        existing_columns=existing_columns,
    )

    table60 = merge_existing_into_soap(
        soap60,
        existing,
        merge_key=args.merge_key,
        existing_columns=existing_columns,
    )

    # --------------------------------------------------------
    # Save run configuration
    # --------------------------------------------------------

    configuration = {
        "existing_features": str(
            existing_path
        ),
        "soap_rcut4p5": str(
            soap45_path
        ),
        "soap_rcut6p0": str(
            soap60_path
        ),
        "target_col": args.target_col,
        "group_col": args.group_col,
        "merge_key": args.merge_key,
        "n_splits": args.n_splits,
        "pca_components": (
            args.pca_components
        ),
        "ridge_alpha": args.alpha,
        "ordering_tolerance": (
            args.order_tolerance
        ),
        "existing_feature_count": len(
            existing_columns
        ),
        "soap45_feature_count": len(
            soap45_columns
        ),
        "soap60_feature_count": len(
            soap60_columns
        ),
        "existing_features_used": (
            existing_columns
        ),
        "output_prefix": OUTPUT_PREFIX,
        "workflow": (
            "processed descriptor-level "
            "SOAP benchmark"
        ),
    }

    output_path(
        output_dir,
        "run_config.json",
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics: list[Metric] = []

    # ========================================================
    # 1. Existing descriptor baseline
    # ========================================================

    print(
        "\n[1/5] Existing structural baseline"
    )

    metrics.extend(
        evaluate_model(
            table=table45,
            feature_columns=existing_columns,
            target_col=args.target_col,
            group_col=args.group_col,
            descriptor_set="Existing",
            cutoff=None,
            n_splits=args.n_splits,
            pca_components=0,
            alpha=args.alpha,
            tolerance=(
                args.order_tolerance
            ),
            output_dir=output_dir,
        )
    )

    # ========================================================
    # 2. SOAP-only rcut=4.5
    # ========================================================

    print(
        "\n[2/5] SOAP-only, rcut=4.5 A"
    )

    metrics.extend(
        evaluate_model(
            table=table45,
            feature_columns=soap45_columns,
            target_col=args.target_col,
            group_col=args.group_col,
            descriptor_set="SOAP-only",
            cutoff=4.5,
            n_splits=args.n_splits,
            pca_components=(
                args.pca_components
            ),
            alpha=args.alpha,
            tolerance=(
                args.order_tolerance
            ),
            output_dir=output_dir,
        )
    )

    # ========================================================
    # 3. Hybrid rcut=4.5
    # ========================================================

    print(
        "\n[3/5] Existing + SOAP, rcut=4.5 A"
    )

    metrics.extend(
        evaluate_model(
            table=table45,
            feature_columns=(
                existing_columns
                + soap45_columns
            ),
            target_col=args.target_col,
            group_col=args.group_col,
            descriptor_set="Hybrid",
            cutoff=4.5,
            n_splits=args.n_splits,
            pca_components=(
                args.pca_components
            ),
            alpha=args.alpha,
            tolerance=(
                args.order_tolerance
            ),
            output_dir=output_dir,
        )
    )

    # ========================================================
    # 4. SOAP-only rcut=6.0
    # ========================================================

    print(
        "\n[4/5] SOAP-only, rcut=6.0 A"
    )

    metrics.extend(
        evaluate_model(
            table=table60,
            feature_columns=soap60_columns,
            target_col=args.target_col,
            group_col=args.group_col,
            descriptor_set="SOAP-only",
            cutoff=6.0,
            n_splits=args.n_splits,
            pca_components=(
                args.pca_components
            ),
            alpha=args.alpha,
            tolerance=(
                args.order_tolerance
            ),
            output_dir=output_dir,
        )
    )

    # ========================================================
    # 5. Hybrid rcut=6.0
    # ========================================================

    print(
        "\n[5/5] Existing + SOAP, rcut=6.0 A"
    )

    metrics.extend(
        evaluate_model(
            table=table60,
            feature_columns=(
                existing_columns
                + soap60_columns
            ),
            target_col=args.target_col,
            group_col=args.group_col,
            descriptor_set="Hybrid",
            cutoff=6.0,
            n_splits=args.n_splits,
            pca_components=(
                args.pca_components
            ),
            alpha=args.alpha,
            tolerance=(
                args.order_tolerance
            ),
            output_dir=output_dir,
        )
    )

    # ========================================================
    # Final summary
    # ========================================================

    metrics_frame = pd.DataFrame(
        [
            asdict(metric)
            for metric in metrics
        ]
    )

    summary_file = output_path(
        output_dir,
        "soap_benchmark_summary.csv",
    )

    metrics_frame.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 76)
    print("SOAP BENCHMARK COMPLETE")
    print("=" * 76)

    print(
        metrics_frame.to_string(
            index=False
        )
    )

    print("=" * 76)
    print(
        f"Summary:\n  {summary_file}"
    )
    print("=" * 76)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Processed-input reviewer-oriented "
            "SOAP benchmark."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--existing-features",
        default=str(
            DEFAULT_EXISTING_FILE
        ),
        help=(
            "Precomputed existing structural "
            "baseline feature table."
        ),
    )

    parser.add_argument(
        "--soap-45",
        default=str(
            DEFAULT_SOAP_45_FILE
        ),
        help=(
            "Precomputed system-anchor-relative "
            "SOAP table for rcut=4.5 A."
        ),
    )

    parser.add_argument(
        "--soap-60",
        default=str(
            DEFAULT_SOAP_60_FILE
        ),
        help=(
            "Precomputed system-anchor-relative "
            "SOAP table for rcut=6.0 A."
        ),
    )

    parser.add_argument(
        "--merge-key",
        default="snapshot_path",
    )

    parser.add_argument(
        "--target-col",
        default="target",
    )

    parser.add_argument(
        "--group-col",
        default="group",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--order-tolerance",
        type=float,
        default=1e-10,
    )

    parser.add_argument(
        "--existing-include-cols",
        nargs="*",
        help=(
            "Optional explicit list of safe "
            "existing structural features."
        ),
    )

    parser.add_argument(
        "--existing-exclude-cols",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            TABLE_DIR
        ),
        help=(
            "Directory for all PRETEST benchmark "
            "outputs."
        ),
    )

    return parser


def main() -> int:

    args = (
        build_parser()
        .parse_args()
    )

    try:

        run_benchmark(args)

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())