#!/usr/bin/env python3
"""
clean_feature_generator_fixed.py

Validation/cleaning step for the reconstructed Baseline-39 SOAP
comparison table.

Input
-----
./table/BL39_existing_structural_features_for_soap.csv

Outputs
-------
./table/BL39_existing_structural_features_for_soap_clean.csv
./table/BL39_existing_feature_duplicate_audit.csv

This script does NOT regenerate structural descriptors and does NOT
require snapshot_split/*.extxyz files.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE / "table"

INPUT_FILE = (
    TABLE_DIR
    / "BL39_existing_structural_features_for_soap.csv"
)

OUTPUT_FILE = (
    TABLE_DIR
    / "BL39_existing_structural_features_for_soap_clean.csv"
)

AUDIT_FILE = (
    TABLE_DIR
    / "BL39_existing_feature_duplicate_audit.csv"
)


KEY_COLUMNS = [
    "snapshot_path",
    "system",
    "temperature",
    "snapshot_index",
]


BASELINE39_FEATURES = [
    "timestep",
    "natoms",

    "SLE_mean",
    "SLE_std",
    "SLE_q25",
    "SLE_q50",
    "SLE_q75",
    "SLE_iqr",

    "Voro_mean",
    "Voro_q25",
    "Voro_q50",
    "Voro_q75",
    "Voro_iqr",

    "q6_mean",
    "q6_std",
    "q6_q25",
    "q6_q50",
    "q6_q75",
    "q6_iqr",

    "q6knn_mean",

    "TSLE_eVatom",

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
]


def require_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def main() -> int:

    if not INPUT_FILE.exists():
        print(
            f"ERROR: input not found: {INPUT_FILE}",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(
        INPUT_FILE
    )

    require_columns(
        df,
        KEY_COLUMNS
        + BASELINE39_FEATURES,
    )

    # --------------------------------------------------------
    # Restrict to exact Baseline-39 schema
    # --------------------------------------------------------

    clean = df[
        KEY_COLUMNS
        + BASELINE39_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Normalize metadata
    # --------------------------------------------------------

    clean["snapshot_path"] = (
        clean["snapshot_path"]
        .astype(str)
        .str.strip()
    )

    clean["system"] = (
        clean["system"]
        .astype(str)
        .str.strip()
    )

    clean["temperature"] = pd.to_numeric(
        clean["temperature"],
        errors="raise",
    ).astype(int)

    clean["snapshot_index"] = pd.to_numeric(
        clean["snapshot_index"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    for col in BASELINE39_FEATURES:
        clean[col] = pd.to_numeric(
            clean[col],
            errors="coerce",
        )

    nan_counts = (
        clean[BASELINE39_FEATURES]
        .isna()
        .sum()
    )

    bad_nan_columns = (
        nan_counts[
            nan_counts > 0
        ]
        .index
        .tolist()
    )

    if bad_nan_columns:
        raise ValueError(
            "NaN values detected in Baseline-39 "
            f"features: {bad_nan_columns}"
        )

    # --------------------------------------------------------
    # Duplicate audit
    # --------------------------------------------------------

    duplicate_mask = (
        clean["snapshot_path"]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        clean.loc[
            duplicate_mask,
            KEY_COLUMNS
            + BASELINE39_FEATURES,
        ]
        .copy()
    )

    if duplicate_rows.empty:

        duplicate_audit = pd.DataFrame(
            columns=[
                "snapshot_path",
                "n_rows",
                "identical_feature_rows",
            ]
        )

    else:

        audit_records = []

        for snapshot_path, subset in (
            duplicate_rows.groupby(
                "snapshot_path",
                sort=False,
            )
        ):

            feature_matrix = (
                subset[
                    BASELINE39_FEATURES
                ]
                .to_numpy(
                    dtype=float
                )
            )

            first = feature_matrix[0]

            identical = bool(
                np.allclose(
                    feature_matrix,
                    first[None, :],
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            )

            audit_records.append(
                {
                    "snapshot_path":
                        snapshot_path,
                    "n_rows":
                        len(subset),
                    "identical_feature_rows":
                        identical,
                }
            )

        duplicate_audit = (
            pd.DataFrame(
                audit_records
            )
        )

        conflicting = (
            duplicate_audit[
                ~duplicate_audit[
                    "identical_feature_rows"
                ]
            ]
        )

        if not conflicting.empty:
            raise ValueError(
                "Conflicting duplicate snapshot rows "
                "were detected:\n"
                + conflicting.head(20)
                .to_string(index=False)
            )

        # Safe collapse only if duplicates are identical.
        clean = (
            clean.drop_duplicates(
                subset=["snapshot_path"],
                keep="first",
            )
        )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if clean[
        "snapshot_path"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate snapshot_path values remain."
        )

    if len(BASELINE39_FEATURES) != 39:
        raise RuntimeError(
            "Expected exactly 39 Baseline features."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    duplicate_audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print("=" * 72)
    print("BASELINE-39 CLEANING COMPLETE")
    print("=" * 72)
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"Audit  : {AUDIT_FILE}")
    print(f"Rows   : {len(clean)}")
    print(
        f"Features: {len(BASELINE39_FEATURES)}"
    )
    print(
        "Duplicate groups audited: "
        f"{len(duplicate_audit)}"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())