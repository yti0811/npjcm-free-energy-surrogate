#!/usr/bin/env python3
"""
prepare_soap_baseline_features.py

Reconstruct the original 39-feature non-SOAP baseline used in the
independent SOAP benchmark.

Input
-----
./table/metadata_soap_backbone.csv

Outputs
-------
./table/BL39_existing_structural_features_for_soap.csv
./table/BL39_baseline_feature_audit.csv

Important
---------
This script does NOT require snapshot_split/*.extxyz files.

The original Baseline-39 consists of:

21 base features:
    timestep
    natoms
    SLE_mean
    SLE_std
    SLE_q25
    SLE_q50
    SLE_q75
    SLE_iqr
    Voro_mean
    Voro_q25
    Voro_q50
    Voro_q75
    Voro_iqr
    q6_mean
    q6_std
    q6_q25
    q6_q50
    q6_q75
    q6_iqr
    q6knn_mean
    TSLE_eVatom

plus 18 system-wise 300 K anchor-relative structural features:
    d_SLE_*_anchor
    d_Voro_*_anchor
    d_q6_*_anchor
    d_q6knn_mean_anchor

Total = 39 features.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE / "table"

INPUT_FILE = TABLE_DIR / "metadata_soap_backbone.csv"

OUTPUT_FILE = (
    TABLE_DIR
    / "BL39_existing_structural_features_for_soap.csv"
)

AUDIT_FILE = (
    TABLE_DIR
    / "BL39_baseline_feature_audit.csv"
)

ANCHOR_TEMPERATURE = 300


# ============================================================
# Exact Baseline-39 definition recovered from original run_config
# ============================================================

BASE_FEATURES = [
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
]


ANCHOR_SOURCE_FEATURES = [
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
]


ANCHOR_OUTPUT_NAMES = {
    "SLE_mean": "d_SLE_mean_anchor",
    "SLE_std": "d_SLE_std_anchor",
    "SLE_q25": "d_SLE_q25_anchor",
    "SLE_q50": "d_SLE_q50_anchor",
    "SLE_q75": "d_SLE_q75_anchor",
    "SLE_iqr": "d_SLE_iqr_anchor",

    "Voro_mean": "d_Voro_mean_anchor",
    "Voro_q25": "d_Voro_q25_anchor",
    "Voro_q50": "d_Voro_q50_anchor",
    "Voro_q75": "d_Voro_q75_anchor",
    "Voro_iqr": "d_Voro_iqr_anchor",

    "q6_mean": "d_q6_mean_anchor",
    "q6_std": "d_q6_std_anchor",
    "q6_q25": "d_q6_q25_anchor",
    "q6_q50": "d_q6_q50_anchor",
    "q6_q75": "d_q6_q75_anchor",
    "q6_iqr": "d_q6_iqr_anchor",

    "q6knn_mean": "d_q6knn_mean_anchor",
}


METADATA_COLUMNS = [
    "snapshot_path",
    "system",
    "temperature",
    "snapshot_index",
]


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}"
        )


def main() -> int:

    if not INPUT_FILE.exists():
        print(
            f"ERROR: input not found: {INPUT_FILE}",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(INPUT_FILE)

    require_columns(
        df,
        METADATA_COLUMNS
        + BASE_FEATURES,
        "metadata_soap_backbone.csv",
    )

    df["system"] = df["system"].astype(str)
    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    if df["snapshot_path"].duplicated().any():
        duplicated = df.loc[
            df["snapshot_path"].duplicated(
                keep=False
            ),
            [
                "snapshot_path",
                "system",
                "temperature",
                "snapshot_index",
            ],
        ]

        raise ValueError(
            "Duplicate snapshot_path values found:\n"
            + duplicated.head(20).to_string(index=False)
        )

    # --------------------------------------------------------
    # Check anchor availability
    # --------------------------------------------------------

    all_systems = sorted(
        df["system"].unique()
    )

    anchor_systems = sorted(
        df.loc[
            df["temperature"]
            == ANCHOR_TEMPERATURE,
            "system",
        ].unique()
    )

    missing_anchor_systems = sorted(
        set(all_systems)
        - set(anchor_systems)
    )

    if missing_anchor_systems:
        raise ValueError(
            "Missing 300 K anchor states for systems: "
            f"{missing_anchor_systems}"
        )

    # --------------------------------------------------------
    # Construct system-wise 300 K anchor references
    # --------------------------------------------------------

    anchor_reference = (
        df.loc[
            df["temperature"]
            == ANCHOR_TEMPERATURE,
            ["system"]
            + ANCHOR_SOURCE_FEATURES,
        ]
        .groupby(
            "system",
            as_index=False,
        )
        .mean()
    )

    reference_rename = {
        feature: f"__anchor_ref__{feature}"
        for feature
        in ANCHOR_SOURCE_FEATURES
    }

    anchor_reference = (
        anchor_reference.rename(
            columns=reference_rename
        )
    )

    out = df.merge(
        anchor_reference,
        on="system",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Generate 18 anchor-relative features
    # --------------------------------------------------------

    anchor_relative_columns = []

    for source_feature in ANCHOR_SOURCE_FEATURES:

        output_name = (
            ANCHOR_OUTPUT_NAMES[
                source_feature
            ]
        )

        reference_name = (
            f"__anchor_ref__{source_feature}"
        )

        out[output_name] = (
            pd.to_numeric(
                out[source_feature],
                errors="raise",
            )
            - pd.to_numeric(
                out[reference_name],
                errors="raise",
            )
        )

        anchor_relative_columns.append(
            output_name
        )

    # --------------------------------------------------------
    # Final 39-feature schema
    # --------------------------------------------------------

    final_columns = (
        METADATA_COLUMNS
        + BASE_FEATURES
        + anchor_relative_columns
    )

    result = out[
        final_columns
    ].copy()

    n_features = (
        len(BASE_FEATURES)
        + len(anchor_relative_columns)
    )

    if n_features != 39:
        raise RuntimeError(
            f"Internal error: expected 39 features, "
            f"got {n_features}"
        )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    feature_columns = (
        BASE_FEATURES
        + anchor_relative_columns
    )

    for col in feature_columns:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        )

    if result[
        feature_columns
    ].isna().any().any():

        bad_columns = (
            result[
                feature_columns
            ]
            .columns[
                result[
                    feature_columns
                ]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "NaN values detected in Baseline-39 "
            f"features: {bad_columns}"
        )

    # --------------------------------------------------------
    # Anchor-relative mean check at 300 K
    # --------------------------------------------------------

    anchor_check = (
        result.loc[
            result["temperature"]
            == ANCHOR_TEMPERATURE
        ]
        .groupby("system")[
            anchor_relative_columns
        ]
        .mean()
    )

    max_abs_anchor_mean = float(
        np.nanmax(
            np.abs(
                anchor_check.to_numpy(
                    dtype=float
                )
            )
        )
    )

    # --------------------------------------------------------
    # Save main output
    # --------------------------------------------------------

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Audit output
    # --------------------------------------------------------

    audit_rows = []

    for system in all_systems:

        subset = result[
            result["system"] == system
        ]

        anchor_subset = subset[
            subset["temperature"]
            == ANCHOR_TEMPERATURE
        ]

        audit_rows.append(
            {
                "system": system,
                "n_rows": len(subset),
                "n_anchor_rows": len(
                    anchor_subset
                ),
                "n_base_features": len(
                    BASE_FEATURES
                ),
                "n_anchor_relative_features": len(
                    anchor_relative_columns
                ),
                "n_total_features": n_features,
                "max_abs_anchor_relative_mean_300K":
                    float(
                        np.nanmax(
                            np.abs(
                                anchor_subset[
                                    anchor_relative_columns
                                ]
                                .mean()
                                .to_numpy(
                                    dtype=float
                                )
                            )
                        )
                    ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print("=" * 72)
    print("BASELINE-39 PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"Audit  : {AUDIT_FILE}")
    print(f"Rows   : {len(result)}")
    print(f"Base features            : {len(BASE_FEATURES)}")
    print(
        "Anchor-relative features: "
        f"{len(anchor_relative_columns)}"
    )
    print(f"Total features           : {n_features}")
    print(
        "Max |300 K anchor-relative mean|: "
        f"{max_abs_anchor_mean:.6e}"
    )
    print("=" * 72)

    print("\nBaseline-39 feature list:")
    for i, col in enumerate(
        feature_columns,
        start=1,
    ):
        print(f"{i:02d}. {col}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())