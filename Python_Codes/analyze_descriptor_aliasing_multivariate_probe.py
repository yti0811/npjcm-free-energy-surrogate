#!/usr/bin/env python3
"""
analyze_descriptor_aliasing_multivariate_probe.py

Compact multivariate representation probe for descriptor aliasing.

Purpose
-------
This is intentionally NOT a universal impossibility test for local descriptors.
It directly asks a narrower question supported by the present manuscript:

    After removing the common thermal backbone, do progressively enriched
    handcrafted descriptor stacks (d1, d2, d3) uniquely determine the remaining
    anchor-relative correction across distinct system-temperature states?

The probe is non-parametric and therefore does not depend on Ridge regression or
one-dimensional SLE binning.

For every Ti-domain snapshot, the script:
  1. constructs the same anchor-relative nested descriptor stacks used in the SI,
  2. standardizes descriptors without using any free-energy target,
  3. finds k nearest descriptor neighbors while EXCLUDING the query's own
     system-temperature state,
  4. predicts the backbone-removed correction by the median target of those
     cross-state neighbors,
  5. reports the remaining local representation mismatch.

If descriptor enrichment fully restored local identifiability within the tested
representation, the cross-state nearest-neighbor mismatch would approach zero.
A finite mismatch after enrichment is direct evidence that residual ambiguity remains
within the tested descriptor family. This does NOT establish that SOAP, ACE, MACE, or
all local descriptors must fail.

Descriptor stacks (matching SI definitions)
--------------------------------------------
d1: anchor-relative SLE summary statistics
d2: d1 + anchor-relative Voronoi-volume summary statistics
d3: d2 + anchor-relative q6 statistics + q6knn_mean

Outputs
-------
descriptor_aliasing_probe/descriptor_aliasing_knn_summary.csv
descriptor_aliasing_probe/descriptor_aliasing_knn_snapshot_errors.csv
descriptor_aliasing_probe/descriptor_aliasing_knn_system_summary.csv
descriptor_aliasing_probe/descriptor_aliasing_knn_control.pdf

No new MD/FL calculations are required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_piml_core_pipeline_backbone as core


STACK_KEYS: Dict[str, List[str]] = {
    "d1_SLE": [
        "SLE_mean", "SLE_std", "SLE_q25", "SLE_q50", "SLE_q75", "SLE_iqr",
    ],
    "d2_SLE_Voro": [
        "SLE_mean", "SLE_std", "SLE_q25", "SLE_q50", "SLE_q75", "SLE_iqr",
        "Voro_mean", "Voro_q25", "Voro_q50", "Voro_q75", "Voro_iqr",
    ],
    "d3_SLE_Voro_q6": [
        "SLE_mean", "SLE_std", "SLE_q25", "SLE_q50", "SLE_q75", "SLE_iqr",
        "Voro_mean", "Voro_q25", "Voro_q50", "Voro_q75", "Voro_iqr",
        "q6_mean", "q6_std", "q6_q25", "q6_q50", "q6_q75", "q6_iqr",
        "q6knn_mean",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--input",
        default=str(HERE / "table" / "F0_dF_by_snapshot.csv"),
    )
    p.add_argument(
        "--output-dir",
        default=str(HERE / "table" ),
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=list(core.TI_SYSTEMS),
        help="Use Ti-domain systems for the manuscript diagnostic.",
    )
    p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[10, 25, 50],
        help="Cross-state nearest-neighbor sensitivity values.",
    )
    return p.parse_args()


def get_stack_columns(
    colmap: Mapping[str, object],
    stack_keys: Sequence[str],
) -> List[str]:
    rel = colmap.get("relative_columns")
    if not isinstance(rel, dict):
        raise KeyError("Expected anchor-relative column map from auto_map_columns().")
    cols: List[str] = []
    for key in stack_keys:
        col = rel.get(key)
        if col is None:
            raise KeyError(f"Missing anchor-relative descriptor for {key}")
        cols.append(str(col))
    return cols


def standardize(X: pd.DataFrame) -> np.ndarray:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return np.asarray(pipe.fit_transform(X), dtype=float)


def cross_state_knn_prediction(
    Z: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median kNN target prediction using neighbors from other states only.

    Returns
    -------
    prediction : (n,) array
    median_neighbor_distance : (n,) array
    neighbor_target_iqr : (n,) array
    """
    D = pairwise_distances(Z, metric="euclidean")
    np.fill_diagonal(D, np.inf)

    pred = np.full(len(target), np.nan)
    med_dist = np.full(len(target), np.nan)
    target_iqr = np.full(len(target), np.nan)

    for i in range(len(target)):
        eligible = np.where(groups != groups[i])[0]
        if len(eligible) < k:
            raise ValueError(
                f"Only {len(eligible)} cross-state neighbors available for row {i}; k={k}."
            )
        local_dist = D[i, eligible]
        # argpartition is O(n) and does not impose a distance threshold.
        nn_local = np.argpartition(local_dist, k - 1)[:k]
        nn = eligible[nn_local]
        vals = target[nn]
        pred[i] = float(np.median(vals))
        med_dist[i] = float(np.median(D[i, nn]))
        target_iqr[i] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))

    return pred, med_dist, target_iqr


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if any(k <= 0 for k in args.k_values):
        raise ValueError("All --k-values must be positive integers.")

    raw = pd.read_csv(input_path)
    df, colmap = core.auto_map_columns(raw)
    system_col = str(colmap["system"])
    temp_col = str(colmap["T"])
    target_col = str(colmap["dFres"])

    present = set(df[system_col].astype(str))
    systems = [s for s in args.systems if s in present]
    if not systems:
        raise ValueError("No requested systems found in the input table.")

    dti = (
        df[df[system_col].astype(str).isin(systems)]
        .copy()
        .reset_index(drop=True)
    )

    # Same thermal backbone definition as the retained production formulation.
    Xprod, _ = core.make_state_features(dti, colmap, include_baseline_terms=True)
    backbone_coef = core.fit_quadratic_backbone(dti, colmap)
    gT = core.predict_backbone_from_X(Xprod, backbone_coef)
    dFres = dti[target_col].to_numpy(dtype=float)
    correction_target = dFres - gT

    groups = (
        dti[system_col].astype(str)
        + "_"
        + dti[temp_col].astype(int).astype(str)
    ).to_numpy()

    snap_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for stack_name, keys in STACK_KEYS.items():
        cols = get_stack_columns(colmap, keys)
        Z = standardize(dti.loc[:, cols])

        for k in args.k_values:
            pred_corr, med_dist, neigh_iqr = cross_state_knn_prediction(
                Z, correction_target, groups, k
            )
            abs_err_mev = np.abs(pred_corr - correction_target) * 1000.0
            neigh_iqr_mev = neigh_iqr * 1000.0

            for i in range(len(dti)):
                snap_rows.append({
                    "descriptor_stack": stack_name,
                    "k": k,
                    "system": str(dti.loc[i, system_col]),
                    "T_K": float(dti.loc[i, temp_col]),
                    "snap": dti.loc[i, "snap"] if "snap" in dti.columns else i,
                    "dFcorr_true_meVatom": float(correction_target[i] * 1000.0),
                    "dFcorr_knn_median_meVatom": float(pred_corr[i] * 1000.0),
                    "abs_representation_mismatch_meVatom": float(abs_err_mev[i]),
                    "median_cross_state_neighbor_distance": float(med_dist[i]),
                    "neighbor_target_IQR_meVatom": float(neigh_iqr_mev[i]),
                })

            summary_rows.append({
                "descriptor_stack": stack_name,
                "k": k,
                "n_snapshots": len(dti),
                "median_abs_representation_mismatch_meVatom": float(
                    np.median(abs_err_mev)
                ),
                "mean_abs_representation_mismatch_meVatom": float(
                    np.mean(abs_err_mev)
                ),
                "q90_abs_representation_mismatch_meVatom": float(
                    np.quantile(abs_err_mev, 0.90)
                ),
                "median_neighbor_target_IQR_meVatom": float(
                    np.median(neigh_iqr_mev)
                ),
                "median_cross_state_neighbor_distance": float(np.median(med_dist)),
                "target_std_meVatom": float(np.std(correction_target, ddof=1) * 1000.0),
                "target_IQR_meVatom": float(
                    (np.quantile(correction_target, 0.75)
                     - np.quantile(correction_target, 0.25)) * 1000.0
                ),
            })

    snap_df = pd.DataFrame(snap_rows)
    summary_df = pd.DataFrame(summary_rows)

    system_summary = (
        snap_df.groupby(["descriptor_stack", "k", "system"], observed=True)
        .agg(
            n_snapshots=("abs_representation_mismatch_meVatom", "size"),
            median_abs_mismatch_meVatom=(
                "abs_representation_mismatch_meVatom", "median"
            ),
            mean_abs_mismatch_meVatom=(
                "abs_representation_mismatch_meVatom", "mean"
            ),
            q90_abs_mismatch_meVatom=(
                "abs_representation_mismatch_meVatom",
                lambda s: float(s.quantile(0.90)),
            ),
            median_neighbor_target_IQR_meVatom=(
                "neighbor_target_IQR_meVatom", "median"
            ),
        )
        .reset_index()
    )

    snap_path = outdir / "descriptor_aliasing_knn_snapshot_errors.csv"
    summary_path = outdir / "descriptor_aliasing_knn_summary.csv"
    system_path = outdir / "descriptor_aliasing_knn_system_summary.csv"

    snap_df.to_csv(snap_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    system_summary.to_csv(system_path, index=False)

    # Compact SI-ready plot: sensitivity to descriptor enrichment and k.
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    stack_order = list(STACK_KEYS)
    x = np.arange(len(stack_order), dtype=float)
    width = 0.22
    k_values = list(args.k_values)
    offsets = (np.arange(len(k_values)) - (len(k_values) - 1) / 2.0) * width

    for off, k in zip(offsets, k_values):
        vals = []
        for stack in stack_order:
            row = summary_df[
                summary_df["descriptor_stack"].eq(stack)
                & summary_df["k"].eq(k)
            ].iloc[0]
            vals.append(float(row["median_abs_representation_mismatch_meVatom"]))
        ax.bar(x + off, vals, width=width, label=f"k={k}")

    print("\n=== Multivariate descriptor-aliasing probe ===")
    print("Systems:", systems)
    print("Backbone coefficients:", backbone_coef.tolist())
    print("\nSummary:")
    print(summary_df[[
        "descriptor_stack",
        "k",
        "median_abs_representation_mismatch_meVatom",
        "mean_abs_representation_mismatch_meVatom",
        "q90_abs_representation_mismatch_meVatom",
    ]].to_string(index=False))
    print("\nInterpretation guardrail:")
    print(
        "A finite d3 mismatch supports residual non-identifiability within the tested "
        "nested local descriptor family. It does not justify a universal claim about "
        "SOAP, ACE, MACE, or all local descriptors."
    )
    print("\nSaved:")
    for p in (summary_path, system_path, snap_path):
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
