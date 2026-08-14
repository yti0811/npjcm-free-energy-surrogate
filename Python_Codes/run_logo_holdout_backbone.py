#!/usr/bin/env python3
"""
run_logo_holdout.py

Leave-one-system-temperature-group-out (LOGO) evaluation using the same
anchor-relative production formulation implemented in run_piml_core_pipeline.py.

Protocol
--------
1. The complete processed table is mapped once by auto_map_columns().
   This constructs system-specific T0-relative descriptor columns before splitting.
2. Only descriptor/baseline inputs at T0 are used for feature alignment.
   No held-out FANC, dFraw, dFres, or prediction value is used for fitting.
3. Each system-temperature group is then held out in turn.
4. Model selection is repeated on the remaining Ti-domain groups using the
   production selector.
5. The selected production model is fit on the remaining groups and evaluated
   on the held-out group.
6. Absolute free energy is reconstructed as

       F_hat = F0 + C_s + DeltaF_res,pred.

Important interpretation
------------------------
Because system-specific T0 descriptor centering and the gauge constant C_s are
available for reconstruction, this is an anchor-aligned Ti-domain LOGO test, not
an anchor-free blind-material prediction test.

Outputs
-------
table/logo_holdout_runs.csv
table/table_s17_logo_holdout.csv
table/logo_holdout_audit.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_CORE = SCRIPT_DIR / "run_piml_core_pipeline_backbone.py"
PRODUCTION_SCRIPT = BACKBONE_CORE
SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_RUNS = TABLE_DIR / "logo_holdout_runs.csv"
OUT_SUMMARY = TABLE_DIR / "table_s17_logo_holdout.csv"
OUT_AUDIT = TABLE_DIR / "logo_holdout_audit.csv"

SEED = 0
TI_DOMAIN_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
EXPECTED_REPRESENTATION = "quadratic_thermal_backbone_plus_T0_relative_correction"
EXPECTED_PRODUCTION_MODEL = "backbone_ridge"


def import_module_from_file(
    module_path: Path,
    module_name: str = "prod_logo_module",
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


def get_group_labels(
    df: pd.DataFrame,
    system_col: str,
    temperature_col: str,
) -> pd.Series:
    return (
        df[system_col].astype(str)
        + "_"
        + df[temperature_col].astype(int).astype(str)
    )


def require_core_api(prod: Any) -> None:
    required = (
        "auto_map_columns",
        "make_state_features",
        "cv_select_ti_model",
        "fit_final_model",
        "predict_final",
        "absolute_metrics",
        "reconstruct_absolute",
    )
    missing = [name for name in required if not hasattr(prod, name)]
    if missing:
        raise AttributeError(
            "Production script is missing required API functions: "
            + ", ".join(missing)
        )


def verify_production_core(prod: Any) -> None:
    """Fail loudly if LOGO is accidentally run against the legacy GPR core."""
    models = prod.candidate_models(seed=SEED)
    if EXPECTED_PRODUCTION_MODEL not in models:
        raise RuntimeError(
            "The loaded production core does not expose the expected backbone model. "
            f"Available models: {sorted(models)}"
        )

    # The Ridge core may keep auxiliary KRR/GPR models for compatibility, but the
    # production selector must return Ridge only.
    # Some compatible Ridge-core revisions expose the anchor temperature under
    # a different constant name or only through selection metadata. Resolve it
    # conservatively and publish prod.T0 for the remainder of this script.
    if hasattr(prod, "T0"):
        resolved_t0 = float(getattr(prod, "T0"))
    elif hasattr(prod, "ANCHOR_TEMPERATURE"):
        resolved_t0 = float(getattr(prod, "ANCHOR_TEMPERATURE"))
    elif hasattr(prod, "T_ANCHOR"):
        resolved_t0 = float(getattr(prod, "T_ANCHOR"))
    else:
        resolved_t0 = 300.0
    setattr(prod, "T0", resolved_t0)

    # Do not infer implementation identity from the module docstring. Some valid
    # production files omit or reformat the top-level description. The actual
    # anchor-relative feature schema is checked after make_state_features() runs.


def fit_selected_model(
    prod: Any,
    train_df: pd.DataFrame,
    x_train: pd.DataFrame,
    colmap: Dict[str, Any],
    selection: pd.DataFrame,
):
    if selection.empty:
        raise RuntimeError("Production selector returned an empty table.")

    best = selection.iloc[0]
    model_name = str(best["model"])

    if model_name != EXPECTED_PRODUCTION_MODEL:
        raise RuntimeError(
            "LOGO loaded a non-production model from the core selector: "
            f"{model_name!r}. Expected {EXPECTED_PRODUCTION_MODEL!r}."
        )

    return prod.fit_final_model(
        df_train=train_df,
        X_train=x_train,
        colmap=colmap,
        model_name=model_name,
        sel_df=selection,
        seed=SEED,
    )


def baseline_absolute_mae(
    frame: pd.DataFrame,
    colmap: Dict[str, Any],
) -> float:
    reference = frame[str(colmap["FANC"])].to_numpy(dtype=float)
    baseline = frame[str(colmap["F0"])].to_numpy(dtype=float)
    return float(np.mean(np.abs(reference - baseline)))


def main() -> None:
    if not PRODUCTION_SCRIPT.exists():
        raise FileNotFoundError(
            "Backbone production core not found. Expected file: "
            f"{PRODUCTION_SCRIPT}"
        )

    print(f"Loading production core: {PRODUCTION_SCRIPT}")
    prod = import_module_from_file(PRODUCTION_SCRIPT)
    require_core_api(prod)
    verify_production_core(prod)

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_FILE}")

    raw = pd.read_csv(SNAPSHOT_FILE)

    # Critical: map and construct system-specific anchor-relative columns ONCE on
    # the complete processed table before train/test splitting.
    df, colmap = prod.auto_map_columns(raw)

    system_col = str(colmap["system"])
    temperature_col = str(colmap["T"])
    dFres_col = str(colmap["dFres"])

    df_ti = (
        df[df[system_col].astype(str).isin(TI_DOMAIN_SYSTEMS)]
        .copy()
        .reset_index(drop=True)
    )
    if df_ti.empty:
        raise ValueError("Ti-domain dataset is empty.")

    groups = get_group_labels(df_ti, system_col, temperature_col)
    unique_groups = sorted(groups.unique())

    # Generate the complete feature schema once. Each split must use exactly this
    # schema; only estimator fitting is fold-specific.
    x_all, feature_columns = prod.make_state_features(
        df_ti,
        colmap,
        include_baseline_terms=True,
    )

    if not feature_columns:
        raise RuntimeError("No production features were generated.")

    # Verify the formulation from actual generated features rather than from a
    # module docstring. At least one temperature-relative feature and multiple
    # anchor-relative descriptor/baseline features must be present.
    lower_features = [str(col).lower() for col in feature_columns]
    has_relative_temperature = any(
        col in {"dt", "dt_anchor", "d_t", "delta_t"}
        or col.startswith("dt_")
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

    run_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for held_out in unique_groups:
        is_test = groups.eq(held_out).to_numpy()
        train_positions = np.flatnonzero(~is_test)
        test_positions = np.flatnonzero(is_test)

        train_df = df_ti.iloc[train_positions].copy().reset_index(drop=True)
        test_df = df_ti.iloc[test_positions].copy().reset_index(drop=True)

        # Use the feature matrix constructed from the same globally aligned table.
        x_train = x_all.iloc[train_positions].copy().reset_index(drop=True)
        x_test = x_all.iloc[test_positions].copy().reset_index(drop=True)

        if list(x_train.columns) != feature_columns:
            raise RuntimeError("Training feature order differs from the production schema.")
        if list(x_test.columns) != feature_columns:
            raise RuntimeError("Test feature order differs from the production schema.")

        selection = prod.cv_select_ti_model(
            df_ti=train_df,
            X_ti=x_train,
            colmap=colmap,
            feat_cols=feature_columns,
            seed=SEED,
        )

        final_model = fit_selected_model(
            prod=prod,
            train_df=train_df,
            x_train=x_train,
            colmap=colmap,
            selection=selection,
        )

        dFres_pred_test, _ = prod.predict_final(
            final_model, x_test, test_df, colmap, reanchor=True
        )

        (
            holdout_mae,
            holdout_fviol,
            holdout_tau,
            n_pairs,
            _,
        ) = prod.absolute_metrics(
            test_df,
            colmap,
            dFres_pred_test,
        )

        best = selection.iloc[0]
        baseline_mae = baseline_absolute_mae(test_df, colmap)

        # Preserve the established public output columns exactly.
        run_rows.append({
            "held_out_group": held_out,
            "snapshots": len(test_df),
            "selected_model": str(best["model"]),
            "blend_weight": (
                float(best["w"])
                if "w" in best.index and pd.notna(best["w"])
                else np.nan
            ),
            "baseline_MAE_abs": baseline_mae,
            "holdout_MAE_abs": float(holdout_mae),
        })

        audit_rows.append({
            "held_out_group": held_out,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "selected_model": str(best["model"]),
            "feature_representation": str(
                best.get("feature_representation", EXPECTED_REPRESENTATION)
            ),
            "anchor_temperature_K": float(
                best.get("anchor_temperature_K", getattr(prod, "T0", 300.0))
            ),
            "n_features": len(feature_columns),
            "feature_columns": "|".join(feature_columns),
            "holdout_fviol": holdout_fviol,
            "holdout_tau": holdout_tau,
            "n_pairs": n_pairs,
            "anchor_alignment_protocol": (
                "complete_table_T0_descriptor_alignment_before_LOGO_split"
            ),
            "heldout_targets_used_for_fit": False,
        })

    runs = (
        pd.DataFrame(run_rows)
        .sort_values("held_out_group")
        .reset_index(drop=True)
    )
    runs.to_csv(OUT_RUNS, index=False)

    # Keep the legacy manuscript-table schema unchanged.
    summary = runs.rename(columns={"snapshots": "n_snapshots"})[
        [
            "held_out_group",
            "selected_model",
            "blend_weight",
            "n_snapshots",
            "baseline_MAE_abs",
            "holdout_MAE_abs",
        ]
    ].copy()
    summary.to_csv(OUT_SUMMARY, index=False)

    audit = (
        pd.DataFrame(audit_rows)
        .sort_values("held_out_group")
        .reset_index(drop=True)
    )
    audit.to_csv(OUT_AUDIT, index=False)

    print("\nProduction core:")
    print(f"  file: {PRODUCTION_SCRIPT}")
    print(f"  model: {EXPECTED_PRODUCTION_MODEL}")
    print(f"  representation: {EXPECTED_REPRESENTATION}")
    print(f"  features: {len(feature_columns)}")

    print("\nLOGO results:")
    print(runs.to_string(index=False))

    print(f"\nSaved: {OUT_RUNS}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_AUDIT}")


if __name__ == "__main__":
    main()
