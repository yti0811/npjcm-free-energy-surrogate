#!/usr/bin/env python3
"""
run_sample_sufficiency.py

Sample-sufficiency analysis for the audited anchor-relative Ridge production
formulation.

For each retained fraction, snapshots are subsampled uniformly within every
Ti-domain system-temperature group. The sampled table is then passed through
the production core from the beginning, so system-specific T0 anchor means are
estimated only from the retained T0 snapshots rather than from the full dataset.

Outputs preserve the existing manuscript-facing schemas:
    table/table_s12_sample_sufficiency.csv
    table/sample_sufficiency_runs.csv

An additional audit file is written:
    table/sample_sufficiency_audit.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, List

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

OUT_CSV = TABLE_DIR / "table_s12_sample_sufficiency.csv"
OUT_RUNS = TABLE_DIR / "sample_sufficiency_runs.csv"
OUT_AUDIT = TABLE_DIR / "sample_sufficiency_audit.csv"

SEED = 0
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
TI_DOMAIN_SYSTEMS = ("alpha-TiAl", "beta-TiV", "Ti64")
EXPECTED_MODEL = "backbone_ridge"
EXPECTED_REPRESENTATION = "quadratic_thermal_backbone_plus_T0_relative_correction"


def import_module_from_file(path: Path, name: str = "prod_sample_module"):
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
        "auto_map_columns", "make_state_features",
        "candidate_models", "cv_select_ti_model",
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


def sample_raw_by_group(
    raw_ti: pd.DataFrame,
    system_col: str,
    temp_col: str,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []

    for _, group in raw_ti.groupby([system_col, temp_col], sort=False):
        n_total = len(group)
        n_keep = min(n_total, max(10, int(round(n_total * fraction))))
        chosen = rng.choice(group.index.to_numpy(), size=n_keep, replace=False)
        parts.append(raw_ti.loc[chosen])

    sampled = pd.concat(parts, axis=0).sort_index().reset_index(drop=True)
    return sampled


def main() -> None:
    prod = import_module_from_file(PRODUCTION_SCRIPT)
    require_core(prod)

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {SNAPSHOT_FILE}")

    raw = pd.read_csv(SNAPSHOT_FILE)

    # Resolve system and temperature column names once. The mapped full table is
    # not used for feature centering in the fraction runs.
    mapped_full, full_colmap = prod.auto_map_columns(raw)
    system_col = str(full_colmap["system"])
    temp_col = str(full_colmap["T"])

    raw_ti = raw[
        raw[system_col].astype(str).isin(TI_DOMAIN_SYSTEMS)
    ].copy()
    if raw_ti.empty:
        raise ValueError("Ti-domain dataset is empty.")

    result_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for fraction in FRACTIONS:
        sampled_raw = sample_raw_by_group(
            raw_ti=raw_ti,
            system_col=system_col,
            temp_col=temp_col,
            fraction=fraction,
            seed=SEED,
        )

        # Re-run production mapping on the sampled data. This is critical:
        # T0 anchor means are estimated from the retained T0 snapshots only.
        sampled, colmap = prod.auto_map_columns(sampled_raw)
        sampled_system_col = str(colmap["system"])
        sampled_temp_col = str(colmap["T"])

        sampled_ti = (
            sampled[
                sampled[sampled_system_col].astype(str).isin(TI_DOMAIN_SYSTEMS)
            ]
            .copy()
            .reset_index(drop=True)
        )

        x_sampled, feature_columns = prod.make_state_features(
            sampled_ti,
            colmap,
            include_baseline_terms=True,
        )

        if not feature_columns:
            raise RuntimeError(
                f"No production features were generated at fraction {fraction:g}."
            )

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
                f"Fraction {fraction:g}: the loaded core did not generate the "
                "expected anchor-relative feature schema. Generated columns: "
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
            raise RuntimeError(
                f"Fraction {fraction:g}: forbidden feature columns detected: "
                f"{forbidden}"
            )

        selection = prod.cv_select_ti_model(
            df_ti=sampled_ti,
            X_ti=x_sampled,
            colmap=colmap,
            feat_cols=feature_columns,
            seed=SEED,
        )
        if selection.empty:
            raise RuntimeError(
                f"No production model selected at fraction {fraction:g}."
            )

        best = selection.iloc[0]
        selected_model = str(best["model"])
        if selected_model != EXPECTED_MODEL:
            raise RuntimeError(
                f"Fraction {fraction:g} selected {selected_model!r}; "
                f"expected {EXPECTED_MODEL!r}."
            )

        n_groups = int(
            sampled_ti.groupby(
                [sampled_system_col, sampled_temp_col], observed=True
            ).ngroups
        )

        result_rows.append({
            "fraction": fraction,
            "n_snapshots": len(sampled_ti),
            "n_groups": n_groups,
            "selected_model": selected_model,
            "MAE_abs": float(best["MAE_abs(Fhat)"]),
            "f_viol": float(best["f_viol_abs(Fhat)"]),
            "tau": float(best["kendall_tau_abs(Fhat)"]),
            "score": float(best["score"]),
        })

        t0 = float(getattr(prod, "T0", 300.0))
        t0_counts = (
            sampled_ti[np.isclose(sampled_ti[sampled_temp_col].astype(float), t0)]
            .groupby(sampled_system_col, observed=True)
            .size()
            .to_dict()
        )

        audit_rows.append({
            "fraction": fraction,
            "sampling_seed": SEED,
            "feature_representation": EXPECTED_REPRESENTATION,
            "production_model": EXPECTED_MODEL,
            "anchor_temperature_K": t0,
            "anchor_means_recomputed_from_retained_sample": True,
            "n_snapshots": len(sampled_ti),
            "n_groups": n_groups,
            "n_features": len(feature_columns),
            "feature_columns": "|".join(feature_columns),
            "T0_counts_by_system": "|".join(
                f"{key}:{value}" for key, value in sorted(t0_counts.items())
            ),
            "contains_target_or_raw_residual_feature": any(
                token in str(col).lower()
                for col in feature_columns
                for token in ("fanc", "dfraw", "dfres", "fhat", "pred")
            ),
        })

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_CSV, index=False)
    results.to_csv(OUT_RUNS, index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_RUNS}")
    print(f"Saved: {OUT_AUDIT}")


if __name__ == "__main__":
    main()
