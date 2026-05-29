from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = TABLE_DIR / "piml_predictions_Ti.csv"

OUT_BREAKDOWN = TABLE_DIR / "fig9_piml_inputs.csv"
OUT_MEAN_BY_SYSTEM_T = TABLE_DIR / "piml_ti_means_by_system_T.csv"
OUT_GLOBAL = TABLE_DIR / "piml_ti_ordering_global.csv"


def safe_kendall(y_true, y_pred):
    tau, _ = kendalltau(y_true, y_pred)
    if pd.isna(tau):
        return np.nan
    return float(tau)


def ordering_metrics_ties_fail(y_pred, y_ref, tol=1e-15):
    y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    conc = 0
    disc = 0
    total = 0

    for i, j in itertools.combinations(range(len(y_pred)), 2):
        dp = y_pred[i] - y_pred[j]
        dr = y_ref[i] - y_ref[j]

        if abs(dr) <= tol:
            continue

        total += 1
        if abs(dp) <= tol or dp * dr < 0:
            disc += 1
        else:
            conc += 1

    if total == 0:
        return np.nan, np.nan, 0

    f_viol = disc / total
    tau = (conc - disc) / total
    return f_viol, tau, total


def main():
    df = pd.read_csv(IN_FILE)

    required = {"system", "T", "FANC_eVatom", "F0_eVatom", "Fhat_final"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for model_name, col in [("Baseline", "F0_eVatom"), ("PIML", "Fhat_final")]:
        f_viol, tau, npairs = ordering_metrics_ties_fail(df[col], df["FANC_eVatom"])
        rows.append(
            {
                "scope": "overall",
                "model": model_name,
                "T": "all",
                "f_viol": f_viol,
                "tau": tau,
                "npairs": npairs,
            }
        )

    for Tval, sub in df.groupby("T", sort=True):
        for model_name, col in [("Baseline", "F0_eVatom"), ("PIML", "Fhat_final")]:
            f_viol, tau, npairs = ordering_metrics_ties_fail(sub[col], sub["FANC_eVatom"])
            rows.append(
                {
                    "scope": "by_T",
                    "model": model_name,
                    "T": int(Tval),
                    "f_viol": f_viol,
                    "tau": tau,
                    "npairs": npairs,
                }
            )

    systems = df["system"].astype(str).to_numpy()
    y_ref = df["FANC_eVatom"].to_numpy(dtype=float)

    for model_name, col in [("Baseline", "F0_eVatom"), ("PIML", "Fhat_final")]:
        y_pred = df[col].to_numpy(dtype=float)

        conc = 0
        disc = 0
        total = 0

        for i, j in itertools.combinations(range(len(df)), 2):
            if not (systems[i] == "Ti64" or systems[j] == "Ti64"):
                continue

            dp = y_pred[i] - y_pred[j]
            dr = y_ref[i] - y_ref[j]

            if abs(dr) <= 1e-15:
                continue

            total += 1
            if abs(dp) <= 1e-15 or dp * dr < 0:
                disc += 1
            else:
                conc += 1

        f_viol = np.nan if total == 0 else disc / total
        tau = np.nan if total == 0 else (conc - disc) / total

        rows.append(
            {
                "scope": "interface_involving",
                "model": model_name,
                "T": "all",
                "f_viol": f_viol,
                "tau": tau,
                "npairs": total,
            }
        )

    breakdown_df = pd.DataFrame(rows).sort_values(["scope", "model", "T"]).reset_index(drop=True)
    breakdown_df.to_csv(OUT_BREAKDOWN, index=False)

    mean_df = (
        df.groupby(["system", "T"], as_index=False)
        .agg(
            FANC_mean=("FANC_eVatom", "mean"),
            F0_mean=("F0_eVatom", "mean"),
            Fhat_mean=("Fhat_final", "mean"),
            n_snap=("FANC_eVatom", "size"),
        )
        .sort_values(["system", "T"])
        .reset_index(drop=True)
    )
    mean_df.to_csv(OUT_MEAN_BY_SYSTEM_T, index=False)

    global_rows = []
    for model_name, col in [("Baseline", "F0_eVatom"), ("PIML", "Fhat_final")]:
        f_viol, tau, npairs = ordering_metrics_ties_fail(df[col], df["FANC_eVatom"])
        global_rows.append(
            {
                "model": model_name,
                "f_viol": f_viol,
                "tau": tau,
                "npairs": npairs,
            }
        )

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(OUT_GLOBAL, index=False)

    print("Saved:")
    print(" -", OUT_BREAKDOWN)
    print(" -", OUT_MEAN_BY_SYSTEM_T)
    print(" -", OUT_GLOBAL)


if __name__ == "__main__":
    main()