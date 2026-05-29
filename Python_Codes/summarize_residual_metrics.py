from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_PRED = TABLE_DIR / "residual_predictions.csv"
OUT_BY_ST = TABLE_DIR / "residual_summary_by_system_T.csv"
OUT_BY_SYS = TABLE_DIR / "residual_summary_by_system.csv"
OUT_GLOBAL = TABLE_DIR / "residual_summary_global.csv"


def fit_predict_group_cv(df, feature_cols, target_col, group_col, model, n_splits_max=5):
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits_max, n_groups)

    if n_splits < 2:
        model.fit(X, y)
        return model.predict(X)

    gkf = GroupKFold(n_splits=n_splits)
    yhat = np.full(len(df), np.nan)

    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        yhat[te] = model.predict(X[te])

    return yhat


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    df = pd.read_csv(IN_FILE)

    required = {
        "system",
        "T",
        "snap",
        "FANC_eVatom",
        "F0_eVatom",
        "dFres_eVatom",
        "SLE_mean",
        "Voro_mean",
        "q6_mean",
        "vMises_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["cv_group"] = df["system"].astype(str) + "_" + df["T"].astype(int).astype(str)

    feature_sets = {
        "enriched_d2": ["SLE_mean", "Voro_mean"],
        "enriched_d3": ["SLE_mean", "Voro_mean", "q6_mean"],
        "temp_only": ["T"],
        "response_only": ["vMises_mean"],
        "piml": ["SLE_mean", "Voro_mean", "q6_mean", "T"],
    }

    models = {
        "enriched_d2": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "enriched_d3": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "temp_only": LinearRegression(),
        "response_only": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "piml": make_pipeline(StandardScaler(), KernelRidge(alpha=1e-4, kernel="rbf", gamma=1.0)),
    }

    for method, feats in feature_sets.items():
        model = models[method]
        df[f"dFpred_{method}"] = fit_predict_group_cv(
            df=df,
            feature_cols=feats,
            target_col="dFres_eVatom",
            group_col="cv_group",
            model=model,
        )
        df[f"Fpred_{method}"] = df["F0_eVatom"] + df[f"dFpred_{method}"]

    keep_cols = [
        "system",
        "T",
        "snap",
        "timestep",
        "dFres_eVatom",
        "dFpred_enriched_d2",
        "dFpred_enriched_d3",
        "dFpred_temp_only",
        "dFpred_response_only",
        "dFpred_piml",
        "FANC_eVatom",
        "F0_eVatom",
        "Fpred_enriched_d2",
        "Fpred_enriched_d3",
        "Fpred_temp_only",
        "Fpred_response_only",
        "Fpred_piml",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUT_PRED, index=False)

    rows_st = []
    for (system, T), sub in df.groupby(["system", "T"], sort=True):
        y_true = sub["dFres_eVatom"].values
        for method in feature_sets:
            y_pred = sub[f"dFpred_{method}"].values
            mae, rmse, r2 = metrics(y_true, y_pred)
            rows_st.append(
                {
                    "system": system,
                    "T": int(T),
                    "method": method,
                    "n_snap": len(sub),
                    "MAE_dFres": mae,
                    "RMSE_dFres": rmse,
                    "R2_dFres": r2,
                }
            )

    tab_st = pd.DataFrame(rows_st).sort_values(["system", "T", "method"]).reset_index(drop=True)
    tab_st.to_csv(OUT_BY_ST, index=False)

    tab_sys = (
        tab_st.groupby(["system", "method"], as_index=False)
        .agg(
            n_T=("T", "count"),
            MAE_mean=("MAE_dFres", "mean"),
            RMSE_mean=("RMSE_dFres", "mean"),
            R2_mean=("R2_dFres", "mean"),
        )
        .sort_values(["system", "method"])
        .reset_index(drop=True)
    )
    tab_sys.to_csv(OUT_BY_SYS, index=False)

    rows_global = []
    y_true_all = df["dFres_eVatom"].values
    for method in feature_sets:
        y_pred_all = df[f"dFpred_{method}"].values
        mae, rmse, r2 = metrics(y_true_all, y_pred_all)
        rows_global.append(
            {
                "method": method,
                "n_snap": len(df),
                "MAE_dFres": mae,
                "RMSE_dFres": rmse,
                "R2_dFres": r2,
            }
        )

    tab_global = pd.DataFrame(rows_global).sort_values("method").reset_index(drop=True)
    tab_global.to_csv(OUT_GLOBAL, index=False)

    print("Saved:")
    print(" -", OUT_PRED)
    print(" -", OUT_BY_ST)
    print(" -", OUT_BY_SYS)
    print(" -", OUT_GLOBAL)


if __name__ == "__main__":
    main()