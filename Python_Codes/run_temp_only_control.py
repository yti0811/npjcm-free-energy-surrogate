from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from scipy.stats import kendalltau


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"
GROUP_FILE = TABLE_DIR / "group_definitions.csv"

OUT_PRED = TABLE_DIR / "temp_only_predictions.csv"
OUT_BY_ST = TABLE_DIR / "temp_only_summary_by_system_T.csv"
OUT_GLOBAL = TABLE_DIR / "temp_only_summary_global.csv"
OUT_BY_T = TABLE_DIR / "temp_only_summary_fixedT_across_systems.csv"
OUT_BY_SYS = TABLE_DIR / "temp_only_summary_within_system_across_T.csv"


def safe_kendall(y_true, y_pred):
    tau, _ = kendalltau(y_true, y_pred)
    if pd.isna(tau):
        return np.nan
    return float(tau)


def pairwise_order_metrics(y_true, y_pred, tol=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if n < 2:
        return np.nan, np.nan

    violations = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]

            if abs(dt) <= tol:
                continue

            total += 1
            if abs(dp) <= tol or np.sign(dt) != np.sign(dp):
                violations += 1

    f_viol = np.nan if total == 0 else violations / total
    tau = safe_kendall(y_true, y_pred)
    return f_viol, tau


def fit_predict_group_cv(df, feature_cols, target_col, group_col, model, n_splits_max=5):
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values

    unique_groups = np.unique(groups)
    n_splits = min(n_splits_max, len(unique_groups))

    if n_splits < 2:
        model.fit(X, y)
        return model.predict(X)

    gkf = GroupKFold(n_splits=n_splits)
    yhat = np.full(len(df), np.nan)

    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        yhat[te] = model.predict(X[te])

    return yhat

def summarize_fixedT_across_systems(df, pred_col, label):
    rows = []
    g = (
        df.groupby(["system", "T"], as_index=False)
        .agg(
            FANC_mean=("FANC_eVatom", "mean"),
            PRED_mean=(pred_col, "mean"),
            n_snap=("FANC_eVatom", "size"),
            MAE=("FANC_eVatom", lambda x: np.nan),  # placeholder
        )
    )

    # 각 (system,T) 평균에서 MAE도 같이 넣기
    mae_map = (
        df.groupby(["system", "T"], as_index=False)
        .apply(lambda sub: pd.Series({
            "MAE": mean_absolute_error(sub["FANC_eVatom"], sub[pred_col])
        }))
        .reset_index(drop=True)
    )
    g["MAE"] = mae_map["MAE"].values

    for T, sub in g.groupby("T", sort=True):
        sub = sub.sort_values("system")
        y_true = sub["FANC_mean"].to_numpy(float)
        y_pred = sub["PRED_mean"].to_numpy(float)
        f_viol, tau = pairwise_order_metrics(y_true, y_pred)

        rows.append({
            "T": int(T),
            "method": label,
            "n_systems": len(sub),
            "mean_group_MAE": sub["MAE"].mean(),
            "f_viol": f_viol,
            "tau": tau,
        })

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

def summarize_within_system_across_T(df, pred_col, label):
    rows = []
    g = (
        df.groupby(["system", "T"], as_index=False)
        .agg(
            FANC_mean=("FANC_eVatom", "mean"),
            PRED_mean=(pred_col, "mean"),
        )
    )

    for system, sub in g.groupby("system", sort=True):
        sub = sub.sort_values("T")
        y_true = sub["FANC_mean"].to_numpy(float)
        y_pred = sub["PRED_mean"].to_numpy(float)
        f_viol, tau = pairwise_order_metrics(y_true, y_pred)

        rows.append({
            "system": system,
            "method": label,
            "n_temps": len(sub),
            "f_viol": f_viol,
            "tau": tau,
        })

    return pd.DataFrame(rows).sort_values("system").reset_index(drop=True)


def summarize(df, pred_col, label):
    rows_st = []
    for (system, T), sub in df.groupby(["system", "T"], sort=True):
        y_true = sub["FANC_eVatom"].values
        y_pred = sub[pred_col].values
        f_viol, tau = pairwise_order_metrics(y_true, y_pred)
        rows_st.append(
            {
                "system": system,
                "T": int(T),
                "method": label,
                "n_snap": len(sub),
                "MAE": mean_absolute_error(y_true, y_pred),
                "f_viol": f_viol,
                "tau": tau,
            }
        )

    by_st = pd.DataFrame(rows_st).sort_values(["system", "T"]).reset_index(drop=True)

    y_true_all = df["FANC_eVatom"].values
    y_pred_all = df[pred_col].values
    f_viol_all, tau_all = pairwise_order_metrics(y_true_all, y_pred_all)

    global_df = pd.DataFrame(
        [
            {
                "method": label,
                "n_snap": len(df),
                "MAE": mean_absolute_error(y_true_all, y_pred_all),
                "f_viol": f_viol_all,
                "tau": tau_all,
            }
        ]
    )

    return by_st, global_df


def main():
    df = pd.read_csv(IN_FILE)

    if GROUP_FILE.exists():
        groups_df = pd.read_csv(GROUP_FILE)
        groups_df["group_id"] = groups_df["group_id"].astype(str)
        df["group_id"] = df["system"].astype(str) + "_" + df["T"].astype(int).astype(str)
        df["cv_group"] = df["group_id"]
    else:
        df["cv_group"] = df["system"].astype(str) + "_" + df["T"].astype(int).astype(str)

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression()),
        ]
    )

    df["dFpred_temp_only"] = fit_predict_group_cv(
        df=df,
        feature_cols=["T"],
        target_col="dFres_eVatom",
        group_col="cv_group",
        model=model,
    )
    df["Fpred_temp_only"] = df["F0_eVatom"] + df["dFpred_temp_only"]

    keep_cols = [
        "system",
        "T",
        "snap",
        "timestep",
        "FANC_eVatom",
        "F0_eVatom",
        "dFres_eVatom",
        "dFpred_temp_only",
        "Fpred_temp_only",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUT_PRED, index=False)

    by_st, global_df = summarize(df, "Fpred_temp_only", "temp_only")
    by_st.to_csv(OUT_BY_ST, index=False)
    global_df.to_csv(OUT_GLOBAL, index=False)

    print("Saved:")
    print(" -", OUT_PRED)
    print(" -", OUT_BY_ST)
    print(" -", OUT_GLOBAL)

    by_st, global_df = summarize(df, "Fpred_temp_only", "temp_only")
    by_T = summarize_fixedT_across_systems(df, "Fpred_temp_only", "temp_only")
    by_sys = summarize_within_system_across_T(df, "Fpred_temp_only", "temp_only")

    by_st.to_csv(OUT_BY_ST, index=False)
    global_df.to_csv(OUT_GLOBAL, index=False)
    by_T.to_csv(OUT_BY_T, index=False)
    by_sys.to_csv(OUT_BY_SYS, index=False)

    print(" -", OUT_BY_T)
    print(" -", OUT_BY_SYS)


if __name__ == "__main__":
    main()