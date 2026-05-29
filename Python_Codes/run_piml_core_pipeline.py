import re
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_FILE = TABLE_DIR / "F0_dF_by_snapshot.csv"

OUT_MODELSEL = TABLE_DIR / "piml_model_selection_summary.csv"
OUT_METRICS = TABLE_DIR / "piml_metrics.csv"
OUT_PRED_TI = TABLE_DIR / "piml_predictions_Ti.csv"
OUT_PRED_AL = TABLE_DIR / "piml_predictions_Al.csv"
OUT_PRED_CU = TABLE_DIR / "piml_predictions_Cu.csv"


SEED = 0

# ============================================================
# COLUMN AUTO-MAPPING
# ============================================================
def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(df, aliases, required=True):
    cols = list(df.columns)
    norm_map = {normalize_name(c): c for c in cols}

    for a in aliases:
        na = normalize_name(a)
        if na in norm_map:
            return norm_map[na]

    for a in aliases:
        na = normalize_name(a)
        for nc, orig in norm_map.items():
            if na in nc:
                return orig

    if required:
        raise KeyError(f"Could not find a column matching aliases: {aliases}")
    return None


def auto_map_columns(df):
    colmap = {}
    colmap["system"] = find_col(df, ["system", "phase", "label", "group", "material"], required=True)
    colmap["T"] = find_col(df, ["T", "temp", "temperature"], required=True)

    colmap["F0"] = find_col(df, ["F0_eVatom", "F0", "f0"], required=True)
    colmap["dFres"] = find_col(df, ["dFres_eVatom", "dFres", "deltafres", "delta_f_res"], required=True)

    colmap["FANC"] = find_col(df, ["FANC_eVatom", "FANC", "F_ANC"], required=False)
    colmap["dFraw_T0_mean"] = find_col(df, ["dFraw_T0_mean", "dfrawt0mean"], required=False)

    optional = {
        "SLE_mean": ["SLE_mean", "slemean", "local_entropy_mean"],
        "SLE_std": ["SLE_std", "slestd"],
        "SLE_q25": ["SLE_q25", "sleq25"],
        "SLE_q50": ["SLE_q50", "sleq50"],
        "SLE_q75": ["SLE_q75", "sleq75"],
        "SLE_iqr": ["SLE_iqr", "sleiqr"],

        "Voro_mean": ["Voro_mean", "voromean", "voronoi_mean", "volume_mean"],
        "Voro_q25": ["Voro_q25", "voroq25"],
        "Voro_q50": ["Voro_q50", "voroq50"],
        "Voro_q75": ["Voro_q75", "voroq75"],
        "Voro_iqr": ["Voro_iqr", "voroiqr"],

        "q6_mean": ["q6_mean", "q6mean"],
        "q6_std": ["q6_std", "q6std"],
        "q6_q25": ["q6_q25", "q6q25"],
        "q6_q50": ["q6_q50", "q6q50"],
        "q6_q75": ["q6_q75", "q6q75"],
        "q6_iqr": ["q6_iqr", "q6iqr"],
        "q6knn_mean": ["q6knn_mean", "q6knnmean"],

        "U_eVatom": ["U_eVatom", "u_evatom", "u"],
        "TSLE_eVatom": ["TSLE_eVatom", "tsle_evatom", "tsle"],
        "natoms": ["natoms", "n_atoms", "num_atoms"],

        "interface_distance": ["interface_distance", "distance_to_interface", "dist_interface", "d"],
        "x": ["x", "coord_x", "posx"],
        "y": ["y", "coord_y", "posy"],
        "z": ["z", "coord_z", "posz"],
    }

    for k, aliases in optional.items():
        colmap[k] = find_col(df, aliases, required=False)

    if colmap["FANC"] is None:
        df["__FANC_reconstructed__"] = (
            df[colmap["F0"]].to_numpy(dtype=float)
            + df[colmap["dFres"]].to_numpy(dtype=float)
        )
        colmap["FANC"] = "__FANC_reconstructed__"

    if colmap["dFraw_T0_mean"] is None:
        df["__dFraw_T0_mean_default__"] = 0.0
        colmap["dFraw_T0_mean"] = "__dFraw_T0_mean_default__"

    return df, colmap


# ============================================================
# METRICS
# ============================================================
def ordering_metrics_ties_fail(y_pred, y_ref):
    y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    conc = disc = total = 0
    for i, j in itertools.combinations(range(len(y_pred)), 2):
        dp = y_pred[i] - y_pred[j]
        dr = y_ref[i] - y_ref[j]
        if dr == 0:
            continue
        total += 1
        if dp == 0 or dp * dr < 0:
            disc += 1
        else:
            conc += 1

    if total == 0:
        return np.nan, np.nan, 0

    f_viol = disc / total
    tau = (conc - disc) / total
    return f_viol, tau, total


def reconstruct_absolute(df_sub, colmap, dFres_pred):
    dFraw_pred = dFres_pred + df_sub[colmap["dFraw_T0_mean"]].to_numpy(dtype=float)
    Fhat = df_sub[colmap["F0"]].to_numpy(dtype=float) + dFraw_pred
    return dFraw_pred, Fhat


def absolute_metrics(df_sub, colmap, dFres_pred):
    _, Fhat = reconstruct_absolute(df_sub, colmap, dFres_pred)
    y_ref = df_sub[colmap["FANC"]].to_numpy(dtype=float)

    mae_abs = mean_absolute_error(y_ref, Fhat)
    f_viol, tau, npairs = ordering_metrics_ties_fail(Fhat, y_ref)
    return mae_abs, f_viol, tau, npairs, Fhat


# ============================================================
# FEATURES
# ============================================================
def safe_fill(df, cols):
    X = df[cols].copy()
    for c in cols:
        if X[c].isna().all():
            X[c] = 0.0
        elif X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X


def make_state_features(df, colmap, include_baseline_terms=True):
    candidates = [
        colmap["T"],
        colmap.get("SLE_mean"), colmap.get("SLE_std"), colmap.get("SLE_q25"),
        colmap.get("SLE_q50"), colmap.get("SLE_q75"), colmap.get("SLE_iqr"),
        colmap.get("Voro_mean"), colmap.get("Voro_q25"), colmap.get("Voro_q50"),
        colmap.get("Voro_q75"), colmap.get("Voro_iqr"),
        colmap.get("q6_mean"), colmap.get("q6_std"), colmap.get("q6_q25"),
        colmap.get("q6_q50"), colmap.get("q6_q75"), colmap.get("q6_iqr"),
        colmap.get("q6knn_mean"),
        colmap.get("natoms"),
    ]

    if include_baseline_terms:
        candidates += [colmap.get("U_eVatom"), colmap.get("TSLE_eVatom"), colmap["F0"]]

    cols = [c for c in candidates if c is not None and c in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_patterns = [
        "vmises", "stress", "sigma", "strain", "force", "response",
        "fanc", "dfres", "deltaf", "fhat", "pred", "uncert"
    ]
    excluded_exact = {
        colmap["FANC"], colmap["dFres"], colmap["dFraw_T0_mean"],
        colmap.get("interface_distance"), colmap.get("x"), colmap.get("y"), colmap.get("z")
    }

    for c in numeric_cols:
        nc = normalize_name(c)
        if c in cols:
            continue
        if c in excluded_exact:
            continue
        if any(p in nc for p in excluded_patterns):
            continue
        cols.append(c)

    cols = [c for c in cols if c is not None]
    cols = list(dict.fromkeys(cols))
    X = safe_fill(df, cols)
    return X, cols


# ============================================================
# MODELS
# ============================================================
def candidate_models(seed=0):
    models = {}

    models["ridge"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=seed))
    ])

    models["krr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KernelRidge(kernel="rbf", alpha=1e-2, gamma=0.2))
    ])

    models["etr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        ))
    ])

    models["hgbr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=400,
            l2_regularization=1e-3,
            random_state=seed
        ))
    ])

    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-8, 1e-1))
    models["gpr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=1
        ))
    ])

    return models


def gpr_predict(pipe, X):
    X_imp = pipe.named_steps["imputer"].transform(X)
    Xs = pipe.named_steps["scaler"].transform(X_imp)
    model = pipe.named_steps["model"]
    mu, std = model.predict(Xs, return_std=True)
    return mu, std


# ============================================================
# MODEL SELECTION
# ============================================================
def composite_score(mae_abs, f_viol, tau, y_ref_scale):
    mae_n = mae_abs / (y_ref_scale + 1e-12)
    return tau - 1.5 * f_viol - 0.25 * mae_n


def cv_select_ti_model(df_ti, X_ti, colmap, feat_cols, seed=0):
    system_col = colmap["system"]
    T_col = colmap["T"]

    groups = df_ti[system_col].astype(str) + "_" + df_ti[T_col].astype(int).astype(str)
    n_groups = groups.nunique()
    n_splits = min(3, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least 2 unique groups for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    models = candidate_models(seed=seed)

    rows = []
    oof_preds = {name: np.full(len(df_ti), np.nan) for name in models}
    y_ref_scale = df_ti[colmap["FANC"]].std()

    for name, model in models.items():
        for tr_idx, va_idx in gkf.split(X_ti, groups=groups):
            Xtr, Xva = X_ti.iloc[tr_idx], X_ti.iloc[va_idx]
            ytr = df_ti.iloc[tr_idx][colmap["dFres"]].to_numpy(dtype=float)

            if name == "gpr":
                model.fit(Xtr, ytr)
                pred, _ = gpr_predict(model, Xva)
            else:
                model.fit(Xtr, ytr)
                pred = model.predict(Xva)

            oof_preds[name][va_idx] = pred

        mae_abs, f_viol, tau, npairs, _ = absolute_metrics(df_ti, colmap, oof_preds[name])
        score = composite_score(mae_abs, f_viol, tau, y_ref_scale)

        rows.append({
            "model": name,
            "MAE_abs(Fhat)": mae_abs,
            "f_viol_abs(Fhat)": f_viol,
            "kendall_tau_abs(Fhat)": tau,
            "Npairs": npairs,
            "score": score,
            "features": ";".join(feat_cols),
            "w": np.nan,
        })

    sel_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    top2 = sel_df["model"].iloc[:2].tolist()
    best_blend = None
    best_score = -np.inf

    if len(top2) == 2:
        m1, m2 = top2
        for w in np.linspace(0.0, 1.0, 21):
            blend = w * oof_preds[m1] + (1.0 - w) * oof_preds[m2]
            mae_abs, f_viol, tau, npairs, _ = absolute_metrics(df_ti, colmap, blend)
            score = composite_score(mae_abs, f_viol, tau, y_ref_scale)
            if score > best_score:
                best_score = score
                best_blend = {
                    "model": f"blend({m1},{m2})",
                    "w": w,
                    "MAE_abs(Fhat)": mae_abs,
                    "f_viol_abs(Fhat)": f_viol,
                    "kendall_tau_abs(Fhat)": tau,
                    "Npairs": npairs,
                    "score": score,
                    "features": ";".join(feat_cols),
                }

    if best_blend is not None and best_blend["score"] > sel_df.iloc[0]["score"]:
        sel_df = pd.concat([pd.DataFrame([best_blend]), sel_df], ignore_index=True)

    return sel_df


def fit_final_model(df_train, X_train, colmap, model_name, sel_df, seed=0):
    models = candidate_models(seed=seed)

    if model_name.startswith("blend("):
        inside = model_name[len("blend("):-1]
        m1, m2 = inside.split(",")
        w = float(sel_df.loc[sel_df["model"] == model_name, "w"].iloc[0])

        ytr = df_train[colmap["dFres"]].to_numpy(dtype=float)
        models[m1].fit(X_train, ytr)
        models[m2].fit(X_train, ytr)

        return {
            "type": "blend",
            "m1_name": m1,
            "m2_name": m2,
            "w": w,
            "m1": models[m1],
            "m2": models[m2],
        }

    model = models[model_name]
    ytr = df_train[colmap["dFres"]].to_numpy(dtype=float)
    model.fit(X_train, ytr)
    return {"type": "single", "name": model_name, "model": model}


def predict_final(final_model, X):
    if final_model["type"] == "single":
        name = final_model["name"]
        model = final_model["model"]
        if name == "gpr":
            mu, std = gpr_predict(model, X)
        else:
            mu = model.predict(X)
            std = None
        return mu, std

    w = final_model["w"]
    m1_name = final_model["m1_name"]
    m2_name = final_model["m2_name"]

    if m1_name == "gpr":
        mu1, _ = gpr_predict(final_model["m1"], X)
    else:
        mu1 = final_model["m1"].predict(X)

    if m2_name == "gpr":
        mu2, _ = gpr_predict(final_model["m2"], X)
    else:
        mu2 = final_model["m2"].predict(X)

    mu = w * mu1 + (1.0 - w) * mu2
    return mu, None


# ============================================================
# RUN COMMON PIPELINE
# ============================================================
def main():
    print("SNAPSHOT_FILE =", SNAPSHOT_FILE)
    df = pd.read_csv(SNAPSHOT_FILE)
    df, colmap = auto_map_columns(df)

    print("\nAuto-mapped columns:")
    for k, v in colmap.items():
        print(f"  {k}: {v}")

    system_col = colmap["system"]

    ti_systems = ["alpha-TiAl", "beta-TiV", "Ti64"]
    present_systems = df[system_col].astype(str).unique().tolist()
    ti_present = [s for s in ti_systems if s in present_systems]

    if len(ti_present) < 2:
        raise ValueError(f"Need at least two Ti-related systems among {ti_systems}, found: {ti_present}")

    df_ti = df[df[system_col].isin(ti_present)].copy()
    X_ti, feat_cols = make_state_features(df_ti, colmap, include_baseline_terms=True)

    print("\nUsing features:")
    print(feat_cols)

    sel_df = cv_select_ti_model(df_ti, X_ti, colmap, feat_cols, seed=SEED)
    sel_df.to_csv(OUT_MODELSEL, index=False)

    best_model_name = sel_df.iloc[0]["model"]
    print("\nSelected best model:", best_model_name)

    final_model = fit_final_model(df_ti, X_ti, colmap, best_model_name, sel_df, seed=SEED)

    dFres_pred_ti, _ = predict_final(final_model, X_ti)
    dFraw_pred_ti, Fhat_ti = reconstruct_absolute(df_ti, colmap, dFres_pred_ti)

    aux_gpr = candidate_models(seed=1)["gpr"]
    aux_gpr.fit(X_ti, df_ti[colmap["dFres"]].to_numpy(dtype=float))
    dFres_mu_gpr_ti, dFres_std_ti = gpr_predict(aux_gpr, X_ti)

    pred_ti = df_ti.copy()
    pred_ti["dFres_pred_final"] = dFres_pred_ti
    pred_ti["dFraw_pred_final"] = dFraw_pred_ti
    pred_ti["Fhat_final"] = Fhat_ti
    pred_ti["dFres_pred_gpr_mu"] = dFres_mu_gpr_ti
    pred_ti["dFres_pred_gpr_std"] = dFres_std_ti
    pred_ti.to_csv(OUT_PRED_TI, index=False)

    f_b, tau_b, _ = ordering_metrics_ties_fail(pred_ti[colmap["F0"]], pred_ti[colmap["FANC"]])
    f_p, tau_p, _ = ordering_metrics_ties_fail(pred_ti["Fhat_final"], pred_ti[colmap["FANC"]])
    mae_b = mean_absolute_error(pred_ti[colmap["FANC"]], pred_ti[colmap["F0"]])
    mae_p = mean_absolute_error(pred_ti[colmap["FANC"]], pred_ti["Fhat_final"])

    metrics_rows = [{
        "domain": "Ti_in_domain",
        "baseline_MAE_abs": mae_b,
        "baseline_f_viol": f_b,
        "baseline_tau": tau_b,
        "piml_model": best_model_name,
        "piml_MAE_abs": mae_p,
        "piml_f_viol": f_p,
        "piml_tau": tau_p,
    }]

    pred_al = None
    pred_cu = None

    if "Al" in present_systems:
        df_al = df[df[system_col] == "Al"].copy()
        X_al, _ = make_state_features(df_al, colmap, include_baseline_terms=True)

        dFres_pred_al, _ = predict_final(final_model, X_al)
        dFraw_pred_al, Fhat_al = reconstruct_absolute(df_al, colmap, dFres_pred_al)

        dFres_mu_gpr_al, dFres_std_al = gpr_predict(aux_gpr, X_al)

        pred_al = df_al.copy()
        pred_al["dFres_pred_final"] = dFres_pred_al
        pred_al["dFraw_pred_final"] = dFraw_pred_al
        pred_al["Fhat_final"] = Fhat_al
        pred_al["dFres_pred_gpr_mu"] = dFres_mu_gpr_al
        pred_al["dFres_pred_gpr_std"] = dFres_std_al
        pred_al.to_csv(OUT_PRED_AL, index=False)

        f_b_al, tau_b_al, _ = ordering_metrics_ties_fail(pred_al[colmap["F0"]], pred_al[colmap["FANC"]])
        f_p_al, tau_p_al, _ = ordering_metrics_ties_fail(pred_al["Fhat_final"], pred_al[colmap["FANC"]])
        mae_b_al = mean_absolute_error(pred_al[colmap["FANC"]], pred_al[colmap["F0"]])
        mae_p_al = mean_absolute_error(pred_al[colmap["FANC"]], pred_al["Fhat_final"])

        metrics_rows.append({
            "domain": "Al_transfer",
            "baseline_MAE_abs": mae_b_al,
            "baseline_f_viol": f_b_al,
            "baseline_tau": tau_b_al,
            "piml_model": best_model_name,
            "piml_MAE_abs": mae_p_al,
            "piml_f_viol": f_p_al,
            "piml_tau": tau_p_al,
        })

    if "Cu" in present_systems:
    	df_cu = df[df[system_col] == "Cu"].copy()
    	X_cu, _ = make_state_features(df_cu, colmap, include_baseline_terms=True)

    	dFres_pred_cu, _ = predict_final(final_model, X_cu)
    	dFraw_pred_cu, Fhat_cu = reconstruct_absolute(df_cu, colmap, dFres_pred_cu)

    	dFres_mu_gpr_cu, dFres_std_cu = gpr_predict(aux_gpr, X_cu)

    	pred_cu = df_cu.copy()
    	pred_cu["dFres_pred_final"] = dFres_pred_cu
    	pred_cu["dFraw_pred_final"] = dFraw_pred_cu
    	pred_cu["Fhat_final"] = Fhat_cu
    	pred_cu["dFres_pred_gpr_mu"] = dFres_mu_gpr_cu
    	pred_cu["dFres_pred_gpr_std"] = dFres_std_cu
    	pred_cu.to_csv(OUT_PRED_CU, index=False)
	
    	f_b_cu, tau_b_cu, _ = ordering_metrics_ties_fail(pred_cu[colmap["F0"]], pred_cu[colmap["FANC"]])
    	f_p_cu, tau_p_cu, _ = ordering_metrics_ties_fail(pred_cu["Fhat_final"], pred_cu[colmap["FANC"]])
    	mae_b_cu = mean_absolute_error(pred_cu[colmap["FANC"]], pred_cu[colmap["F0"]])
    	mae_p_cu = mean_absolute_error(pred_cu[colmap["FANC"]], pred_cu["Fhat_final"])

    	metrics_rows.append({
            "domain": "Cu_transfer",
            "baseline_MAE_abs": mae_b_cu,
            "baseline_f_viol": f_b_cu,
            "baseline_tau": tau_b_cu,
            "piml_model": best_model_name,
            "piml_MAE_abs": mae_p_cu,
            "piml_f_viol": f_p_cu,
            "piml_tau": tau_p_cu,
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_METRICS, index=False)

    print("\nSaved:")
    print(" ", OUT_MODELSEL)
    print(" ", OUT_METRICS)
    print(" ", OUT_PRED_TI)
    if pred_al is not None:
        print(" ", OUT_PRED_AL)
    if pred_cu is not None:
        print(" ", OUT_PRED_CU)


if __name__ == "__main__":
    main()