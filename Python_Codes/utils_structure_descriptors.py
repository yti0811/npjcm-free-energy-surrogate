import numpy as np
import pandas as pd

KB_eV_per_K = 8.617333262e-5

_HAVE_FREUD = False
try:
    import freud
    _HAVE_FREUD = True
except Exception:
    _HAVE_FREUD = False


def iter_snapshots(path: str):
    def read_line(handle):
        line = handle.readline()
        return None if not line else line.strip()

    with open(path, "r") as handle:
        while True:
            line = read_line(handle)
            if line is None:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(read_line(handle))
            read_line(handle)
            natoms = int(read_line(handle))
            read_line(handle)

            bounds = []
            for _ in range(3):
                lo, hi = read_line(handle).split()
                bounds.append((float(lo), float(hi)))

            bounds_lo = np.array([b[0] for b in bounds], dtype=float)
            bounds_hi = np.array([b[1] for b in bounds], dtype=float)

            header = read_line(handle)
            colnames = header.split()[2:]
            ncols = len(colnames)

            data = np.empty((natoms, ncols), dtype=float)
            for i in range(natoms):
                row = read_line(handle)
                vals = row.split()
                if len(vals) != ncols:
                    raise ValueError(f"Column mismatch in {path}: got {len(vals)}, expected {ncols}")
                data[i] = np.asarray(vals, dtype=float)

            cols = {name: data[:, j] for j, name in enumerate(colnames)}

            yield {
                "timestep": timestep,
                "natoms": natoms,
                "bounds_lo": bounds_lo,
                "bounds_hi": bounds_hi,
                "cols": cols,
            }


def _quantiles(values):
    q25, q50, q75 = np.nanpercentile(values, [25, 50, 75])
    return float(q25), float(q50), float(q75), float(q75 - q25)


def von_mises_from_stress_peratom(stress6, atomvol):
    sxx, syy, szz, sxy, sxz, syz = [stress6[:, i] for i in range(6)]
    volume = np.where(atomvol > 0.0, atomvol, np.nan)

    term = (
        0.5 * (sxx - syy) ** 2
        + 0.5 * (sxx - szz) ** 2
        + 0.5 * (szz - syy) ** 2
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )
    return np.sqrt(term) / volume


def compute_q6_freud_knn(pos, bounds_lo, bounds_hi, k=16):
    if not _HAVE_FREUD:
        raise ImportError("freud-analysis is not installed")

    lengths = (bounds_hi - bounds_lo).astype(np.float32)
    box = freud.box.Box(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], is2D=False)
    pos0 = (pos - bounds_lo[None, :]).astype(np.float32)

    query = freud.locality.AABBQuery(box, pos0)
    nlist = query.query(pos0, dict(num_neighbors=int(k), exclude_ii=True)).toNeighborList()

    steinhardt = freud.order.Steinhardt(l=6, average=False)
    steinhardt.compute((box, pos0), neighbors=nlist)

    q6 = np.asarray(getattr(steinhardt, "particle_order", steinhardt.order), dtype=float)
    if np.ndim(q6) == 0:
        q6 = np.full(len(pos0), float(q6), dtype=float)

    n_atoms = len(q6)
    sums = np.zeros(n_atoms, dtype=float)
    counts = np.zeros(n_atoms, dtype=int)

    query_idx = np.asarray(nlist.query_point_indices, dtype=int)
    point_idx = np.asarray(nlist.point_indices, dtype=int)

    sums[query_idx] += q6[point_idx]
    counts[query_idx] += 1

    q6_knn = np.where(counts > 0, sums / counts, np.nan)
    return q6, q6_knn


def snapshot_stats(snap, T, knn_k=16):
    cols = snap["cols"]
    pos = np.vstack([cols["x"], cols["y"], cols["z"]]).T

    pe = cols["c_peatom"]
    ke = cols["c_keatom"]
    total_u = pe + ke

    sle = cols["c_ent"]
    sle_mean = float(np.nanmean(sle))
    sle_std = float(np.nanstd(sle))
    sle_q25, sle_q50, sle_q75, sle_iqr = _quantiles(sle)

    voro = cols["c_atomvol[1]"]
    voro_mean = float(np.nanmean(voro))
    voro_q25, voro_q50, voro_q75, voro_iqr = _quantiles(voro)

    stress6 = np.vstack([cols[f"c_satom[{i}]"] for i in range(1, 7)]).T
    vms = von_mises_from_stress_peratom(stress6, voro)
    vms_mean = float(np.nanmean(vms))
    vms_q25, vms_q50, vms_q75, vms_iqr = _quantiles(vms)

    q6_mean = np.nan
    q6_std = np.nan
    q6_q25 = np.nan
    q6_q50 = np.nan
    q6_q75 = np.nan
    q6_iqr = np.nan
    q6knn_mean = np.nan

    if knn_k and knn_k > 0 and _HAVE_FREUD:
        q6, q6knn = compute_q6_freud_knn(pos, snap["bounds_lo"], snap["bounds_hi"], k=knn_k)
        q6_mean = float(np.nanmean(q6))
        q6_std = float(np.nanstd(q6))
        q6_q25, q6_q50, q6_q75, q6_iqr = _quantiles(q6)
        q6knn_mean = float(np.nanmean(q6knn))

    return {
        "timestep": snap["timestep"],
        "natoms": snap["natoms"],
        "U_eVatom": float(np.nanmean(total_u)),
        "PE_eVatom": float(np.nanmean(pe)),
        "KE_eVatom": float(np.nanmean(ke)),
        "SLE_mean": sle_mean,
        "SLE_std": sle_std,
        "SLE_q25": sle_q25,
        "SLE_q50": sle_q50,
        "SLE_q75": sle_q75,
        "SLE_iqr": sle_iqr,
        "Voro_mean": voro_mean,
        "Voro_q25": voro_q25,
        "Voro_q50": voro_q50,
        "Voro_q75": voro_q75,
        "Voro_iqr": voro_iqr,
        "q6_mean": q6_mean,
        "q6_std": q6_std,
        "q6_q25": float(q6_q25) if np.isfinite(q6_q25) else np.nan,
        "q6_q50": float(q6_q50) if np.isfinite(q6_q50) else np.nan,
        "q6_q75": float(q6_q75) if np.isfinite(q6_q75) else np.nan,
        "q6_iqr": float(q6_iqr) if np.isfinite(q6_iqr) else np.nan,
        "q6knn_mean": q6knn_mean,
        "vMises_mean": vms_mean,
        "vMises_q25": vms_q25,
        "vMises_q50": vms_q50,
        "vMises_q75": vms_q75,
        "vMises_iqr": vms_iqr,
    }

def compute_F0_dF(df_snap, FANC_eVatom, T0=300):
    df = df_snap.copy()

    if not isinstance(FANC_eVatom, dict):
        raise ValueError("FANC_eVatom must be a dict mapping temperature to free energy.")

    df["FANC_eVatom"] = df["T"].map(FANC_eVatom)
    if df["FANC_eVatom"].isna().any():
        missing_T = sorted(df.loc[df["FANC_eVatom"].isna(), "T"].unique())
        raise ValueError(f"Missing FANC mapping for temperatures: {missing_T}")

    df["TSLE_eVatom"] = df["T"] * KB_eV_per_K * df["SLE_mean"]
    df["F0_eVatom"] = df["U_eVatom"] - df["TSLE_eVatom"]
    df["dFraw_eVatom"] = df["FANC_eVatom"] - df["F0_eVatom"]

    d0 = df.loc[df["T"] == T0, "dFraw_eVatom"].mean()
    if pd.isna(d0):
        raise ValueError(f"No snapshots found at T0={T0} K")

    df["dFraw_T0_mean"] = d0
    df["dFres_eVatom"] = df["dFraw_eVatom"] - d0
    return df