from pathlib import Path
import pandas as pd

from utils_structure_descriptors import iter_snapshots, snapshot_stats, compute_F0_dF

FANC_EVATOM = {
    "alpha-TiAl": {300: -4.80882399, 400: -4.84201127, 500: -4.88184236, 600: -4.92700903, 700: -4.97663412},
    "beta-TiV": {300: -4.88796052, 400: -4.92292887, 500: -4.96463792, 600: -5.01175502, 700: -5.06352247},
    "Ti64": {300: -4.79827426, 400:  -4.83216935, 500: -4.87341393, 600: -4.92132028, 700: -4.97995281},
    "Al": {300: -3.3669, 400: -3.3990, 500: -3.4378, 600: -3.4820, 700: -3.5308},
    "Cu": {300: -3.5616, 400: -3.5990, 500: -3.6431, 600: -3.6927, 700: -3.7469},
}

DUMP_FILES = {
    "alpha-TiAl": {
        300: "/Users/yti0811/Desktop/FE_data/npj_CM01/aTiAl/T300/TiAl_300K_3.dat",
        400: "/Users/yti0811/Desktop/FE_data/npj_CM01/aTiAl/T400/TiAl_400K_3.dat",
        500: "/Users/yti0811/Desktop/FE_data/npj_CM01/aTiAl/T500/TiAl_500K_3.dat",
        600: "/Users/yti0811/Desktop/FE_data/npj_CM01/aTiAl/T600/TiAl_600K_3.dat",
        700: "/Users/yti0811/Desktop/FE_data/npj_CM01/aTiAl/T700/TiAl_700K_3.dat",
    },
    "beta-TiV": {
        300: "/Users/yti0811/Desktop/FE_data/npj_CM01/bTiV/T300/Ti_300K_3.dat",
        400: "/Users/yti0811/Desktop/FE_data/npj_CM01/bTiV/T400/Ti_400K_3.dat",
        500: "/Users/yti0811/Desktop/FE_data/npj_CM01/bTiV/T500/Ti_500K_3.dat",
        600: "/Users/yti0811/Desktop/FE_data/npj_CM01/bTiV/T600/Ti_600K_3.dat",
        700: "/Users/yti0811/Desktop/FE_data/npj_CM01/bTiV/T700/Ti_700K_3.dat",
    },
    "Ti64": {
        300: "/Users/yti0811/Desktop/FE_data/npj_CM01/Ti64/Ti64_2/MSD/T300/Ti64_300K_3.dat",
        400: "/Users/yti0811/Desktop/FE_data/npj_CM01/Ti64/Ti64_2/MSD/T400/Ti64_400K_3.dat",
        500: "/Users/yti0811/Desktop/FE_data/npj_CM01/Ti64/Ti64_2/MSD/T500/Ti64_500K_3.dat",
        600: "/Users/yti0811/Desktop/FE_data/npj_CM01/Ti64/Ti64_2/MSD/T600/Ti64_600K_3.dat",
        700: "/Users/yti0811/Desktop/FE_data/npj_CM01/Ti64/Ti64_2/MSD/T700/Ti64_700K_3.dat",
    },
    "Al": {
        300: "/Users/yti0811/Desktop/FE_data/npj_CM01/Al/T300/Al_300K_3.dat",
        400: "/Users/yti0811/Desktop/FE_data/npj_CM01/Al/T400/Al_400K_3.dat",
        500: "/Users/yti0811/Desktop/FE_data/npj_CM01/Al/T500/Al_500K_3.dat",
        600: "/Users/yti0811/Desktop/FE_data/npj_CM01/Al/T600/Al_600K_3.dat",
        700: "/Users/yti0811/Desktop/FE_data/npj_CM01/Al/T700/Al_700K_3.dat",
    },
    "Cu": {
        300: "/Users/yti0811/Desktop/FE_data/npj_CM01/Copper/T300/Cu_300K_3.dat",
        400: "/Users/yti0811/Desktop/FE_data/npj_CM01/Copper/T400/Cu_400K_3.dat",
        500: "/Users/yti0811/Desktop/FE_data/npj_CM01/Copper/T500/Cu_500K_3.dat",
        600: "/Users/yti0811/Desktop/FE_data/npj_CM01/Copper/T600/Cu_600K_3.dat",
        700: "/Users/yti0811/Desktop/FE_data/npj_CM01/Copper/T700/Cu_700K_3.dat",
    },

}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "table"
SNAPSHOT_DIR = OUTDIR / "snapshots"
DERIVED_DIR = OUTDIR
TABLES_DIR = OUTDIR

KNN_K = 16
T0 = 300


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    meta = []

    for system, temps in DUMP_FILES.items():
        for T, path in temps.items():
            rows = []
            for sidx, snap in enumerate(iter_snapshots(path)):
                st = snapshot_stats(snap, T=T, knn_k=KNN_K)
                st["system"] = system
                st["T"] = T
                st["snap"] = sidx
                rows.append(st)

            df_snap = pd.DataFrame(rows)
            df_snap.to_csv(SNAPSHOT_DIR / f"{system}_{T}K_snap.csv", index=False)

            meta.append({
                "system": system,
                "T": T,
                "file": path,
                "nsnap": len(df_snap),
                "natoms": int(df_snap["natoms"].iloc[0]),
            })

            all_rows.append(df_snap)

    pd.DataFrame(meta).to_csv(TABLES_DIR / "dump_meta.csv", index=False)

    df_all = pd.concat(all_rows, ignore_index=True)

    out_list = []
    for system in df_all["system"].unique():
        dsys = df_all[df_all["system"] == system].copy()
        dsys = compute_F0_dF(dsys, FANC_EVATOM[system], T0=T0)
        out_list.append(dsys)

    df_F = pd.concat(out_list, ignore_index=True)
    df_F.to_csv(DERIVED_DIR / "F0_dF_by_snapshot.csv", index=False)

    df_sum = (
        df_F.groupby(["system", "T"], as_index=False)
        .agg({
            "FANC_eVatom": "mean",
            "U_eVatom": "mean",
            "SLE_mean": "mean",
            "F0_eVatom": "mean",
            "dFraw_eVatom": "mean",
            "dFres_eVatom": "mean",
        })
    )
    df_sum.to_csv(DERIVED_DIR / "F_summary_by_systemT.csv", index=False)

    print("DONE. Outputs written to:", OUTDIR)


if __name__ == "__main__":
    main()