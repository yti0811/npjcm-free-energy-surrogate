from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "table"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "Al": TABLE_DIR / "piml_predictions_Al.csv",
    "Cu": TABLE_DIR / "piml_predictions_Cu.csv",
}

def process_transfer_system(system_name, in_file):
    df = pd.read_csv(in_file)

    required = {"T", "FANC_eVatom", "F0_eVatom", "Fhat_final"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{system_name}] Missing required columns: {missing}")

    mean_df = (
        df.groupby("T", as_index=False)
        .agg(
            FANC_mean=("FANC_eVatom", "mean"),
            F0_mean=("F0_eVatom", "mean"),
            Fhat_mean=("Fhat_final", "mean"),
            n_snap=("FANC_eVatom", "size"),
        )
        .sort_values("T")
        .reset_index(drop=True)
    )

    mean_df["system"] = system_name

    out_inputs = TABLE_DIR / f"fig11_{system_name.lower()}_transfer_inputs.csv"
    out_summary = TABLE_DIR / f"piml_{system_name.lower()}_summary_by_T.csv"

    mean_df.to_csv(out_inputs, index=False)
    mean_df.to_csv(out_summary, index=False)

    return mean_df, out_inputs, out_summary


def main():
    all_rows = []

    for system_name, in_file in INPUTS.items():
        mean_df, out_inputs, out_summary = process_transfer_system(system_name, in_file)
        all_rows.append(mean_df)

        print(f"[{system_name}] Saved:")
        print(" -", out_inputs)
        print(" -", out_summary)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_out = TABLE_DIR / "fig11_transfer_inputs_combined.csv"
    combined.to_csv(combined_out, index=False)

    print("Combined saved:")
    print(" -", combined_out)


if __name__ == "__main__":
    main()