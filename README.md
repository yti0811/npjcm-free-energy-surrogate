# Thermodynamically Admissible Free-Energy Surrogates

## Reproducibility Package

**Manuscript**  
*Frenkel–Ladd-Anchored Thermal Backbone Learning for Admissibility-Aware Free-Energy Reconstruction under Descriptor Aliasing*

This repository provides a minimal processed-data reproducibility package for the accompanying manuscript and Supplementary Information.

The package contains the processed snapshot-level datasets, the retained Backbone Ridge analysis workflow, validation and control scripts, selected descriptor-aliasing diagnostics, processed-input SOAP benchmark files, representative output tables, and representative upstream LAMMPS inputs.

Reproduction is intentionally organized at the **processed-data level**. Large raw molecular-dynamics trajectories, restart files, production dump files, and the complete `snapshot_split/*.extxyz` archive are not included because of their aggregate size and file count.

Some filenames retain the historical `piml` prefix for compatibility with the original analysis workflow. These filenames refer to the retained Backbone Ridge implementation described in the manuscript and do not indicate a separate production model.

---

# 1. Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── Python_Codes/
│   │
│   ├── run_piml_core_pipeline_backbone.py
│   ├── run_sample_sufficiency_backbone.py
│   ├── run_repeated_training_backbone.py
│   ├── run_logo_holdout_backbone.py
│   ├── run_stricter_split_backbone.py
│   ├── run_ti_trained_controls_transfer.py
│   │
│   ├── analyze_descriptor_aliasing_multivariate_probe.py
│   ├── run_monolithic_ridge_control.py
│   │
│   ├── prepare_soap_baseline_features.py
│   ├── clean_feature_generator_fixed_BL39.py
│   ├── run_SOAP_benchmark.py
│   │
│   ├── postprocess_piml_ti_domain.py
│   ├── postprocess_piml_al_cu_transfer.py
│   ├── summarize_residual_metrics.py
│   ├── utils_structure_descriptors.py
│   │
│   └── table/
│       ├── F0_dF_by_snapshot.csv
│       ├── metadata_soap_backbone.csv
│       ├── soap_anchor_relative_rcut4.5.csv
│       ├── soap_anchor_relative_rcut6.csv
│       └── [generated and representative CSV outputs]
│
├── Representative_Outputs/
│
└── LAMMPS_References/
    ├── aTiAl/
    ├── stacked_Ti64/
    └── Conf/
```

---

# 2. Processed Data and Required Inputs

## 2.1 Core Backbone Ridge input

The principal processed input for the production Backbone Ridge workflow is:

- `F0_dF_by_snapshot.csv`

For direct execution, this file should be located at:

```text
Python_Codes/table/F0_dF_by_snapshot.csv
```

The processed table contains the thermodynamic and descriptor information used to construct the physics baseline, Frenkel–Ladd reference, system-specific gauge alignment, gauge-fixed residual, common thermal backbone, and anchor-relative descriptor correction.

The processed dataset includes the Ti-domain systems:

- alpha-TiAl
- beta-TiV
- stacked Ti64

and the external transfer systems:

- FCC Al
- FCC Cu

at the temperature states evaluated in the manuscript.

Large raw MD trajectories are intentionally omitted because they are not required for the downstream processed-data analyses.

## 2.2 Additional precomputed inputs for the SOAP benchmark

The processed-input SOAP workflow uses the following three **precomputed processed inputs**:

- `metadata_soap_backbone.csv`
- `soap_anchor_relative_rcut4.5.csv`
- `soap_anchor_relative_rcut6.csv`

These files should be placed in:

```text
Python_Codes/table/
```

### `metadata_soap_backbone.csv`

This table contains the snapshot-level metadata, target quantities, and non-SOAP source variables required to reconstruct the 39-feature non-SOAP benchmark baseline used in the independent SOAP analysis.

It is itself a precomputed upstream output. The minimal reproducibility workflow does **not** regenerate this file from raw trajectories.

### `soap_anchor_relative_rcut4.5.csv`

This file contains the precomputed system-anchor-relative SOAP representation for a cutoff radius of 4.5 Å.

### `soap_anchor_relative_rcut6.csv`

This file contains the corresponding precomputed system-anchor-relative SOAP representation for a cutoff radius of 6.0 Å.

The two SOAP tables are descriptor-level inputs used directly by `run_SOAP_benchmark.py`.

Full regeneration of the SOAP descriptors requires the complete atomic snapshot archive (`snapshot_split/*.extxyz`) and the corresponding upstream trajectory data. These files are not included in the minimal processed-date package because of their aggregate size and file count. Therefore, the distributed SOAP workflow reproduces the **processed-representation statistical benchmark**, rather than regenerating SOAP descriptors from raw atomic structures.

---

# 3. Core Backbone Ridge Workflow

## 3.1 `run_piml_core_pipeline_backbone.py`

This is the main production implementation of the retained Frenkel–Ladd-anchored thermal-backbone / Backbone Ridge workflow.

The absolute free-energy reconstruction is represented hierarchically as

```text
F_ANC = F0 + C_s + DeltaF_res
```

with

```text
DeltaF_res = g_hat(T) + r_hat(delta x)
```

where:

- `F0` is the explicit physics baseline,
- `C_s` is the system-specific gauge-alignment constant,
- `g_hat(T)` is the common thermal backbone, and
- `r_hat(delta x)` is the anchor-relative descriptor-dependent Ridge correction.

The final reconstructed free energy is

```text
F_hat = F0 + C_s + g_hat(T) + r_hat(delta x)
```

The script performs:

- construction of gauge-fixed residual targets,
- fitting of the common thermal backbone,
- anchor-relative descriptor preprocessing,
- Ridge-model selection,
- grouped out-of-fold evaluation,
- admissibility-aware model assessment,
- final Ti-domain fitting,
- Ti-domain reconstruction,
- anchor-aligned external Al/Cu prediction without material-specific retraining, and
- generation of downstream prediction and audit tables.

Representative outputs include:

- `piml_model_selection_summary.csv`
- `piml_metrics.csv`
- `piml_predictions_Ti.csv`
- `piml_predictions_Al.csv`
- `piml_predictions_Cu.csv`
- `piml_feature_audit.csv`
- `piml_backbone_coefficients.csv`

The generated Ti/Al/Cu prediction files are subsequently used by several downstream post-processing and control scripts.

---

# 4. Standard Validation and Transfer Scripts

## 4.1 `run_sample_sufficiency_backbone.py`

Evaluates robustness of the retained Backbone Ridge workflow under reduced training-snapshot counts.

Representative outputs include:

- `table_s12_sample_sufficiency.csv`
- `sample_sufficiency_runs.csv`
- `sample_sufficiency_audit.csv`

## 4.2 `run_repeated_training_backbone.py`

Repeats the retained deterministic Ridge/model-selection workflow as a pipeline reproducibility audit.

Agreement across nominal seeds should be interpreted as computational/pipeline reproducibility rather than stochastic-training uncertainty.

Representative outputs include:

- `table_s13_repeated_training_summary.csv`
- `repeated_training_runs.csv`
- `repeated_training_audit.csv`

## 4.3 `run_logo_holdout_backbone.py`

Performs Leave-One-Group-Out validation over Ti-domain system-temperature groups.

This is an anchor-aligned Ti-domain LOGO evaluation and should not be interpreted as an anchor-free blind-material prediction test.

Representative outputs include:

- `table_s17_logo_holdout.csv`
- `logo_holdout_runs.csv`
- `logo_holdout_audit.csv`

## 4.4 `run_stricter_split_backbone.py`

Performs stricter internal validation beyond the standard grouped-OOF / LOGO analysis.

The script evaluates:

- leave-one-system-out validation,
- bulk-to-interface transfer,
- leave-one-temperature-out validation, and
- low-to-high-temperature transfer.

Representative outputs include:

- `stricter_system_holdout_runs.csv`
- `stricter_system_holdout_summary.csv`
- `stricter_bulk_to_interface_runs.csv`
- `stricter_bulk_to_interface_summary.csv`
- `stricter_temperature_holdout_runs.csv`
- `stricter_temperature_holdout_summary.csv`
- `stricter_low_to_high_runs.csv`
- `stricter_low_to_high_summary.csv`

## 4.5 `run_ti_trained_controls_transfer.py`

Evaluates deliberately restricted negative-control models under the Ti-training to Al/Cu no-retraining transfer hierarchy.

The retained controls include:

- a temperature-only control, and
- a temperature plus local-response control.

This script uses prediction tables generated by the production Backbone Ridge workflow.

---

# 5. Direct Multivariate Descriptor-Aliasing Probe

## `analyze_descriptor_aliasing_multivariate_probe.py`

This script performs the direct non-parametric multivariate representation-space probe reported in the Supplementary Information.

It evaluates the backbone-removed correction target using nested anchor-relative descriptor stacks:

- `d1`: SLE-based statistics,
- `d2`: SLE + Voronoi-volume statistics,
- `d3`: SLE + Voronoi-volume + q6-based statistics.

Cross-state nearest-neighbor mismatch is evaluated for multiple neighborhood sizes (`k = 10, 25, 50`) after excluding neighbors from the query snapshot's own system-temperature state.

Representative outputs include:

- `descriptor_aliasing_knn_summary.csv`
- `descriptor_aliasing_knn_system_summary.csv`
- `descriptor_aliasing_knn_snapshot_errors.csv`

The analysis corresponds to the direct multivariate descriptor-aliasing assessment reported in the Supplementary Information (including the results summarized in Table S5 / Fig. S4 in the current manuscript version).

This is a representation-space diagnostic and does not use a fitted Ridge prediction as the local estimator.

---

# 6. Matched Backbone Ridge vs Monolithic Ridge Control

## `run_monolithic_ridge_control.py`

This script performs the matched architectural control comparing explicit Backbone Ridge decomposition with a monolithic Ridge formulation.

The comparison is designed so that the two models use the same anchor-relative correction descriptors, grouped fold structure, preprocessing logic, and Ridge regularization. The monolithic model additionally receives the explicit thermal terms

```text
dT_anchor
dT_anchor_sq
```

so that it has access to the same quadratic temperature information represented explicitly by the Backbone formulation.

The script uses the following principal inputs from `./table`:

- `F0_dF_by_snapshot.csv`
- `piml_predictions_Ti.csv`
- `piml_backbone_coefficients.csv`
- `piml_model_selection_summary.csv`
- `SI_Table_S6_oof_predictions_existing.csv`

The `SI_Table_S6_oof_predictions_existing.csv` file supplies the grouped OOF fold mapping used for the matched comparison. In the recommended reproduction workflow, this file is generated by `run_SOAP_benchmark.py`; therefore, the SOAP benchmark should be executed before the matched monolithic control unless an identical verified fold-mapping file is already supplied.

Representative outputs include:

- `matched_backbone_monolithic_metrics.csv`
- `matched_backbone_oof_predictions.csv`
- `monolithic_ridge_oof_predictions.csv`
- `matched_backbone_fold_coefficients.csv`
- `matched_control_fold_assignment.csv`
- `matched_control_state_mean.csv`
- `matched_control_state_mean_metrics.csv`
- `target_burden_summary.csv`
- `target_burden_by_fold.csv`
- `matched_control_run_summary.txt`

The results correspond to the matched Backbone-versus-monolithic control and target-burden analysis reported in the Supplementary Information (Tables S7 and S8 in the current manuscript version).

### Production OOF reference versus matched-control OOF

The production Backbone Ridge OOF record and the recomputed Backbone Ridge record used in the matched control are intentionally distinct evaluation records.

The production reference is imported from the production model-selection workflow. The matched Backbone result is recomputed under the fold mapping used for the direct Backbone-versus-monolithic comparison. Therefore, the production Backbone OOF MAE and the matched-control Backbone OOF MAE are not required to be numerically identical.

For the architectural control, the relevant direct comparison is between the **recomputed matched Backbone Ridge** and **matched Monolithic Ridge** rows produced under the same fold mapping.

---

# 7. Independent Processed-Input SOAP Benchmark

The SOAP analysis provides an independent stress test using a substantially richer local many-body representation.

The minimal reproducibility workflow uses three supplied precomputed inputs:

```text
metadata_soap_backbone.csv
soap_anchor_relative_rcut4.5.csv
soap_anchor_relative_rcut6.csv
```

and three executable processing/analysis scripts:

```text
prepare_soap_baseline_features.py
clean_feature_generator_fixed_BL39.py
run_SOAP_benchmark.py
```

## 7.1 `prepare_soap_baseline_features.py`

This script reconstructs the original 39-feature non-SOAP benchmark baseline directly from the supplied `metadata_soap_backbone.csv`.

The Baseline-39 representation consists of:

- 21 source variables retained in the original independent SOAP benchmark, and
- 18 system-wise 300 K anchor-relative descriptor variables.

Thus,

```text
21 + 18 = 39 non-SOAP benchmark features
```

The 39-feature definition is the one recorded in the original SOAP benchmark configuration used for the manuscript analysis.

Representative outputs include:

- `BL39_existing_structural_features_for_soap.csv`
- `BL39_baseline_feature_audit.csv`

Despite the historical output filename containing `structural_features`, the Baseline-39 table should be understood as the **non-SOAP benchmark baseline**, rather than as the 18-feature `d3` descriptor stack used in the main descriptor-enrichment analysis.

This step does not require raw MD trajectories or the `snapshot_split/*.extxyz` archive.

## 7.2 `clean_feature_generator_fixed_BL39.py`

This script validates and cleans the reconstructed Baseline-39 table before it is used by the SOAP benchmark.

The script checks:

- the exact 39-feature schema,
- numerical validity,
- snapshot-level uniqueness,
- duplicate consistency, and
- the expected feature count.

Representative outputs include:

- `BL39_existing_structural_features_for_soap_clean.csv`
- `BL39_existing_feature_duplicate_audit.csv`

The cleaning step does not regenerate structural descriptors and does not require raw atomic snapshot files.

## 7.3 `run_SOAP_benchmark.py`

This script performs the downstream processed-input SOAP statistical benchmark.

It reads:

- `BL39_existing_structural_features_for_soap_clean.csv`
- `soap_anchor_relative_rcut4.5.csv`
- `soap_anchor_relative_rcut6.csv`

and evaluates:

1. Baseline-39,
2. SOAP-only (`rcut = 4.5 Å`),
3. Baseline-39 + SOAP (`rcut = 4.5 Å`),
4. SOAP-only (`rcut = 6.0 Å`), and
5. Baseline-39 + SOAP (`rcut = 6.0 Å`).

The corresponding input dimensions are:

```text
Baseline-39            =   39 features
SOAP-only              = 3000 features
Baseline-39 + SOAP     = 3039 features
```

The statistical protocol includes:

- grouped Ti-domain out-of-fold evaluation,
- fold-local imputation and standardization,
- PCA for SOAP-containing models,
- Ridge regression,
- final Ti-domain fitting, and
- anchor-aligned fixed-model evaluation on Al and Cu without material-specific retraining.

Representative outputs include:

- `soap_benchmark_summary.csv`
- `SI_Table_S6_oof_predictions_existing.csv`
- `oof_predictions_soap-only_rcut4.5.csv`
- `oof_predictions_hybrid_rcut4.5.csv`
- `oof_predictions_soap-only_rcut6.csv`
- `oof_predictions_hybrid_rcut6.csv`
- corresponding `transfer_*.csv` files
- corresponding fitted-model files
- run-configuration/audit information

The processed-input workflow reproduces the independent SOAP benchmark reported in the Supplementary Information (Table S6 in the current manuscript version), including the 39-feature baseline, 3000-feature SOAP-only representations, 3039-feature hybrid representations, grouped Ti-domain OOF evaluation, and fixed-model Al/Cu transfer.

### Scope of SOAP reproducibility

The distributed package reproduces the following stage:

```text
precomputed processed SOAP representation
        ↓
statistical learning
        ↓
grouped Ti-domain OOF evaluation
        ↓
fixed-model Al/Cu evaluation
        ↓
reported SOAP benchmark
```

It does not regenerate the SOAP descriptor matrices from the complete atomic structure archive.

Full upstream SOAP regeneration would require:

```text
raw MD trajectories
        ↓
snapshot_split/*.extxyz
        ↓
SOAP descriptor construction
        ↓
system-wise anchor processing
        ↓
soap_anchor_relative_*.csv
```

The complete raw trajectory and snapshot archive are intentionally omitted from the minimal processed-date package.

---

# 8. Post-Processing Scripts

## 8.1 `postprocess_piml_ti_domain.py`

Reads:

- `table/piml_predictions_Ti.csv`

and generates manuscript-facing Ti-domain reconstruction and ordering summaries.

Representative outputs include:

- `fig7_ti_ordering_inputs.csv`
- `piml_ti_means_by_system_T.csv`
- `piml_ti_ordering_global.csv`

## 8.2 `postprocess_piml_al_cu_transfer.py`

Reads:

- `table/piml_predictions_Al.csv`
- `table/piml_predictions_Cu.csv`

and generates temperature-resolved external-transfer summaries for FCC Al and FCC Cu.

Representative outputs include:

- `piml_al_summary_by_T.csv`
- `piml_cu_summary_by_T.csv`
- `fig8_transfer_inputs_combined.csv`
- `fig8_al_transfer_inputs.csv`
- `fig8_cu_transfer_inputs.csv`

## 8.3 `summarize_residual_metrics.py`

Generates aggregated residual-learning and staged absolute-reconstruction summaries for:

1. `F0`,
2. `F0 + C_s`,
3. `F0 + C_s + g_hat(T)`, and
4. `F0 + C_s + g_hat(T) + r_hat(delta x)`.

Representative outputs include:

- `residual_predictions.csv`
- `residual_summary_by_system_T.csv`
- `residual_summary_by_system.csv`
- `residual_summary_global.csv`
- `residual_summary_audit.csv`
- `reconstruction_stage_summary_by_system_T.csv`
- `reconstruction_stage_summary_by_system.csv`
- `reconstruction_stage_summary_global.csv`

---

# 9. Descriptor Utility

## `utils_structure_descriptors.py`

Provides utility routines for constructing compatible snapshot-level structural descriptors from upstream LAMMPS data.

The distributed processed datasets have already been generated. Therefore, this utility is not required for reproducing the downstream Backbone Ridge, validation, control, multivariate-probe, or processed-input SOAP analyses.

Raw descriptor generation requires the corresponding upstream per-atom trajectory/dump data, which are not included in the minimal processed-data package.

---

# 10. Representative LAMMPS Inputs

`LAMMPS_References/` contains representative upstream molecular-dynamics and Frenkel–Ladd input files documenting the simulation protocol.

Examples include:

```text
aTiAl/
    aTiAl_annealing.in
    aTiAl_msd_500K.in
    aTiAl_FE_new_500K.in

stacked_Ti64/
    stacked_Ti64_annealing.in
    stacked_Ti64_msd_500K.in
    stacked_Ti64_FE_all_500K.in

Conf/
    representative structure/configuration files
```

These files are provided for methodological documentation and are not required for the downstream processed-data reviewer workflow.

---

# 11. Recommended Reproduction Workflow

Before execution, confirm that the following processed inputs are present in `Python_Codes/table/`:

```text
F0_dF_by_snapshot.csv
metadata_soap_backbone.csv
soap_anchor_relative_rcut4.5.csv
soap_anchor_relative_rcut6.csv
```

Then change to the analysis directory:

```bash
cd Python_Codes
```

## Step 1 — Production Backbone Ridge workflow

```bash
python run_piml_core_pipeline_backbone.py
```

This step generates the principal Ti/Al/Cu prediction files and production model-selection outputs used by several downstream analyses.

## Step 2 — Standard robustness and grouped validation

```bash
python run_sample_sufficiency_backbone.py
python run_repeated_training_backbone.py
python run_logo_holdout_backbone.py
```

## Step 3 — Stricter structural and temperature splits

```bash
python run_stricter_split_backbone.py
```

## Step 4 — Restricted negative-control transfer

```bash
python run_ti_trained_controls_transfer.py
```

## Step 5 — Direct multivariate descriptor-aliasing probe

```bash
python analyze_descriptor_aliasing_multivariate_probe.py
```

## Step 6 — Reconstruct and validate Baseline-39 for the SOAP benchmark

```bash
python prepare_soap_baseline_features.py
python clean_feature_generator_fixed_BL39.py
```

This produces the cleaned Baseline-39 input used by the SOAP benchmark.

## Step 7 — Run the independent processed-input SOAP benchmark

```bash
python run_SOAP_benchmark.py
```

This step uses the supplied precomputed SOAP representations and generates, among other outputs, `SI_Table_S6_oof_predictions_existing.csv`.

## Step 8 — Run the matched Backbone-versus-monolithic control

```bash
python run_monolithic_ridge_control.py
```

This step reuses the verified grouped fold mapping from `SI_Table_S6_oof_predictions_existing.csv` for the matched architectural comparison.

## Step 9 — Manuscript-facing postprocessing

```bash
python postprocess_piml_ti_domain.py
python postprocess_piml_al_cu_transfer.py
python summarize_residual_metrics.py
```

Unless otherwise specified by an individual script, generated CSV outputs are written to:

```text
Python_Codes/table/
```

---

# 12. Input/Output Dependency Summary

```text
CORE BACKBONE BRANCH

F0_dF_by_snapshot.csv
        |
        +--> run_piml_core_pipeline_backbone.py
        |       |
        |       +--> piml_predictions_Ti.csv
        |       +--> piml_predictions_Al.csv
        |       +--> piml_predictions_Cu.csv
        |       +--> piml_model_selection_summary.csv
        |       +--> piml_backbone_coefficients.csv
        |       +--> other model-selection / metrics / audit CSVs
        |               |
        |               +--> postprocess_piml_ti_domain.py
        |               +--> postprocess_piml_al_cu_transfer.py
        |               +--> run_ti_trained_controls_transfer.py
        |               +--> summarize_residual_metrics.py
        |
        +--> run_sample_sufficiency_backbone.py
        +--> run_repeated_training_backbone.py
        +--> run_logo_holdout_backbone.py
        +--> run_stricter_split_backbone.py
        +--> analyze_descriptor_aliasing_multivariate_probe.py


SOAP PROCESSED-DATA BRANCH

metadata_soap_backbone.csv
        |
        +--> prepare_soap_baseline_features.py
                |
                +--> BL39_existing_structural_features_for_soap.csv
                        |
                        +--> clean_feature_generator_fixed_BL39.py
                                |
                                +--> BL39_existing_structural_features_for_soap_clean.csv
                                         |
                                         |
soap_anchor_relative_rcut4.5.csv -------+
                                         |
soap_anchor_relative_rcut6.csv ---------+
                                         |
                                         +--> run_SOAP_benchmark.py
                                                 |
                                                 +--> soap_benchmark_summary.csv
                                                 +--> SI_Table_S6_oof_predictions_existing.csv
                                                 +--> SOAP/hybrid OOF outputs
                                                 +--> Al/Cu transfer outputs
                                                          |
                                                          +--> run_monolithic_ridge_control.py

Additional production inputs used by run_monolithic_ridge_control.py:
    F0_dF_by_snapshot.csv
    piml_predictions_Ti.csv
    piml_backbone_coefficients.csv
    piml_model_selection_summary.csv
```

The SOAP branch starts from supplied processed snapshot-/descriptor-level inputs. The complete raw atomic snapshot archive is not required for this downstream reproduction workflow.

---

# 13. Software Requirements

Recommended Python version:

- Python 3.10+

Core downstream packages:

- numpy
- pandas
- scipy
- scikit-learn
- joblib

Additional packages used by supplied scripts where applicable:

- matplotlib

The processed-input SOAP benchmark does not require regeneration of SOAP descriptors from raw atomic structures. Therefore, packages required only for upstream raw SOAP construction are not necessary for the minimal reproducibility workflow.

Dependencies may be installed using:

```bash
pip install -r requirements.txt
```

---

# 14. Data and Reproducibility Scope

This repository supports processed-data reproduction and verification of the principal numerical analyses reported in the manuscript and Supplementary Information, including:

- Backbone Ridge model selection and fitting,
- gauge-aligned residual reconstruction,
- common thermal-backbone decomposition,
- Ti-domain grouped OOF evaluation,
- sample-sufficiency analysis,
- repeated-execution reproducibility,
- LOGO evaluation,
- leave-one-system-out validation,
- bulk-to-interface transfer,
- leave-one-temperature-out validation,
- low-to-high-temperature transfer,
- external Al/Cu no-retraining transfer,
- restricted negative-control transfer analyses,
- direct multivariate descriptor-aliasing diagnostics,
- matched Backbone-versus-monolithic Ridge controls,
- target-burden analysis,
- independent processed-input SOAP representation tests, and
- manuscript-facing staged-reconstruction and summary tables.

Large MD trajectory files, restart files, production dump files, and the complete `snapshot_split/*.extxyz` archive are not included because this package is intentionally designed around the processed-data reproducibility layer.

For the SOAP analysis, `metadata_soap_backbone.csv`, `soap_anchor_relative_rcut4.5.csv`, and `soap_anchor_relative_rcut6.csv` are supplied precomputed inputs. Therefore, the package reproduces Baseline-39 reconstruction and the downstream SOAP statistical-learning/evaluation workflow, but does not regenerate the SOAP descriptors from the complete raw atomic structure archive.

Representative LAMMPS input and configuration files are provided separately to document the upstream simulation methodology.

---

# 15. Notes on File Naming

Several files retain the historical `piml` prefix for compatibility with the development and manuscript-analysis workflow. They refer to the retained Backbone Ridge pipeline and should not be interpreted as a separate production model.

Pipeline-level intermediate files such as:

- `piml_predictions_Ti.csv`
- `piml_predictions_Al.csv`
- `piml_predictions_Cu.csv`

retain stable filenames because they are shared by multiple downstream scripts.

The `BL39_` prefix identifies files associated with reconstruction and validation of the 39-feature non-SOAP baseline used in the independent SOAP benchmark.

The historical filename `existing_structural_features_for_soap` should not be interpreted as meaning that Baseline-39 is identical to the 18-feature `d3` descriptor stack used elsewhere in the manuscript. The independent SOAP benchmark uses a separate 39-feature non-SOAP baseline.

Manuscript-facing filenames may contain Supplementary Table or main-text Figure numbering. If numbering changes during editorial processing, the numerical content of the corresponding files is unchanged.

---

# 16. Citation

D. Y. Kim, Y. Hwang, H. W. Park, D. H. Wang, and T. Yi,  
“Frenkel–Ladd-anchored thermal backbone learning for admissibility-aware free-energy reconstruction under descriptor aliasing,” manuscript (2026).

