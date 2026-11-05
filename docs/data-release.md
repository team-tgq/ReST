# ReST research data, version 1.0.1

**English** | [简体中文](data-release.zh-CN.md)

Supporting data for *Reliable Spatiotemporal Generalization for Snow Water Equivalent Forecasting*.
Code and Temporal checkpoints: https://github.com/team-tgq/ReST/releases/tag/v1.0.1.
Data release: https://github.com/team-tgq/ReST/releases/tag/data-v1.0.1.
The data are archived on [Zenodo](https://doi.org/10.5281/zenodo.23968133), DOI **10.5281/zenodo.23968133**.

## Contents and use

The package contains the actual active-station task inputs, fixed priors, split indices,
archived publication result tables, figure source values, and external ReST prediction/observation
pairs. External baseline and ablation **result values** are included. Their model implementations
and checkpoints are not distributed.

| Directory/file | Contents |
|---|---|
| `task_datasets/` | Four sources, seven files each; 399 active stations; daily arrays for 2010–2020 |
| `station_index_map.csv` | Original and released station indices and upstream provider identity |
| `assets/` | Fixed priors for all 942 original candidate stations, active station inventory, source provenance, checkpoint download links |
| `splits/window_index.csv.gz` | All 590,988 source/site/initialization-date window identities |
| `splits/indices.npz` | Numeric window references and train/validation indices for all seven protocols |
| `splits/counts.csv` | Verified training/validation counts |
| `figure_data/` | Numerical inputs to manuscript Figures 2–6 |
| `results/` | Publication summaries, seed-level comparisons, ablations, missingness results and external metrics |
| `predictions/` | Three external ReST prediction CSVs, 245,100 rows per seed; 31 Alaska stations |
| `provenance/` | Source identities, analysis definitions, original file hashes and numerical validation |
| `manifest.json`, `SHA256SUMS` | Inventory and checksums of every payload file |
| `DATA_DICTIONARY.md`, `schema.json` | Variable definitions, units, aggregation rules and file schemas |
| `DATA_RIGHTS.md` | Third-party attribution and component-specific terms; this is not an MIT-only dataset |

The released arrays retain every daily record for the 399 stations that contribute forecast
windows: SNOTEL 190, CanSWE 62, CDEC 130, and NorSWE 17. Inactive candidates were omitted.
Each numerical array is an exact station slice of the audited research input; dates are unchanged.
Station order is preserved within each source, but station indices are compacted. The mapping
file records that change. Metadata omits private machine paths and operational records.
The original 942-candidate checksums in `provenance/original_dataset_manifest.json` describe
the original archive, **not** this subset; use this package's `manifest.json` to verify downloads.

After extracting the ZIP and installing ReST v1.0.1, run from this directory:

```bash
rest-train --data task_datasets --priors assets/station_priors.csv --protocol temporal --seed 52 --output runs/temporal/52
rest-predict --checkpoint /path/to/rest_temporal_seed_52.pt --data task_datasets --priors assets/station_priors.csv --protocol temporal --output runs/temporal_predictions.npz
```

Download checkpoints using `assets/checkpoints.json`. Outputs must use fresh paths.
Load `splits/indices.npz` with `numpy.load(..., allow_pickle=False)`.
Arrays named `<protocol>_train` and `<protocol>_validation` index rows in `window_index.csv.gz`.
`refs` columns are source index, issue-day index and released station index, with source order
`snotel`, `canswe`, `cdec_sw1_bulk`, `norswe_region_europe_nonoverlap`.
Windows retain the original December–May initialization filter and complete ten-day labels.

## Figure and table correspondence

Numbering follows the currently supplied ReST manuscript; renumber this map if the HESS
manuscript is reorganized. Figure 1 is a conceptual architecture diagram without numerical source data.

| Item | Files | Calculation |
|---|---|---|
| Figure 2 | `figure_data/fig02_mae_reduction_percent.csv`; `results/main_comparison/protocol_means.csv` | 100 × (baseline mean MAE − ReST mean MAE) / baseline mean MAE |
| Figure 3 | `figure_data/fig03_external_lead_by_seed.csv`; `results/external_station_lead_pairs.csv` | Station-equal MAE by lead within seed; then mean/sample SD across three seeds |
| Figure 4 | `figure_data/fig04_missing_swe_history.csv`; `results/missingness_all_condition_seed_metrics.csv` | Scenario-specific NSE mean/sample SD across seeds |
| Figure 5 | `figure_data/fig05_paired_seed_differences.csv`; `figure_data/fig05_plotted_summary.csv` | Transformer-replacement minus ReST, paired within protocol and seed; mean/sample SD |
| Figure 6 | `figure_data/fig06_protocol_seed_cells.csv`; `fig06_macro_by_seed.csv` and `fig06_mean_sd.csv` in the same directory | Protocol-equal stratum means within seed, then seed mean/sample SD |
| Table 1 | `splits/counts.csv`; `station_index_map.csv` | Counts from the released task dataset |
| Table 2 and main comparison summaries | `results/main_comparison/` | Seven-protocol equal means within seed; three-seed mean/sample SD |
| Table 3 | `results/ablation_cell_metrics.csv`, `ablation_macro_by_seed.csv`, `ablation_protocol_means.csv`, `ablation_summary.csv` | Final nonnegative publication evaluation, not earlier training-log metrics |
| Table 4 | `results/missingness_all_condition_seed_metrics.csv`, `missingness_category_balanced_by_seed.csv` | Average scenarios within each of three categories, then give categories equal weight within seed |
| Supporting subgroup/threshold analyses | `results/module_diagnostic_cells.csv`, `results/threshold_sensitivity.csv` | See `provenance/module_analysis_plan.json` |
| External result checks | `predictions/`, `results/external/`, `results/external_station_pairs.csv` | Recompute station and lead errors from released observation/prediction pairs |

For Figure 3, both ReST and N-HITS plotted lead values are provided; row-level prediction CSVs
are supplied for ReST. The external ReST station-equal MAE recomputes to 14.5145572 mm,
which rounds to the reported 14.515 mm. Main/ablation publication tables reflect final
nonnegative-output evaluation. Earlier training-log values were checked and excluded.

## Reproducibility boundaries

All 590,988 window identities are preserved. Seven split counts match the manuscript, and
32 sampled full input/target records match the original dataset exactly. Figure 2 percentages,
Figure 3 ReST lead means, Figure 4 source values, Figure 5 error bars, Figure 6 aggregation,
and Table 4 category weighting were checked; see `provenance/numerical_validation.json`.

This package supports primary-dataset training/evaluation and numerical reanalysis of
the supplied publication result and figure tables. External benchmark data are provided as
archived prediction/observation pairs for metric and figure reproduction. The contents are
listed above; this is not an archive of every historical experiment artifact. Comparative
model implementations are outside the full-ReST code release.

For questions about these data or requests concerning additional research materials, email [202512070422@std.uestc.edu.cn](mailto:202512070422@std.uestc.edu.cn).

## Attribution and citation

This is a derived research package assembled from the third-party sources listed in
`DATA_RIGHTS.md` and `assets/data_provenance.json`; it does not claim ownership of the
original observations. Cite the original dataset providers and versions as well as the ReST
paper, code version and this data release. Preserve component-specific licenses, including
Hydro-Québec's non-commercial/share-alike terms. Cite the version-specific data DOI
[10.5281/zenodo.23968133](https://doi.org/10.5281/zenodo.23968133); the software release remains separately identified by its GitHub version.
