# ReST

**English** | [简体中文](README.zh-CN.md)

**Reliable Spatiotemporal Generalization for Snow Water Equivalent Forecasting**

PyTorch implementation accompanying the ReST manuscript. ReST predicts the next
1–10 days of snow water equivalent (SWE, mm) from multiscale historical observations,
validity masks, station attributes, and an offline process-informed prior.

The public package contains the complete ReST model,
the seven evaluation splits, training and evaluation commands, fixed station
priors, and three pretrained Temporal models. The manuscript is not yet assigned
a publication DOI.

## Model

```text
Short history (30 days) ───┐
Long history (30 samples) ─┼─ MPE ─ AMRM context adaptation ─ SDEF ─ SWE (10 days)
Analogue history (75 days) ┘         │                         ↑
Station attributes + offline prior ─┘     Conditional anchor ──┘
```

- **MPE** preserves each variable's temporal structure before organizing evidence
  into process groups across three historical scales.
- **AMRM** represents observation coverage, age, and gaps; adapts the context;
  and estimates a shared state anchor when current SWE is missing.
- **SDEF** combines five trend/residual experts with Top-2 routing and restores
  SWE magnitude using masked historical statistics.

The final inference forecast is nonnegative. Training uses raw forecasts so
negative values retain gradients. No language model is called during training
or inference. Attention does not mix different forecast samples.

## Install and run

Python 3.10+ and PyTorch 2.2+ are required. Install a PyTorch build appropriate for
your CPU or GPU, then:

```bash
git clone https://github.com/team-tgq/ReST.git
cd ReST
python -m pip install -e ".[dev]"
rest-demo
pytest -q
```

The demo uses synthetic data and checks forward/backward execution. It does not
reproduce the reported forecasting accuracy.

## Train

Download the processed study arrays and result/figure data from the separate
[data-v1.0.1 release](https://github.com/team-tgq/ReST/releases/tag/data-v1.0.1). The archive contains 399 active stations,
590,988 windows and seven split definitions. See [Data](docs/data.md) for setup
and [the data release guide](docs/data-release.md) for contents and limitations.
After extracting the archive, point `--data` to its `task_datasets` directory.

```bash
rest-train --data data/task_datasets --priors assets/station_priors.csv --protocol temporal --seed 52 --output runs/temporal/52
```

The defaults follow the manuscript: FP32, batch size 1,024, AdamW with learning
rate and weight decay of `1e-4`, at most 30 epochs, a plateau learning-rate
schedule, and early stopping. Best checkpoints maximize validation NSE; the
scheduler and early stopping monitor validation loss. Use a fresh output
directory for each run.

| Paper dataset | `--protocol` | Training windows | Validation windows |
|---|---|---:|---:|
| Temporal | `temporal` | 468,188 | 122,800 |
| CS-SNOTEL | `cs_snotel` | 277,108 | 313,880 |
| CS-CanSWE | `cs_canswe` | 541,742 | 49,246 |
| CS-CDEC | `cs_cdec` | 386,660 | 204,328 |
| CS-NorSWE | `cs_norswe` | 567,454 | 23,534 |
| CE-High | `ce_high` | 345,975 | 245,013 |
| CE-Lower | `ce_lower` | 245,013 | 345,975 |

Run each split with seeds `52`, `101`, and `211`. This repository publishes only
the complete ReST model. External baselines and module-ablation implementations
are not distributed. See [Method](docs/method.md) for the manuscript-to-code mapping.

## Evaluate and transfer

Download the three Temporal checkpoints from the [v1.0.1 release](https://github.com/team-tgq/ReST/releases/tag/v1.0.1).
Each file is approximately 17 MB and includes its training-fitted normalization.
File hashes and original checkpoint hashes are in [assets/checkpoints.json](assets/checkpoints.json).

```bash
rest-predict --checkpoint rest_temporal_seed_52.pt --data data/task_datasets --priors assets/station_priors.csv --protocol temporal --output runs/temporal_predictions.npz
```

The output contains predictions, targets, and `[source_index, issue_day, site_index]`
references. Printed metrics are pooled over forecast windows and lead times;
R² is squared Pearson correlation, matching the study implementation.

This command evaluates labeled historical windows and requires complete ten-day
targets. For forecasting without future labels, construct historical input
tensors and use `load_checkpoint`, its returned `Normalizer.inputs`, and the
model directly; these APIs do not require targets.

For new stations, provide their daily history, static attributes, and offline
priors in the same schema. Use `--protocol all --sources YOUR_SOURCE` and set
`--start`/`--end` to the covered data period. `--issue-start` restricts evaluated
initialization dates while retaining earlier records as history. No parameters
or normalization statistics are fitted during evaluation. For the 31-station Alaska
benchmark, the data release provides archived prediction/observation pairs and
plotted results;
pooled command-line metrics are not its reported station-weighted aggregates.

## Repository guide

| File | Purpose |
|---|---|
| `src/rest/model.py` | ReST assembly and output contract |
| `src/rest/mpe.py`, `process.py` | Multi-scale Process Encoder |
| `src/rest/amrm.py`, `reliability.py` | AMRM, anchors, training augmentation |
| `src/rest/sdef.py` | Standardization, routing, decomposed forecasts |
| `src/rest/data.py` | Historical windows, study splits, normalization |
| `src/rest/train.py`, `predict.py` | Training and checkpoint evaluation |
| `assets/` | Priors, prompt, station inventory, provenance, checkpoint hashes |
| `tests/` | Model, missingness, causality, data, and checkpoint checks |

See [reproducibility notes](docs/reproducibility.md) for the verified equivalence
checks and the precise scope of this release. Historical candidate models,
multi-machine dispatch, exploratory outputs, and manuscript files are excluded.

## Citation and license

Please cite the accompanying manuscript by its title and this repository's
version/commit. [CITATION.cff](CITATION.cff) identifies this software release;
paper author metadata and DOI can be added when finalized.

Code and author-created release assets are distributed under the [MIT license](LICENSE).
Original observational and reanalysis datasets retain their providers' terms.

## Contact

For questions about these data or requests concerning additional research materials, email [202512070422@std.uestc.edu.cn](mailto:202512070422@std.uestc.edu.cn).
