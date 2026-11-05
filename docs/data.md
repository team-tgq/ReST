# Data and access

**English** | [简体中文](data.zh-CN.md)

[Overview](../README.md) · [Method](method.md) · [Reproducibility](reproducibility.md)

## Sources used in the study

The primary study uses 2010–2020 observations, with forecast initialization dates
from December 1 to May 30 and complete ten-day target windows. The local processing
archive identifies the following products and access routes:

| Data | Product or version | Provider |
|---|---|---|
| SNOTEL SWE and station records | NRCS station time series | [NRCS snow and water data](https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-water-interactive-map) |
| CanSWE | 1928–2025, v8 | [Zenodo record 19075529](https://doi.org/10.5281/zenodo.19075529) |
| CDEC SWE | Official SW1 snow-water sensor records | [California Data Exchange Center](https://cdec.water.ca.gov/) |
| NorSWE | 1979–2021, v3.0, study non-overlap subset | [Zenodo record 15263370](https://doi.org/10.5281/zenodo.15263370) |
| Meteorological forcing | ERA5-Land daily aggregates and source-specific station records | [Earth Engine ERA5-Land catalog](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) |
| Meteorological gap filling | Source-specific historical weather archive | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) |
| Microwave brightness temperatures | AMSR-E/AMSR2 Unified AU_SI25 v1, where available | [NSIDC AU_SI25](https://doi.org/10.5067/TRUIAL3WPAUP) |
| Topographic and land-cover context | Copernicus DEM GLO-30 and ESA WorldCover v200 (2021) in the processing metadata | [DEM catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30), [WorldCover catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) |

Provider products are inputs to the processing archive, not interchangeable
copies of the final task dataset. Source-specific forcing provenance and unit
conversion notes are retained in `assets/data_provenance.json`. For example,
shortwave radiation is stored as J m⁻² day⁻¹; station flux in W m⁻² is multiplied
by 86,400 and Open-Meteo MJ m⁻² day⁻¹ by 10⁶. Precipitation is in mm, temperatures
in °C, wind in m s⁻¹, relative humidity in percent, and specific humidity in kg kg⁻¹.

## Seven-file dataset format

Place each source under `data/task_datasets/SOURCE/`:

| File | Format |
|---|---|
| `dates.npy` | Unique ascending daily dates, `[time]`, fixed-width strings or NumPy dates |
| `sites.csv` | Unique `site_id`, `elevation`, `latitude`/`lat`, `longitude`/`lon`, `landcover_code`, `southness` |
| `inputs.npy` | Physical features, float32 `[time, site, feature]` |
| `feature_mask.npy` | Observation validity, same shape as inputs |
| `targets.npy` | SWE in mm, float32 `[time, site, 1]` |
| `target_mask.npy` | Target validity, same shape as targets |
| `meta.json` | Ordered `feature_names` and `target_unit: "mm"` |

The ordered dynamic feature names are:

```text
swe_prev, swe_prev_observed, day_sin, day_cos, tmax, tmin,
precipitation, shortwave_radiation, wind_velocity,
relative_humidity_max, relative_humidity_min, specific_humidity,
tb_19ve, tb_37ve, tb_diff
```

Additional static columns in `inputs.npy` are ignored; station attributes come
from `sites.csv`. Do not insert future SWE, fill missing targets by interpolation,
or fit preprocessing statistics on validation stations/windows. Masks are 1 for
observed values and 0 for missing values. The loader rejects active non-finite
values and non-daily date axes.

The full processed source arrays contain 942 candidate stations. Applying the
paper's complete-target and season filters leaves 399 stations and 590,988
windows. `assets/stations.csv` identifies these 399 active stations;
`assets/dataset_manifest.json` records the hashes and shapes of the locally
verified study inputs. The original preprocessing and deduplication decisions
cannot be reconstructed from provider links alone.

## Offline priors

`assets/station_priors.csv` contains the five model inputs for all 942 candidate
stations, keyed by `source,site_id`. The final component retains its historical
column name `prior_sparse_source` but means **insufficient static evidence**;
it is not an estimate of observed missingness. The archived prompt is in
`assets/static_prior_prompt.json`. Its evidence is limited to latitude,
longitude, elevation, and southness; source/site identifiers only align rows.

New stations require their own offline prior rows. The loader rejects missing
rows rather than substituting an undocumented prior. The archived table is the
reproducible input; regenerating stochastic language-model responses need not
produce identical numbers. The recorded model identifier and table hashes are
in `assets/priors_provenance.json`.

## Release status

The [data-v1.0.1 release](https://github.com/team-tgq/ReST/releases/tag/data-v1.0.1) provides the processed arrays for the 399
active stations, all 590,988 window identities, seven splits, fixed priors,
publication result tables, numerical source data for Figures 2–6, and external
ReST prediction/observation pairs for three seeds. See the
[data release guide](data-release.md) for the file-to-figure map and verified checks.
Figure numbering follows the supplied manuscript and may change in the HESS draft.

Download `ReST_reproduction_data_v1_0_1.zip` from that page and extract it. Its size is
73,251,127 bytes and its SHA-256 is:

```text
881802779daf19fa5b93e6fbbb367f011fad9a150ab4f6e6343785339ef49886
```

From the extracted `ReST_reproduction_data_v1_0_1` directory, after installing ReST:

```bash
rest-train --data task_datasets --priors assets/station_priors.csv --protocol temporal --seed 52 --output runs/temporal/52
```

The release removes inactive candidate stations while preserving all daily records
for active stations. `station_index_map.csv` records compacted indices. Repository
`assets/dataset_manifest.json` describes the original 942-candidate archive;
use the downloaded package's `manifest.json` and `SHA256SUMS` for this subset.
Machine-readable download metadata is in [assets/data_release.json](../assets/data_release.json).

The archive covers the primary task datasets and the published numerical results;
external benchmark data are supplied as prediction/observation pairs. Comparative
result values are provided, while baseline and ablation implementations remain
outside this repository.

For questions about these data or requests concerning additional research materials, email [202512070422@std.uestc.edu.cn](mailto:202512070422@std.uestc.edu.cn).
The data are archived on [Zenodo](https://doi.org/10.5281/zenodo.23968133), DOI **10.5281/zenodo.23968133**.

Read the package's `DATA_RIGHTS.md` before reuse. The derived dataset retains
component-specific provider terms, including Hydro-Québec non-commercial/share-alike
conditions; the code's MIT license does not replace these terms.
