# 数据与访问方式

[English](data.md) | **简体中文**

[概览](../README.zh-CN.md) · [方法说明](method.zh-CN.md) · [复现说明](reproducibility.zh-CN.md)

## 研究使用的数据来源

主研究使用 2010–2020 年观测，起报日期为每年 12 月 1 日至次年 5 月 30 日，并要求未来十天目标完整。本地处理归档记录了以下产品与访问入口：

| 数据 | 产品或版本 | 提供方 |
|---|---|---|
| SNOTEL SWE 与站点记录 | NRCS 站点时间序列 | [NRCS 雪与水数据](https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-water-interactive-map) |
| CanSWE | 1928–2025，v8 | [Zenodo 记录 19075529](https://doi.org/10.5281/zenodo.19075529) |
| CDEC SWE | 官方 SW1 雪水传感器记录 | [California Data Exchange Center](https://cdec.water.ca.gov/) |
| NorSWE | 1979–2021，v3.0，研究使用的非重叠子集 | [Zenodo 记录 15263370](https://doi.org/10.5281/zenodo.15263370) |
| 气象强迫 | ERA5-Land 日聚合及来源特定的站点记录 | [Earth Engine ERA5-Land 目录](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) |
| 气象补缺 | 来源特定的历史天气归档 | [Open-Meteo 历史天气 API](https://open-meteo.com/en/docs/historical-weather-api) |
| 微波亮温 | AMSR-E/AMSR2 Unified AU_SI25 v1，可用时使用 | [NSIDC AU_SI25](https://doi.org/10.5067/TRUIAL3WPAUP) |
| 地形与土地覆盖 | 处理元数据中的 Copernicus DEM GLO-30 和 ESA WorldCover v200（2021） | [DEM 目录](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30)、[WorldCover 目录](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) |

提供方产品是处理流程的输入，不能直接当作最终任务数据集的等价副本。各来源的强迫数据出处和单位转换记录保存在 `assets/data_provenance.json`。例如，短波辐射存储为 J m⁻² day⁻¹：站点 W m⁻² 通量乘以 86,400，Open-Meteo 的 MJ m⁻² day⁻¹ 乘以 10⁶。降水单位为 mm，温度为 °C，风速为 m s⁻¹，相对湿度为百分比，比湿为 kg kg⁻¹。

## 七文件数据集格式

每个来源放在 `data/task_datasets/SOURCE/` 下：

| 文件 | 格式 |
|---|---|
| `dates.npy` | 唯一、递增、逐日连续的日期，`[time]`，固定宽度字符串或 NumPy 日期 |
| `sites.csv` | 唯一的 `site_id`，以及 `elevation`、`latitude`/`lat`、`longitude`/`lon`、`landcover_code`、`southness` |
| `inputs.npy` | 原始物理特征，float32 `[time, site, feature]` |
| `feature_mask.npy` | 观测有效性，与 inputs 同形状 |
| `targets.npy` | SWE，单位 mm，float32 `[time, site, 1]` |
| `target_mask.npy` | 目标有效性，与 targets 同形状 |
| `meta.json` | 有序的 `feature_names` 和 `target_unit: "mm"` |

动态特征的顺序为：

```text
swe_prev, swe_prev_observed, day_sin, day_cos, tmax, tmin,
precipitation, shortwave_radiation, wind_velocity,
relative_humidity_max, relative_humidity_min, specific_humidity,
tb_19ve, tb_37ve, tb_diff
```

`inputs.npy` 中额外的静态列会被忽略，站点属性从 `sites.csv` 读取。不要混入未来 SWE，不要通过插值补齐缺失目标，也不要在验证站点或窗口上拟合预处理统计。掩码 1 表示有效观测，0 表示缺失。加载器会拒绝被标为有效的非有限值，以及非逐日连续的日期轴。

完整处理后源数组包含 942 个候选站点。应用论文的完整目标与雪季筛选后，保留 399 个站点和 590,988 个窗口。`assets/stations.csv` 标识这 399 个有效站点；`assets/dataset_manifest.json` 记录本地核验过的研究输入的哈希与形状。仅依靠提供方链接，无法重建原有的预处理和去重决策。

## 离线先验

`assets/station_priors.csv` 包含全部 942 个候选站点的五维模型先验，以 `source,site_id` 为联合键。最后一维保留历史列名 `prior_sparse_source`，含义是**静态证据不足**，并非对观测缺失程度的估计。归档提示词位于 `assets/static_prior_prompt.json`，输入证据仅限纬度、经度、海拔和 southness；来源与站点标识只用于行对齐。

新站点需要提供对应的离线先验行。加载器遇到缺失行会报错，不会自动替换为未经记录的先验。归档数值表是可复现的输入；重新生成具有随机性的语言模型响应，不保证得到相同数值。记录中的模型标识与表格哈希见 `assets/priors_provenance.json`。

## 发布状态

本次发布提供代码、固定先验、站点身份、数据来源记录、输入文件校验和，以及三个 Temporal 推理 checkpoint。大型处理后数组、Alaska 评估包、完整的逐图源数据表和其他协议的训练权重尚未在此公开归档。

当前不宣称存在处理后数据 DOI。只有完成独立、可核验的归档后，才能在论文声明中将这些材料描述为公开可得。MIT 许可证不替代第三方数据的使用条款。
