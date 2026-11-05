# ReST 研究复现数据包，版本 1.0.1

[English](data-release.md) | **简体中文**

本包用于 *Reliable Spatiotemporal Generalization for Snow Water Equivalent Forecasting* 的数据获取、结果核对与绘图复现。配套代码和 Temporal 权重见 [ReST v1.0.1](https://github.com/team-tgq/ReST/releases/tag/v1.0.1)，数据发布入口为 [data-v1.0.1](https://github.com/team-tgq/ReST/releases/tag/data-v1.0.1)。当前尚未取得本数据包的 Zenodo DOI。

## 包含什么

- 实际产生实验窗口的 399 个站点：SNOTEL 190、CanSWE 62、CDEC 130、NorSWE 17。每个来源保留七文件格式和完整逐日历史。
- 全部 590,988 个窗口的身份与七协议划分，以及固定站点先验、来源记录和原始站点索引映射。
- 主文图 2–6 的绘图数值、主比较与消融的论文冻结评估表、2,709 行缺失情形结果、误差分组和阈值敏感性数据。
- 31 个 Alaska 站点、三个随机种子共 735,300 行 ReST 预测与观测配对数据。

对比方法和消融的结果数值用于核对论文图表；其模型代码和权重不在本包中。未参与有效任务窗口的候选站点已省去，原始数值不作修改。站点索引压缩后的对应关系见 `station_index_map.csv`。原 942 站点数组的哈希记录仅用于追溯；核验本包应使用新生成的 `manifest.json` 和 `SHA256SUMS`。

## 如何使用

解压后安装 ReST v1.0.1，在数据包根目录执行：

```bash
rest-train --data task_datasets --priors assets/station_priors.csv --protocol temporal --seed 52 --output runs/temporal/52
rest-predict --checkpoint /path/to/rest_temporal_seed_52.pt --data task_datasets --priors assets/station_priors.csv --protocol temporal --output runs/temporal_predictions.npz
```

权重下载链接和哈希见 `assets/checkpoints.json`。训练输出目录及预测输出文件应使用新路径。`splits/indices.npz` 以 `allow_pickle=False` 加载；其中各协议的索引对应 `splits/window_index.csv.gz` 的行。

图表与文件的逐项对应关系见[英文说明的对照表](data-release.md#figure-and-table-correspondence)，字段、单位及统计口径见 [DATA_DICTIONARY.md](https://github.com/team-tgq/ReST/releases/download/data-v1.0.1/DATA_DICTIONARY.md)。当前编号以现有 ReST 稿件为参考，HESS 稿件如有重排，需同步调整编号。图 1 为模型概念图，没有数值源数据。

## 已完成的核验

全部窗口的来源、站点和起报日与原数据一致；七协议计数一致；抽查的 32 个完整输入/目标样本逐值一致。图 2 的改进百分比、图 3 的 ReST 逐提前期误差、图 4 的数值、图 5 的配对差及误差条、图 6 的分组汇总和表 4 的类别等权结果已复核。外部站点等权 MAE 为 14.5145572 mm，按论文精度取 14.515 mm。详情见 `provenance/numerical_validation.json`。

论文采用最终非负输出评估，部分早期训练日志的指标与之不同；本包使用与论文一致的冻结结果表。

## 范围与来源

本包支持主研究数据上的模型运行，以及所提供结果表、图表的数值复核。外部评估数据以归档的预测—观测配对形式提供，用于指标和绘图核算。具体内容以上方清单为准，本包并非全部历史实验产物的归档；对比模型实现不在完整 ReST 代码发布范围内。

如需咨询数据或申请其他研究材料，请发送邮件至 [202512070422@std.uestc.edu.cn](mailto:202512070422@std.uestc.edu.cn)。

本包基于第三方公开数据加工而来，须保留原始提供方、数据版本和各组成部分的许可。尤其 Hydro-Québec 子集带有非商业及相同方式共享条件；整个数据包不套用代码的 MIT 许可证。详细来源、许可和署名见 [DATA_RIGHTS.md](https://github.com/team-tgq/ReST/releases/download/data-v1.0.1/DATA_RIGHTS.md) 及 `assets/data_provenance.json`。Zenodo DOI 归档尚待完成，不应在投稿时把 GitHub 发布误写为已取得 DOI。
