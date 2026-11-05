# ReST

[English](README.md) | **简体中文**

**Reliable Spatiotemporal Generalization for Snow Water Equivalent Forecasting**

**面向雪水当量预测的可靠时空泛化**

本仓库提供 ReST 论文配套的 PyTorch 实现。ReST 利用多尺度历史观测、有效性掩码、站点属性和离线过程先验，预测未来 1–10 天的雪水当量（SWE，单位 mm）。

公开内容包括完整 ReST 模型、七种评估划分、训练与评估命令、固定站点先验，以及三个预训练 Temporal 模型。论文尚未获得正式出版 DOI。

## 模型

```text
Short 历史（30 天）────────┐
Long 历史（30 个采样位置）──┼─ MPE ─ AMRM 上下文适配 ─ SDEF ─ SWE（10 天）
Analogue 历史（75 天）─────┘          │                 ↑
站点属性 + 离线先验 ─────────────────┘     条件锚点 ────┘
```

- **MPE（多尺度过程编码器）**：先保留各变量的时间结构，再跨三个历史尺度组织过程组证据。
- **AMRM（自适应缺失可靠性模块）**：表示观测覆盖率、观测年龄和缺口，适配上下文，并在当前 SWE 缺失时估计共享状态锚点。
- **SDEF（尺度一致的分解式专家预测）**：通过 Top-2 路由组合五个趋势/残差专家，并利用带掩码的历史统计恢复 SWE 量纲。

最终推理结果非负。训练使用未截断的预测值，使负值区域仍保留梯度。训练和推理过程中不调用语言模型。注意力计算不会混合不同预测样本。

## 安装与运行

需要 Python 3.10+ 和 PyTorch 2.2+。先安装适合本机 CPU 或 GPU 的 PyTorch 版本，然后执行：

```bash
git clone https://github.com/team-tgq/ReST.git
cd ReST
python -m pip install -e ".[dev]"
rest-demo
pytest -q
```

演示程序使用合成数据检查前向与反向传播，不用于复现论文报告的预测精度。

## 训练

从 [Zenodo 数据归档](https://doi.org/10.5281/zenodo.23968133)（DOI：10.5281/zenodo.23968133）或 [GitHub 镜像](https://github.com/team-tgq/ReST/releases/tag/data-v1.0.1)下载处理后实验数据、结果表及绘图源数据。数据包包含 399 个有效站点、590,988 个窗口及七种划分。使用方法见[数据说明](docs/data.zh-CN.md)，内容与限制见[数据发布详解](docs/data-release.zh-CN.md)。解压后，将下方命令的 `--data` 指向包内 `task_datasets` 目录。

```bash
rest-train --data data/task_datasets --priors assets/station_priors.csv --protocol temporal --seed 52 --output runs/temporal/52
```

默认设置与论文一致：FP32、batch size 1,024、AdamW、学习率和权重衰减均为 `1e-4`，最多训练 30 个 epoch，使用验证损失平台期学习率调度与提前停止。最佳 checkpoint 按验证 NSE 最大值选取；学习率调度和提前停止监测验证损失。每次运行使用新的输出目录。

| 论文数据划分 | `--protocol` | 训练窗口 | 验证窗口 |
|---|---|---:|---:|
| Temporal | `temporal` | 468,188 | 122,800 |
| CS-SNOTEL | `cs_snotel` | 277,108 | 313,880 |
| CS-CanSWE | `cs_canswe` | 541,742 | 49,246 |
| CS-CDEC | `cs_cdec` | 386,660 | 204,328 |
| CS-NorSWE | `cs_norswe` | 567,454 | 23,534 |
| CE-High | `ce_high` | 345,975 | 245,013 |
| CE-Lower | `ce_lower` | 245,013 | 345,975 |

各划分使用随机种子 `52`、`101` 和 `211`。本仓库仅公开完整 ReST，不提供外部基线或模块消融实现。论文与代码的对应关系见[方法说明](docs/method.zh-CN.md)。

## 评估与迁移

从 [v1.0.1 发布页](https://github.com/team-tgq/ReST/releases/tag/v1.0.1)下载三个 Temporal checkpoint。每个文件约 17 MB，包含训练集拟合的归一化参数。文件哈希及原始 checkpoint 哈希见 [assets/checkpoints.json](assets/checkpoints.json)。

```bash
rest-predict --checkpoint rest_temporal_seed_52.pt --data data/task_datasets --priors assets/station_priors.csv --protocol temporal --output runs/temporal_predictions.npz
```

输出包含预测、目标和 `[source_index, issue_day, site_index]` 索引。打印的指标将所有预测窗口与预测步合并计算；R² 定义为 Pearson 相关系数的平方，与研究实现一致。

该命令用于评估有标签的历史窗口，需要完整的未来十天目标。没有未来标签时，应构造历史输入张量，直接使用 `load_checkpoint`、其返回的 `Normalizer.inputs` 和模型进行预测；这些 API 不需要目标标签。

迁移到新站点时，需按相同格式提供逐日历史、静态属性和离线先验。使用 `--protocol all --sources YOUR_SOURCE`，并通过 `--start`/`--end` 指定覆盖的数据区间。`--issue-start` 限制参与评估的起报日期，同时保留更早的历史记录。评估时不拟合模型参数或归一化统计。对于 Alaska 31 站点评估，数据发布包提供该实验归档的预测—观测配对及绘图结果。命令行的合并指标不等同于该实验按站点加权聚合的结果。

## 仓库导览

| 文件 | 用途 |
|---|---|
| `src/rest/model.py` | ReST 总装与输出契约 |
| `src/rest/mpe.py`、`process.py` | 多尺度过程编码器 |
| `src/rest/amrm.py`、`reliability.py` | AMRM、锚点和训练增强 |
| `src/rest/sdef.py` | 标准化、路由与分解式预测 |
| `src/rest/data.py` | 历史窗口、研究划分与归一化 |
| `src/rest/train.py`、`predict.py` | 训练与 checkpoint 评估 |
| `assets/` | 先验、提示词、站点清单、来源记录与 checkpoint 哈希 |
| `tests/` | 模型、缺失、因果性、数据与 checkpoint 检查 |

已核验的等价性检查和发布范围详见[复现说明](docs/reproducibility.zh-CN.md)。本仓库不包含历史候选模型、多机调度、探索性输出或论文稿件。

## 引用与许可证

请引用配套论文标题，以及本仓库的版本或 commit。[CITATION.cff](CITATION.cff) 标识当前软件版本；论文作者元数据与 DOI 将在确定后补充。

代码和作者制作的发布资产采用 [MIT 许可证](LICENSE)。原始观测与再分析数据仍遵循各提供方的使用条款。

使用实验数据或归档结果时，请同时引用[研究数据归档](https://doi.org/10.5281/zenodo.23968133)。

## 联系方式

如需咨询数据或申请其他研究材料，请发送邮件至 [202512070422@std.uestc.edu.cn](mailto:202512070422@std.uestc.edu.cn)。
