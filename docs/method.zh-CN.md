# 论文与代码实现的对应关系

[English](method.md) | **简体中文**

[概览](../README.zh-CN.md) · [数据说明](data.zh-CN.md) · [复现说明](reproducibility.zh-CN.md)

| 论文内容 | 实现位置 |
|---|---|
| §2.1：30/30/75 个历史位置 | `data.history_indices` |
| §2.2：变量分块与过程组 | `process.PhysicsGuidedHydroProcessEncoder`、`mpe.MultiScaleProcessEncoder` |
| §2.3：可靠性状态与上下文适配 | `amrm.AdaptiveMissingnessReliabilityModule` |
| §2.3：尾部缺口修正 | `reliability.GapAwareSWEStateAnchor` |
| Short SWE 缺失时的跨变量锚点 | `reliability.SWEFreeCrossVariableAnchor` |
| §2.4：历史标准化与输出尺度恢复 | `sdef._masked_standardize`、`ScaleConsistentDecomposedExpertForecasting` |
| §2.4：五个专家的 Top-2 组合 | `sdef._ProcessRouter`、`_ProcessExperts` |
| §2.5：掩码 MAE + 0.01 均衡损失 + 0.1 锚点损失 | `train.run_epoch`、`model.ReST.forward` |

模型采用 128 维 token、4 个注意力头、1 层变量内时间编码、0.1 dropout，以及 192 维专家隐藏层。完整模型具有 4,209,294 个参数。动态通道遵循 `config.py` 中的 `DEFAULT_FEATURE_NAMES`；改变通道顺序会改变方法含义。

## 输入与输出契约

每个 `ObservedHistory` 包含三个对齐的 `[batch, positions, 15]` 张量：`normalized`、`raw` 和 `observed`。原始物理值必须保留，用于锚点计算和样本级标准化。缺失值的掩码为零；观测到的零 SWE 是有效的无雪记录。五个静态属性依次为海拔、纬度、经度、土地覆盖代码和 southness。离线先验包含五个分量，顺序见 `data.PRIOR_NAMES`。

Short 覆盖 `t-29` 至 `t`；Long 采样位置为 `t-210, t-203, ..., t-7`。Analogue 按前第 5 个至前第 1 个水文年排列，围绕相同水文年日序取 −7 至 +7 的偏移，水文年起点为 10 月 1 日。在闰日附近，这种构造不同于直接替换公历年份。缺失的早期年份以零掩码填充。

历史 z-score 仅使用有效观测，按样本、尺度和变量独立计算。有效值少于两个或尺度退化时，使用均值 0、尺度 1。数据集级 min-max 统计则单独在训练窗口可见的坐标上拟合。

起报日 SWE 可用时，以该观测作为锚点；出现尾部缺口时，以最后有效 SWE 加有界修正作为锚点。Short 中没有 SWE 时，跨变量锚点可以使用其余非雪过程 token、静态属性和先验。学习型标准化基础状态提供最终兜底。趋势与残差可以为负，只有最终推理 SWE 才截断到非负范围。

训练增强仅同步修改 Short 的值和掩码：SWE 尾部缺失概率为 0.30，最多遮蔽 7 个位置；外生变量块缺失概率为每变量 0.15，块长度为 5 个位置。目标、Long 和 Analogue 保持原样，调用 `eval()` 后关闭增强。

## 公开实现范围

仅发布完整 ReST。外部基线、模块消融、替代编码器和替代预测头均不包含在内。公开训练入口不提供模型变体选择参数。

推理与训练计算图保留投稿实现的行为，公开名称统一为 ReST、MPE 和 SDEF。在经过核验的权重导出过程中，历史 checkpoint 键前缀 `mpcde` 和 `sc_dpef` 分别改名为 `mpe` 和 `sdef`；导出后的 checkpoint 可由本包直接加载。
