# 历史规范：配对趋势优先（2026-09-03晚停止作为默认流程）

本版原文保留用于复现；当前入口见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

更新：2026-09-03。新方法默认采用 `paired_trend_review_v1`。
目标是可靠判断“比固定基线上升还是下降”，不是拟合官方隐藏集绝对分数。
旧版规范完整保存在 [历史规范](EXPERIMENT_PROTOCOL_PRE_TREND_20260903.md)，只供复现。

**2026-09-03实跑校验警示：** 固定基线/A/B已完成，但预声明S1024→P40比较在开发为正、
确认反向，未通过已知官方趋势核验。当前规则和原始结果保留；A可以继续作为同条件实验
证据，不能单独作为正式提交依据。详见[实跑闭环](../reports/experiments/PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)。

## 1. 唯一计分口径

[configs/project.yaml](../configs/project.yaml) 中
`metric_protocol=platform_observed_20260831` 为当前依据。

- 同细类匹配；船/飞机IoU=0.50，车辆=0.35；重复框算FP。
- 粗类R/FDR分别为内部细类等权平均，三粗类均值用于硬门。
- 三个Recall子分、三个FDR子分、一个时延子分，合计除以7。
- pooled和25细类总体macro仅作附表，不能选候选。
- 不再使用“7项排名二次排序”或“pooled硬门”作为新实验决策。
- 没有可比时延时只报六个质量子分之和/7的差，不填虚构总分。

接口仍为 `contract_v1`，匹配版本仍为 `official_eval_v1`。
旧 `ranking_version` 标签保留用于兼容历史taxonomy聚合代码，不代表现在按名次计分。

## 2. 默认只保留两套测试

### A：固定配对验证

一次性冻结来源组隔离的 train / development / confirmation，之后复用。
基线与新方法在相同训练来源、训练成熟度和推理流程下比较，只变声明的方法因素。
主模型和所有初始化/外部预训练资产均须审计，不得含development/confirmation来源。

- development：日常筛选、按相同规则选择每个模型自己的工作点；
- confirmation：只对有希望的候选运行，使用该模型在development冻结的工作点；
- 二者是同一个验证协议的两部分，不再自动增加第三折、第四套主榜。
- 校准必须来自被评模型自身，不再借另一个fold模型的置信度阈值。
- confirmation不得参与本候选的参数/阈值选择；已被反复分析的旧Sentinel不称全新盲集。

固定划分已经物化：[paired_trend_v1](../data/splits/paired_trend_v1/contract.json)。
训练3,136图/190组，开发673图/31组，确认672图/34组，三部分均覆盖25类。
GT、图像和标签SHA已冻结，来源组及完全相同图像无跨区；历史CV3不改。
训练区限定的成熟基线已完成并缓存，不能以全量已见权重代替或重复初始化。
可执行入口、当前状态和限制见[落地说明](PAIRED_TREND_QUICKSTART.md)。

### B：部署回归

固定少量大图、稀疏/密集样本和已审核真负图；检查召回损失、背景FP/100MP、输出一致性、
速度和内存。含训练来源时明确为工程回归，不称隐藏集分数预测。
复用当前safe切片、382张/100.139008MP审核背景和入口逐框检查。
固定六张训练来源的稀疏/密集原图，原像素、不缩放不截断地放入两张100MP画布，
改变摆放偏移检查切片接缝。它们只作工程回归，黑色留白不是对真实背景分布的模拟。
没有改动的环境/镜像检查按SHA复用；Linux入口对照不冒充Docker GPU测试。

不再让Normal、Hard、Sentinel、MacroRisk、多个FDR前沿分别成为互相矛盾的必过榜单。
旧CV3仅用于历史研究或确有必要的一次额外确认，不是每个新方法的默认前置步骤。

## 3. 一个目标、一个工作点、一张表

开发区主工作点按官方七项综合分选择；时延与阈值无关时，等价于最大化六项质量子分和。
默认全局阈值，保留底阈值0.001的预测；沿用现有0.005开发网格，无需对每个想法强制0.001扫描。
确认区只应用选定工作点，不重新扫描。复杂细类校准作为被测方法本身，而不是所有实验的必选项。

表格固定包含：基线/候选总分、Δ总分、三类R/FDR和对应得分贡献、时延、GT/负图覆盖、
训练/划分/代码/预测SHA以及是否同源。主判断为分差；单类变化用于解释风险。
相同数值阈值的对照可辅助识别校准变化，但不同模型分数尺度不同，不能把它当成唯一公平标准。

- 开发正向超过0.5分：值得进入确认；小于−0.5分：通常停止。
- ±0.5分内：收益小或不确定，不自动延长训练。
- 确认同方向且没有部署回归故障：进入候选讨论；与开发方向相反则暂不占正式机会。
- 0.5只是预先固定的工程筛选带，不是统计显著性或官方得分保证。
- 不再硬性要求所有类的Recall/FDR同时不下降；合理取舍由总分和过线风险共同解释。
- CV开发分数低于85%Recall不等于正式候选必然失败，不能拿绝对硬门冻结所有研究。
- 当前正式v2.0仍是76.6010、三门通过的回退资产；新的正向报告不自动授权full训练或提交。

## 4. 代码入口与缓存

**日常唯一入口是 `scripts/run_paired_trend.py`**，依次负责准备、基线训练/缓存、
候选开发校准、正向才确认，并绑定checkpoint、训练来源、预测与GT的SHA。
**交付回归入口是 `scripts/run_paired_deployment_regression.py`**。
完整命令见[快速使用](PAIRED_TREND_QUICKSTART.md)。

下列底层工具仍可用于历史结果回放，不代替主入口的血缘检查：

```bash
PYTHONPATH=src python scripts/evaluate_fixed_score_threshold.py \
  --gt CHECK_GT.json --pred CANDIDATE_LOW.json --threshold FROZEN_THRESHOLD \
  --latency-seconds MEASURED_SECONDS --output candidate_metrics.json
PYTHONPATH=src python scripts/compare_candidate_trend.py \
  --baseline baseline_metrics.json --candidate candidate_metrics.json \
  --output comparison.json
```

这些是参数占位示例。基线只评估一次并缓存；同一预测可重放后处理，无需再次GPU推理。
同源诊断使用 `--diagnostic-only`。无时延时省略相应参数，仅比较质量贡献。
GT必须含完整images清单（包括无框图），预测不能包含清单之外的ID。
比较器校验GT SHA及当前协议，重算得分、不相信外部总分字段；不代替训练血缘或计时环境审计。

## 5. 产物与记录

每个候选保留一份小型 `comparison.json` 和一份结论记录；权重、低阈值预测、完整日志留在
`outputs/<experiment_id>/`。既有[实验总表](../reports/experiments/leaderboard.csv)的旧列
不改义，新比较产物通过artifact_ref/notes链接；历史失败不反写为成功。

只有排名靠前或方向矛盾时，按结果打开细类/大小/背景错误分析。不再每次生成全套长报告。
对正式提交，提前保存预测的“总分方向和逐类风险”，回传后逐条核对；积累真实趋势命中率，
不以事后更换测试集证明自己原来判断正确。
