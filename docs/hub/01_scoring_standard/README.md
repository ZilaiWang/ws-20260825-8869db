# 官方评分方案 V1.6 速查与团队工具

更新日期：2026-08-10
状态：`current`

> 本文档把官方《基于不均衡小样本学习的光学遥感卫星陆上目标检测识别比赛评分方案
> V1.6》翻译成团队可直接执行的口径，并说明配套 CLI 工具的使用方法。所有成员提交
> 正式结果前，必须用本文档的入口自检，禁止自行解释规则。

## 1. 官方规则要点（原文摘要）

### 1.1 初赛刚性门槛（不过则淘汰）

| 门槛 | 数值 | 说明 |
|---|---|---|
| 整体检测召回率 | ≥ 85% | 三类目标**合并**计算（不分大类） |
| 虚警率（FDR） | ≤ 20% | 同上，`FDR = FP / (FP + TP)` |
| 单幅 10000×10000 推理 | ≤ 20 秒 | 单张 NVIDIA RTX3090 或同等算力国产 GPU/NPU；**不含数据读取** |

三项全部通过才能获得性能基础分 70 分，否则初赛不予通过。

### 1.2 匹配规则（Recall/FDR 怎么数）

1. 预测按 score **从高到低**排序后依次匹配；
2. 每个预测框最多匹配一个真实框，每个真实框最多被一个预测框匹配；
3. **细类必须一致**才能匹配（预测 `category_id` 与 GT `category_id` 相同）；
4. 多个预测框匹配同一 GT 时，置信度最高的计为 TP，其余计为 FP；
5. 未匹配到 GT 的预测框计 FP，未被匹配的 GT 计 FN；
6. IoU 匹配阈值：**车辆 0.35**，舰船/飞机 0.50。

> ⚠️ 匹配**在细类内进行**，三大类映射只用于选择 IoU 阈值和汇总指标，
> **不能**在匹配前把细类合并。模型输出 25 个细类，一个都不能丢。

### 1.3 V1.6 排名口径（决定方案/创新/落地打分）

- 三大类各自的 Recall/FDR = **大类内细类指标的简单平均**：
  - 船 = 4 型（驱护舰、航母、两栖船、民船）各 1/4 的平均；
  - 飞机 = 20 型各 1/20 的平均；
  - 车辆 = 1 型（FSC）本身。
- 每队共 **7 项排名**：船 Recall、船 FDR、飞机 Recall、飞机 FDR、车辆
  Recall、车辆 FDR、总时效性。
- 正式排名聚合固定使用完整 25 细类税表（4/20/1）。局部 fold、小样本
  子集若缺类，只能用 `--allow-partial-taxonomy` 作诊断，不得冒充官方排名值。
- 对 7 个排名求和后二次排序（和越小越靠前），二次排序的从前往后百分比
  决定方案合理性（10 分）、技术创新程度（10 分）、工程可落地性（10 分）
  的打分区间：`(100% - p) ± 20%`，每项下限 0、上限 10。
  - 例：二次排序第 30% → 三项各 5-9 分。

> 💡 关键含义：**先过刚性门槛（pooled Recall≥0.85 / FDR≤0.20 / ≤20s），再优化
> 7 项排名**。即使刚过线，若 7 项排名垫底，方案/创新/落地三项分数也会被压到
> 很低。**舰船 macro FDR 0.52 是当前最大官方排名风险。**

排名脚本的保守规则：7 项任一缺失的队伍标记为 incomplete，不参与任一单项
排名和二次排名；不允许用“只报 6 项”获得虚假排名优势。官方未公布完整的并列名次
和百分位细则，团队模拟统一使用竞赛名次与 `position / complete_team_count`，并在
产物中明示为团队推定口径。

## 2. 团队配套工具

### 2.1 单模型评估（`scripts/evaluate.py`）

每个模型的正式结果用同一入口评估，同时输出 pooled（门槛）与官方 macro（排名）
两种口径，并支持时效门槛判定：

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --gt outputs/gt_cv3_v2.json \
  --pred outputs/实验ID/predictions.json \
  --project-config configs/project.yaml \
  --latency-seconds 6.2 \
  --output outputs/实验ID/result.json
```

输出 JSON 关键块：

| 字段 | 含义 |
|---|---|
| `detection_gate` | 刚性门槛判定（Recall≥0.85 / FDR≤0.20） |
| `timing_gate` | 时效门槛判定（≤20s，需传 `--latency-seconds`） |
| `official_ranking.seven_ranking_metrics_v1_6` | 7 项排名的原始指标值 |
| `official_ranking.per_coarse` | 三大类 macro 与 pooled 对照 |
| `official_ranking.per_fine` | 25 细类明细 |

> `--latency-seconds` 缺省时 `timing_gate` 不出现、`seven_ranking_metrics` 中
> `latency_seconds` 为 `null`。
>
> 正式评估默认要求 GT 覆盖完整 25 细类。`--allow-partial-taxonomy` 只用于
> fold/子集诊断，输出的 macro 不可填入正式 leaderboard。

### 2.2 多队排名模拟（`scripts/rank_official.py`）

把多支队伍（M1 / M3 / 加了各种模块的变体）的 `result.json` 放一起算 7 项排名
与二次排序，模拟"我们最终可能排到第几、三项打分区间多大"：

```bash
# 方式一：直接传多份 evaluate.py 产物
PYTHONPATH=src python scripts/rank_official.py \
  --results outputs/M1/result.json --tag M1 \
  --results outputs/M3/result.json --tag M3 \
  --output outputs/rank_compare.json

# 方式二：传汇总 JSON（结构见下）
PYTHONPATH=src python scripts/rank_official.py \
  --teams teams.json --output outputs/rank_compare.json
```

`teams.json` 结构：

```json
[
  {"team_id": "M1", "ship_recall": 0.7235, "ship_fdr": 0.5201,
   "aircraft_recall": 0.9076, "aircraft_fdr": 0.1571,
   "vehicle_recall": 0.6169, "vehicle_fdr": 0.6161,
   "latency_seconds": 6.2},
  ...
]
```

输出：每队 7 项指标、7 项排名、排名和、二次排序名次/百分比、三项打分区间。

> ⚠️ 这是**自我模拟**（我们队内部排名），不能代表真实对手分布；只用于判断
> "哪个模块改进能改善我们的相对位置"。真实排名取决于所有参赛队，无法预知。

## 3. 团队纪律

1. **所有正式实验必须同时报 pooled 与官方 macro 双口径**（`evaluate.py` 默认
   已同时输出），leaderboard 的 `*_macro_*` 列填 macro，门槛列填 pooled。
2. **阈值统一由 A 选**，成员只交低阈值原始预测，不在同一验证集上拟合阈值后
   报成绩。
3. **时效数据**必须来自 E 的正式 10K 流水线（`runtime_10k` 契约）。官方
   20 秒口径不含图像磁盘读取，但应包含读取之后的切片/预处理、模型推理、坐标
   恢复、跨 tile 融合和结果整理；同时另报 disk read 和 complete wall time。不拿
   model forward 时间冒充正式时延。
4. 内部目标（如官方口径 FDR≤0.17）一律按官方 macro 口径计，不用 pooled 替代。
5. 协议版本见 `configs/project.yaml`：匹配为 `official_eval_v1`，
   V1.6 排名聚合为 `official_ranking_v1_6`。历史结果不改写、
   不跨版本比较。

## 4. 相关文件

- 官方评分方案 PDF：`../../../../doc/比赛评分方案-V1.6.pdf`
- 官方细类匹配依据：`../../../../doc/官方QA.md` 第 10–11 条
- 评估核心实现：`src/rsdet/evaluation/official_metric.py`、`official_ranking.py`
- 阈值扫描：`scripts/sweep_thresholds.py`
- 实验总表：`reports/experiments/leaderboard.csv`
- 执行总纲：`reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`
