# M1/M3 正式 CV3 OOF 后处理分析方案 v1

状态：`implemented_waiting_for_M1_M3_formal_aggregates`  
任务 ID：`M1-M3-OOF-ANALYSIS-TASK-01`  
计算资源：CPU；不需要 GPU

## 1. 本任务回答什么

M1 与 M3 各自完成三折训练，只说明每张图都得到了一次未见来源组上的低阈值
预测。要决定主检测器、异构互补和 P05/P06 的后续准入，还必须在同一份正式
GT 上回答：

1. 依照官方细类匹配规则，M1/M3 的 Recall、FDR、TP、FP、FN 是多少；
2. 分数阈值变化时 Recall/FDR 如何变化；
3. FP 是重复、细类混淆、定位不足还是其余未归因候选；
4. FN 是细类混淆、定位不足还是完全漏检；
5. 对同一个 GT，哪些对象两者都检出、仅 M1 检出、仅 M3 检出或都漏掉；
6. 不考虑部署代价时，“任一模型命中即算命中”的 object oracle recall 上限
   是多少。

本任务不训练模型，也不把同一 OOF 上选择的阈值冒充独立测试成绩。

## 2. 唯一输入链

```text
formal_crop_manifest_v2
SHA a3bed44f...e484128
  └── crop_policy=tight：每个 annotation_uid 恰好一条 GT

M1-CV3-OOF-aggregate
  ├── oof_metadata.json
  ├── oof_images.csv
  ├── oof_proposals.csv
  └── predictions_oof_low.json

M3-CV3-OOF-aggregate
  └── 同上
```

两个 aggregate 都必须满足：

- `contract_version=cv3_oof_v1`；
- `status=complete_downstream_ready`；
- `downstream_admission=true`；
- `source_manifest_sha256=27b2eef4...b577331`；
- `image_count=4481`；
- `low_score_threshold=0.001`；
- 已绑定 formal crop 精确 SHA；
- 三个小型 aggregate 文件逐一与 metadata 内 SHA 闭环；
- `oof_images.csv` 的 4481 个 image ID 唯一，模型键不能混入。

还会做跨文件逐条闭环，而不只检查“文件存在”：

- `metadata.proposal_count == oof_proposals.csv 行数 == prediction JSON 行数`；
- image ledger 的每图 `prediction_count` 与 JSON 完全一致，包含零预测图；
- proposal CSV 与原 JSON 按输出顺序逐条核验 image、category、bbox、score；
- `.10g` 固定序列化后按绝对误差 `1e-12` 比较数值；
- `source_prediction_index` 必须在每个 fold 内从 0 连续递增；
- proposal 的 fold、model key、checkpoint SHA 必须与 image ledger 一致；
- metadata 记录的 artifact 路径若仍存在，必须就是当前实体；回传包搬迁后仅在
  文件名和 SHA 均一致时允许标记为 relocated audit。

任何一个条件失败都停止；诊断 aggregate 不能顶替正式输入。

## 3. GT 构造为何严谨

正式 crop 每个对象有 tight、context、jitter 三行。检测 GT 只读取
`crop_policy=tight`，但使用的是其中原始 `gt_x0/y0/x1/y1`，不是 crop
窗口或 jitter proposal。每个 `annotation_uid` 必须唯一，坐标语义必须是
`continuous_float_xyxy_half_open`。

由此恢复：

- 4481 张图；
- 20,933 个 GT；
- 25 个 `category_id=0..24`；
- 原始像素 xyxy；
- stable `annotation_uid`、fold、group_id。

同时导出 `ground_truth_from_formal_crop.json` 供复核。它只是冻结 GT 的 COCO
视图，不另造标注。

## 4. 官方指标与匹配轨迹

评估核心仍是 `src/rsdet/evaluation/official_metric.py`：

1. 按三大类选择官方 IoU 阈值：舰船/飞机 0.50，车辆 0.35；
2. 预测按 score 降序；
3. 只有相同 25 细类 `category_id` 才能匹配；
4. 每个 GT 最多匹配一次；
5. 未匹配预测计 FP，未匹配 GT 计 FN；
6. 完成细类匹配后才汇总三大类与总体 Recall/FDR。

实现增加 `evaluate_predictions_with_trace`，它与原评估函数同源返回每个 TP、
FP、FN 的原始列表下标。后续诊断不重新实现官方匹配。原有指标测试加上轨迹
计数和下标一致性测试。

## 5. 阈值曲线

冻结网格：

```text
start=0.001
stop=1.0
step=0.01
```

官方匹配按分数降序，因此删除低分后缀不会改变更高分预测先前的 TP/FP 状态。
实现从 0.001 的一次官方匹配轨迹构造精确 score-prefix 曲线，避免上百次重复
贪心匹配；同时在最低、中间、最高三个阈值调用官方评估器直接重算，任一点
TP/FP/FN 不一致即停止。

每个阈值报告总体和三大类：

- Recall、FDR；
- TP、FP、FN；
- 保留候选数。

在同一 OOF 上按“FDR 不超过 0.20 时 Recall 最高，再按 FDR 低、阈值高”给出
一个描述性工作点。该工作点明确写：

```text
same_oof_selection=true
exploratory_only=true
deployment_admission=false
```

它适合定位错误和决定下一轮实验，不是独立验证后的最终阈值。

## 6. 错误分解

先固定某一阈值并完成官方匹配，再仅对官方 unmatched 项按以下顺序归因：

1. `FP_DUP`：同细类 GT 上的 IoU 已达到官方阈值，但 GT 已被更早预测占用；
2. `FP_CLS/FN_CLS`：剩余预测与剩余 GT 类别不同，但 IoU 达到该 GT 大类
   阈值；按 IoU 高、score 高的确定性一对一规则配对；
3. `FP_LOC/FN_LOC`：剩余预测与剩余 GT 细类相同、IoU 大于 0 但低于官方
   阈值；确定性一对一配对；
4. `FP_BG`：以上规则均不能归因的剩余预测；
5. `FN_MISS`：以上规则均不能归因的剩余 GT。

必须满足：

```text
FP_DUP + FP_CLS + FP_LOC + FP_BG == official FP
FN_CLS + FN_LOC + FN_MISS == official FN
```

这里 `FP_BG` 的精确语义是“在已冻结规则后未归因”，它可能含极差定位的真实
目标候选，不能表述成已经人工确认的纯背景。错误分解是诊断，不是官方第二套
指标。

分解在两个位置各做一次：

- `candidate_floor=0.001`：候选召回和错误容量；
- same-OOF exploratory workpoint：接近可用工作点的错误构成。

描述性工作点的逐案例 CSV 保留 image ID、预测或 annotation UID、类别、框、
score、配对对象和 IoU，便于后续抽图审查。0.001 候选层可能达到百万行，
因此只保留完整守恒计数，不重复写出一个庞大逐案例 CSV；原始 OOF proposal
仍由 aggregate 完整保存。

## 7. M1/M3 对象级配对与 oracle

对每个 GT `annotation_uid`，直接读取两个模型的官方 TP 轨迹：

```text
both / M1_only / M3_only / neither
```

分别在 candidate floor 和各模型的描述性工作点生成 20,933 行对象表，记录：

- fold、group、细类和大类；
- M1/M3 阈值、是否命中、TP score、IoU；
- 未命中时的 `FN_CLS/FN_LOC/FN_MISS`；
- object oracle 是否命中。

oracle 定义为“任一模型对该 GT 产生官方 TP”。它只给 Recall 上限：

```text
oracle_union_recall =
  (both + M1_only + M3_only) / total_gt
```

oracle 没有 FDR，不能作为可部署集成。若 `M3_only` 很少，M3 不进入最终系统；
若它集中在明确困难子集，则后续只研究门控，不默认全量双模型。

## 8. 标准输出

```text
M1-M3-CV3-OOF-ANALYSIS/
├── analysis_metadata.json
├── ground_truth_from_formal_crop.json
├── M1/
│   ├── threshold_curve.csv
│   ├── exploratory_workpoint.json
│   ├── candidate_floor_metrics_and_errors.json
│   ├── exploratory_workpoint_metrics_and_errors.json
│   └── exploratory_workpoint_error_cases.csv
├── M3/
│   └── 同上
└── paired/
    ├── object_outcomes_candidate_floor.csv
    ├── object_outcomes_exploratory_workpoints.csv
    └── complementarity_and_oracle.json
```

`analysis_metadata.json` 绑定 config、project protocol、formal crop、两个
aggregate metadata 和两份 prediction JSON 的 SHA，并记录所有输出的大小与
SHA。

## 9. 结果如何指导下一步

1. M1 若已接近官方门槛，优先做阈值校准、融合和 10K 工程闭环；
2. `FN_CLS` 主导时，P03/P04 的对象精分类器才有真实端到端准入依据；
3. `FN_LOC` 主导且有大量阈值跨越空间时，P06 确定性框修正先于扩散框教师；
4. `FP_BG` 主导时，重新构建 P05 hard-negative，且样本来自真实 OOF FP；
5. `FP_DUP` 主导时，优先全局坐标融合/NMS/WBF，不新增重模型；
6. M3 仅在 `M3_only` 稳定且足以覆盖额外时延时保留。

正式阈值仍需预注册后在独立数据或后续冻结协议中确认。本任务不会越过这条
科学边界。
