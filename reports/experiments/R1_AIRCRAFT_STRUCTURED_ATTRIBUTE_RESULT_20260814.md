# R1-4 飞机物理属性辅助监督结果（2026-08-14）

## 1. 结论

R1-4 技术执行完整，但科学上不准入。五个训练期物理属性头没有提供独立于细类标签的
跨来源结构信息；正式工作点继续保留 R1-1 的 proposal-domain CE + full D4。

主条件 `structured_attribute_identity` 相对 `ce_identity` 仅净增 2 TP，同时增加 2 FP；
full D4 相对当前最强 CE + D4 也仅净增 2 TP、减少 3 FP，却使飞机 macro Recall 下降
`0.001065`。这属于几十个对象上的预测置换，不是稳定增益。停止当前 class-derived
attribute 路线，不扫描属性权重、属性组合或学习率。

## 2. 完整性与复现

- 服务器状态：`complete`，三折各固定 5 epoch，无 OOM、NaN 或重试；
- 回传包：`outputs/R1-4-AIRCRAFT-STRUCTURED-ATTRIBUTE-return-no-checkpoints.tar.gz`；
- SHA256：`7ff510f1f2151c92853d60af7bdedee7f260a93980300fb750e0fd41e8a97a0c`；
- 解压目录：`outputs/R1-4-AIRCRAFT-STRUCTURED-ATTRIBUTE-formal/`；
- 三折 bundle、runtime、run summary、audit 和五份 evaluation JSON 齐全；
- 本地快速 evaluator 与服务器原 evaluator 的五份核心 JSON 逐对象完全一致；
- ship/vehicle 四项指标最大差为 0，结构性旁路有效；
- 属性辅助头没有写入部署 checkpoint，推理结构与 CE 完全相同。

服务器生成的 `.sha256` 文件记录了服务器绝对路径，因此不能直接在本机执行
`shasum -c`；本次分别计算服务器和本地文件摘要并确认二者均为上述 SHA。这是交付脚本
可移植性问题，不影响包内容。

## 3. Identity 主条件

| 指标 | CE identity | Attribute identity | 差值 |
|---|---:|---:|---:|
| Overall Recall | 0.926480 | 0.926575 | +0.000096（+2 TP） |
| Overall FDR | 0.150541 | 0.150602 | **+0.000061（+2 FP）** |
| Overall macro Recall | 0.886262 | 0.885300 | **-0.000961** |
| Overall macro FDR | 0.203530 | 0.204400 | **+0.000870** |
| Aircraft macro Recall | 0.934442 | 0.933241 | **-0.001202** |
| Aircraft macro FDR | 0.132540 | 0.133628 | **+0.001088** |
| Aircraft pooled Recall | 0.947112 | 0.947224 | +0.000112 |
| Aircraft pooled FDR | 0.115709 | 0.115789 | +0.000080 |

配对转移为 `new_tp=43, broken_tp=41, net_tp=2, fp_delta=2`。配置中的宽松门禁因
容许 macro Recall/FDR 各退化 `0.002` 而返回 `primary_gate_passed=true`，但它只说明没有
灾难性退化，不能把统计上极小且宏平均更差的结果解释成准入。

## 4. Full D4 与当前最强工作点

| 指标 | CE + D4 | Attribute + D4 | 差值 |
|---|---:|---:|---:|
| Overall Recall | 0.930110 | 0.930206 | +0.000096（+2 TP） |
| Overall FDR | 0.146015 | 0.145890 | -0.000125（-3 FP） |
| Overall macro Recall | 0.891217 | 0.890365 | **-0.000852** |
| Overall macro FDR | 0.200975 | 0.200879 | -0.000095 |
| Aircraft macro Recall | 0.940637 | 0.939572 | **-0.001065** |
| Aircraft macro FDR | 0.129347 | 0.129228 | -0.000119 |
| Aircraft pooled Recall | 0.951370 | 0.951482 | +0.000112 |
| Aircraft pooled FDR | 0.110244 | 0.110092 | -0.000151 |

严格配对结果为 `new_tp=28, broken_tp=26, retained_tp=19444, net_tp=2`。候选数和模型
规模很大时，净 2 TP 远小于随机种子、折间组成和下一次训练波动，不足以形成工程价值。

## 5. 错误分解说明了什么

CE + D4 的正式剩余错误为：

- FP：`DUP=750, CLS=777, LOC=62, BG=1740`；
- FN：`CLS=777, LOC=62, MISS=624`；
- 飞机内部：`FP_CLS=667, FN_CLS=666, FP_BG=785, FN_MISS=196`。

属性头相对 CE + D4 只把错误重新分配为：

- `FP_DUP -12, FP_CLS -3, FP_BG +12`；
- `FN_CLS -3, FN_MISS +1`；
- TU-160 的 `FN_CLS +10`，FA-18 的 `FN_CLS -9`；
- `TU-160→TU-22` 从 110 增至 114，`TU-160→B-1B` 从 9 墨增至 14；
- `FA-18→SU-35` 从 46 降至 41。

原假设认为发动机数、翼型等属性可帮助 TU-160 等结构性混淆，实际结果相反。属性训练
accuracy 在第 2--5 epoch 已接近 99.9%，说明这些标签几乎可由已有细类特征直接推导；
它们没有注入新的图像证据，主要改变既有决策边界。

## 6. 对下一步的约束

1. 停止训练域类中心、同视图 KD 和 class-derived attribute 三类辅助损失；它们共同说明
   在已近 100% 拟合的 proposal 训练域继续压缩类别表示不能解决来源隔离泛化。
2. 保留 CE + D4 为飞机对象头最强点。D4 的收益已经跨对照成立，因此下一项只针对
   **同一对象不同 90° 视图预测不一致**，而不是再引入类别语义先验。
3. R1-5 采用双 D4 视图监督 CE + 对称一致性约束，目标是让 identity 接近 CE + D4，
   同时要求 full D4 不退化。只用一个预注册权重，不做网格搜索。
4. 端到端剩余 `FN_MISS=624` 和 `FP_BG=1740` 不可能由飞机 crop 细分类器单独消除；
   M3 互补检测器和经人工确认的 PU/background 分支仍是更高上限方向。

## 7. 评估基础设施修正

本轮补充了两项不改变任何历史结果的公共能力：

- `scripts/analyze_r1_frozen_condition.py`：按冻结的每折 variant/C2 重建预测，输出守恒的
  `FP_DUP/FP_CLS/FP_LOC/FP_BG` 与 `FN_CLS/FN_LOC/FN_MISS`；禁止在诊断时重新选模；
- `crossfit_thresholds.py`：将每个阈值重复做完整官方匹配改为一次候选底线匹配和精确
  分数前缀累计，并对低/中/高阈值作直接 official parity；R1-1 六条件八份 JSON 与旧实现
  完全一致，31 点扫描约 `0.49s`，阈值扫描部分约加速 30 倍。

这使后续短实验可以在数分钟内完成严格 cross-fit 评估，避免训练很短、评估却重复耗时。
