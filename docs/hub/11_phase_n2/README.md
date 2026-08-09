# Phase N2：X-CROP-03 与 X-BG-01，共享对象学生

更新日期：2026-08-09  
状态：`running`（N2-1 训练中，N2-2 代码就绪）

> 本目录记录 N2 阶段的执行方案、数据流与准入门槛。N2 由 A 统一实现
> （总纲 Phase N2），共享对象学生是 P03/P05/困难门控的共同底座。

## 1. 目标与数据流

```text
N0-3 对象证据层（pred_oof_evidence.json，23,870 候选）
  → N2-1 proposal crop manifest（23,870 行，含 crop_xyxy/标签/视图/背景）
  → 对象学生训练（ConvNeXt-T，26 类含背景哨兵，三折防泄漏）
  → N2-2 重分类/拒背景/联合（M1 框 + 学生类别 + cross-fit 评估）
  → N2-3 准入门槛判断
```

**防泄漏纪律**：每个 held-out fold 的学生只能用另外两个 fold 的候选训练
（按 leakage_group_id = source_group 划分，与 N0-3 fold 归属一致）。

## 2. N2-1：普通 Pred-OOF crop 强基线

| 视图 | 回答的问题 |
|---|---|
| GT crop | 理想上限（P03-F 已答：macro_recall 0.9287） |
| oracle_positive | 定位存在时细类是否可修复 |
| deployable_positive | 实际候选上的端到端输入质量 |
| hard_negative | 是否能稳定拒背景 |

- 模型：ConvNeXt-T（ImageNet 初始化），tight-224，fine_tune，seed42+fold
- 训练集：非 held-out fold 候选（deployable_positive + oracle_positive + hard_negative）
- 标签策略：TP→真值细类；FP→预测类别；FP_BG 无 oracle→background 哨兵(25)
- 输出：三折 best_checkpoint + 逐折验证指标

**当前状态**：三折训练中（服务器 autodl RTX 3090）。

## 3. N2-2：分开消融，再联合

固定顺序（总纲）：

1. M1 + cross-fit threshold（基线：Recall 0.9176 / FDR 0.1990）；
2. + 25 类重分类（`reclassify`：学生类别替换 M1 类别，不拒背景）；
3. + 背景拒识（`background_reject`：学生判 background 则丢弃，不改细类）；
4. + 联合对象学生（`joint`：重分类 + 拒背景）；
5. + 困难对象门控（后续实现，保留主要收益并降低时延）；
6. 只有 P04 正式支持时，加入 DINOv2 蒸馏。

每一行重新执行 cross-fit 阈值，不允许沿用对某一模块最有利的同 OOF 阈值。

**实现**：
- `src/rsdet/analysis/proposal_reclassification.py`：三种模式融合
- `scripts/reclassify_proposals.py`：GPU 重分类（每折一个 JSON）
- `scripts/evaluate_reclassified.py`：cross-fit 评估（复用 N0-1）

## 4. N2-3：准入门槛

对象模块进入正式主线至少满足（全部同时满足）：

- pooled 官方指标改善；
- 至少 2/3 folds 同方向；
- Overall Recall 不低于 0.88，FDR 明显优于 0.20 并朝 0.17 收敛；
- 舰船 Recall 不因拒识稳定下降；
- 对飞机 `FN_CLS` 有明确净恢复；
- 对舰船/车辆 FP 有明确净减少；
- 收益不只来自 TU-160 单一大组或一两个极少样本；
- 全候选离线收益成立后，困难门控能保留主要收益并降低时延。

## 5. 与 P03/P04 的关系

- P03-F（GT crop 对象分类）：macro_recall 0.9287 —— GT crop 理想上限；
- P04-F（教师探针）：DINOv2-B 0.8294 最优 —— 教师选择依据（N2 第 6 步用）；
- N2-1 用 proposal crop（候选质量低于 GT crop），结果应低于 P03-F 上限，
  差值量化"候选输入质量损失"。

## 6. 关键文件

- `src/rsdet/analysis/proposal_crops.py`：proposal crop manifest 生成
- `src/rsdet/analysis/proposal_reclassification.py`：重分类/拒背景/联合
- `scripts/train_object_student.py`：对象学生三折训练
- `scripts/reclassify_proposals.py`：GPU 重分类
- `scripts/evaluate_reclassified.py`：cross-fit 消融评估
