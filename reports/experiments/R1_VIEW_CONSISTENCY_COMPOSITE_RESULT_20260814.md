# R1-7/R1-8 一致性头组合与概率融合结果（2026-08-14）

## 1. 最终决策

R1-7 将 consistency+D4 接入已准入的飞机后 NMS，得到一个高召回但
高虚警的 Pareto 备选；R1-8 固定 50/50 平均 CE 与 consistency 概率，
未形成折中优势。

主工作点保留 **R1-6 CE+D4+C2+飞机后 NMS**：

- 相对 R1-7 少 50 TP，但少 61 FP；
- 飞机 macro FDR 优 `0.00250`，pooled FDR 优 `0.00291`；
- R1-7 只作高召回备选，不作为背景盲审和当前系统冻结输入；
- 停止搜概率融合权重。

## 2. R1-7：consistency+D4 + 固定 NMS

NMS 仍仅作用于飞机细类，IoU 固定 `0.50`。三次外层敏感性审计都
选回 0.50，与直接固定预测逐条一致。

| 指标 | R1-6 | R1-7 | R1-7 减 R1-6 |
|---|---:|---:|---:|
| TP / FP / FN | 19470 / 2482 / 1463 | 19520 / 2543 / 1413 | **+50 / +61 / -50** |
| pooled Recall | 0.930110 | 0.932499 | **+0.002389** |
| pooled FDR | 0.113065 | 0.115261 | +0.002196 |
| macro Recall | 0.891217 | 0.891803 | **+0.000586** |
| macro FDR | 0.155794 | 0.157797 | +0.002003 |
| aircraft pooled Recall | 0.951370 | 0.954171 | **+0.002801** |
| aircraft pooled FDR | 0.068922 | 0.071830 | +0.002908 |
| aircraft macro Recall | 0.940637 | 0.941369 | **+0.000732** |
| aircraft macro FDR | 0.072871 | 0.075375 | +0.002504 |

R1-7 自身的 NMS 从 3361 FP 中删除 818 个，且 TP/FN 不变；但分类头
带来的新虚警未被 NMS 全部消化。

## 3. R1-8：固定等权概率融合

三折严格按 proposal UID 对齐 CE+D4 与 consistency+D4，对 20 维飞机条件
概率做 `0.5/0.5` 算术平均。权重是预先固定的，没有网格搜索。

相对 CE+D4：

- `new_tp=78, broken_tp=74, net_tp=4`；
- FP `+20`，pooled FDR `+0.000723`；
- aircraft macro Recall `-0.003386`；
- 主门禁失败，`next_action=retain_reference_and_stop_current_refinement_method`。

这表明两头的错误不能通过朴素平均稳定抵消，不再扩展到 0.25/0.75
或逐类权重，避免继续在同一 OOF 上搜参。

## 4. 产物

- R1-7 配置：`configs/experiments/r1_view_consistency_post_nms_v1.yaml`
- R1-7 决策：`outputs/R1-7-VIEW-CONSISTENCY-POST-NMS/decision.json`
- R1-8 配置：`configs/experiments/r1_equal_probability_ensemble_v1.yaml`
- R1-8 bundle 构建：`scripts/build_r1_equal_ensemble_bundle.py`
- R1-8 决策：`outputs/R1-8-EQUAL-PROBABILITY-ENSEMBLE/evaluation/decision.json`

