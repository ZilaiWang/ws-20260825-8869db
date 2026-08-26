# M3 教师证据交付（T3-TEACHER-EVIDENCE）

> 交付方：D（吴事凡）| 消费方：**C**（联合训练样本发现）
> 生成日期：2026-08-18 | 冻结口径见 `evidence_manifest.json`

## 1. 这是什么

M3（RT-DETR-L）在官方 CV3 三折 OOF 上，找回的 **M1 漏检目标**（hard positives）
逐对象清单 + 冻结元数据。这些目标：官方 GT 存在、M1 低阈值（0.001）无匹配候选、
M3 有匹配候选（同细类 + 官方 IoU 阈值）。

**总数：1313 个**，全部带 M3 预测框、score、IoU 和 GT 对照，可直接作为 C 的
训练样本发现输入（总纲 5.3 节第 5 条"与 D 的 M3 真阳性差集做联合训练样本发现"）。

## 2. 门禁判定（为什么交给 C 而不是 A）

总纲 5.4 停止条件（优先级从上到下）：

1. **车辆条件（优先）**：M3 在 ≥2/3 fold、≥3 source groups 找回 ≥21 唯一车辆 GT
   → 教师证据唯一交给 C；
2. 若车辆条件不通过，但纠正 ≥30 个 M1 细类错误 → 交给 A；
3. 均不通过 → 仅保留诊断。

**判定结果：**

| 门禁 | 要求 | 实际 | 结论 |
|---|---|---|---|
| gate1 | ≥2/3 fold 有 M3-only | fold0=584 / fold1=324 / fold2=405（3/3） | PASS |
| gate2 | ≥3 source groups | 126 组 | PASS |
| gate3 | 净增 ≥30 TP 或车辆 ≥21 | 总 1313 / **车辆 56** | PASS |
| **车辆条件** | ≥2/3 fold + ≥3 组 + 车辆 ≥21 | 3/3 折 + 126 组 + **56** | **PASS → 交给 C** |

## 3. 文件清单

| 文件 | 内容 |
|---|---|
| `hard_positives.csv` | 1313 行逐对象（gt_uid/image/fold/source_group/类/GT框/尺寸/M3框/score/IoU） |
| `hard_positives.json` | 同上 JSON（便于程序消费） |
| `stratified_stats.json` | 类/尺寸/折叠/来源组分层统计 |
| `evidence_manifest.json` | 冻结：M3 checkpoint SHA、协议版本、输入 SHA、门禁判定 |

## 4. 分层快照（C 消费重点）

- **类**：aircraft 1036 / ship 221 / **vehicle 56**；
- **尺寸**：small 854 / tiny 64 / medium 388 / large 7；
- **折叠**：fold0 584 / fold1 324 / fold2 405（三折均衡）；
- **来源组**：126 组，Top：mar20-airport-proxy-018=259（非单组集中）；
- 车辆 hard positives 56 个：对应 C 的车辆准入门槛（Recall 0.85 需净找回 ≥21
  唯一车辆；0.90 需 ≥41）——**M3 补回量已覆盖该门槛**。

## 5. C 怎么用（建议路径，消费方式由 C 决定）

1. **联合训练样本发现**：把 vehicle/tiny/small 的 hard positives 并入 C 的
   N0-HEAD / N0-NEARMISS 候选池，作为 P2 恢复头的监督信号或 hard-mining 正样本；
2. **真阳性差集**：`hard_positives.csv` 即"M3 真阳性 − M1 真阳性"差集
   （paired 分析中的 m3_only），与 C 的 N0 差集取交集可精确定位共同漏检；
3. 逐对象含 M3 的 bbox/score，C 可直接 crop 做可视化或特征级参考。

## 6. 边界与后续

- 本证据的"GT 确认"基于官方 OOF GT（自动配对），未做逐张人工复核；
  如需更严格可后续抽检 contact sheets（建议 10%）；
- M3 独有候选的 precision 不属于本交付（属 T4-GATED-INFER 评估范围）；
- 后续：T4 决策框架（gate4：N2-CFG 后 FDR≤0.17；gate5：10K p95≤18s）
  就绪后单独判定，不影响本证据使用。

## 7. 可追溯性

- M3 checkpoint：fold0 last.pt，SHA `72dd3b4d…`（与 E 正式测速同源）；
- 配对口径：同细类 + 粗类 IoU 阈值，min_score=0.001，贪心一对一（官方 V1.6）；
- 生成脚本：`scripts/build_m3_teacher_evidence.py`；
- 输入 SHA 全部记录在 `evidence_manifest.json`。
