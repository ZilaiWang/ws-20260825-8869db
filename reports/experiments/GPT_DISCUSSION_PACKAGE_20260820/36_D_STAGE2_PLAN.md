# D 第二阶段剩余工作：深度执行规划（替 D 完成）

> 日期：2026-08-18 | 依据：NEXT_STAGE_TEAM_INNOVATION_EXECUTION_MASTER_v1.md 5.4 节
> 现状：T0-M3-FINISH ✅、T1-PAIR ✅（gate1/2/3 PASS）；剩 T2-HARDPOS、T3-TEACHER-EVIDENCE、T4-GATED-INFER

## 1. 总纲要求复述（5.4 节原文提炼）

D 的连续主线：**先完成 M3，再判断其最有价值的角色；不预设 RT-DETR 永久参与正式推理。**

| 实验 | 内容 | 状态 |
|---|---|---|
| T0-M3-FINISH | 完成冻结 RT-DETR-L/1024 三折 OOF | ✅ 已完成 |
| T1-PAIR | M1-only/M3-only/共同 TP/各自唯一 TP/FP 对象配对 | ✅ 已完成（门禁 3/5 PASS） |
| T2-HARDPOS | 筛选 M3 找到、M1 漏掉且人工/GT 确认的 hard positives | ⏳ **本次做** |
| T3-TEACHER-EVIDENCE | 交付冻结教师证据（不由 D 写消费模型） | ⏳ **本次做** |
| T4-GATED-INFER | 仅当 M3 直接互补有稳定净收益时做门控推理 | ⏳ 决策框架先备（待 gate4/5） |

## 2. 停止条件对照（决定 T3 消费方）

总纲停止条件（按优先级）：

1. **车辆条件优先**：M3 在 ≥2/3 fold、≥3 source groups 中额外找回 ≥21 个唯一车辆 GT
   → **教师证据唯一交给 C**；
2. 若车辆条件不通过，但纠正 ≥30 个 M1 细类错误（≥2/3 fold、≥3 groups）
   → 教师证据唯一交给 A；
3. 两项均未通过 → 仅保留诊断，不做 T3 消费实验。

**当前判定（来自 M3_PAIRED_OOF_ANALYSIS.json）：**

- gate1（≥2/3 fold 有 M3-only）：PASS，fold0=584 / fold1=324 / fold2=405（3/3 折）；
- gate2（≥3 source groups）：PASS，126 个组；
- gate3（≥30 TP 或车辆 ≥21）：PASS，总 1313 / **车辆 56**；
- **结论：车辆条件（56 ≥ 21）在 3/3 折、126 组上成立 → T3 教师证据唯一交给 C**
  （与 C 的 5.3 节"与 D 的 M3 真阳性差集做联合训练样本发现"对接）。

## 3. T2-HARDPOS 设计

### 3.1 定义

hard positive = 官方 GT 存在、**M1 低阈值（0.001）无匹配候选、M3 有匹配候选** 的对象。
即配对分析中的 `m3_only`（1313 个）。GT 确认 = 官方 OOF GT 本身（已与 M3 预测按
官方口径配对，同细类 + IoU 阈值）。

### 3.2 输入（本地已有）

| 资产 | 路径 | 用途 |
|---|---|---|
| M3 OOF proposals | `outputs/M3-CV3-OOF-aggregate/oof_proposals.csv`（218MB） | M3 匹配候选 |
| M1 OOF proposals | `outputs/M1-CV3-OOF-.../oof_proposals.csv` | M1 匹配候选 |
| 官方 GT | `outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv` | GT 真值 |
| split | `data/splits/cv3_airport_proxy_k60_v2.json` | fold/source_group |
| 协议 | `configs/project.yaml` | IoU 阈值/类别映射 |

### 3.3 输出（逐 GT 明细）

`outputs/M3-TEACHER-EVIDENCE/hard_positives.csv`：

```csv
gt_uid,image_id,fold,source_group,class_name,category_id,gt_bbox_xyxy,
gt_short_edge,size_bin,m3_proposal_uid,m3_score,m3_bbox_xyxy,iou
```

每条 = 一个 m3_only GT + 其 M3 匹配候选（score/bbox/IoU）。
排序：vehicle → ship → aircraft；同粗类内按 score 降序（教师证据强度）。

### 3.4 分层统计

- 类（vehicle/ship/aircraft）；
- 尺寸（tiny/small/medium/large，按 GT short_edge）；
- fold（0/1/2）；
- source_group（top 分布，确认 ≥3 组、非单组集中）。

## 4. T3-TEACHER-EVIDENCE 设计

### 4.1 交付物（冻结、可追溯）

```
outputs/M3-TEACHER-EVIDENCE/
├── evidence_manifest.json      # 冻结：proposal 来源/score/checkpoint SHA/协议版本/判定
├── hard_positives.csv          # 逐 GT 明细（T2 输出）
├── hard_positives.json         # 同上，JSON 格式
├── stratified_stats.json       # 类/尺寸/折叠/组分层统计
├── contact_sheets/             # 可视化复核页（可选，若图可用）
└── TEACHER_EVIDENCE_README.md  # 交付说明（消费方 C 怎么用）
```

### 4.2 冻结信息（evidence_manifest.json）

- M3 checkpoint SHA：`72dd3b4d…`（fold0 last.pt，与 E 正式测速同源）；
- 配对口径：同细类 + 粗类 IoU 阈值（官方 V1.6）；
- min_score：0.001；
- 门禁判定：gate1/2/3 PASS + 车辆条件通过 → **教师证据唯一交给 C**；
- 生成时间 / 生成脚本 / 输入 SHA。

### 4.3 C 的消费面（总纲 5.3 节第 5 条）

- 与 D 的 M3 真阳性差集做**联合训练样本发现**：
  - C 的 N0-HEAD/N0-NEARMISS 漏检目标 ↔ 本清单（M3 找回的）；
  - 车辆 hard positives（56 个）优先：C 的车辆准入门槛 Recall 0.85 需净找回 21 个
    唯一车辆——**M3 的 56 个车辆补回正是 C 的候选下限来源**；
  - 尺寸分布（tiny 64 / small 854 等）供 C 的小尺度恢复头参考。

## 5. T4-GATED-INFER 决策框架

### 5.1 依赖（当前不可执行）

- gate4：N2-CFG 后 Overall FDR≤0.17（等盲审白名单 → N2-CFG 三折）；
- gate5：10K p95≤18s（等 E 组合正式测速，入口已就绪）。

### 5.2 决策判据（总纲原文）

T4 仅当：**校准后额外净增 ≥30 TP、Overall FDR≤0.17 且 10K p95≤18 秒** 才保留；
否则不进入最终系统。

### 5.3 本阶段可做

- 预写 `evaluate_gated_infer.py`：输入 M1/M3 预测 + 门控掩码（source_group/score/
  coarse 白名单），输出"门控后 TP 净增 + FDR + 时延预算"；
- 等 gate4/5 数据到后一键判定。

## 6. 执行顺序

1. **本规划文档**（本文件）；
2. **T2**：扩展配对脚本输出逐 GT hard positives（复用 analyze_m3_paired_oof.py 的
   匹配逻辑，新增明细导出）→ 产出 hard_positives.csv + 分层统计；
3. **T3**：组 evidence_manifest + README，交付到 `outputs/M3-TEACHER-EVIDENCE/`
   并归档 `reports/members/D/`（D 的交付）→ 抄送 C；
4. **T4**：决策框架脚本就绪（不执行），标注依赖 gate4/5。

## 7. 验收

- hard_positives.csv 计数 = m3_only = 1313（与 M3_PAIRED_OOF_ANALYSIS 一致）；
- 车辆 56、tiny 64、small 854 等分层与已冻结分析逐项一致；
- evidence_manifest 含 checkpoint SHA、协议版本、门禁判定（可追溯）；
- README 明确消费方 = C、消费方式 = 联合训练样本发现。

## 8. 风险与边界

- hard positive 的"人工确认"：本方案以官方 OOF GT 作为确认（GT 确认），
  未做逐张人工复核；如需更严格，可后续用 contact sheets 抽检（如 10%）；
- M3 独有候选的 precision：未在 T2 内计算（属于 T4 门控评估范围）；
- 若 N2-CFG 后 FDR 无法 ≤0.17，T4 自动不保留，不影响 T2/T3 交付。
