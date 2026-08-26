# Phase N0 执行报告：M1 决策证据补齐

日期：2026-08-09
执行人：A（王子莱）
模型：M1 = YOLO26-s（seed 42，正式 CV3 OOF）
数据：`cv3_airport_proxy_k60_v2`（4,481 图 / 20,933 GT / 55,548 低阈值候选）
环境：本机 CPU（MacBook），全部 N0 任务为纯后处理，未占用 GPU

> **2026-08-10 收尾更正**：N0-1/N0-2 的官方匹配与解耦结论保持有效。
> N0-3 v1 存在 oracle 命中按“同图+同预测类”误传给其他候选的缺陷，
> 且 FP 子类是 nearest-overlap 诊断分类，不是官方计数守恒错误分解。
> `outputs/N0-EVIDENCE-M1` 与由它生成的 N0-4 抽样包已冻结为
> **superseded_invalid**。修复重放已完成：`outputs/N0-EVIDENCE-M1-v2`，
> 合同为 `pred_oof_evidence_v2`，主文件 SHA256 为
> `ae5ffc8d3559a8eb87d38d0c2851ad53d1f8a5ec1c274b3117aae9f5a8bbe900`。

## 1. N0-1：cross-fit 阈值基线（已完成 ✅）

**当前产出**：`outputs/N0-CROSSFIT-M1-v2/crossfit_result.json`（V1.6 macro 补算；
SHA256 `53f985039b3bb72694538e353fc4a32a92b380e9df844e8693813b3a7cc73b06`）

| 项 | 值 |
|---|---|
| **合并 held-out（正式无偏 baseline）** | **Recall 0.9176 / FDR 0.1990**（gate PASS） |
| 内部目标 FDR≤0.17 | 未达成（0.1990） |
| fold 0 | 选阈 0.041 → Recall 0.8996 / FDR 0.2259（**gate FAIL**） |
| fold 1 | 选阈 0.071 → Recall 0.9379 / FDR 0.1401（gate PASS） |
| fold 2 | 选阈 0.041 → Recall 0.9155 / FDR 0.2294（**gate FAIL**） |
| 阈值离散度 | mean 0.051 / std 0.014 / spread 0.030 |

cross-fit 合并 held-out 的完整 V1.6 macro：

| 大类 | macro Recall | macro FDR |
|---|---:|---:|
| 舰船 | 0.7162 | 0.5389 |
| 飞机 | 0.9080 | 0.1589 |
| 车辆 | 0.6119 | 0.6239 |
| 25 细类 overall diagnostic | 0.8654 | 0.2383 |

**结论**：
- 与同 OOF 探索值（Recall 0.9172 / FDR 0.1957）对比，cross-fit FDR 差约 **0.003**
  —— 自证偏差量级小（阈值选择稳定），但确认存在。
- **fold 0 与 fold 2 的 FDR 结构性超线**（0.226 / 0.229），不是阈值抖动。
  总体过线是"平均掩盖个别折不过线"的统计假象。
- 部署阈值定案建议：`0.051`（cross-fit 均值），标注 `formal_baseline`。

## 2. N0-2：定位/分类解耦（已完成 ✅）

**产出**：`outputs/N0-DECOUPLED-M1/decoupled_result.json`

| 指标 | 值 |
|---|---|
| **R_loc@oracle-class（预测细类免费）** | **0.9705** |
| **Acc_fine@localized（几何匹配上的细类正确率）** | **0.9297** |
| source-group bootstrap 95% CI | [0.9585, 0.9792]（255 组，2000 迭代） |

**分层定位召回**：

| 层 | 对象数 | oracle 定位召回 |
|---|---|---|
| overall | 20,933 | 0.9705 |
| size_large_ge64 | 16,636 | 0.9838 |
| size_medium_32_64 | 4,009 | 0.9439 |
| **size_small_16_32** | 258 | **0.6163** |
| **size_tiny_lt16** | 30 | **0.2000** |
| category_24（FSC 车辆） | 402 | **0.6169** |

**结论（修正总纲粗结论）**：
- **总体而言"分类主导"成立**：oracle 定位召回 0.9705 vs 官方 recall 0.9172，
  分类环节贡献约 0.053 的损失。
- **但对小目标（<32px）"定位才是瓶颈"**：tiny/small 的 oracle 召回仅
  0.20/0.62，远低于总体。车辆（FSC，402 个）正是小目标重灾区（0.617）。
- 结论：**N1/N2 的对象分类器解决"报出来但认错"（FP_CLS/细类混淆）；C 分工的
  小目标候选恢复解决"根本没报出来"（FN_MISS）**。二者不是替代关系，是叠加。

## 3. N0-3：Pred-OOF 对象证据 manifest（v1 废弃，v2 完成）

**当前产出**：`outputs/N0-EVIDENCE-M1-v2/pred_oof_evidence.json`

阈值 0.051 工作点：

| 项 | 值 |
|---|---|
| 候选总数（score≥0.051） | 23,870 |
| 官方 TP | 19,199 |
| 官方 FP | 4,671 |
| FP 类型 | FP_CLS 2,103 / FP_BG 1,826 / FP_LOC 556 / FP_DUP 186 |
| oracle_positive 视图 | 20,315 |
| deployable_positive 视图 | 19,199 |
| hard_negative 视图 | 3,242 |

**说明**：每条候选含原始 `source_prediction_index`，oracle 只能赋给
实际匹配到 GT 的那条候选，不再在“同图+同预测类”之间传播。表中 FP 类型
仍是 `nearest_overlap_diagnostic_v1`，只用于抽样与研究；正式计数守恒的
`FP_BG/FP_CLS/FP_DUP/FP_LOC` 仍以 M1 正式错误分解为准。

## 4. N0-4：FP_BG 人工语义审计抽检包（v2 抽样包已重放）

**当前产出**：`outputs/N0-FP-BG-AUDIT-v2/`
- `audit_samples.csv`：人工标注表（label/labeler 待 B 填写）
- `audit_samples.json`：抽检包明细

| 项 | 值 |
|---|---|
| FP_BG 池 | 1,826 |
| 抽样正卡 | 270（3 大类 × 3 折 × 3 分位 = 27 分层单元 × 10/层） |
| 盲重复卡 | 54（20%） |
| 总审计样本 | 324 |

**待办**：按 `docs/hub/01_scoring_standard` 协议的 5 类标签盲审；
一致性率 ≥ 0.85 通过人工程序质检。只有 `clear_background` 可作背景训练样本。

## 5. 代码资产（全部合入主项目）

| 文件 | 说明 |
|---|---|
| `src/rsdet/analysis/crossfit_thresholds.py` | N0-1 核心（加载聚合/GT、扫描、cross-fit） |
| `src/rsdet/analysis/decoupled_errors.py` | N0-2 核心（oracle 匹配、bootstrap、分层） |
| `src/rsdet/analysis/object_evidence.py` | N0-3 核心（manifest、三种视图、FP 分类） |
| `src/rsdet/analysis/fp_bg_audit.py` | N0-4 核心（分层抽检、盲重复卡、标签协议） |
| `scripts/n0_1_crossfit_thresholds.py` | CLI |
| `scripts/n0_2_decoupled_errors.py` | CLI |
| `scripts/n0_3_build_evidence_manifest.py` | CLI |
| `scripts/n0_4_fp_bg_audit.py` | CLI |
| `tests/test_crossfit_thresholds.py` 等 4 个 | 44 个新测试 |

测试：**400 passed, 5 skipped**（无回归）。

## 6. 结论与建议

1. **M1 正式无偏 baseline 已定**：Recall 0.9176 / FDR 0.1990（cross-fit，
   可进 leaderboard）。
2. **风险点明确**：fold 0/2 的 FDR 超线（>0.22）+ 官方 macro 船 FDR 0.52
   —— 排名维度风险高于门槛维度。
3. **资源方向修正**：小目标（<32px）定位是硬瓶颈（oracle 0.20-0.62），
   C 分工的小目标候选恢复优先级应高于常规细类调优；但 FP_CLS 2,103 仍是
   N1/N2 对象分类器的主战场（FP_BG 1,826 待 B 审计后决定背景样本资格）。
4. **下一步**：N1 已完成；N0-4 v2 只在重启真实背景拒识时交人工
   盲审，未确认的 hard negative 不得自动标为背景。
