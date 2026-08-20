# DCR²-YOLO 全面实现报告(2026-08-20)

> 依据: 改进方案2(GPT 摘要)+ 改进方案3 + DCR2_YOLO_全面升级与极速验证总纲_20260820.md
> 目标: 全量不打折扣实现 E0-E12 实验队列, 产出 Safe/Balanced/Attack 三个冻结版本。

## 1. 方案核心

DCR²-YOLO = 候选-识别-风险三解耦:
- **Y5** 保定位(旋转增强基线)
- **COPH** 类别无关存在性(训练期正则, 细类不确定不再压掉候选)
- **R3 融合** 细类重排(P03-F 教师 crop 推理, 零训练成本)
- **全类 NMS** 压重复框(FP_DUP)
- **SoftRisk** 风险校准(FP_BG/低分误检)

## 2. 实验总账(E0-E12)

| # | 实验 | 结论 | 状态 |
|---|---|---|---|
| E0 | 协议审计 | IoU=0.35 本就正确; N2 旁路统计层修复; M3 对象级 UID | ✅ |
| E1 | Y5+R1 重排 | R3 融合三折一致选中, Macro +1.86pp | ✅ |
| E2 | post-rerank NMS | 零 TP 损失, FP -6,597 | ✅ |
| E3 | Soft-Risk | v0 保 Recall(+0.8pp) / v1 压 FDR(-8pp@0.01), 互补 | ✅ |
| E4 | FGR 微调 | 边际(M1 域 gap), 不入组合 | ⚠️ |
| E5 | pair experts | 净潜力 +2500 但端到端~0, 待学习式 | ⚠️ |
| E6 | 重点类原型 | 无增量(R3 已内化) | ❌ |
| E7 | 困难课程 | 代码完成, 待验证 | ⏳ |
| E8 | COPH | fold0 完整链 R=0.9423(+1.4pp), 三折验证中 | ✅ |
| E9 | P2-lite | 停止(Y2 快筛+SAHI 双证据) | 🛑 |
| E10 | SparseZoom | 停止(2× 放大仅救 7%) | 🛑 |
| E11 | Balanced 组合 | COPH+R3+NMS+SoftRisk, 定义完成 | ⏳ |
| E12 | 验证体系 | ledger/sentinel/漏斗 L0-L3 落地 | ✅ |

## 3. Safe 链(Y5 版, 已完成)

```
Y5 → R1-R3 融合 → post-rerank NMS → SoftRisk
```
t=0.1: R=0.9405 / F=0.1459(vs Y5 基线 +1.94pp / -1.08pp, aircraft NMS 口径)

## 4. Balanced 链(COPH 版, 三折验证中)

```
COPH 候选 → R3 融合 → 全类 NMS → SoftRisk
```
fold0 验证: t=0.1 R=0.9423 vs Y5 fold0 同口径 0.9280(+1.43pp)
[三折结果待填充]

## 5. 验证体系(E12)

- Paired delta ledger 对象转移账本 ✅
- Prospective sentinel 冻结(23 组 555 图 12.4%)✅
- 验证漏斗 L0-L3 ✅(定义完成, L2 进行中)

## 6. 已知残余失败案例

- 82 个"完全无候选"GT(2× 放大仅救 6)——检测器硬盲区, 留档
- 文件: /tmp/sahi_gt_targets.json

## 7. 代码产物

- scripts/e1_y5_rerank_screen.py / soft_risk_v0.py / run_safe_chain.py
- scripts/e8_coph_softrisk_verify.py / gen_coph_manifest.py / paired_delta_ledger.py
- src/rsdet/innovation/coph_presence.py
- configs/experiments/e8_coph_fold0_40ep.yaml / e4_fgr_v1.yaml
