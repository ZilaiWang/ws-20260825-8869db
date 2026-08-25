# HERA-YOLO 批次 A 实施总结(2026-08-20)

> 依据《改进方案4》(HERA-YOLO 方法级重构)。批次 A = 立即执行、全部 replay/缓存级,
> 不需要训练新 detector。本批次已全部完成(A0-A3 + sentinel 泛化)。

## 一、方案4 的核心判断(已被实验验证)

方案4 主张: **候选、细类、有效性、唯一性必须"对象级联合裁决",而非"融合一次、改类一次、
SoftRisk 一次、NMS 一次"的串行补丁**。并明确:
- COPH 只证明"能抬候选",未证明"提升可部署前沿"→ 降级 Candidate-Heavy;
- 下一批最高 ROI = OER 对象证据解析器 + 可观测性路由器(成本最低、证据最充分)。

## 二、批次 A 实验总账

| ID | 实验 | 关键结果 | 结论 |
|---|---|---|---|
| A0 | PR frontier 重算 | COPH Recall@FDR=.12=0.9295 < Y5 0.9357 | 取消 COPH 主提交, Safe=0.9357 为靶点 |
| A1 | OOF 对象图 | 65,301 节点 / 234,127 边(same 34,549 / different 18,149 / bg 181,429) | OER 训练数据就绪 |
| A2 | OER-v0 node_validity | HistGB 排序 OER+NMS R@FDR=.12=**0.9415** | 单模型超 Safe 三模块串行链 |
| A3 | 可观测性路由器 | 全改 OER+NMS R@FDR=.12=**0.9584**, broken=0 | 破解 E5 broken, 超 HERA-Core 目标 |
| A3-sentinel | 泛化验证 | 555 冻结图改类 +1.42pp(0.9698→0.9840) | 收益非 overfit |

## 三、核心指标推进(固定风险前沿 Recall@FDR=0.12)

| 版本 | Recall@FDR=.12 | FDR@Recall=.94 | 来源 |
|---|---:|---:|---|
| Y5 基线 | 0.9211(固定阈值口径) | — | 前批 |
| Safe(冻结) | 0.9357 | 0.1310 | A0 |
| OER + NMS | 0.9415 | 0.1136 | A2 |
| **OER + NMS + 改类** | **0.9584** | **0.0677** | A3 |
| sentinel(改类) | 0.9840 | 0.0190 | A3-sentinel |

**HERA-Core 目标 0.945 已超过(0.9584), 逼近 Attack 目标 0.96。**

## 四、三个关键机制发现

1. **对象级联合裁决优于串行补丁**: OER 用 HistGB + crop 证据(top1/margin/entropy/agree)
   + 几何 + 密度, 单模型吸收了 R3 融合(细类证据)与 SoftRisk(风险)的信息, 且树模型
   捕捉了逻辑回归看不到的非线性交互(0.9415 vs Safe 0.9357);
2. **改类天然不 broken**: 只在"yolo 错细类"候选上改类(它们本来就不是 TP), 改对变 TP
   (+8,227), 改错仍 FP_CLS(不损 TP)——直接破解 E5 规则版的 broken 267-393;
3. **crop 纠错可信度可预测**: 路由器 AUC 0.9459, 目标越大/crop margin 越高/entropy 越低
   → crop 可信。

## 五、待 GPU 的下一批(批次 A 剩余 + B/C 主线)

| ID | 内容 | GPU 需求 |
|---|---|---|
| A4 | 基础模型 proposal probe(DINOv2/RemoteCLIP 线性探针) | 需 GPU + 下载模型 |
| A5 | OTM/OTO 零训练诊断(one-to-one 是否提供 precision 证据) | 需 GPU(YOLO26 OTO 推理) |
| A3-full | 可观测性路由补视觉特征(对比度/模糊/边缘能量) | 需 GPU + 原图 |
| A6 | 低保真相关性(20/40/80/160 epoch Kendall tau) | 需 GPU(历史 checkpoint 推理) |
| B1-B5 | CAQ 类别无关候选头 → 密集前景蒸馏 | 需 GPU 训练 |

## 六、产物索引

- scripts/a0_pr_frontier.py / a1_build_object_graph.py / a2_oer_v0.py
- scripts/a2_oer_edge.py / a3_observability_router.py / a3_router_e2e.py / a3_sentinel_check.py
- reports/experiments/A0_PR_FRONTIER_20260820/ / A2_OER_V0_20260820/ / A3_OBSERVABILITY_ROUTER_20260820/

---

## 补充: 批次 A 完整收尾(18:35-19:10, 需 GPU 部分)

### A5: OTM/OTO 诊断 → has_oto_strong 特征突破 0.96 ✅
- YOLO26 含 one2one head, OTO 高置信(score>0.5)支持是 FP_BG 强判别信号(TP 0.892 vs FP 0.094);
- OER + has_oto_strong + 改类: **Recall@FDR=0.12 = 0.9607**(+0.24pp), **超过 Attack 目标 0.96**;
- sentinel 泛化 0.9847。

### A3-full: 视觉特征 → 停 ✅(负结论)
- 视觉特征对路由器 AUC +0.005(边际), 被 short_edge/crop_margin 覆盖;
- 更正 A3 口径: 5折随机CV AUC 0.9459 高估, 严格 fold cross-fit = 0.8649。

### A4: foundation probe → 🛑 网络受限
- 服务器外网受限, DINOv2/RemoteCLIP 无法下载, 降级 ConvNeXt-T 维持。

### A6: 低保真相关性 → ⏸️ 暂缓
- 历史无中间 epoch checkpoint, 需重训; 且服务对象(B系列)前景不明。

## 最终里程碑数字链(Recall@FDR=0.12)

Safe 0.9357 → OER+NMS 0.9415 → +改类 0.9584 → **+has_oto强 0.9607** → sentinel 0.9847

**0.9607 已超过 Attack 目标 0.96, 且 sentinel 泛化确认。**
