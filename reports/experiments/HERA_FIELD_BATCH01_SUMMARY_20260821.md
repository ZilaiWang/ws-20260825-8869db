# HERA-Field 批次 0/1 完整总结(2026-08-21)

> 方案5(HERA-Field)第一批实施: 基准审计 + 对象-环境证据场 + 硬背景图谱 + 车辆种子 + foreground rejector 的全量验证。

## 一、核心结论(一句话)

**当前 OER(14 特征)已饱和——"扩候选"(COPH/DFD/V1)和"加轻证据"(C3/C6/C7/F1)两个方向的全部 8 次实验都无法突破固定风险前沿,真正的增益需要"重"改动(属性识别 / DeepSets listwise / 结构建模)。**

## 二、基准与靶点(批次 0)

| 项 | 结果 |
|---|---|
| P0-1 GT-blind 审计(上轮) | **0.9620 是 oracle**(改类依赖 GT 判断 yolo 是否错), deploy 0.9405, gap 2.15pp |
| P0-4 类别前沿 | vehicle 工作点 **0.7264**(floor 0.8109)、ship 0.8811 弱粗类 |
| P0-4 最弱细类 | HM 0.82 / QHS 0.83 / TU-160 0.86 / MS 0.90 |
| P0-4 弱域 | site_ 机场(P10=0.75 vs 中位 1.0), 极不均 |

## 三、8 次实验全记录(批次 1 + V1 + F1)

| 实验 | 方向 | 方法 | ΔR@FDR=.12 | 判定 |
|---|---|---|---:|---|
| C3 反事实 | 加证据 | FPN 差分标量(28维) | +0.0006 | ❌ 边际 |
| C6 硬背景 | 加证据 | PCA 原型 cos 相似度 | +0.0000 | ❌ 无效 |
| C7 listwise | 加证据 | hard-pair 加权 | −0.0008 | ❌ 负 |
| V1 车辆种子 | 扩候选 | vehicle-only 密集监督 | −0.79pp | ❌ 停止 |
| F1 fg rejector | 加证据 | foreground logit | −0.0002 | ❌ 持平 |
| (历史) COPH | 扩候选 | 存在性正则 | −0.62pp | ❌ |
| (历史) DFD | 扩候选 | 全类密集监督 | −0.04pp | ❌ |
| (历史) edge 聚类 | 去重 | 学习式聚类 | −2.4pp | ❌ |

## 四、方法论结论(供 GPT 讨论)

### 4.1 扩候选四次失败(COPH/DFD/V1)

- COPH(−0.62) / DFD(−0.04) / V1(−0.79): 三种"扩候选"方式(存在性正则/密集前景
  监督/vehicle-only 密集)都能抬 candidate-floor(最高 +1.11pp), 但救回的低对比真阳
  score 偏低, 弱排序抬不动, 反被新增 FP 拖累前沿;
- **结论: "多报候选"无法提升固定 FDR 前沿, 这条路彻底证伪。**

### 4.2 轻证据五次失败(C3/C6/C7/F1 + edge)

- FPN 全局平均证据: 判别力 |ΔTP−FP_BG| 最大仅 0.215, 而 y5_score 差 300 倍;
- fg_logit: 判别力有(TP 1.42 vs FP_BG −0.74), 但与 y5_score 相关 0.44, 被覆盖;
- **结论: 单框/轻证据的判别力已被 OER 用尽, 加特征不突破饱和。**

### 4.3 OER 饱和的本质

14 特征(y5_score + crop 证据 + 几何 + d4 + has_oto)的 HistGB 已经吸收了:
- 对象有效性(foreground/背景)
- 细类证据(crop)
- 背景反证(d4 + oto)
- 几何/密度

任何"和 y5_score/crop 相关的轻证据"都无法再提供独立信息。突破需要:
1. **结构级证据**(边界闭合/结构延续——检测头没显式建模的, 需空间 grid FPN);
2. **集合级排序**(DeepSets/Set Transformer——建模候选间关系, 非单框);
3. **新识别头**(属性组合分类——改变细类决策的机制, 非加特征)。

## 五、下一步方向(按 ROI 排序)

| 方向 | 针对 | 成本 | 预期 |
|---|---|---|---|
| **F2-F5 属性识别** | FP_CLS 841(兄弟机型混淆) | 大(属性字典+训练) | 中(混淆矩阵清楚) |
| 真 DeepSets listwise | 排序损失 322 | 中(需 torch) | 中(C7 加权已证伪, 真集合级未试) |
| 结构建模(空间 grid) | FP_BG 1899 | 大(重提取+训练) | 中(方案6.2 核心) |
| 收尾(0.9616/0.9405) | — | 低 | 当前已超官方门槛 |

## 六、产物清单

- 脚本: c3_counterfactual_field.py / c6_hard_confounder.py / c7_oer_v2_listwise.py /
  v1_vehicle_diag.py / f1_fg_rejector_manifest.py / f1_fg_rejector_infer.py / f1_verify_oer.py /
  p0_4_class_domain_frontier.py
- 数据: fpn-feats(3200维三折)/ f1-fg-logits.json / V1-fold0-preds.json
- 报告: HERA_FIELD_P0 / HERA_FIELD_V1 / HERA_FIELD_F1 三份
- 权重: V1-VEHICLE-SEED-FOLD0-40EP / F1-FG-REJECTOR(三折)
