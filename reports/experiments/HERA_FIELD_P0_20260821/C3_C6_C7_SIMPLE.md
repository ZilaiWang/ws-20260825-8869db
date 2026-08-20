# HERA-Field 批次1: C3/C6/C7 简单版证据 —— 边际/负, 方法论定位(2026-08-21)

> 方案5 §六/§七/§十一 的"轻量"先行验证。

## 结果汇总(全量 OOF 三折 cross-fit, oracle 改类)

| 实验 | 方法 | R@FDR=.12 | Δ |
|---|---:|---:|---:|
| baseline | OER 14特征 | 0.9616 | — |
| C3 反事实 | +28维 FPN 差分标量 | 0.9623 | +0.0006 |
| C6 硬背景 | +PCA原型 cos 相似度 | 0.9623 | +0.0000 |
| C7 listwise | hard-pair sample_weight 加权 | 0.9609 | −0.0008 |

## 判别力诊断(为什么无效)

1. **FPN 全局平均证据判别力极弱**: 反事实标量 |ΔTP−FP_BG| 最大仅 0.215(log 尺度),
   而 y5_score 的判别力是 TP 0.908 vs FP_BG 0.003(差 300 倍);
2. **FPN 特征无独立信息**: FPN 是 Y5 检测器的中间特征, 分类头已用其分类,
   spatial-average 后的信息已被 y5_score/crop 证据覆盖;
3. **硬背景原型 cos 相似度太弱**: TP 均值 0.006 vs FP_BG 0.06(差异 0.054),
   高混淆(>0.5)比例几乎为 0。

## 关键方法论结论

**14 特征的 OER 已接近饱和, "轻"证据拼接(FPN 全局平均/加权/原型)无法带来增益。**

这印证方案5 的核心判断, 同时精确化了实现要求:
- C3 反事实要真正有效, 必须建模**结构延续/边界闭合**(方案6.2)——需要空间 grid
  的 FPN 特征(中心 vs 边界对比), 而非全局平均;
- C6 硬背景要有效, 必须用**训练 foreground rejector**(F1, 开放拒绝)或独立教师,
  而非 KMeans + cos 相似度;
- C7 要有效, 必须上**DeepSets/Set Transformer 集合排序**, 而非样本加权。

## 结论: 增益来自"重"改动, 不是"轻"拼接

三大增长空间(FP_BG 1899 / 排序损失 322 / vehicle 无候选 19%)都需要:
1. **vehicle 0.7264(最大短板)→ V1 独立车辆种子头(GPU 训练)**;
2. **FP_BG 1899 → F1 foreground rejector(GPU 训练 crop 开放拒绝)**;
3. **排序损失 322 → 结构建模 / 真正的 listwise**。

下一步优先: V1 车辆种子(vehicle 是唯一工作点 <0.9 的粗类)。

## 产物

- scripts/c3_counterfactual_field.py / c6_hard_confounder.py / c7_oer_v2_listwise.py
- outputs/Y5-OER-RESTORE/fpn-feats/(三折 FPN 特征, 3200维, 已持久化)
- outputs/Y5-OER-RESTORE/c3-counterfactual.json / c6-confounder.json / c7-listwise.json
