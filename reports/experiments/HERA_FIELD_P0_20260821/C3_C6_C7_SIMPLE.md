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

---

## 追加: 深挖集合级证据(C7-lite 正向 / DeepSets 负, 2026-08-21 02:00-03:00)

### C7-lite 手工集合上下文 —— ✅ 唯一正向(+0.26pp)

| 特征数 | R@FDR=.12 | Δ |
|---|---:|---:|
| baseline(14特征) | 0.9616 | — |
| +8 同图邻居统计 | 0.9635 | +0.19pp |
| +11(空间邻居) | 0.9638 | +0.21pp |
| **+15(局部密度/相对面积/top3)** | **0.9642** | **+0.26pp** |

R@FDR=.11 更明显: 0.9591 → 0.9623(+0.32pp)。

**结论**: 集合级证据(候选间关系)是 OER 唯一缺失的维度, 手工统计有效且
趋势随特征丰富度持续上涨。

### 真 DeepSets(Set-Attention)—— ❌ 训练退化

- Set-Attention(点积注意力聚合邻居 + MLP), 三折 cross-fit, 30 epoch;
- 纯排序质量: R@FDR=.12 = 0.1704(vs y5_score 0.9559, −0.79);
- 分数分布退化: p25=0.449 中位=0.449 p75=0.451(大量候选挤在 0.45, 无区分度);
- **失败原因**: 简单 Set-Attention + 30 epoch 训练不充分(BCE loss 0.21 不下降),
  小集合(图内 1-2 候选)attention 无意义, class-balance 权重倾向输出常数;
- **结论**: 学习性集合聚合需要更好的设计(更多 epoch/更好初始化/处理小集合),
  手工统计(C7-lite)当前更可靠。

### F-lite 家族感知改类 —— ❌ 无判别

- crop 改类正确率: 家族一致 0.789 vs 不同 0.795(几乎无区分);
- 家族标签不能区分"该改/不该改", F2-F5 需真正的属性组合分类器。

### C4 结构证据(空间 grid FPN)—— ❌ 无效

- closure 判别力 0.56(远弱于 y5_score 300x), OER +0.0002;
- 根因: Y5 backbone/head 已用 FPN 特征, 空间 grid 无独立信息。
