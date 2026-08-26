# SCOPE Gate 2/3/5 判断 + 方案5剩余实验收尾（2026-08-26）

## 一、SCOPE 候选级动作：结构性天花板已彻底定位

### 核心结论链（3 轮诊断，30+ 实验）

1. **oracle 改类收益 97% 来自「减 FP」**（+1.86pp 三折），机制是 DROP 掉「位置对类别错」的
   冗余候选（改类后被同 image 同类候选 NMS 抑制 → FP 9829→6296，TP 几乎不变）。
2. **pairwise Δ_ij 证伪**（1350 对非零 0.4% 且全负）——方案6 Gate 2 的 U4 关系建模不成立。
3. **收益全部来自 OER≥0.1 的高分 fn_wrong（3176 个）**，OER<0.1 的 7222 个贡献为 0。
4. **broken 的本质 = crop 分类器误差**：tp 里 5.5% 候选 crop_top1_class 判错，被误判为
   fn_wrong，drop 掉即损失高分 TP。
5. **frontier 非对称**：broken（OER≈0.98 真实 TP）破坏力 >> 减 FP 收益，任何门控
   （trust/y5/OER/crop_top1/同类抑制）都只能恢复 7-12%。

### 验证过的路径（全负/边际）

| 路径 | 结果 |
|---|---|
| fine_correct 标签替代 is_valid | −0.0043（贪心匹配非唯一）|
| 加 disagree_x_crop 交互特征 | +0.0000（特征已用尽）|
| trust_label 路由器 + 各门控 | 恢复 7-12% |
| pairwise interaction（Gate 2 U4）| 证伪 |

### 结论
- **is_valid（贪心匹配 TP）是 OER 唯一正确标签**，fine_correct 因贪心非唯一不可用。
- **deploy 改类/DROP 天花板 = crop 分类器精度**，关系建模无法突破，不上 torch/GPU。
- **base 0.9421 已接近安全 deploy 上限**。

## 二、Gate 5 判断：需要更强 teacher，项目里没有现成的

- P03 ConvNeXt-tiny 已是当前 crop 特征来源（最强类别证据，config 含 224/336 分辨率）。
- R1-0-P03-TEACHER-M1-OOF 用的是 **M1 detector** 的 proposal（`m1-f0-i1-...`），当前项目是
  Y5 detector（`y5-f0-i1-...`），proposal 不同，logits 不能直接复用。
- 结论：Gate 5 需要先训练更强分类器（更大模型 / 336 分辨率），是 GPU 任务，且只能边际
  改善 crop 精度（5.5%→~4%），无法根本改变 frontier 非对称的结构性矛盾。

## 三、方案5 剩余实验评估（L2 双折 frontier@FDR0.12）

| 实验 | fold0(最难) | fold1(counter) | L2 判定 |
|---|---:|---:|---|
| Y5 基线 | 0.8871 | 0.9478 | — |
| F4 可观测性掩码 | −0.0103 | +0.0025 | ❌ 不同向 |
| F6 尾类重加权 | +0.0029 | +0.0001 | ❌ 持平 |
| V2 车辆中心-周围 | −0.0080 | −0.0047 | ❌ 负 |
| D3 worst-group 采样 | +0.0763 ⚠️ | +0.0189 ⚠️ | ⚠️ 数据泄漏 |
| D4 worst-group loss | +0.0785 ⚠️ | +0.0262 ⚠️ | ⚠️ 数据泄漏 |

### D3/D4 数据泄漏（本轮的 bug 发现）

- **根因**：`d3_worst_group_curriculum.py` 用**全量 formal.boxes（含三折 val）**诊断 worst-group，
  映射 hard_images 时未区分 fold；`train_cv3_oof.py` 的 `build_dataset_yaml` 里 `id_to_rel`
  含 train+val 全部样本，`hard_rel` 从它映射，**val 图被加进训练集**。
- **后果**：D3/D4 frontier 假性 +7.6pp。
- **修复**：`hard_rel` 只从 `split=="train"` 的图映射（1410→904 张，排除 506 张泄漏 val 图）。
- **状态**：已提交修复，D3/D4 并行重跑中（GPU 98%）。

## 四、当前服务器状态

```
D3 fold0 🔄(重跑, 1/40)   D4 fold0 🔄(重跑, 1/40)   并行, GPU 98%
→ 各两折约 3.5h
```

## 五、下一步建议

方案6 全部 Gates 的判断：
- **Gate 0/1** ✅ 完成（底座冻结 0.9421 + 动作价值可学习 AUC 0.98）
- **Gate 2** ❌ 证伪（pairwise 无协同）
- **Gate 3** ❌ 天花板（crop 精度 + frontier 非对称）
- **Gate 4** ⏳ 待 D3/D4 重跑结果（但域平衡在 SCOPE 框架下意义有限）
- **Gate 5** ⚠️ 需要先训更强 teacher，边际收益有限
- **Gate 6** 最终集成

**核心判断**：SCOPE 的「候选级动作」路线在 deploy 口径下已触达天花板（0.9421），
唯一持续正向的是「集合级证据」（F5 集合上下文 oracle +0.26pp / deploy +0.06pp）。
建议把资源转向：**F5 集合上下文的深挖 + SCOPE 减 FP 机制的整合**，而非继续投入
Gate 2/3/5 的重网络/重 teacher。
