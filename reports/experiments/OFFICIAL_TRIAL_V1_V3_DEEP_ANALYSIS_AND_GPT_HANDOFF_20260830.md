# 官方预测评 trial-v1～v3 深度分析、实验总账与下一阶段讨论材料

更新日期：2026-08-30
状态：`official_trial_v3_analyzed_v2_remains_incumbent`
用途：项目内部决策与提交给 GPT 的下一阶段讨论材料
证据等级：官方平台截图 + 冻结部署配置 + 正式 OOF/来源互斥实验账本

---

## 0. 一页结论

1. `trial-v3.0` 不是 `trial-v2.0` 的全面升级。综合分从 **86.2274 降到
   85.0018（-1.2256）**，当前最优提交仍是 `trial-v2.0`。
2. v3 相对 v2 的正向项是：船 Recall +1.1327pp、飞机 Recall +0.0754pp、飞机
   FDR -0.5069pp、车辆 FDR -8.1588pp；负向项是：船 FDR +3.8707pp、车辆
   Recall -4.0269pp、时延 +80.74%。这是明显的 Pareto 交换，不满足“所有类都提升”。
3. v3 同时改变了推理视图和三粗类阈值，因此一次官方 A/B **不能**把变化归因于某一个
   因素。可以确认的是整套策略对车辆过于保守、对船舶增加的候选不够干净。
4. 内部“全量权重 + 精确 Docker 配置 + trial-mix + V1.6 macro”部署审计有效：它准确
   预测了三类变化方向，并几乎精确预测船 FDR 的恶化和车辆 FDR 的改善。但它只预测
   车辆 Recall 下降约 0.54pp，官方实际下降 4.03pp，说明隐藏集车辆分数分布比同源
   回看更低，`vehicle=0.366` 对域偏移不稳健。
5. 后续固定保留两类内部测评：
   - `Normal-CV3`：防止正常识别、定位和尾类能力退化；
   - `Hard10K-CV3 + source-disjoint sentinel`：筛复杂背景、风险排序和域迁移方向。
   全量部署回看只作提交前配置审计，不再用于反复选方法和阈值。
6. 达到综合分 93 不能靠继续扫描阈值。官方综合分依赖七项相对排名，没有公开可由单队
   指标直接复算的公式。最可靠的工程目标是先 Pareto 支配 v2：船和车辆同时提高
   Recall、降低 FDR，飞机保持接近饱和，并把时延控制在合理区间。
7. 方案 7～9 的优先验证器主线已执行较完整，但**并非所有方法都已尝试**。尚未正式执行
   且最有可能形成新信息的方向是：
   - 外部遥感数据的粗类/objectness 预训练后再做官方 25 类微调；
   - DEIM/D-FINE 或现有 M3 的正式快速异构筛选；
   - 只对不确定图块/候选启用的稀疏、按类别路由 TTA；
   - 车辆/船舶专用的质量与定位头，而不是再加一个 tight-crop 前景标量。
8. 下一次官方提交前必须先在内部解耦“第二视图”和“高阈值”。目前没有证据支持直接
   提交另一个任意阈值版本。v2 继续作为安全回退，v3 作为重要负向/域校准证据。

---

## 1. 官方规则与综合分应如何理解

### 1.1 刚性门槛

官方 V1.6 初赛门槛是：

- 三类合并 pooled Recall ≥ 0.85；
- 三类合并 pooled FDR ≤ 0.20；
- RTX 3090 或同等算力上单幅 10000×10000 推理 ≤ 20 秒。

平台页面同时展示船、飞机、车辆各自的 Recall/FDR 和平均时延。细类必须一致才算 TP，
车辆 IoU 阈值为 0.35，船/飞机为 0.50。

### 1.2 七项排名而非一个可微标量

V1.6 的正式相对排名由七项组成：

1. ship Recall；
2. ship FDR；
3. aircraft Recall；
4. aircraft FDR；
5. vehicle Recall；
6. vehicle FDR；
7. inference time。

各项先在全部队伍中排名，再把七个名次求和后二次排序。因此：

- `86.2274` 和 `85.0018` 不是准确率、mAP 或六指标简单平均；
- 其他队伍继续提交时，同一镜像的动态名次和百分位也可能变化；
- 不能从我们单队七个数精确计算“怎样一定得到 93 分”；
- 同时改善多个弱项通常比把已饱和的飞机 Recall 从 0.999 提到 1.000 更有排名价值。

### 1.3 本轮优化准则

“综合分达到 93”保留为目标，但不用虚构公式倒推。内部准入改用可验证的 Pareto 条件：

- ship Recall 不低于 0.95，FDR 目标不高于 0.12～0.13；
- aircraft Recall 保持 ≥0.995，FDR 不高于 0.03；
- vehicle Recall 不低于 0.95，FDR 优先压到 ≤0.15～0.18；
- 端到端时延优先控制在 4 秒附近，结构性增益足够大时可接受更高，但不以“低于 20 秒”
  作为无限增加计算量的理由。

这些是下一轮工程目标，不是对隐藏集结果的保证。

### 1.4 平台展示值与 V1.6 macro 的口径边界

平台字段名只有 `ship.recall`、`ship.false_detection_rate` 等，没有在页面公开其聚合代码。
trial-v1 的六个小数可被整数 TP/FP/FN 精确还原，强烈提示页面显示的是各粗类 pooled
计数；V1.6 文档用于七项排名的定义则是粗类内细类简单平均。当前报告因此遵守：

- “官方 v1/v2/v3”表格只称为**平台展示粗类指标**；
- 内部 V1.6 macro 只用于经验校准，不能宣称与平台显示聚合完全同构；
- 正式实验继续同时输出 pooled 与 25 细类 macro 两套账本；
- 不用三粗类简单平均替代官方刚性 pooled 门槛，也不用 pooled 替代七项排名口径。

---

## 2. 三次官方提交的完整记录

### 2.1 官方指标

| 标签 | 综合分 | 船 R/FDR | 飞机 R/FDR | 车辆 R/FDR | 平均时延 |
|---|---:|---:|---:|---:|---:|
| trial-v1.0 | 67.0171 | 0.845739 / 0.177335 | 0.806938 / 0.209749 | 0.510067 / 0.419847 | 2.331833s |
| **trial-v2.0** | **86.2274** | **0.942287 / 0.126937** | **0.999246 / 0.024300** | **0.946309 / 0.237838** | **2.704833s** |
| trial-v3.0 | 85.0018 | 0.953614 / 0.165644 | 1.000000 / 0.019231 | 0.906040 / 0.156250 | 4.888833s |

截图时 v2 为第 33 名并保持最佳标签；v3 未取代 v2。

### 2.2 v3 相对 v2 的精确变化

| 项目 | v2 | v3 | v3-v2 | 方向 |
|---|---:|---:|---:|---|
| 综合分 | 86.2274 | 85.0018 | **-1.2256** | 负 |
| ship Recall | 0.942287 | 0.953614 | **+1.1327pp** | 正 |
| ship FDR | 0.126937 | 0.165644 | **+3.8707pp** | 负 |
| aircraft Recall | 0.999246 | 1.000000 | +0.0754pp | 小幅正 |
| aircraft FDR | 0.024300 | 0.019231 | -0.5069pp | 正 |
| vehicle Recall | 0.946309 | 0.906040 | **-4.0269pp** | 明显负 |
| vehicle FDR | 0.237838 | 0.156250 | **-8.1588pp** | 明显正 |
| 平均时延 | 2.704833s | 4.888833s | +2.184000s / **+80.74%** | 负 |

### 2.3 结果的科学含义

v3 改善了六个精度指标中的四个，但恶化的三项（含时延）恰好都是重要排名风险：

- 车辆 Recall 一次损失 4.03pp，超过车辆 FDR 改善所能安全解释的范围；
- 船舶增加的候选使 Recall 上升，但 FP 增长更快，FDR 恶化 3.87pp；
- 双视图把时延提高 80.74%，直接损失第七项排名。

因此，不能用“前两个类别提升，车辆 FDR 提升”概括 v3；准确结论是：

> 双视图 + 高车辆阈值把系统从高召回高车辆 FDR，移动到了较低车辆 Recall、较低车辆
> FDR 的另一个工作点，同时让船舶和时延变差。它扩展了我们对隐藏域 PR 曲线的认识，
> 但没有形成更优部署策略。

---

## 3. v2 与 v3 的版本身份

### 3.1 trial-v2.0

- 模型：Y5-S（YOLO26-s 架构链）；
- 训练：全部 4,481 张官方训练图、160 fixed epochs；
- 权重：`last.pt`，SHA256
  `f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229`；
- 推理：单视图 identity；
- 工作点：三粗类统一阈值 0.15；
- 平台平均时延：2.704833 秒。

### 3.2 trial-v3.0

- 使用与 v2 **完全相同的全量 Y5-S 权重**；
- 推理视图：identity + 90°；
- 融合：细类内 safe NMS；
- 冻结阈值：ship=0.150、aircraft=0.301、vehicle=0.366；
- 本地镜像：`xh-detector:y5s-rot90tta-trial-v2-calibrated-v1`；
- 镜像 digest：
  `sha256:734a3c0ba755ecd1ec5ce3737124ddfc4f91d8c403a4f13c2d1540bded1d3fe8`；
- 平台平均时延：4.888833 秒。

### 3.3 官方 A/B 的可解释边界

v2→v3 同时改变了：

1. 是否运行第二个旋转视图；
2. aircraft 阈值；
3. vehicle 阈值。

所以这次官方结果只能判断“整套 v3 策略”不如 v2，不能严格回答：

- 如果只增加第二视图、仍统一 0.15，会怎样；
- 如果仍单视图、只改变车辆阈值，会怎样；
- 船 FDR 恶化中有多少来自第二视图新增 FP；
- 车辆 Recall 损失中有多少来自阈值、多少来自旋转融合/NMS。

下一轮内部实验必须先解耦，不能继续把两个变量绑定后提交。

---

## 4. 内部部署审计与官方结果的校准

### 4.1 trial-v2 的提交前预测

全量权重、精确部署配置在 trial-mix 上按官方 V1.6 粗类宏平均复算：

| 粗类 | 内部 v2 R/FDR | 官方 v2 R/FDR | 官方-内部 |
|---|---:|---:|---:|
| ship | 0.948936 / 0.142899 | 0.942287 / 0.126937 | -0.6649pp / -1.5962pp |
| aircraft | 0.997373 / 0.078783 | 0.999246 / 0.024300 | +0.1873pp / -5.4483pp |
| vehicle | 0.945652 / 0.236842 | 0.946309 / 0.237838 | +0.0657pp / +0.0996pp |

结论：三类 Recall 和车辆 FDR 非常接近；船/飞机 FDR 的内部估计更悲观。

### 4.2 trial-v3 的提交前预测

同一全量模型部署回看得到：

| 粗类 | 内部 v3 R/FDR | 官方 v3 R/FDR | 官方-内部 |
|---|---:|---:|---:|
| ship | 0.953260 / 0.179614 | 0.953614 / 0.165644 | +0.0354pp / -1.3970pp |
| aircraft | 0.998077 / 0.058559 | 1.000000 / 0.019231 | +0.1923pp / -3.9328pp |
| vehicle | 0.940217 / 0.135000 | 0.906040 / 0.156250 | **-3.4177pp / +2.1250pp** |

注：v3 的这组三粗类内部值是 V1.6 macro 部署审计口径。项目早期报告中也有 pooled
风险代理值，两者不可互换。

### 4.3 对“内测是否有效”的回答

把 v3-v2 的变化量对比：

| 粗类/指标 | 内部预测变化 | 官方变化 | 判断 |
|---|---:|---:|---|
| ship Recall | +0.4324pp | +1.1327pp | 方向对，低估收益 |
| ship FDR | +3.6715pp | +3.8707pp | **高度准确** |
| aircraft Recall | +0.0704pp | +0.0754pp | **高度准确** |
| aircraft FDR | -2.0224pp | -0.5069pp | 方向对，内部更乐观 |
| vehicle Recall | -0.5435pp | -4.0269pp | 方向对，**严重低估损失** |
| vehicle FDR | -10.1842pp | -8.1588pp | 方向对，幅度接近 |

结论不是“内测无效”，而是：

- **方法方向筛选有效**：六个变化方向全部正确；
- **船 FDR、飞机 Recall、车辆 FDR 具备较好定量预测力**；
- **高车辆阈值下的 Recall 绝对值不可靠**，同源全量回看存在明显乐观偏差；
- Hard10K 的绝对 Recall 偏悲观，但能揭示“车辆阈值/排序跨域不稳定”；
- source-disjoint sentinel 是防止只在六张开发拼图上过拟合的必要账本。

### 4.4 必须修正的评估纪律

1. Normal-CV3 和 Hard10K-CV3 保持冻结，继续用于方法选择；
2. source-disjoint sentinel 只用于冻结候选的方向复验，不反向调参；
3. 全量部署回看只检查镜像、工作点、分数尺度，不把其 Recall 当独立泛化结果；
4. 任何把 vehicle 阈值明显提高到 0.15 以上的候选，都必须增加域偏移压力项：
   `min(内部回看 Recall, OOF/Hard 下保守 Recall)`，不能只看同源回看；
5. 下一次官方提交只允许一次已经在上述账本中解耦变量的冻结候选。

---

## 5. 当前问题的错误分解

### 5.1 飞机不是主要瓶颈

- v2 aircraft Recall 已为 0.999246，FDR 0.024300；
- v3 只增加约一个千分点以内的 Recall，综合分仍下降；
- 继续为飞机全量增加第二视图，边际排名收益很小，却消耗约一倍推理成本。

飞机后续任务是“保持”，而不是继续全局加算力。

### 5.2 船舶是候选召回和背景排序的双重问题

- v3 第二视图使 ship Recall +1.13pp，说明旋转视图确实补到真目标；
- ship FDR 同时 +3.87pp，说明新增候选的精度不足；
- 历史 Hard10K 分解中，ship 的高分背景占主体，单一 crop verifier、DINO 前景概率、
  coarse binary 只能带来小幅或负向收益；
- 因此船舶不应简单关闭第二视图，也不应全量无条件融合第二视图。合理形式是：只在
  第一视图证据不足的图块/位置使用第二视图，并要求跨视图几何/类别支持。

### 5.3 车辆是工作点跨域漂移和小目标能力共同作用

- v2 vehicle 0.946309/0.237838：召回强，但虚警偏高；
- v3 vehicle 0.906040/0.156250：虚警明显改善，但损失约 4.03pp Recall；
- 内部只预计损失 0.54pp，说明同源 trial-mix 上车辆真目标分数显著高于隐藏域；
- 旧方案已观察到车辆无候选、小目标、机场/城市背景混淆和不同来源域漂移；
- 车辆不能再用单一高阈值解决，必须提高真目标分数/质量排序，或通过小目标专家补回
  候选，再在较低阈值下控制 FP。

### 5.4 时延不是硬门槛问题，但已成为排名问题

- v2 2.70 秒，v3 4.89 秒，都远低于 20 秒；
- 但第七项本身参与排名，v3 用 80.7% 额外计算只换来一个不优的 Pareto 点；
- “仍低于 20 秒”不足以证明全量 TTA 合理；
- 新计算只应用于不确定图块/候选，才能形成更好的精度—时延前沿。

---

## 6. 历史实验结果总账

本节只列对当前决策有作用的结果。旧结果若后来发现评估合同错误，会明确标记，禁止与
当前官方匹配结果直接比较。

### 6.1 评估协议与数据分组

| 路线 | 结果 | 当前状态 |
|---|---|---|
| formal CV3 / source-group OOF | 4481 图、完整 OOF、机场代理组和来源隔离 | 正式保留 |
| 官方 prediction-first 一对一匹配 | 修复 tie block、细类匹配、工作点角色 | 正式唯一 scorer |
| corrected-OER | R@FDR=.12 = 0.943104；R@.10 = 0.936655 | 可信旧基线 |
| 旧 OER 约 0.9620 | 924 TP/FP 身份互换，共 1848 候选受错误合同影响 | **作废** |
| trial-mix Hard10K | 6 张 pseudo-10K、2158 GT | 压力测试，不作隐藏分预测 |
| source-disjoint sentinel | 600 新来源、1969 GT，与开发来源互斥 | 方向复验保留 |

### 6.2 检测器与训练主干

| 方法 | 关键结果 | 决策 |
|---|---|---|
| M1 临时 fold0 | 官方 v1 分 67.0171，明显跨域不足 | 仅工程历史 |
| Y5-S full 单视图 | 官方 v2 分 86.2274；当前最佳 | **部署 incumbent** |
| Y5-L full | 同合同 FDR≈.15 为 0.955514/0.148989，低于 S 的 0.961075/0.145095；更慢 | 否决 |
| background-complete Y5 | candidate Recall 0.8679，固定风险 0.7210/0.1511 | 否决 |
| 低强度 hard replay | candidate-floor -7.18pp，R@.15 -0.79pp | 否决 |
| 3-coarse detector + P03 | candidate 0.8684，R@.15 0.7836 | 否决 |
| Y3 作为候选补充 | 四源+Y3 candidate 0.9727，未超过四源 0.9731 | 否决 |
| M3 RT-DETR-L | 已完成候选资产；提供异构候选，但简单融合排序差 | 保留异构研究资产 |
| DEIM/D-FINE | 文档建议，未完成正式快筛/全训练 | **未执行** |

### 6.3 视图、尺度和候选生成

| 方法 | 关键结果 | 决策 |
|---|---|---|
| Y5-S identity+90° | 内部候选地板和固定风险方向正；官方船 R 正、车辆 FDR 正，但车辆 R/船 FDR/时延负 | 有条件保留，不全量准入 |
| Y5-800 + M3 + 旋转候选 | Hard10K 候选上限约 0.9713 | 候选有价值，排序未解决 |
| 四源候选 | 开发 candidate oracle 0.9731；sentinel 0.98172 | 候选上限较高 |
| SAHI/P2/简单多尺度 | 正式筛选负向或无增益 | 停止 |
| coarse NMS | vehicle 可改善但 ship 损失；全局无净增 | 不独立部署 |
| 稀疏按需 TTA | 尚未按“只处理不确定图块”正式实现 | **未执行** |
| sparse recenter | 代码有窗口原语，因上游 E4 负向未正式接入 | 条件未满足 |

### 6.4 标量校准、层级阈值和风险头

| 方法 | 关键结果 | 决策 |
|---|---|---|
| 全局/粗类阈值 | 同集上界也不足以跨越 Hard 缺口 | 不再大网格扫描 |
| 25 细类层级阈值 | Normal R -1.75pp；macro R 0.910099→0.876519 | 否决 |
| E2 residual BCE | +3 TP/+4 FP；0.863299→0.864690 | 未过门禁 |
| RankNet | TP 不增、FP -4（粗网格） | 不独立准入 |
| soft-FDR / one-winner | 无增益或负向 | 停止 |
| multi-source support | 0.8156/0.1423，显著低于基线 | 否决 |
| PAV V1 score-only | corrected-OER +0.1385pp，三折正但低于 +0.2pp 门禁 | 弱正向消融 |
| PAV V2 | active-FP AP 0.2155；最佳 +0.0408pp | 停止 |
| MAR proxy | +0.0430pp，但六项最差 -1.1037pp | 停止 |

### 6.5 像素验证与教师特征

| 方法 | 关键结果 | 决策 |
|---|---|---|
| ConvNeXt coarse binary | identity 0.8624/0.1467，仅约 +0.23pp | 方向小，不足 |
| score-sqrt hard negative | 9 个 checkpoint 训练完成；正式 eval 运行 64 分钟后为让出 full audit 预算而停止 | **无科学结论** |
| E1 clean coarse manifest | 381 条 cross-coarse 污染已修正；服务器状态为 `superseded_by_formal_dual_view_20260830`，没有唯一变量重训结果 | **未执行重训** |
| DINOv2-B open-set/foreground | foreground AUC≈0.97，但风险头 0.8411/0.1467 | 否决集成 |
| CleanDIFT | P04 probe 低于 DINO/ConvNeXt，且重计算成本高 | 不进入当前链 |
| natural tight verifier | 0.8582/0.1493，未超过旧点 | 否决 |
| balanced vehicle expert | vehicle 局部改善，overall 小幅下降 | 只作互补证据 |
| E3 7-channel dual-view | formal CV3 R -1.285pp；ship -6.60pp、vehicle -12.44pp | 否决 |
| E4 VOI | 32/64/128/256 budget 均少 TP | 否决 |
| FPN context/core/ring/scene | 诊断 AUC 有价值，加入 OER 全部负 | 不再轻拼接 |

### 6.6 数据与域泛化

| 方法 | 关键结果 | 决策 |
|---|---|---|
| 机场代理 60 组 + formal CV3 | 已解决明显场景泄漏和折分稳定性 | 正式保留 |
| D1 source-domain 聚类 | 12 个高误检域贡献约 30% FP_BG；20 个低召回域 R≈0.382 | 重要诊断 |
| D3/D4 旧 hard/worst-group | curriculum 使用全量折信息且混合因素 | 无正式资格 |
| 干净 nested domain-balanced/worst-group | 未按每个 outer-train 重建 | **未执行** |
| FAIR1M/RarePlanes/xView/AI-TOD 预训练 | 只完成方案论证，未完成当前模型的正式训练 | **未执行** |

### 6.7 旧方案 5 中尚未完成的专项

以下项目不能写成“已经证伪”，因为尚未完成等价正式实验：

- F2 family→attribute→fine 属性组合；
- F3 counterfactual attribute negatives；
- F4 observability mask；
- F6 tail prototype residual；
- V2 中心—周围残差/频域增强；
- V3 车辆支持面 gating；
- V4 density-conditioned top-K；
- V5 vehicle q_match@IoU=.35；
- V6 DFD→seed head distillation；
- D2～D6 的干净嵌套域泛化/域专家；
- 多分支蒸馏、重参数化和 TensorRT。

其中一部分概念已被后续 PAV/E3/VOI/背景实验覆盖并显示弱信号，但不是逐项等价实现。
后续应按“能否提供旧系统没有的新信息”重新排序，而不是机械补齐所有旧 ID。

---

## 7. “方案 7～9 是否已经全部尝试”的正式回答

### 7.1 已充分执行并可收尾的内容

- 官方 scorer、formal OOF、来源隔离和固定风险前沿；
- PAV V1/V2、MAR 代理和 corrected-OER 收尾；
- 单 Y5、Y5-L、Y3、M3 候选、四源候选；
- 单/双视图部署候选及一次官方 A/B；
- 全局、粗类、细类层级阈值；
- 粗类 detector、coarse binary、DINO/CleanDIFT 诊断；
- hard background、background-complete、低强度 replay；
- E0 标签审计、E2 风险头、E3 七通道、E4 VOI；
- NMS、简单多尺度、SAHI/P2、轻量上下文特征。

这些方向已经足以支持结论：继续堆 tight-crop 标量、阈值、NMS 或普通背景负样本，不能
让系统从 86 分跃迁到 93 分。

E1 的干净标签清单已经生成，但没有与旧 coarse-binary 完全同参数的三折 clean retrain
最终 readout，因此不能把 E1 写成已证伪。score-sqrt 也只能写成“训练完成、评估未完成”，
不能从缺失 readout 推断正负。二者可以作为低成本补账项，但即使达到此前 coarse-binary
的小幅增益，也不应阻塞 B1/B2 的新模型路线。

### 7.2 尚未执行或没有形成正式结论的内容

1. 外部遥感数据 objectness/粗类预训练；
2. DEIM/D-FINE-L 的正式单折快筛和异构替换；
3. 基于缓存双视图输出的稀疏、类别条件 TTA；
4. 车辆/船舶专用 quality/IoU/centerness 头；
5. 干净 nested domain-balanced/worst-group 训练；
6. 方案 5 的属性组合与车辆专项系列；
7. 强模型训练后再进行知识蒸馏/重参数化部署。

因此正确表述是：

> 现有检测器上的常规后处理与轻验证器空间已经搜索得比较充分；真正未开发的是“更强的
> 遥感预训练/异构检测器”和“只在弱类临界位置增加新信息”的模型级空间。

---

## 8. 为什么当前方法难以达到 93

### 8.1 v2 已不是简单漏调阈值

v2 车辆已经处于 0.946 R / 0.238 FDR。若单纯提高阈值，v3 证明会把 Recall 拉到
0.906；若继续使用低阈值，车辆 FDR 又是七项中的明显弱项。缺少的是让车辆 TP 分数
高于背景 FP 的新表征，而不是找到另一个神奇阈值。

### 8.2 v3 的额外计算没有被选择性利用

第二视图对船候选有帮助，但对所有图块完整执行：

- 为已饱和飞机支付了重复成本；
- 把船背景一起带入；
- 然后又靠高车辆阈值一次性删除不确定车辆。

更合理的流程是先用单视图快速判断，只对以下区域运行第二视图/专家：

- score 接近工作点；
- 小目标、tile 边界、旋转敏感；
- 第一视图与上下文/类别证据冲突；
- 船/车辆候选密集或疑似背景纹理区域。

### 8.3 旧验证器一直在重复同类信息

PAV、coarse binary、DINO foreground、E3 等虽结构不同，但大多仍是对当前 proposal crop
做前景判断。它们与 detector score 高相关，无法稳定恢复隐藏域低分 TP。下一轮若仍做
验证器，必须引入以下至少一种新信息：

- 外部遥感背景/尺度预训练；
- 不同检测架构的候选与定位分布；
- 跨视图几何一致性；
- 预测框质量/IoU 的端到端监督；
- 真实 source-domain 均衡训练。

---

## 9. 下一阶段建议实验树

### Phase A：先用已有缓存解耦 v3（低成本，必须先做）

#### A1. 四格因果对照

固定同一 Y5-S full 权重，使用缓存输出比较：

| 编号 | 视图 | 阈值 |
|---|---|---|
| A1-0 | identity | 0.15/0.15/0.15（v2） |
| A1-1 | identity+90° | 0.15/0.15/0.15 |
| A1-2 | identity | 0.15/0.301/候选 vehicle 阈值 |
| A1-3 | identity+90° | 0.15/0.301/同一 vehicle 阈值 |

只在 Normal-CV3、Hard10K-CV3、sentinel 上运行，不提交官方。目的：分离视图收益与阈值
损失。vehicle 阈值从训练折/官方两点约束下预注册一个，不做几十点 leaderboard 拟合。

#### A2. 隐藏域保守车辆工作点

v2 与 v3 提供了两个官方锚点：

- threshold 0.15：R 0.9463 / FDR 0.2378；
- threshold 0.366（伴随 TTA）：R 0.9060 / FDR 0.1563。

由于视图也变化，不能直接线性插值。内部先画单视图和双视图各自冻结 PR 曲线，并使用
`min(Normal, source-disjoint/hard conservative)` 选一个目标：vehicle R≥0.94、
FDR≤0.18。若没有任何点满足，阈值路线正式封顶，不再提交阈值版本。

#### A3. 类别条件稀疏 TTA 仿真

利用已经缓存的 identity/90°预测，不重新推理，模拟：

- aircraft 永远只用 identity；
- ship 只在 identity 低置信/边界候选上接受第二视图新增框；
- vehicle 的第二视图只用于补候选，不把所有第二视图背景纳入；
- 新增框必须满足跨视图同细类几何支持，或进入专门质量头；
- 输出每 10K 图需要第二视图的 tile 比例和理论时延。

准入：相对 v2，三粗类 Recall 均不下降超过 0.2pp，ship/vehicle FDR 同时下降，理论时延
显著低于 v3。失败即关闭现有 TTA 工程。

### Phase B：获得真正的新模型信息

#### B1. 外部遥感 objectness/粗类预训练（第一优先）

建议数据角色：

- FAIR1M：机场、港口、车辆、船舶和大规模旋转目标；
- RarePlanes：飞机结构、机场背景与飞机 objectness；
- xView/AI-TOD：小车辆、复杂城市背景和 tiny-object 表征；
- DOTA/DIOR：通用遥感物体与场景多样性。

训练原则：

1. 外部标签只映射到 objectness 或三粗类，不冒充官方 25 个军机/舰船细类；
2. 先外部域预训练 detector backbone/neck/objectness；
3. 再用官方 4,481 图完成 25 类微调；
4. 最后短程 source-balanced 微调，不使用隐藏/预测评标签；
5. fold0 快筛只看 Normal candidate floor、Hard fixed-risk、ship/vehicle 分项；
6. 通过后才扩三折和 full。

这是目前最可能同时改善车辆召回、船背景和跨域稳定性的方向。

#### B2. DEIM/D-FINE 或 M3 异构快速筛选（第二优先）

目标不是简单并入第五路候选，而是回答：它是否在 vehicle tiny object、ship 定位或隐藏
背景上提供 Y5 没有的正交错误。

快筛合同：

- 单 fold 或固定训练预算；
- 同一训练/验证来源；
- 输出 candidate oracle、R@FDR .10/.15/.20、三粗类、候选数和时延；
- 与 Y5-only、M3-only、Y5+异构分别比较；
- 至少在 ship/vehicle 中一类提升 candidate ≥1pp，且简单单调校准后 fixed-risk ≥0.5pp，
  才进入 full；
- 未达门禁不训练完整三折。

#### B3. 车辆/船舶质量头（第三优先）

若 B1/B2 提供更强候选，再训练质量头而不是旧 tight 前景分类器：

- 监督对象：official canonical TP、duplicate、wrong-fine、localization、background；
- 预测：objectness、fine class、IoU/quality、canonical winner 概率；
- 重点使用跨模型/跨视图一致性和框质量，而不是只输入 RGB crop；
- 输出是 detector logit 的小残差，保持原排序安全；
- 外层 cross-fit，禁止 held-out 阈值参与训练。

### Phase C：只为胜出候选做完整训练与 Docker

1. A 或 B 的候选先通过 Normal、Hard、sentinel；
2. 只保留一个主候选和一个安全回退；
3. 完整 4,481 图固定训练，不用全量回看选 epoch；
4. 冻结权重 SHA、配置、镜像 digest；
5. 真实 10K 端到端 GPU 复测；
6. 官方提交只用于最后外部校准，不用五次机会做网格搜索。

---

## 10. 建议的下一官方候选形态

### 10.1 当前立即决定

- **保留 v2 为最佳与回退镜像**；
- v3 不再作为主候选；
- 不立即提交“随手把 vehicle 阈值降一点”的 v4；
- 先完成 A1/A2 解耦，得到单视图和双视图各自的车辆 PR 曲线及域偏移下界。

### 10.2 如果必须在短期内构造候选

候选应满足：

- 同一 Y5-S full 权重；
- aircraft 维持单视图，避免为已饱和类别增加时延；
- ship 保留 v2 主分支，仅对不确定 tile 启用旋转补候选；
- vehicle 阈值低于 v3 的 0.366，但必须由 A2 的保守工作点确定；
- 第二视图新增框不能无条件进入，需要几何/类别支持；
- 先在两套固定基准和来源互斥 sentinel 上通过，再构建镜像。

这比再次提交一个全量双视图阈值版本更有因果信息，也更可能同时保住车辆 Recall 和时延。

---

## 11. 后续准入与停止条件

### 11.1 方法准入

候选相对 v2 或同协议内部基线必须同时满足：

1. Normal-CV3 Recall 下降 ≤0.3pp；
2. 任一粗类 Recall 下降 ≤0.5pp，车辆优先要求不下降；
3. Hard10K-CV3 在 FDR≤0.15 时 Recall +0.5pp；
4. source-disjoint sentinel 方向一致；
5. full 部署审计中 ship/vehicle 不出现新的单项退化；
6. 时延增量与精度增量成比例；
7. 只有一个因素变化，或有完整四格消融能够解释。

### 11.2 停止条件

- 只降低 FDR、但任一弱类 Recall 损失 >0.5pp：停止；
- 只提高候选 oracle、fixed-risk 不升：不进入 Docker；
- 只在 trial-mix 正向、sentinel 反向：判代理过拟合；
- 新 verifier 与 detector score 高相关且增益 <0.2pp：停止；
- 全量 TTA 时延翻倍但三类不能 Pareto 改善：改为稀疏路由或停止；
- 外部数据预训练破坏 25 细类 Normal macro：停止或加强官方细类微调；
- 快筛未过门禁：不扩三折，不用更多 epoch 追结果。

---

## 12. 给 GPT 讨论时需要重点回答的问题

1. 在只有 v2/v3 两个官方锚点、且 v3 同时改变 TTA 和阈值时，如何设计最少的内部对照
   来估计 vehicle 的隐藏域安全工作点？
2. 如何把双视图改成稀疏 tile/candidate routing，使 ship 新 TP 保留而不引入对应 FP？
3. 对于 4,481 张官方小样本和允许使用外部数据的规则，FAIR1M、RarePlanes、xView、
   AI-TOD 应按什么顺序与标签层级预训练，才能最大化 objectness/背景迁移而不污染 25 细类？
4. D-FINE/DEIM 与现有 Y5/M3 的最低成本筛选合同应如何冻结，避免再投入一整套无效长训？
5. 车辆 0.946/0.238 与 0.906/0.156 两点之间，怎样通过质量学习而不是阈值插值达到
   R≥0.95、FDR≤0.18？
6. 官方综合分由七项相对排名构成时，怎样设置更合理的多目标准入函数，避免再次出现
   “看起来五项变好、总分反而下降”？
7. 属性组合、车辆 q_match、域泛化和外部预训练四类未完成方向中，哪一类最可能在数日内
   形成足够大的真实增益？

---

## 13. 证据与文件索引

### 官方规则与提交

- `docs/hub/01_scoring_standard/README.md`
- `reports/submission/DOCKER_PREFLIGHT_20260829.md`
- `reports/submission/MODEL_AND_WEIGHT_FREEZE_AUDIT_20260829.md`
- `submission/docker/configs/y5_full_s_safe_1024_thr015.json`
- `submission/docker/configs/y5_full_s_safe_1024_rot90cwtta_trial_v2_calibrated_v1.json`

### 固定测评与官方校准

- `reports/experiments/FIXED_BENCHMARK_AND_NEXT_MODEL_PLAN_20260830.md`
- `outputs/OFFICIAL-TRIAL-CALIBRATION-20260830/trial_v1_v3_metrics.json`
- `configs/experiments/fixed_benchmark_v1.yaml`
- `src/rsdet/evaluation/official_metric.py`
- `src/rsdet/evaluation/official_ranking.py`

### 方案 7～9 执行账本

- `reports/experiments/IMPROVEMENT_PLAN7_EXECUTION_CLOSURE_20260826.md`
- `reports/experiments/HERA_GUARD_V2_94_RECALL_EXECUTION_20260829.md`
- `reports/experiments/HERA_GUARD_V3_METRIC_ALIGNED_EXECUTION_20260830.md`
- `reports/experiments/HERA_FIELD_BATCH02_SUMMARY_20260821.md`
- `reports/experiments/HERA_FULL_ANALYSIS_20260820.md`
- `reports/experiments/HERA_SCOPE_FULL_DIAGNOSIS_20260826.md`

### 当前关键本地产物

- `outputs/FIXED-BENCHMARK-V1/`
- `outputs/HERA-GUARD-V3-20260830/`
- `outputs/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1/`
- `outputs/HERA-GUARD-V2-STRUCTURAL-READOUT-20260830/`
- `outputs/Y5-HARD-REPLAY-FOLD0-SCREEN-V1/`
- `outputs/FOUR-SOURCE-SENTINEL-V1/`

---

## 14. 最终判断

v3 是一次有价值但明确失败的部署实验。它证明：

- 旋转第二视图确实能补船舶和少量飞机目标；
- 高车辆阈值确实能显著压低车辆 FDR；
- 但两者当前组合无法同时保住船 FDR、车辆 Recall 和时延排名；
- 我们的内部评估已经具备较强方向预测能力，但对隐藏域高阈值车辆 Recall 仍然乐观；
- 继续在同一个 Y5-S 分数上做阈值、NMS、普通 crop verifier，不足以到 93 分。

下一轮的中心应从“调当前分数”转向“产生新信息”：外部遥感 objectness/背景预训练、
异构检测器、跨视图稀疏一致性与车辆/船舶质量学习。先用已有缓存完成 v3 因果解耦，
随后只为通过固定双基准和来源互斥 sentinel 的一个候选投入完整训练与官方提交。
