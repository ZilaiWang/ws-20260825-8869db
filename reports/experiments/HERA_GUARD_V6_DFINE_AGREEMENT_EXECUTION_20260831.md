# HERA-Guard V6：D-FINE 车辆一致性执行记录（2026-08-31）

状态：`normal_cv3_positive / full_dfine_training / hard_and_sentinel_pending`

本轮只解决一个问题：D-FINE 中已经观察到的车辆互补信号，能否在不引入
D-FINE 框的前提下改善 Y5 的车辆 TP–FP 排序。所有统计均使用仓库的官方
prediction-first matcher，ship/aircraft 完全旁路。

## 1. 冻结合同

1. 主候选、bbox 和 25 类标签只来自 Y5；
2. 对每个 Y5 vehicle 框，寻找 IoU >= 0.35 的 D-FINE 同细类框；
3. 证据固定为 `Y5 score × max D-FINE support score`；
4. D-FINE 不允许新增、替换或移动任何框；
5. 阈值只由另外两个折选择，留出折只评一次；
6. 正式部署仍使用同细类 safe fusion；
7. full 权重固定 40 epoch，不按含训练图的 validation 结果选 epoch。

输入三折预测：

- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_*/y5_predictions.json`
- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_*/dfine_predictions.json`
- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_*/instances_val.json`

## 2. 严格增量恢复审计

正式脚本：`scripts/analyze_dfine_agreement_recovery.py`。

在每个 outer fold 上，先由另外两折选择 Y5 的 FDR15 工作点，再只恢复主阈值
以下、满足产品证据阈值的 Y5 vehicle 框：

|增量风险门|新增 TP|新增 FP|vehicle Recall|vehicle FDR|
|---:|---:|---:|---:|---:|
|0.15|+7|+1|+1.741pp|-0.117pp|
|0.20|+11|+2|+2.736pp|-0.005pp|

三个留出折在两档风险下均增加 TP。该结果替代早期一次性诊断中的 `16/4`、
`22/6` 估计；早期估计未使用最终冻结阈值选择实现，不能再作为正式数字引用。

输出：

- `outputs/HERA-GUARD-V6-DFINE-AGREEMENT-AUDIT-V1/audit.json`
- SHA256 `6fe4b90533d969f7c1c71058315849162e90824e5e4ace74fc08f4f7e853f1dd`

## 3. 单 Y5 蒸馏快筛：负向

### 3.1 元数据学生

教师缓存包含 65,301 个对齐候选、5,131 个 vehicle 候选。34-D metadata 与
63-D metadata+crop 轻量学生对教师产品的相关系数虽达到约 0.94，但主要因为
教师产品与 Y5 原始分数相关系数本身为 0.956。严格外层结果：

|学生|增量风险 0.15|增量风险 0.20|结论|
|---|---:|---:|---|
|base34|0 TP / 0 FP|+2 TP / 0 FP|只有 fold2 微增益|
|base_crop63|0 / 0|0 / 0|停止|

### 3.2 ConvNeXt 真实视觉嵌入学生

随后使用冻结的 tight/context 各 768-D ConvNeXt 嵌入。先回归教师产品，再将
监督解耦为直接回归 D-FINE support。两种实现均未通过跨折门禁：

- 产品学生的增益只出现在 fold2，边际 FDR 0.44--0.56；
- support 学生的 tight 分支 pooled 可增加 15 TP，但同时增加 16 FP，且仍只由
  fold2 贡献；context 分支 0 TP / +1 FP；
- 学生输出尺度在三个域间明显漂移，训练折阈值无法迁移到 fold0/1。

结论：冻结 crop 表示不足以稳定蒸馏 D-FINE 的关键尾部排序。停止 MLP、隐藏层、
loss 权重和分辨率扫描。若最终必须单模型，应把蒸馏前移到 Y5 的 FPN/检测头。

## 4. 重排审计更正

早期 `product_rerank_fdr15.json` 曾被表述为 Y5 与产品重排的对照。复核发现
`--route rerank` 先把 `score` 覆盖为产品分，随后 baseline 与 candidate 都评估了
同一个产品分集合。因此该文件中的 `198 TP / 36 FP` 只是产品路线自身的
工作点，**不是相对 Y5 的 +34 TP / +6 FP**；后一结论撤销，不再引用。

当前有效的无偏对照仅保留第 2 节的增量恢复审计，以及第 5 节实现中明确
分离原始 Y5 分数和产品分数的 Recall-guard 审计。这一更正不改变“D-FINE
存在互补信号”的结论，但降低了对收益幅度的估计。

## 5. 与 v2 阈值 0.15 对齐的 Recall guard

官方 incumbent 不是 FDR15 工作点，因此又以原 Y5 `threshold=0.15` 为基线，
在训练折限制 Recall 损失后选择产品阈值。

### 5.1 pooled guard（仅诊断）

|允许训练 Recall 损失|pooled TP 变化|pooled FP 变化|Recall 变化|FDR 变化|
|---:|---:|---:|---:|---:|
|0.0pp|0|-39|0.000pp|-8.475pp|
|0.5pp|0|-45|0.000pp|-9.978pp|
|1.0pp|-1|-48|-0.249pp|-10.670pp|

该结果很强，但 pooled 约束会掩盖 fold0 召回损失，因此不能单独准入。

### 5.2 每训练折独立 guard

|允许训练 Recall 损失|pooled TP 变化|pooled FP 变化|Recall 变化|FDR 变化|
|---:|---:|---:|---:|---:|
|0.0pp|+4|-22|+0.995pp|-4.902pp|
|0.5pp|+4|-22|+0.995pp|-4.902pp|
|1.0pp|+2|-32|+0.498pp|-6.977pp|

仍存在 fold0 域偏移，但方向比 pooled guard 更稳健。

### 5.3 最终全数据阈值校准候选

三折完成后，用全部 OOF 做最终部署超参数校准：

|固定产品阈值|三折 TP/FP 变化|三折 Recall 变化|三折 FDR 变化|最坏单折 Recall 变化|
|---:|---:|---:|---:|---:|
|0.055|+9/+3|**+2.239pp**|-0.335pp|+0.752pp|
|0.059|+9/0|**+2.239pp**|-0.900pp|+0.752pp|
|0.065|+6/-10|+1.493pp|-2.569pp|-0.752pp|
|0.080|-1/-34|-0.249pp|-7.177pp|-2.256pp|

按“所有折 Recall 不下降、相同 Recall 时 FP 更少”选择 `0.059` 作为当前 vehicle
Attack 候选。该表属于全部 OOF
上的最终校准，不是无偏外层性能估计；无偏证据仍以第 2、5 节为准。

### 5.4 飞机产品路线的预注册候选

使用同一三折 OOF，将 Y5=0.15 与产品阈值 0.05 直接配对：三折合计
`+80 TP / -387 FP`，Recall `+0.448pp`，FDR `-1.939pp`，三个折 Recall 分别
增加 `+0.618pp / +0.421pp / +0.280pp`。因此新增第二个完全冻结候选：

- route A：仅 vehicle 乘一致性分，vehicle=0.059；
- route B：aircraft + vehicle 乘同细类一致性分，aircraft=0.05、vehicle=0.059；
- ship 始终保持 Y5=0.15，不参与产品重排。

route B 不在 Hard10K/Sentinel 上调阈值；只有它在两者上同时保护三大类时，
才可以替代 route A。

## 6. 全量训练与部署状态

服务器任务：`/workspace/results/DFINE-M-FULL-40EP-AGREEMENT-V1`

- 4,481 images / 20,933 annotations / 25 categories；
- train/val COCO 合并 image overlap=0、annotation overlap=0；
- full ledger SHA256 `779f548274484dc503b76bb5075f429770328f28c938848802ce96a885213dc4`；
- COCO 预训练 D-FINE-M SHA256
  `b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195`；
- 40 epoch、seed 42、1024、batch 8、AMP；
- 3090 训练已启动，预计 3.5--4 小时。

部署实现：

- `src/rsdet/submission/agreement.py`；
- `src/rsdet/submission/competition.py` 中 `_SubmissionDfineDetector` 与
  `_SubmissionAgreementDetector`；
- D-FINE 只产生 support，结果 JSON 永远使用 Y5 bbox/label；
- vehicle 产品阈值候选 0.059；另有 aircraft=0.05 的显式预注册候选；
- 16 项部署合同测试通过。

## 7. 尚未通过的门禁

在以下工作完成前，不得宣称该方法优于 trial-v2，也不得直接覆盖 Safe 镜像：

1. full D-FINE checkpoint 40/40、SHA 和 epoch 验收；
2. Hard10K 使用相同 0.059 固定 vehicle 产品阈值，不重新调参；
3. source-disjoint sentinel 同方向；
4. 3090 双模型端到端结果等价与真实 10K 时延；
5. Docker 内冻结 D-FINE 源码 commit、配置、权重和依赖；
6. 对官方 v2 工作点的 quality gain 足以覆盖时延排名损失。

当前科学判断：**双检测器产品重排是强候选，但仍是 Attack 路线，不是新
incumbent。**
