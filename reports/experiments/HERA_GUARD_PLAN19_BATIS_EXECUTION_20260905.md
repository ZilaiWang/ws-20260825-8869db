# 方案19：HERA-Guard BATIS 执行、机制审计与停止结论

日期：2026-09-05

状态：**实验闭环；所有 BATIS 候选拒绝进入正式部署**

生产结论：**保持 P40 主检测器及既有 Aircraft-D4 路径，不启用 BATIS、稀疏重居中、增大 overlap 或跨细类去重。**

## 1. 本轮回答的问题

《改进方案19》提出：正式大图由 `1024×1024`、`overlap=256` 的窗口推理，而当前 Safe Fusion 先以低阈值保留候选，再按“非边界优先”选 canonical，最后才执行 `0.536` 输出阈值。这个顺序可能把同一簇内高于阈值的框替换为低于阈值的框，造成可恢复的融合漏检。

本轮没有假定该漏洞一定能提高正式分数，而是完成了三层验证：

1. 代码层：确认阈值反转机制真实存在，并实现不改变旧默认行为的可审计修复；
2. 机制层：一次低阈值推理保存 tile ledger，离线重放 H0/H1/H2，严格区分恢复 TP 与新增 FP；
3. 外层层：在原生连续图、Natural pseudo-10K、Trial-mix pseudo-10K 上核验方向，并补做 phase oracle、K=8 稀疏重居中、overlap320 和 Ship 跨细类严格去重。

最终结果是：**方案19定位的工程漏洞真实存在，但发生频率和 TP/FP 质量不足以构成稳定收益。切片相位也确实影响召回，但固定替代相位、稀疏补窗和更密 overlap 都不能跨测试集稳定提高分数。**

## 2. 冻结合同

- 主检测器：P40 sanitized full，SHA256 `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012`；
- tile：`1024`；生产 overlap：`256`；对照 overlap：`320`；
- 模型输入：`1280`；tile 候选阈值：`0.001`；最终输出阈值：`0.536`；
- Safe Fusion：同细类，merge IoU/IoS=`0.50/0.75`，fine NMS=`0.70`；
- BATIS 只允许接管 Ship `0–3` 与 Vehicle `24`；Aircraft `4–23` 必须保持旧 Safe Fusion，既有 D4 也不在本轮改动范围内；
- owner 候选仅在距最高分不超过 `0.2 logit` 时可优先；
- 稀疏重居中：每图最多 `K=8`，rescue floor=`0.25`，safe core=`128`，query IoU=`0.25`，只接纳真实模型框且替换必须更高分；
- H3：仅 Ship 跨细类，IoU≥`0.75` 或 IoS≥`0.90`，且归一化中心距离≤`0.20`；
- 指标：`platform_observed_20260831`，三个粗类内部先做细类宏平均，再对粗类取门禁均值；
- pseudo-10K 只作工程方向证据。其人工拼接缝会改变 Ship/Vehicle 统计，不能当作正式分数预测。

冻结配置见 [`configs/experiments/hera_guard_batis_plan_v1.json`](../../configs/experiments/hera_guard_batis_plan_v1.json)。

## 3. 实现内容

### 3.1 几何、ownership 与可见性

- [`src/rsdet/tiling/boundary_geometry.py`](../../src/rsdet/tiling/boundary_geometry.py)：生产/移相网格、反射 padding、完整包含、可见率、上下文量和不规则末端网格的 Voronoi ownership；
- [`scripts/audit_batis_tiling_geometry.py`](../../scripts/audit_batis_tiling_geometry.py)：逐 GT 生成边界画像，输出当前相位是否完整、是否由 overlap 几何保证以及各相位可见性。

### 3.2 阈值一致 Safe Fusion

- [`src/rsdet/postprocess/safe_tile_fusion.py`](../../src/rsdet/postprocess/safe_tile_fusion.py)：新增双阈值 canonical、类别范围、owner logit slack 和机制计数；所有参数默认关闭，旧调用输出保持不变；
- [`src/rsdet/pipeline/large_image.py`](../../src/rsdet/pipeline/large_image.py)：把 BATIS 参数接入大图管线；
- [`src/rsdet/submission/competition.py`](../../src/rsdet/submission/competition.py)：部署配置 fail-closed 校验；BATIS 输出阈值必须与 post-fusion 阈值一致，且只可用于 safe fusion；
- [`scripts/run_batis_tile_ledger.py`](../../scripts/run_batis_tile_ledger.py)：一次推理同时物化 H0、H1-A、H1-B、H2 和 audit；
- [`scripts/analyze_batis_replays.py`](../../scripts/analyze_batis_replays.py)：按固定官方观测指标做配对评分，并追踪 recovered/lost GT 与 FP 变化。

### 3.3 诊断与旁路

- [`scripts/run_batis_phase_oracle.py`](../../scripts/run_batis_phase_oracle.py)：生产相位加三种反射 padding 移相，输出逐相位指标和只用于诊断的 recall oracle；
- [`src/rsdet/pipeline/sparse_recenter.py`](../../src/rsdet/pipeline/sparse_recenter.py)、[`scripts/run_batis_sparse_recenter.py`](../../scripts/run_batis_sparse_recenter.py)：确定性风险簇、窗口去重和 K=8 补窗；
- [`src/rsdet/postprocess/strict_ship_cross_fine.py`](../../src/rsdet/postprocess/strict_ship_cross_fine.py)、[`scripts/apply_strict_ship_cross_fine_dedup.py`](../../scripts/apply_strict_ship_cross_fine_dedup.py)：H3 Ship-only 严格跨 fine 去重；
- [`configs/experiments/p40_native_tiled1024_overlap320_probe_v1.json`](../../configs/experiments/p40_native_tiled1024_overlap320_probe_v1.json)：E5 单因素配置。

对应单测为 [`tests/test_boundary_geometry.py`](../../tests/test_boundary_geometry.py)、[`tests/test_safe_tile_fusion.py`](../../tests/test_safe_tile_fusion.py)、[`tests/test_sparse_recenter.py`](../../tests/test_sparse_recenter.py)、[`tests/test_strict_ship_cross_fine.py`](../../tests/test_strict_ship_cross_fine.py) 和 [`tests/test_submission_contract.py`](../../tests/test_submission_contract.py)。

## 4. E0：几何与机制审计

### 4.1 公共 4,481 图的几何边界

公共数据共 20,933 个 GT。当前相位恰好完整包含全部 GT，但这主要因为只有 30 张图大于 1024，不能代表正式大图的随机相位分布。

|范围|GT|当前相位完整|由 overlap=256 保证完整|
|---|---:|---:|---:|
|全部|20,933|20,933|20,201|
|Ship|2,682|2,682|2,018|
|Aircraft|17,849|17,849|17,781|
|Vehicle|402|402|402|
|最大边 ≥256|736|736|4|

因此，“公共小图上没有截断”不能证明正式大图没有边界风险；但它也说明公共 OOF 不适合直接估计该风险的频率。完整逐框证据保存在本机 `outputs/HERA-GUARD-PLAN19-20260905/e0_geometry_full4481.json`，不进入公开仓库。

### 4.2 阈值反转确实存在

|测试|Safe Fusion 簇数|全部类别阈值反转|实际作用于 Ship/Vehicle|
|---|---:|---:|---:|
|26 张原生连续图|625|1|1|
|Natural pseudo-10K|13,650|29|13|
|Trial-mix pseudo-10K|15,936|32|28|

这证明方案19指出的代码路径不是理论猜测。但反转只占约 `0.16%–0.20%` 的簇，能否部署取决于这些恢复框的 TP/FP 质量。

## 5. H0–H2 离线重放结果

H1-A 是先删 `<0.536` 再融合；H1-B 保留低分成员参与聚类，但 canonical 必须 `≥0.536`。三套测试中 H1-A 与 H1-B 的最终预测逐框、逐分、逐类完全一致，说明当前数据上低分 support 没有提供额外输出差异。

|候选|测试|门禁 Recall Δ|门禁 FDR Δ|同延迟分数 Δ|主要配对变化|
|---|---|---:|---:|---:|---|
|H1-A/B|原生 26 图|不完整 taxonomy|Ship +1.128pp|不可算|0 TP、+1 FP|
|H1-A/B|Natural 10K|+0.246pp|+0.676pp|**−0.505**|Ship +4 TP/+8 FP；Vehicle +0 TP/+1 FP|
|H1-A/B|Trial 10K|+0.434pp|+0.363pp|**+0.033**|Ship +13 TP/+13 FP；Vehicle +1 TP/+1 FP|
|H2 owner|Natural 10K|+0.181pp|+0.748pp|**−0.586**|Ship 净 +2 TP/+10 FP，且 2 个原 TP 丢失|
|H2 owner|Trial 10K|+0.338pp|+0.468pp|**−0.167**|Ship 净 +7 TP/+19 FP，且 8 个原 TP 丢失|

Aircraft 在 H1/H2 中保持逐指标完全相同，验证了类别边界没有串线。H1 只在 Trial 10K 极小正向，在 Natural 和原生图负向；H2 又比 H1 更差。因此阈值一致修复虽纠正了代码语义，却不能自动视为精度修复。

## 6. 网格相位诊断

原生连续图的生产相位为 213 TP/4 FP/3 FN；三个单独移相均未超过生产相位。Ship 的 phase recall oracle 从 `93.421%` 提高到 `96.053%`，但实际只多覆盖 1 个 GT。

Trial 10K 的生产相位门禁 Recall/FDR 为 `88.838%/3.217%`；三个移相的 Recall 分别为 `88.022%/86.300%/86.699%`，均更低。把四相位“见过的 GT”并集计算的不可部署 recall oracle 为 `91.927%`，较生产相位 `+3.089pp`，其中 Ship/Aircraft/Vehicle oracle 分别为 `92.908%/99.177%/83.696%`。

解释：相位敏感性存在，但没有一个固定移相在各粗类上更好；oracle 没有 FP、延迟和可执行选择规则，不能作为候选或分数。

## 7. E3：Sparse Recenter K=8

|测试|请求/实际窗口|接纳|Recall Δ|FDR Δ|同延迟分数 Δ|含估算额外时延|
|---|---:|---:|---:|---:|---:|---:|
|原生 26 图|48/9|1|0|0|0|负向成本|
|Natural 10K|365/48|17|+0.187pp|+0.473pp|−0.349|约 −0.365|
|Trial 10K|1,034/48|16|+0.241pp|−0.012pp|+0.134|约 +0.117|

Trial 10K 恢复了 Ship +3 TP、Vehicle +1 TP 且未增加 pooled FP，是本轮最接近正向的信号；但 Natural 10K 新增背景 FP，原生图没有收益。它不满足跨测试稳健性，故按冻结停止条件不运行 K=16，也不允许进入 full 或 Docker。

## 8. E5：overlap=320 上界对照

Natural 10K tile 数由 1,014 增至 1,176，增加 `15.98%`；Recall `+0.865pp`，但 FDR `+1.788pp`，同延迟分数 `−1.473`，计入约 `+0.395s` 的推理成本后更差。Trial 10K Recall `+0.629pp`、FDR `+0.419pp`，同延迟仅 `+0.055`，计入时延后约 `−0.001`。原生 26 图完全不变。

结论：更密 overlap 能暴露更多候选，但新增暴露同时放大背景 FP；它不是当前精度—时间权衡的有效解。

## 9. H3：Ship 跨细类严格去重

- 原生图：0 个删除；
- Natural 10K：0 个删除，指标完全相同；
- Trial 10K：删除 2 个框，恰好 1 TP + 1 FP；门禁 Recall `−0.012pp`、FDR `−0.011pp`，同延迟绝对分 `−0.0039`。

当前严格规则几乎没有容量，且仅有的变化没有正收益。不能通过放松阈值扩大删除量，因为这会变成事后搜索，并直接增加误删真实并排或跨类目标的风险。

## 10. 方案19逐项闭环

|方案19项目|状态|结论|
|---|---|---|
|E0 tile ledger/切片损失日志|完成|阈值反转真实但稀少；公开小图不足以模拟正式相位|
|E1-A 最终阈值前移|完成|与 E1-B 输出一致；跨测试不稳健|
|E1-B 双阈值 canonical|完成|语义正确、无稳定分数收益，不部署|
|E2 valid-core owner|完成|较 H1 进一步增加 FP/丢失 TP，不部署|
|H2b 非 owner 双 tile 支持|按停止条件不启动|H2 已负向；更严格衍生规则没有正向依据|
|H3 Ship 跨 fine 去重|完成|几乎无容量，Trial 轻微负向|
|phase mirror/oracle|完成|证明相位敏感；无固定移相可部署|
|E3 sparse recenter K=8|完成|Trial 弱正向、Natural 负向、原生零收益|
|E4 sparse recenter K=16|按停止条件不启动|K=8 未证明稳健恢复，扩大预算只会增加成本/FP 暴露|
|E5 overlap320|完成|召回增益被 FDR 与时延抵消|
|E6 随机相位短微调|按停止条件不启动|E3/E4 未证明 pipeline 可稳定恢复真实目标|
|Background-100MP、全新 Sentinel-B、正式外层折|未消耗|没有候选通过前置准入，不应为失败候选继续消耗正式门禁|

这些“未启动”是方案19明示停止条件的执行结果，不是遗漏。

## 11. 深层结论

1. **正式差距不是一个单独的 canonical bug。** H1 能恢复若干目标，但新增框的 TP:FP 质量在 Natural 10K 仅 4:9，在 Trial 10K 为 14:14；修复输出权限无法解决低分真目标与背景响应的排序重叠。
2. **切片确有上限，但当前没有可靠的无监督选择器。** 四相位 oracle 说明某些 GT 在别的相位可见；然而每个单独移相都更差，且重居中触发器不能在 Natural 与 Trial 间保持质量。
3. **增加候选暴露不是主线答案。** H1、E3 和 E5 都表现为“Recall 上升同时 FP 上升”，overlap 越密越明显。下一次改进必须提升新增候选的条件精度，而不是继续增加窗口或降低门槛。
4. **Ship/Vehicle 的主要瓶颈回到表征与排序。** 更有价值的方向是能在真实背景上把低分 TP 与背景 FP 分开的模型或对象级判别证据，并用现有固定评测做单因素验证。
5. **现有安全主线不应因实现完成而改变。** BATIS 代码可以保留为审计/研究设施，但生产配置继续保持原 Safe Fusion；Aircraft-D4 全程未受影响。

## 12. 产物索引

本地小型证据根目录：`outputs/HERA-GUARD-PLAN19-20260905/`（按仓库约定由 `.gitignore` 排除；本报告保留全部决策数字与代码入口）。

- H0–H2：`HERA-GUARD-PLAN19-BATIS-{NATIVE,NATURAL10K,TRIAL10K}-SV-V1/`；
- phase：`HERA-GUARD-PLAN19-PHASE-{NATIVE,TRIAL10K}-V1/`；
- E3：`HERA-GUARD-PLAN19-E3K8-{NATIVE,NATURAL10K,TRIAL10K}-V1/`；
- E5：`HERA-GUARD-PLAN19-E5OVERLAP320-{NATIVE,NATURAL10K,TRIAL10K}-V1/`；
- H3：`h3_{native,natural10k,trial10k}/`；
- E0：`e0_geometry_full4481.json`。

服务器保留完整 ledger、预测和日志；仓库只收纳可复核的小型 summary、metrics 与 paired analysis，不收 checkpoint 或大预测。

## 13. 验收

- Ruff：方案19新增/修改代码全部通过；
- 方案19相关回归：70 tests passed；服务器锁定环境全仓回归：1,159 passed、2 skipped；
- H1-A/H1-B：三套测试预测完全一致；
- Aircraft：H0/H1/H2 逐指标完全一致；
- tile max-det saturation：三套 H0–H2 与 overlap320 均为 0；
- 生产状态：未修改正式配置，未训练 full，未打包 Docker，未发起正式提交。
