# HERA-Guard 方案18：APRR 反事实与真实/伪大图切片审计

日期：2026-09-05。状态：**方案18离线资格赛完成，APRR 未通过门禁并按方案停止；
大图差距审计完成，生产切片实现未见系统性退化，10×10伪大图构造被确认会显著污染
类别趋势。未训练 full、未实现 APRR 运行时、未打包、未提交。**

## 1. 结论先行

1. 方案18的总体实验纪律是正确的：P40 保有框、分数和细类标签所有权，专家只提供
   布尔支持；先用冻结 OOF 账本做反事实，未过门即停止运行时和 full。该纪律避免了
   v3 的第二检测器全面接管和 task-vector 的全局 top-k 旁效应。
2. 方案18提出的 APRR 具体组合不成立。Ship `S012` 虽使 Ship macro Recall
   `+14.26pp`，却使 macro FDR `+44.12pp`，三折质量变化全部为负；Vehicle `V065`
   三折质量均正并使 FDR `-10.81pp`，但 Recall 损失 `4.73pp`，远超预注册的
   `0.5pp`。组合 `S012_V065` 三折仍全部为负，source-group bootstrap P10 为
   `-4.8764`。因此 `stop_aprr_before_runtime` 是唯一合规结论。
3. 用户提出的“大图来源差异”假设成立，但要精确表述：**问题主要不在 1024/256
   切片器或 safe fusion，而在把 100 张互不相干的小图随机拼成一张 10K 图。**
4. 在 26 张原生连续、长边位于 `(1024,1280]` 的真实训练图上，同一 P40 权重和
   `0.536` 阈值，生产 `safe1024/overlap256` 相对单块 `1280` 为 `+2 TP / 0 FP`，
   pooled Recall `97.685% -> 98.611%`。这排除了切片实现普遍丢框的解释，但样本中没有
   Vehicle，不能覆盖隐藏大图分布。
5. 伪大图的缩放其实做得较好：原图直接进入 1280 与“先缩到 1000 cell、再把 1024
   tile 输入 1280”的目标尺度比恒为 `0.9765625`，只小 `2.34375%`。主要缺陷是生产
   每个 1024 tile 都同时跨越横纵人工缝，并固定混入 4 个无关来源场景。
6. 该缺陷已经被逐对象实验证实。距人工拼接缝 32 px 内，Natural 的条件漏检率为
   `67.57%`，远离缝为 `2.64%`；Trial-mix 为 `76.92%` 对 `7.07%`。这不是抽象风险，
   而是已经进入当前 P40 误差账本的强混杂。
7. 把同一批 600 张原始来源图的 P40 预测精确投影回 pseudo cell 后再与伪大图直接
   切片推理比较，Trial-mix 伪构造令总体 Recall `-1.251pp`、FDR `+0.801pp`；其中
   Ship Recall `-3.249pp`、FDR `+2.285pp`，Vehicle 却 Recall `+2.174pp`、FDR
   `-3.180pp`。**同一种伪构造会把不同粗类推向相反方向，所以它不再适合作为
   Ship/Vehicle 创新模块的准入或排序依据。**
8. 伪大图绝对指标还有另一个相反偏差：其 600 个来源来自公开训练集，full P40 已见过
   这些来源，因此绝对 Recall/FDR 又会偏乐观。训练来源泄漏与人工拼接缝惩罚同时存在，
   不能相互抵消，也不能用一个总分校正系数修复。

## 2. 方案18审查

### 2.1 保留的正确部分

- P40 是唯一 proposal owner；任何辅助模型都不能输出自己的 bbox、score 或 label。
- RFS 仅允许支持 P40 的低分 Ship proposal；hierarchy 仅允许支持 P40 Vehicle 风险带。
- 同细类匹配：Ship IoU `0.50`，Vehicle IoU `0.35`。
- Aircraft 在反事实阶段完全旁路，避免把 D4 的固定收益与新模块混在一起。
- 只比较预注册的 `S012/S0123`、Vehicle protect `0.60/0.65`，不做融合权重和连续阈值
  扫描。
- 离线资格赛未通过即停止选择性 tile runtime、共享头和 full RFS。

实现位于：

- [APRR 核心](../../src/rsdet/submission/aprr.py)；
- [冻结 OOF 回放](../../scripts/replay_aprr_cv3.py)；
- [核心测试](../../tests/test_aprr.py)；
- [机器可读结果](../../outputs/HERA-GUARD-PLAN18-20260905/aprr/result.json)。

### 2.2 必须修正的旧推断

方案18成文时使用了部分已被后续诊断更新的量：

1. D4 的 `4.01--4.65s` 总时延投影已经过时。同进程 AB/BA 复测得到官网总时延投影
   `4.872--5.053s`，D4-only 预计只比 P40 的 `76.601` 净增约 `0.107--0.133`。
2. 把离线质量变化机械加到官方 `76.601` 得到 `80.2`，不具有预测效力。短 OOF
   Vehicle 在 `0.536` 只有 `32.59%` Recall，而正式 full P40 为 `83.16%`；两者不在
   同一成熟度和分数工作区。
3. RFS 在自身工作点上对某些稀有 Ship 类的改善，不等价于“RFS 支持的 P40 低分框具有
   高纯度”。本次回放正好否定了这个隐含前提。
4. 因此前述名义正式投影没有被用作准入条件；保留其历史文字，但不制造伪精确正式分。

## 3. APRR 冻结反事实

### 3.1 输入与规则

- P40 OOF proposals：hierarchy 评测中的冻结 baseline 低阈值账本；
- RFS proposals：`P40-WEAK-RFS-V1` 三折 OOF；
- hierarchy proposals：`HERA-GUARD-MPSR-HIER-VEHICLE-ROUTE-CV3-OOF-3090-V1`；
- P40 每折阈值：`0.546 / 0.516 / 0.501`；
- RFS 每折支持阈值：`0.471 / 0.451 / 0.446`；
- hierarchy 支持阈值：`0.546`；
- Vehicle protect：`0.60 / 0.65`；
- 255 个 source group、2,000 次 bootstrap、seed `20260905`。

两个 GT 文件的字节 SHA 不同，但图像集合、20,933 条 annotation 的
`(image_id, category_id, bbox)` 集合完全一致；回放统一采用 hierarchy GT。

### 3.2 单模块结果

|变体|三折质量变化|合并质量变化|关键粗类变化|结论|
|---|---|---:|---|---|
|S012|`-6.704/-5.614/-2.770`|`-4.142`|Ship R `+14.261pp`，FDR `+44.117pp`|拒绝|
|S0123|`-7.599/-6.583/-3.703`|`-5.077`|Ship R `+16.355pp`，FDR `+54.805pp`|拒绝|
|V060|`+1.960/-0.396/+0.871`|`+0.655`|Vehicle R `-2.488pp`，FDR `-5.121pp`|拒绝|
|V065|`+2.180/+1.902/+1.740`|`+2.056`|Vehicle R `-4.726pp`，FDR `-10.814pp`|拒绝|

`S012` 共救回 2,272 个 P40 Ship 框。主要问题集中于 class2/QHS：基线为
`367 TP / 74 FP`，救援后变成 `475 TP / 2,182 FP`。三个稀有类的最坏折 FDR 增量
分别达到 class0 `+94.44pp`、class1 `+72.73pp`、class2 `+67.10pp`。RFS 与 P40
共享了大量水面/港口背景混淆，布尔一致性并没有形成独立证据。

`V065` 从 P40 Vehicle 的 `133 TP / 40 FP / 269 FN` 变为
`114 / 16 / 288`：它确实删除 24 FP，但同时删除 19 TP。三折质量为正只说明当前
评分权衡偏好更低 FDR，不能覆盖预注册“任一折 Recall 损失不超过 0.5pp”的安全条件，
更不能在正式 P40 Vehicle 只有 95 GT、Recall 仅 83.16% 时冒险部署。

### 3.3 组合结果

最佳的组合枚举仍是 `S012_V065`，但：

- 三折质量变化：`-4.524 / -3.711 / -1.029`；
- 合并变化：`-2.087`；
- source-group bootstrap：P10 `-4.876`、P50 `-2.657`、P90 `-0.025`；
- Aircraft 逐指标完全不变；
- 最终决策：`stop_aprr_before_runtime`。

这说明失败不是由 Aircraft 旁效应、框所有权或实现 bug 造成，而是 Ship 支持信号本身
不具备所需精度。根据方案18自己的停止规则，不启动 full RFS、选择性 tile 专家、共享头、
Hard/Sentinel 或 Docker。

## 4. 伪大图几何审计

### 4.1 当前构造的精确行为

`build_pseudo_10k_mosaics.py` 将每张来源图等比例缩入 `1000×1000` cell，居中后用
RGB 114 填充；10×10 cell 构成一张 `10000×10000` JPEG。生产推理为：

```text
tile = 1024
overlap = 256
stride = 768
model input = 1280
safe fusion: merge IoU 0.50 / IoS 0.75 / border 8
```

量化结果：

|项|Natural|Trial-mix|
|---|---:|---:|
|伪图 / 来源图 / GT|6 / 600 / 2,875|6 / 600 / 2,158|
|每张 10K 的 tile|169|169|
|每个 tile 相交的来源 cell|恒为 4|恒为 4|
|来源→cell 缩放中位数|1.250|1.185|
|伪路径/直接路径网络目标尺度|恒为 0.9765625|恒为 0.9765625|
|tile 平坦填充比例 p90 / max|3.12% / 29.37%|— / 42.53%|
|无完整安全 tile 视图|Aircraft 6、Ship 20、Vehicle 0|Aircraft 1、Ship 53、Vehicle 1|

机器可读几何审计：

- [Natural](../../outputs/HERA-GUARD-PLAN18-20260905/tile_fidelity/natural.json)；
- [Trial-mix](../../outputs/HERA-GUARD-PLAN18-20260905/tile_fidelity/trial_mix.json)；
- [审计脚本](../../scripts/audit_pseudo10k_tiling_fidelity.py)。

### 4.2 为什么尺度不是主因

设来源图长边为 `L`。直接输入 1280 的比例为 `1280/L`；伪路径先缩放为
`1000/L`，再随 1024 tile 输入 1280，比例为 `1000/L × 1280/1024`。两者之比：

```text
(1000/L × 1280/1024) / (1280/L) = 1000/1024 = 0.9765625
```

它与来源尺寸无关。因而当前构造并非把目标缩小一倍或造成随机尺度漂移；`2.34%` 的固定
差异可能影响阈值边缘，但不足以解释长期数十分的代理—官方差距。

### 4.3 真正的问题：人工缝和不连续上下文

1024 大于 1000，所以没有任何生产 tile 只看到一个来源图。169 个 tile 全部同时越过
人工横缝和竖缝，把四个无关机场、港口或道路背景放进同一张 detector 输入。真实官方
大图虽然也被切片，但 tile 内是连续地物，不存在这种语义跳变。

使用 full P40、正式 `0.536` 工作点直接重跑两套伪图后，条件误差为：

|伪集|距人工缝≤32px 漏检率|远离人工缝漏检率|缝附近FDR|远处FDR|
|---|---:|---:|---:|---:|
|Natural|67.57%|2.64%|7.69%|2.33%|
|Trial-mix|76.92%|7.07%|11.76%|3.86%|

放宽到 64 px 后，Natural 漏检仍为 `24.83% vs 2.31%`，Trial-mix 为
`43.72% vs 5.97%`。平坦填充区中心没有产生 TP/FP，说明 RGB 114 空带不是当前
`0.536` 工作点的直接 FP 来源；主要污染来自来源边缘附近的目标、被替换为无关场景的
外部上下文和交叉 tile 的预测变化。

证据：

- [Natural seam error](../../outputs/HERA-GUARD-PLAN18-20260905/pseudo_seam/natural/analysis.json)；
- [Trial seam error](../../outputs/HERA-GUARD-PLAN18-20260905/pseudo_seam/trial/analysis.json)；
- [分析脚本](../../scripts/analyze_pseudo10k_seam_errors.py)。

## 5. 两组因果配对实验

### 5.1 原生连续图：整图 vs 生产切片

从 4,481 张真实训练图中选出长边 `1025--1280` 的 26 张。这些图在 1280 路径中恰好
一个 tile，在生产 1024 路径中会真正被切开。两次运行使用完全相同的：

- P40 权重 SHA `b0df7981...c8012`；
- model imgsz 1280、conf 0.001、NMS 与 safe fusion；
- 最终阈值 0.536；
- RTX 3090、同一代码和图像字节。

|路径|TP / FP / FN|pooled Recall|pooled FDR|平均单图时间|
|---|---:|---:|---:|---:|
|单块1280|211 / 4 / 5|97.685%|1.860%|0.101s|
|safe1024/256|213 / 4 / 3|98.611%|1.843%|0.117s|
|切片−整图|`+2 / 0 / -2`|`+0.926pp`|`-0.017pp`|`+0.016s`|

216 个 GT 中 210 个两条路径都命中、3 个仅切片命中、1 个仅整图命中、2 个都未命中。
全部 159 个 Aircraft 均由两条路径命中；净增益来自 Ship。现有真实连续样本未显示切片
器的系统性退化。

局限很明确：只有 26 图、57 Ship 和159 Aircraft，没有 Vehicle；图像最大只有 1280，
不能复制官方超大图的 tile 数量、目标密度和长距离背景结构。

证据与实现：

- [配对分析](../../outputs/HERA-GUARD-PLAN18-20260905/native_tiling/analysis.json)；
- [子集构建审计](../../outputs/HERA-GUARD-PLAN18-20260905/native_tiling/build_audit.json)；
- [子集构建器](../../scripts/build_native_tiling_probe.py)；
- [配对分析器](../../scripts/analyze_native_tiling_probe.py)；
- [整图配置](../../configs/experiments/p40_native_whole1280_probe_v1.json)；
- [生产切片配置](../../configs/experiments/p40_native_tiled1024_probe_v1.json)。

### 5.2 同源像素：逐源推理投影 vs 伪图切片推理

为进一步隔离伪构造影响，对 Natural 和 Trial-mix 各自的 600 张来源图执行 P40；再用
构造时的同一 scale、cell 偏移把检测框投影回 10K 坐标，与同一 GT 下的伪图切片预测
比较。两侧权重、工作点和输出字段一致；剩余差异只来自固定 `2.34%` 尺度、随机邻居、
人工缝和 tile fusion。

|集合|总体 Recall 变化|总体 FDR 变化|TP / FP / FN 变化|
|---|---:|---:|---:|
|Natural 伪图−逐源投影|`-0.348pp`|`+0.907pp`|`-10 / +26 / +10`|
|Trial 伪图−逐源投影|`-1.251pp`|`+0.801pp`|`-27 / +16 / +27`|

Trial-mix 分粗类变化尤其说明趋势污染：

|粗类|Recall 变化|FDR 变化|TP / FP 变化|
|---|---:|---:|---:|
|Ship|`-3.249pp`|`+2.285pp`|`-31 / +19`|
|Aircraft|`0`|`+0.190pp`|`0 / +2`|
|Vehicle|`+2.174pp`|`-3.180pp`|`+4 / -5`|

Natural 也同向显示 Ship `-2.717pp`、Vehicle `+3.659pp`。所以把 10K 总分看成一个
“接近官方难度”的标量会掩盖结构性偏差：它会系统性惩罚当前 Ship，同时让 Vehicle
看起来比逐源推理更好。这足以造成 class-specific rescue、阈值或专家路由的方向误判。

证据与实现：

- [Natural comparison](../../outputs/HERA-GUARD-PLAN18-20260905/source_projection/natural/comparison.json)；
- [Trial comparison](../../outputs/HERA-GUARD-PLAN18-20260905/source_projection/trial/comparison.json)；
- [来源子集与映射构建](../../scripts/build_pseudo_source_projection_probe.py)；
- [框投影](../../scripts/project_source_predictions_to_pseudo10k.py)；
- [配对比较](../../scripts/compare_pseudo_source_projection.py)。

## 6. 这能否解释本地与官方的全部差距

不能简单说“官方是大图，所以我们的切片错了”。当前证据支持三层结论：

1. **部署实现层：未发现普遍错误。** 真实连续图切片没有丢失性能，已有坐标、边界和
   safe fusion 单元测试也继续通过。
2. **代理构造层：确认存在显著且类别相关的偏差。** 人工缝强烈集中 FN，并让 Ship 与
   Vehicle 朝相反方向变化。这可以解释本地方法排名和官方排名为何经常不一致。
3. **分布层：仍是绝对差距主因。** pseudo 的来源图参与过 full P40 训练，隐藏官方图
   则是未见站点、未见连续背景和可能不同 GSD/尺寸分布。伪图即使增加人工难度，也不能
   抵消训练来源泄漏，更不能复原官方 620 Ship、2,849 Aircraft、95 Vehicle 的未知场景。

因此不应给现有 pseudo 得分加减 `1.25pp` 后当作官方预测。它不是单调校准误差，而是
按类别、来源边缘距离和方法上下文敏感性变化的交互偏差。

## 7. 从现在起的固定测试职责

### 7.1 保留 10K，但降级为工程测试

伪 10K 继续用于：

- Docker 输入/输出合同；
- 坐标恢复、越界框、重复框和 safe fusion 回归；
- 10K 显存、时延、batch、tile 数和 max-detections 压力；
- 同一算法改代码前后的逐框等价性。

它不再用于：

- 选择 Ship/Vehicle 阈值；
- 宣布 semantic verifier、rescue 或 reject 准入；
- 预测官方绝对 Recall/FDR/总分；
- 在多个候选中按 10K 总分挑 winner。

### 7.2 方法选择回到两个固定证据层

1. **主筛：原生小图 source-disjoint OOF。** 只看同权重成熟度、同阈值政策下的配对
   方向、逐折同 FP 预算、细类和 source-group 稳健性；不把短 OOF 绝对阈值平移到 full。
2. **切片回归：原生连续 whole-vs-tiled probe。** 每次改 tiler/fusion 才运行；要求
   paired TP 不下降、不得新增长边界 FP。当前 26 图是最小版本，待有真实连续大图后直接
   扩展，不更改分析代码。

Hard、Sentinel-B、Background-100MP 仍可作为一次冻结确认，但不得反复参与选择。

### 7.3 怎样得到更接近官方的大图代理

优先级如下：

1. 获取 E 阶段或官方提供的真实连续大图及原始标注，这是唯一能直接覆盖问题的方案。
2. 检查公开 crop 是否带可信的原图坐标/窗口元数据；只有坐标和像素重叠都一致时，才能
   重建同一 parent scene。文件名中的 `crop1/crop2` 不能自行解释为邻接顺序。
3. 若没有坐标，不再随机拼 10×10。可以做单图 padding/canvas 工程压力，但必须明确它
   不模拟场景语义；不能把无关图像放入同一个 1024 detector tile 后评价识别能力。
4. 新获得的连续大图先做 direct/tiled 配对和边界分层，再冻结为从未参与模型选择的
   `Continuous-Sentinel`。

## 8. 代码、测试和产物索引

新增核心代码：

- `src/rsdet/submission/aprr.py`；
- `scripts/replay_aprr_cv3.py`；
- `scripts/audit_pseudo10k_tiling_fidelity.py`；
- `scripts/build_native_tiling_probe.py`；
- `scripts/analyze_native_tiling_probe.py`；
- `scripts/analyze_pseudo10k_seam_errors.py`；
- `scripts/build_pseudo_source_projection_probe.py`；
- `scripts/project_source_predictions_to_pseudo10k.py`；
- `scripts/compare_pseudo_source_projection.py`。

专项测试：

- `tests/test_aprr.py`；
- `tests/test_pseudo10k_tiling_fidelity.py`；
- `tests/test_native_tiling_probe.py`；
- `tests/test_pseudo_source_projection_probe.py`。

本轮专项结果根目录：

- [HERA-GUARD-PLAN18-20260905](../../outputs/HERA-GUARD-PLAN18-20260905)；
- [本地全文件 SHA256](../../outputs/HERA-GUARD-PLAN18-20260905/LOCAL_SHA256SUMS.txt)。

本轮专项测试为 `8 passed`，Ruff 与 `git diff --check` 通过。结果包约 3.1MB；所有大权重
仍留在服务器，本地只保留 GT、预测账本、小型映射、分析和校验文件。

## 9. 最终决策

方案18没有产生可提交的 APRR 候选；按其预注册规则立即停止是成功的风险控制，不是执行
不完整。当前可证明的最重要新结论，是长期使用的随机 10×10 伪大图并不具备语义评估
资格。它的目标尺度近似正确、工程压力有价值，但人工拼接缝会显著集中漏检，并且让
Ship 与 Vehicle 产生相反偏差。

后续若继续优化识别，主线必须回到“原生、来源隔离、成熟度可比”的证据；若继续优化
切片，必须使用真实连续图做同图 paired test。两条问题从此分开管理，避免再把模型失败、
代理构造失败和切片实现失败混成一个分数。
