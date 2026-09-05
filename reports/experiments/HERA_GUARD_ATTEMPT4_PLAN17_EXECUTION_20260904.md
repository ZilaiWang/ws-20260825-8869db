# HERA-Guard Attempt 4：方案 17 执行记录

日期：2026-09-04。当前状态：**两条冻结实验链已得到结论；Vehicle
task-vector 拒绝，Aircraft-D4-only 保留为边际正向候选。未训练
full、未打包、未提交**。

## 1. 决策背景

正式 Attempt 3 已把因果拆清：Aircraft-D4 在隐藏集上净增 `21 TP`、减少
`21 FP`，但层级 Vehicle 路由只增加 `1 TP`、同时增加 `8 FP`，第二个检测器还令
时延从 `3.551833s` 增至 `6.964667s`。因此本轮不再搜索 hierarchy 阈值或 rescue，
而是严格执行[方案 17](../../../改进方案17.md)的两个最小方向：

1. `P40 + Aircraft-D4 only`：保留正式隐藏集已经证明的飞机增益，Ship/Vehicle
   完全恢复 P40；
2. `P40 -> hierarchy` 的 Vehicle class-row task vector：只改 6 个最终分类卷积中
   `class_id=24` 的行，其他类别、几何头、归一化状态和所有中间特征完全继承 P40。

正式最佳仍是 v2.0 / `76.6010`。本轮不把历史 Hard/Sentinel 绝对分当成官网分数
预测；CV3 用于选择和准入，压力集只做冻结确认。

## 2. 实现与冻结合同

### 2.1 Vehicle class-row task vector

- 基线：每折 P40 progressive-resolution 40e checkpoint；
- donor：同折 `S256/V128` hierarchy checkpoint；
- 同构核验：708 个 state tensor key/shape 完全一致；
- 目标模块：
  `model.23.(one2one_)?cv3.[0-2].2`，恰好 6 个 `Conv2d(out=25)`；
- 唯一可变张量：上述 6 个模块的 weight/bias 第 24 行；
- α：固定为 `0 / 0.125 / 0.25 / 0.5`；
- 固定工作点：`score >= 0.546`；
- 三张 4080 SUPER 各负责一个 held-out fold，fold 内依次推理三个非零 α；
- α=0 不重复浪费完整推理，但必须通过真实 checkpoint 的逐张量 bitwise parity，
  且生成权重必须能由 Ultralytics 正常加载；
- 每个 held-out fold 的 α 仅由另外两折合并指标选择；同分时取更小 α；
- 最终报告三个 outer-fold delta、来源组 paired bootstrap P10、Vehicle 增量 FP
  六倍压力、Ship/Aircraft 逐框 parity；
- 只有每折正增益、Vehicle Recall 每折损失不超过 0.5pp、bootstrap P10>0、
  6x FP stress>0，才允许进入 Background-100MP；否则立即停止。

实现索引：

- [任务向量核心](../../src/rsdet/experiments/class_task_vector.py)
- [checkpoint 物化器](../../scripts/merge_yolo_class_task_vector.py)
- [outer-policy 与稳健性分析](../../scripts/analyze_vehicle_task_vector_policy.py)
- [三卡驱动](../../scripts/server/run_attempt4_vehicle_task_vector_cv3_3gpu_v1.sh)
- [核心测试](../../tests/test_class_task_vector.py)
- [策略测试](../../tests/test_task_vector_policy.py)

### 2.2 P40 + Aircraft-D4 only

这条链没有改动 D4 决策：

- P40 full 权重与 v2 相同，SHA256
  `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012`；
- Aircraft full checkpoint 与 v3 相同，SHA256
  `5f1b175f8b2c310a3c0583e652f5cf7cfd444d0b0d74d794a9ac59e4537832d5`；
- 固定 `relabel_min_probability=0.9`、Aircraft same-fine NMS `0.5`；
- 完全撤掉 hierarchy、Vehicle rescue 和第二检测器。

只做三项等价提速：

1. 将最终 `0.536` 阈值提前到 D4 crop 生成前。D4 不改变 score，且 NMS 按 score
   降序，所以被最终阈值拒绝的低分框不可能抑制保留框；
2. full checkpoint 已覆盖 ConvNeXt-T 的全部 state，直接构建架构，取消先加载再
   覆盖的 109MB ImageNet state；旧配置仍兼容；
3. 对象 batch 先从 16 增至 32，再经过配对等价验证冻结为 64；只改变
   批处理分块，不改变最终输出。

3090 配对验收同时运行 P40 control 与 D4-only，并要求：

- Ship、Vehicle 与 P40 逐框完全一致；
- Aircraft 与正式 v3 所用 D4 分支在相同代理图上逐框完全一致；
- 配对增量时延投影低于 break-even `5.8023s`，目标低于 `4.5s`；
- Hard 与 Sentinel 均报告完整六率和时间，但不借它们重新选择 D4 参数。

实现索引：

- [D4 runtime](../../src/rsdet/submission/aircraft_d4.py)
- [Docker 主路径](../../src/rsdet/submission/competition.py)
- [冻结候选配置](../../configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json)
- [逐框与时延比较](../../scripts/compare_d4_only_runtime.py)
- [3090 驱动](../../scripts/server/run_attempt4_p40_aircraft_d4_only_3090_v1.sh)

## 3. 已完成工程验收

- 本机最终方案 17 专项：`35 passed`；新增 macOS AppleDouble 数据旁路文件
  回归后，数据/CV3 专项另得 `11 passed`；Ruff、`bash -n` 与
  `git diff --check` 全通过；
- 3090 当前代码全量回归：`1147 passed, 2 skipped in 76.93s`。两项 skip
  均为既有条件型测试，不是失败；
- 三卡服务器：真实 fold0 P40/donor 找到恰好 6 个目标头；α=0 所有 708 个 state
  tensor bitwise 一致，Ultralytics 重载为 25 类成功；
- 3090：完整 Aircraft checkpoint 无 ImageNet state 成功构建，参数量
  `27,839,353`，最终对象 batch=64；
- 两台所需资产 SHA 均已现场核验；GPU 起跑前无其他训练/推理进程。

全量回归还暴露并修复了一项与实验本身无关、但会影响 Linux 复现的真实问题：
从 macOS 复制的数据可能同时带入 `._*.jpg` 与 `._*.txt` AppleDouble 资源叉文件。
它们有合法扩展名却不是图像/UTF-8 标签。`XHDataset` 现明确忽略这类旁路文件，
并有独立回归测试；服务器上 4,481 张真实图与 4,481 个对应标签可以稳定重建
CV3，不再误计为 8,962 个样本。

首次启动因服务器克隆较旧、缺少若干调用脚本，在进入 GPU 推理前分别留下
`failed_exit_1/2`。没有权重、参数或输出被复用。同步完整当前代码后改用全新 R2
结果目录重启，保留初次失败现场，避免覆盖审计历史。

## 4. 执行位置与终态

|链|服务器|screen|结果目录|状态|
|---|---|---|---|---|
|Vehicle task vector CV3|3×4080 SUPER / port 10007|`attempt4-task-vector-r2`|`/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-VEHICLE-TASK-VECTOR-CV3-R2`|complete，拒绝|
|P40 + D4-only runtime|RTX 3090 / port 19864|`attempt4-d4-only-r2`|`/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-P40-AIRCRAFT-D4-ONLY-3090-R2`|complete，保留|
|D4 batch 32/64 配对时延|RTX 3090 / port 19864|`attempt4-d4-b64`|`/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-AIRCRAFT-D4-BATCH64-3090-V1`|complete，batch=64 准入|
|D4 tensorized + channels-last|RTX 3090 / port 19864|`attempt4-d4-tensor`|`/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-AIRCRAFT-D4-TENSORIZED-3090-V1`|complete，组合不准入|
|D4 tensorized-only 拆分实验|RTX 3090 / port 19864|`attempt4-d4-tensor-only`|`/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-AIRCRAFT-D4-TENSORIZED-ONLY-3090-V1`|complete，准入|

两条链均不自动 full、不自动打包 Docker、不自动正式提交。结果完成后先下载小型
audit/summary/SHA，再按上述门禁更新本报告。

### 4.1 任务向量的 coarse-ownership 现场发现

九组非零 α 推理完成后，严格审计发现：虽然 checkpoint 中 0..23 类权重行和全部
几何 state 均 bitwise 继承 P40，Ultralytics 的端到端后处理会先按 25 类最大分数做
全局 top-k；class24 分数变化因而会改变少数进入 top-k 的非 Vehicle anchor。相对历史
P40 缓存，三折高于 0.546 的非 Vehicle `(image,class)` 数量差分别为 `4/4/9`，不能把
单 checkpoint 直接宣称为类别旁路。

按方案 17 的 fail-safe 要求，分析改为显式 class-disjoint：0..23 始终取原 P40，
仅 class24 取 task-vector 输出。这样可以科学回答 Vehicle delta 是否有价值，但在它
通过所有统计门禁前不实现部署；若通过，仍必须实现并验收一次 forward 的
coarse-preserving delta head，离线双账本组合本身不得打包。

原驱动因上述 fail-safe 在分析阶段保留了 `failed_exit_1`，并非 GPU 推理或
checkpoint 失败。冻结输入上的 class-disjoint 恢复分析已生成
`policy_result_r2.json(status=complete)`；原失败现场未覆盖，详见
`task_vector/RECOVERY_RECEIPT.json`。

## 5. Vehicle task-vector 结果：拒绝

三个 held-out fold 分别由其余两折选出 `alpha=0.5 / 0 / 0.5`。
严格 outer-fold 结果为：

|held-out fold|选定 α|分数变化|Vehicle Recall 变化|Vehicle FDR|
|---:|---:|---:|---:|---:|
|0|0.5|`-0.2872`|`+4.5113pp`|`18.1818% -> 22.0779%`|
|1|0|`0`|`0`|`30.9524% -> 30.9524%`|
|2|0.5|`+0.1452`|`+4.4444pp`|`21.0526% -> 23.8806%`|

合并后 Vehicle 从 `128 TP / 37 FP / 274 FN` 变为
`140 TP / 46 FP / 262 FN`：召回增加了 12 个，但新增 9 个 FP。直接合并分数
仅 `+0.05384`，来源组 bootstrap `P10=-0.37720`，对新增 FP 施加官网
观察到的 6 倍风险后为 `-1.51717`。因此同时失败：

- 三折全正增益；
- 来源组 bootstrap P10 为正；
- 6× 增量 FP 压力下仍为正。

这不是“调小 α”可修复的边界问题：fold1 已经自动退回 α=0，其他两折
都表现为用较多 FP 换 Recall，且一折净负。故按冻结合同停止：不跑
Background-100MP、不实现 coarse-preserving delta head、不训练 full、不测
Ship task-vector。

## 6. P40 + Aircraft-D4-only 结果：边际正向保留

### 6.1 逐框所有权

Hard 与 Sentinel 两套数据上的四项断言均通过：

- Ship 和 Vehicle 与 P40 逐框完全一致；
- Aircraft 与正式 v3 的 D4 分支逐框完全一致；
- P40 Ship 与 v3 primary 参考完全一致。

这证明 Attempt 4 确实删除了 v3 失败的 hierarchy/Vehicle 分支，没有暗中
更改其他类别。

### 6.2 质量与时延

|proxy|Aircraft Recall|Aircraft FDR|D4 增量时延|投影官网总时延|break-even|
|---|---:|---:|---:|---:|---:|
|Hard|`99.0344% -> 99.2267%`|`2.8929% -> 2.2227%`|`+1.9915s`|`5.5433s`|通过|
|Sentinel|Recall 不变 `98.9635%`|`2.6784% -> 2.3827%`|`+1.5536s`|`5.1054s`|通过|

Hard/Sentinel 绝对分数中含 6 张超大图的代理时延，不用于预测官网分数。
对正式 v3 已知的 D4 质量增益 `+0.3215`，当前配对时延投影的时间扣分
约为 `0.222-0.284`，因此估计净增益只有约 `+0.04 至 +0.10`。该路线
是可解释且低风险的正向候选，但幅度很小，不应被描述为大幅升级。

batch=64 只作等价提速尝试，不是新的模型或阈值搜索。配对验证结果：

|proxy|预测逐框一致|batch32|batch64|提速|修正后官网总时延投影|
|---|---:|---:|---:|---:|---:|
|Hard|完全一致|`8.1082s`|`7.5008s`|`8.10%`|`4.9359s`|
|Sentinel|完全一致|`7.5556s`|`7.0915s`|`6.54%`|`4.6413s`|

修正方式是在原 D4-only 增量时延上加配对的 `batch64-batch32` 差，避免把
两次独立基线抖动当作优化收益。按正式 D4 质量收益复算，相对 v2 估计净
增益约为 `+0.124 至 +0.166`。batch=64 因而准入；它仍未达到 4.5 秒的理想目标。

`tensorized_views + channels_last` 的组合也保持两套最终 JSON 逐框完全一致，
但配对时延在 Hard 快 `11.67%`、Sentinel 却慢 `1.03%`。因为未在两个冻结
压力集同向，该组合不准入。随后只保留“单次归一化后用 tensor 生成 D4
视图”，关闭 channels-last 做单因素复核。该拆分实验在 Hard 快 `14.01%`，
Sentinel 慢 `0.0043s` / `0.06%`，属于 6 张图时延测量噪声；两套输出仍逐框完全一致，
且它在两套上都优于带 channels-last 的组合，因此准入。

最终工程配方固定为 `batch_objects=64`、`tensorized_views=true`、
`channels_last=false`。经两次配对差值累加修正，官网总时延投影约为
`4.0100-4.6455s`，对应 D4-only 相对 v2 的预计净增益约 `+0.165 至
+0.256`。这仍是代理投影，不是官网实测承诺。

## 7. 产物与决策

小型证据已回传到
[`outputs/HERA-GUARD-ATTEMPT4-20260904`](../../outputs/HERA-GUARD-ATTEMPT4-20260904/)，
其本地文件级 SHA 记录在 `LOCAL_SHA256SUMS.txt`。

当前决策：

1. **拒绝 Vehicle task-vector**，不做任何后验调参、full 或部署实现；
2. **保留 P40 + Aircraft-D4-only，并冻结 `batch_objects=64` + tensorized
   D4 views，不启用 channels-last**；
3. 本轮没有产生新的全量权重，因为 D4-only 复用已验收的 P40 full 和
   Aircraft full；
4. 尚未获得用户打包/提交指令，因此不会生成 Docker 或提交官网。

## 8. 2026-09-05 深层诊断补充

在不重训、不改变上述冻结结论的前提下，已对全部现有 Vehicle OOF 缓存做阈值工作区、
逐折同 FP 预算、低分 proposal 容量和新增框错误结构诊断；同时在 3090 上完成 P40 与
最终优化 D4 的同进程 AB/BA 重复测速。完整结论见
[Attempt 4 深层诊断](HERA_GUARD_ATTEMPT4_DEEP_DIAGNOSIS_20260905.md)。

新增证据对原结论作两项收紧：

1. Vehicle task-vector 在合并 FP 预算下表面的 `+3 TP/-2 FP` 无法逐折复现；每折在
   自己的基线 FP 预算内 TP 增益均为 0。其 46 个新增框为 29 TP、17 FP，TP 与背景 FP
   分数重叠，故拒绝原因已定位为站点相关的 proposal 校准而非 alpha 网格不足。
2. 旧 D4 时延投影由多次独立运行差值累加，偏乐观。更严格的 6 图×3 次同进程配对得到
   Hard `+1.501s`、Sentinel `+1.320s`；若正式 Aircraft 收益完全复现，D4-only 相对
   P40 的预计净分仅约 `+0.107--+0.133`。D4 仍为边际正向候选，但原第 6.2 节末尾的
   `4.0100--4.6455s / +0.165--+0.256` 投影由本次结果取代，不再作为当前估计。
