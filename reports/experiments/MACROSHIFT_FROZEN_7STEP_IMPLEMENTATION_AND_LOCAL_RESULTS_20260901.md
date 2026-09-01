# MacroShift 冻结七步：实现、审计与本地结果

日期：2026-09-01
正式指标协议：`platform_observed_20260831`
状态：本地可完成部分已完成；服务器训练与真正未见 Sentinel-B 仍按门禁等待。

## 1. 输入与范围

本轮同时核对了两份方案全文，而不是只读取项目根目录的一份：

- `../改进方案13.md`：913 行，SHA256 `3579829a2e7c9118ddf1e758dbd771d33e49cdaa2d12e432d2d3d5cfd3b359e3`。
- `/Users/suzuku/Downloads/HERA_GUARD_MACROSHIFT_CHAMPIONSHIP_PLAN_20260901.md`：817 行，SHA256 `1354e9a633b69dc6b0b7c846b6bedeba94e2604a2e696ef85f740d1c0ee95ddc`。

遗漏的下载文档已经完整纳入。文档中关于协议迁移、MacroRisk、Vehicle
rescue、Ship 分流、Background-100MP、Sentinel-B 和唯一 full 配方的内容均有对应实现。
Composite-10K 属于后续训练资产，不得在本轮七步门禁之前替代 Background-100MP 或
Sentinel-B，也没有被伪装成独立验证集。

## 2. 一页结论

| 步骤 | 本地状态 | 结论 |
|---|---|---|
| 1. 协议迁移 | 完成 | 13 个当前正式入口均绑定 `platform_observed_20260831`，审计通过；历史脚本只允许复现，不得准入 |
| 2. 细类阈值部署 | 完成 | 25 类完整性、优先级和 direct/safe/global/Docker 逐框一致性均有测试 |
| 3. MacroRisk V2 | 完成并拒绝 | 相对 0.15 基线宏召回下降 8.596pp；1000 次组 bootstrap 联合通过概率 0，禁止部署 |
| 4. Vehicle reject/rescue | 完成并拒绝 | Vehicle Recall +1.244pp，但 Vehicle FDR +1.169pp；不满足无 FDR 回退门禁 |
| 5. Ship 分流 | 完成并拒绝 | 分层审核只授权测试 `objectness_quality`；单卡三折正式回放后 Ship Recall -1.058pp、FDR +1.150pp，停止该方向 |
| 6a. Background-100MP | 完成并冻结 | 382 图、100.139008MP、GT 64px 排除、382/382 全量视觉审核、38 个可疑候选剔除 |
| 6b. Sentinel-B | 代码完成，资产阻塞 | 当前所有有标签比赛组均参与过开发；把它们重命名为 Sentinel-B 会形成假独立，故 fail-closed |
| 7. 独立组合/full | 完成，当前阻塞 | 当前没有独立通过全部门禁的新模块，因此 recipe 接受模块数为 0，唯一 full 训练被正确阻止 |

这不是“没完成”，而是门禁工作正常：两条看似可提升的路线被严格外层回放拒绝，避免浪费
唯一 full 训练和官方提交次数。

## 3. Step 1：统一正式指标协议

### 3.1 唯一口径

实现：

- `src/rsdet/evaluation/platform_protocol.py`：三大类宏 Recall/FDR、官方绝对分和硬门槛的唯一构造器。
- `src/rsdet/evaluation/protocol.py`：协议解析与 fail-closed。
- `configs/project.yaml`：固定 `metric_protocol: platform_observed_20260831`。
- `configs/evaluation/metric_protocol_registry.json`：当前正式入口白名单。
- `scripts/audit_metric_protocol_migration.py`：入口内容和协议绑定审计。

正式 gate 定义为三个大类分别由细类宏平均，再对 Ship/Aircraft/Vehicle 等权平均。全细类 pooled
计数只保留为诊断，不能再决定正式准入。部分 taxonomy 的单元测试/诊断允许回退到历史诊断值，
但 `submission/competition.py` 和正式准入仍要求完整协议与完整 taxonomy。

本地证据：

- `outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/metric_protocol_audit.json`
- `status=pass`，正式入口 13/13 绑定。

历史研究脚本没有被删除；它们是复现实验资产，不属于新的正式准入入口。后续新增正式入口必须先加入
registry 并通过审计，否则不能进入 recipe。

## 4. Step 2：`score_threshold_by_fine`

### 4.1 统一解析和优先级

实现：

- `src/rsdet/postprocess/thresholds.py`
- `src/rsdet/postprocess/safe_tile_fusion.py`
- `src/rsdet/postprocess/global_aggregation.py`
- `src/rsdet/pipeline/large_image.py`
- `src/rsdet/submission/competition.py`

逐框阈值优先级固定为：

1. `score_threshold_by_fine[category_id]`
2. `score_threshold_by_coarse[coarse_class]`
3. `score_threshold`

Docker 正式配置一旦声明 fine map，必须恰好覆盖 `0..24`，拒绝缺项、重复/非法 key、NaN、越界值。
阈值在全局聚合之后施加，safe/global/direct 三条路径使用同一个函数，防止离线和容器次序不同。

测试：`tests/test_fine_threshold_deployment.py` 覆盖 direct、safe、global 和 Docker 配置；逐框保留集合、
原始框、细类标签一致。

### 4.2 当前部署决策

部署能力已经就绪，但 MacroRisk V2 没有通过门禁，所以没有把其 25 阈值写入提交配置。能力完成不等于
候选被准入。

## 5. Step 3：MacroRisk V2

实现：

- `src/rsdet/evaluation/macro_risk_v2.py`
- `scripts/analyze_macro_risk_v2.py`
- `tests/test_macro_risk_v2.py`

每个 held-out fold 的 25 个细类阈值只用另外两折拟合。细类 raw 阈值在 logit 空间向大类 anchor
收缩，并按 GT 数和独立 group 数施加 `0.2/0.4/0.8` 最大移动。最后以 group 而非图像进行
1000 次 bootstrap。

输入：

- `tmp/macroshift_full_oof_gt.json`
- `tmp/macroshift_y5_oof_predictions.json`
- `tmp/macroshift_group_map.csv`

结果：`outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/macro_risk_v2_cv3.json`

| 指标 | 0.15 基线 | MacroRisk V2 | 差值 |
|---|---:|---:|---:|
| 三大类宏 Recall | 72.4984% | 63.9024% | -8.5960pp |
| 三大类宏 FDR | 26.1985% | 15.7944% | -10.4041pp |
| 绝对分代理 | 62.6758 | 64.4382 | +1.7624 |

它通过牺牲大量 Recall 换 FDR/分数代理，在硬 Recall 门槛下不可用。bootstrap：Recall P10
60.7629%，FDR P90 19.2451%，联合门槛通过概率 0/1000。`admitted=false`。

## 6. Step 4：Vehicle Reject + Rescue 与 selective D-FINE

实现：

- `src/rsdet/submission/vehicle_rescue.py`
- `scripts/analyze_vehicle_reject_rescue.py`
- `configs/experiments/macroshift_vehicle_rescue_v1.json`
- `tests/test_vehicle_rescue.py`

约束：

- Ship/Aircraft 逐框原样保留。
- Vehicle core 原样保留。
- D-FINE/specialist 只能支持已有 primary tail 框，不能生成新框、改框或改细类。
- specialist 只在含 Vehicle tail proposal 的 tile 上执行，并记录 primary/specialist tile 数。
- 配置只在另外两折选择；held-out fold 不参与选择。

结果：`outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/vehicle_reject_rescue_cv3.json`

| Vehicle | 基线 | 候选 | 差值 |
|---|---:|---:|---:|
| Recall | 54.9751% | 56.2189% | +1.2438pp |
| FDR | 34.8083% | 35.9773% | +1.1690pp |

总 gate Recall 只增加 0.4146pp，gate FDR 恶化 0.3897pp，绝对分近乎不变。结论：代码保留，
候选拒绝；不能因为 Recall 单项好看而进入组合。

## 7. Step 5：Ship 分层审核和唯一方向

实现与证据链：

- `scripts/decompose_coco_oof_errors.py`：按冻结 official-match hierarchy 重新分解当前 Y5 OOF。
- `scripts/build_ship_error_review.py`：按 reason × fine class 分层抽取。
- `scripts/render_ship_error_review.py`：生成 129 张上下文卡，红框为 FP、绿框为 FN。
- `scripts/decide_ship_training_direction.py`
- `src/rsdet/evaluation/error_route.py`
- `configs/experiments/macroshift_ship_objectness_quality_v1.json`
- `scripts/train_official_quality_head.py --coarse-filter ship`
- `scripts/server/run_macroshift_ship_quality_single_gpu.sh`：单卡逐折持锁执行器。
- `scripts/export_sparse_quality_oof.py`：Ship 稀疏质量分数与非 Ship identity 的逐框合并、NMS 和覆盖审计。
- `scripts/analyze_ship_quality_cv3.py`：两折选择 Ship 阈值、一折只评估的正式协议外层 CV3。

当前 Y5 OOF、0.15 工作点的 Ship 分解：

- FP_DUP 166、FP_CLS 83、FP_LOC 37、FP_BG 576。
- FN_CLS 83、FN_LOC 37、FN_MISS 295。
- 在决定训练方向的 BG-vs-CLS 子集中：FP_BG 占 87.4052%；FN_MISS 占 78.0423%。

分层审核队列共 129 张，覆盖 15 个 reason × fine strata。视觉审核发现一个关键事实：不少官方
`FP_BG` 视觉上像真实船，但没有可匹配 GT，因而在比赛规则里仍是 FP。这意味着：

- 不应把所有 FP_BG 当“纯海面背景”做粗暴负样本；
- 应学习 official-match quality：同细类可匹配性、当前工作点 protected TP、active FP 和受限残差；
- `fine_tail` 主要解决细分类，不对应当前主导错误，明确排除；
- `objectness_quality` 是唯一允许进入服务器 CV3 的 Ship 方向；“方向选中”不等于模块准入。

当前输出：

- `outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/y5_oof_error_decomposition/`
- `outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/ship_error_review.csv`
- `outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/ship_error_review_sheets/`
- `outputs/MACROSHIFT-LOCAL-PREFLIGHT-20260901/ship_training_direction.json`

### 7.1 单卡服务器 CV3 实测（2026-09-01）

服务器 `17879` 的 RTX 3090 使用既有 deployable metadata 缓存完成三折，无需重新提取图像特征：

- cache SHA256：`9efe398bfb115accc596ce735be716fd3b9bcf2e0d941a7940baa13d5672f74f`；
- 65,301 行、实际 63 维、Ship 20,310 行；非 Ship 逐框保持 identity；
- fold 0/1/2 固定第 20 轮 held-out TP-vs-FP pair accuracy：0.8041 / 0.8928 / 0.8170；
- 三折训练产物、TorchScript 和稀疏 OOF 覆盖均完整。

第一份“训练折绝对分最大”诊断把 Ship 阈值推到 0.626–0.901，虽然 gate FDR -1.296pp、
代理分 +1.049，但 gate Recall -0.184pp、Ship Recall -0.551pp，拒绝。它揭示目标函数过度偏向
FDR，不作为正式模块判断。

随后按冻结因果问题重跑：baseline Ship 阈值固定 0.15；候选阈值只能在另外两折中选择，目标为在
不高于训练折 baseline Ship macro FDR 时最大化 Ship macro Recall。held-out 三折合并结果：

| 指标 | baseline | Ship quality | 差值 |
|---|---:|---:|---:|
| gate Recall | 68.1367% | 67.7842% | -0.3525pp |
| gate FDR | 18.6737% | 19.0571% | +0.3834pp |
| Ship Recall | 69.9314% | 68.8738% | -1.0576pp |
| Ship FDR | 28.8640% | 30.0141% | +1.1501pp |
| 绝对分代理 | 64.1533 | 63.9162 | -0.2370 |

`preliminary_normal_cv3_admission=false`。根据预注册停止条件，不继续跑 Hard、Sentinel-A、
Background-100MP，也不把该模块写入 final recipe。小型回传位于
`outputs/MACROSHIFT-SHIP-QUALITY-V1-20260901/`；正式分析 SHA256 为
`84d4aa41705995adea47534c5f420438b1c170821367e98988d22ae317a8029d`。

## 8. Step 6：Background-100MP 与 Sentinel-B

### 8.1 Background-100MP

实现：

- `scripts/build_background_100mp.py`
- `scripts/build_background_review_sheets.py`
- `scripts/compile_background_visual_review.py`
- `scripts/freeze_background_100mp.py`
- `scripts/evaluate_background_100mp.py`
- `src/rsdet/evaluation/background_stress.py`
- `configs/evaluation/background_100mp_visual_exclusions_v1.json`

冻结资产：`outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN/`

- 512×512 crops 382 张，共 100.139008MP。
- 来源图 299 张。
- 所有 crop 与 GT 扩张 64px 区域无交叉。
- 初始 382 张全量审核、后续替换增量审核；38 个可见或含混船只候选永久排除。
- 最终 382/382 覆盖再次编译。
- manifest SHA256：`ed3cbbe6952ea5a7792821a316bd3b0ed93888f74a50eda2630f630c9c9020e7`。
- `freeze_decision.json`: `formal_admission=true`。

它只用于压力测试 FP/100MP，不能替代有 GT 的 Recall/FDR 集。

### 8.2 Sentinel-B

实现：

- `scripts/freeze_sentinel_b.py`
- `configs/evaluation/sentinel_b_registry_template.csv`

状态：`outputs/MACROSHIFT-SENTINEL-B-V1/status.json` 为
`blocked_no_unseen_source_registry`。当前所有本地有标签比赛组都参与过训练、OOF、Hard 或既有
sentinel 开发，不能通过改名制造“新未见集”。freeze 脚本已经实现 source/group/SHA/预测前冻结
门禁；拿到真正新来源 registry 后可以立即冻结。

## 9. Step 7：只组合独立通过模块并训练唯一 full

实现：

- `src/rsdet/evaluation/module_admission.py`
- `scripts/compose_macroshift_final_recipe.py`
- `configs/experiments/macroshift_final_baseline_v1.json`
- `scripts/train_macroshift_full.py`
- `tests/test_error_route_background_admission.py`

每个模块必须同时满足：协议正确、独立外层评估、Recall +0.5pp、FDR 不回退、绝对分上升、任一大类
Recall 下降不超过 0.5pp、时延回退不超过 2 秒、Background 不恶化、若有 Sentinel-B 则必须通过。
Ship fine-tail 与 objectness-quality 互斥。

当前 recipe（服务器 Ship 回放后仍不变）：

- `accepted_modules=[]`
- `unique_full_training_admission=false`
- `train_macroshift_full.py` 会拒绝启动。

这是正确的当前状态。Ship objectness-quality 已和 MacroRisk、Vehicle 一样被独立外层门禁拒绝；
不得重生成可训练 recipe，也不得启动唯一 full。

## 10. 验证

- 新增/相关专项：20 passed。
- 全仓第一次：895 passed、5 skipped、3 个协议迁移兼容失败。
- 三处均已修复：配置测试加入新协议；部分 taxonomy 仅保留诊断回退，不影响正式完整 taxonomy
  fail-closed。
- 最终全仓：898 passed、5 skipped。
- 变更 Python 文件 `ruff`：all checks passed。

## 11. 当前停止状态与后续解锁条件

1. 七步本轮结束：MacroRisk、Vehicle、Ship 三个候选均已拒绝。
2. 不运行 Ship fine-tail，不组合三个拒绝模块，不训练新 full。
3. 只有新的独立候选先通过 Normal 外层门禁，才运行 Hard/Sentinel/Background。
4. 真正全新来源出现后，先冻结 Sentinel-B，再允许任何候选查看其预测。
5. 独立候选通过全部门禁后才能生成 module admission JSON → recipe SHA → 唯一 full。

完整命令见 `docs/server/MACROSHIFT_FROZEN_7STEP_SERVER_RUNBOOK_20260901.md`。

## 12. 同日遗留候选补充读出

七步之后又完成了三个此前已有资产但未按新正式协议收口的方向：有限阈值迁移、coarse
score-sqrt、base+crop quality（含标签无关分位数校准）。三者均未通过完整门禁，
`accepted_modules=[]` 和 `unique_full_training_admission=false` 保持不变。详细数字、正确
Hard/Sentinel 资产 SHA、技术无效复跑与最终结论见
`reports/experiments/MACROSHIFT_ATTEMPT2_CANDIDATE_TOURNAMENT_20260901.md`。
