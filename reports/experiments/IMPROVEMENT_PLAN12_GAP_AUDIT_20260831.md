# 《改进方案12》全量问题核验与后续执行清单

日期：2026-08-31  
状态：`attempt_1_fdr_gate_failed / ext_v_2x2_rejected / protocol_repairs_complete`

## 1. 审计边界与当前结论

本文逐项核验《改进方案12》中的评分口径、P0-1至 P0-5、外部训练实现、数据安全、
DataLoader 采样、路线取舍、五次提交策略与最终交付。核验结论如下：

1. Attempt 1 是官方数据训练的 Y5-S Safe v2 同构镜像，不含外部数据、D-FINE、HAD 或新
   patch。正式得分 72.1331；Recall/时延通过，三粗类平均 FDR 0.236623，FDR 硬门失败。
2. 新绝对评分器、Hard 阈值冻结到 Sentinel、三种粗类聚合解释和 nested 综合分阈值选择
   已在本地代码完成。
3. EXT-V 重复采样已通过真实 Ultralytics Dataset 加载审计，没有被去重；
   batch 30 等效 batch 60 的假设成立。
4. partial-label patch 原实现缺少框去重，role view 使用硬链接，Detect 迁移只报兼容数；
   这三项已修复为 fail-closed 合同。
5. 当前服务器的 EXT-V-v1 不中断。它仍使用 drop-difficult 资产，只能作为第一版
   诊断。corrected-difficult 的转换代码已就绪，但只在 v1 明显正向时才值得重训。
6. 当前服务器快照仍只会跑三个 fine cell；独立 follow-up 已排队，会在第一张空闲
   GPU 自动补第四格 `EXT-V→omit`。
7. 40 epoch 只是配方快筛，绝不是正式提交权重。准入后仍需要 8+152=160 epoch full、
   Normal/Hard/Frozen-Sentinel 三折、3090 时延和 Docker 离线复验。

## 2. 评分口径核验

### 2.1 已实现

- `src/rsdet/evaluation/absolute_score.py`：逐段 Recall/FDR/time 子分和 `3/7,3/7,1/7`
  加权总分；对 `t=20` 按公告字面合同保留不连续边界并有单测。
- `scripts/score_absolute_preliminary.py`：同时输出
  `pooled_counts`(有计数时)、`macro_raw_then_score`和`mean_per_coarse_score`。
- `scripts/analyze_cv3_oof_pseudo_frontier.py`：新增两折选一折应用的 nested 综合分阈值；
  约束默认为 Recall>=0.87、FDR<=0.18，目标为发布的绝对综合分。
- `scripts/decide_hera_guard_final_candidate.py`：不再只用单一 pooled 解释；Hard 和
  Frozen-Sentinel 的三种可用聚合解释的最坏分差均不能为负。

### 2.2 正式回传已确认的聚合规则

使用 v2 页面展示的三类 Recall/FDR 和 2.704833 s 计算：

- macro raw 后计分：84.2245；
- 逐粗类计分后平均：84.9002；
- 当时页面展示：86.2274。

v3 两种 macro 解释都约为 84.2087，当时页面展示 85.0018。因此旧页面显示分不是
当前新公式对六个展示小数的简单 macro；可能是旧规则、隐藏 pooled 计数或更高精度数据。
正式 Attempt 1 的接口回传已确认：平台分别计算船、飞机、车辆的 Recall/FDR 子分，加上一个
时延子分，七项直接平均；硬门 Recall/FDR 标志使用三个展示粗类指标的算术平均。Attempt 1
三类平均 Recall 0.898414、FDR 0.236623，故只有 FDR 门失败。V1.6 所述 pooled 合并计数不再
是当前平台硬门的实际实现。

## 3. P0 逐项核验

|P0|文档问题|核验状态|证据/后续|
|---|---|---|---|
|P0-1|仓库驱动与真实 3GPU 命令漂移|本地修复，当前运行快照已单独锁定|本地 `run_hera_guard_final_extv_3gpu_fast.sh`固定 EXT-V-only/batch30/3GPU/环境锁；服务器真实入口 SHA `f7a64350...`，驱动 SHA `5f3cd559...`|
|P0-2|Sentinel 自己选阈值|已修复|候选驱动先从 Hard frontier 读取每折阈值，Sentinel 只应用；decision 必须提供 frozen 两件套|
|P0-3|仍围绕 FDR15，不直接优化新分|已实现，待候选实测|nested score selector+三聚合 scorer+最坏分差门；原 FDR 前沿仅保留为诊断|
|P0-4|1,637 difficult 被删标但像素保留|转换代码已修，当前 v1 不中断|`difficult_policy=keep_primary` 只保留 plane/ship/small-vehicle/large-vehicle；场景结构仍为背景。只在 v1 明显正向时重物化/重训|
|P0-5|缺 `EXT-V→omit`|已排队|`run_hera_guard_final_extv_omit_followup.sh` 在第一个 fine cell 结束后使用其空闲 GPU 补第四格，得到完整 2x2 因果设计|

## 4. 外部训练实现核验

### 4.1 Detect head 与迁移

保留 fresh native `DetectionModel` 重建整个 Detect head 的正确设计。新硬门包括：

- 目标 backbone/neck 全部 tensor 必须在来源 checkpoint 中存在且 shape 一致；
- 所有这些 tensor 在 head reset 前必须与来源值完全相等；
- 审计输出完整 tensor 名列表与名列表 SHA；
- 源/目标 model YAML 的结构签名必须完全一致，仅忽略 `nc/names` 类别元数据。

服务器使用官方 yolo26s 做真实构造验证：源/目标结构 SHA 均为
`0d87b686...e4f50a`，468/468 个 backbone/neck tensor 精确覆盖。

### 4.2 partial-label-safe patch

新实现在写入 class 24 前强制检查 `FINE_NAMES[24] == FSC`，并对两类重复框做 IoU>=0.5
去重：

- confirmed 与原始 class-24 框；
- confirmed 与已接受 confirmed 框。

审计 JSON 分别记录两类去重数。仍需在最终采用 patch 前对实际加入的 18 框做一次图像紧密度
复核；这一步只在 patch cell 真正正向时执行。

### 4.3 role view 与真实采样

role view 标签改为 `shutil.copy2`，不再共享 inode；图像仍是只读目录软链接。当前已运行资产
无需重做，但最终审计必须重验源标签 SHA。

对当前 EXT-V 资产的真实 Ultralytics 8.4.103 Dataset 审计：

| 字段 | 实测 |
|---|---:|
|train list rows|13,831|
|Dataset loaded rows|13,831|
|unique images|9,153|
|1x / 2x / 4x unique tiles|5,927 / 2,500 / 726|
|aircraft loaded instances|12,868|
|ship loaded instances|43,287|
|vehicle loaded instances|100,040|
|other loaded instances|26,364|
|global batch / world size|30 / 3|
|trainer nbs / accumulate / effective batch|64 / 2 / 60|
|actual optimizer weight decay|0.00046875|

审计产物：
`outputs/HERA-GUARD-FINAL-AUDIT-20260831/ext-v-role-runtime-audit.json`，SHA256
`09f7f2a55fba434fa430f4e61afcd6dcf8b0e010c88024c4fdbc0b26a66cc4d2`。

## 5. 当前服务器真实状态与漂移处理

22:00 实查：EXT-V coarse 已写入 67 行 `results.csv`（含表头，即完成 66/80 epoch），
3 GPU 负载健康，继续运行。

真实服务器入口是隔离快照，不是污染的日常 checkout：

- wrapper：`/root/hera-ops/run_extv_fast.sh`，SHA `f7a643508d...`；
- frozen repo：`/root/autodl-tmp/hera-final`；
- driver SHA：`5f3cd5598c...`；
- Python 3.10.13 / torch 2.5.1+cu121 / CUDA 12.1 / Ultralytics 8.4.103 /
  NumPy 1.26.4 / Pillow 10.4.0 / PyYAML 6.0.2；
- initial weight SHA：`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`。

不在运行中覆盖这个快照。筛选完成后，候选评估必须使用包含本文修复的新冻结快照，
否则 Sentinel 和新综合分结论无效。

## 6. 筛选完成后的唯一执行树

1. 完成四格：official-omit、official-patch、EXT-V-omit、EXT-V-patch。
2. 固定配对对比，不调阈值/融合权重：
   - external 主效应：同 patch 与同 omit 下的 EXT-V vs official；
   - patch 主效应：同初始化下 patch vs omit；
   - interaction：两个 patch 差的差。
3. 候选只有在 Normal、Hard、Frozen-Sentinel 和三种综合分解释上通过才扩三折。
4. v1 如果增益很小或为负：整条 external/patch 路线停止，不为 sunk cost 消耗官方机会。
5. v1 如果明显正向：物化 corrected-difficult 资产并做唯一重训对照；它必须胜过 v1 才取代。
6. 唯一入选配方训练 8 epoch fresh-head warmup + 152 epoch full，不使用 held-out 选 epoch。
7. 复验 3090 latency、断网 Docker、schema、max_det 触顶、资产 SHA。
8. 仅当保守预估相对已有官方最高分 >=+0.5 时才提 Attempt 2。

## 7. 五次正式机会的当前版本

|Attempt|用途|当前状态|
|---|---|---|
|1|Safe official-only 真实锚点|已提交，等待评测|
|2|唯一 external full 候选|只在四格+三折+三外层+160 epoch 全通过时提交|
|3|同一最强权重的 nested 综合分全局阈值|相对已提交最强版本内部保守增益 >=+0.7 才用|
|4|corrected-difficult 或 patch interaction 形成的真独立候选|不用于小阈值/TTA/D-FINE union|
|5|异常恢复或最强版的最后冻结|保留，9 月 4 日前不用|

## 8. 明确停止的路线（不得重复消耗）

- HAD branch-only 与 terminal-FPN；
- D-FINE 双模型部署（教师权重只作训练/审计资产）；
- EXT-G；
- DIOR（在 EXT-V 明确正向前）；
- Q0/FPN/PAV/新 verifier；
- 常规阈值、NMS、TTA 网格搜索；
- 只在单折正向的候选。

## 9. 仍待完成的交付门禁

- [x] Attempt 1 官方分项与总分回传，更新聚合口径推断。
- [x] EXT-V-v1 80 epoch 完成、checkpoint/result/environment/code SHA 验收。
- [x] 四格 40 epoch 完成；确认每格同 fold、同数据排除、同 seed、同训练预算。
- [ ] 若 patch 正向，代理对 18 个实际加入框做图像紧密度复核。
- [x] 候选评估用新冻结代码快照，Sentinel 显示 `frozen_from_hard=true`；四格全部拒绝。
- [ ] 若入选，扩三折并用新 scorer 选择配方，不用 pooled oracle。
- [ ] 若 v1 强正向，才审核 corrected-difficult 资产和重训成本。
- [ ] 最终 full 160 epoch、单一 last checkpoint、不用 validation 选 epoch。
- [ ] 源标签 SHA 复验；role labels 必须是独立 inode。
- [ ] 3090 延迟、断网启动、三类 smoke、10K smoke、schema、max_det、Docker digest。
- [ ] 代码、训练合同、数据审计、权重 SHA、Docker 五者引用同一候选 ID。

## 10. 文档中其他问题的处理结论

- 外部数据不是单独创新点；有效表述是 Partial-Label-Safe Training、
  Structured-Background-Aware Coarse Pretraining、Audited Coarse-to-Fine Transfer 和
  Official-Score-Aligned Deployment 组成的整体闭环。
- 最终部署仍必须是单视图、单 Y5-S；外部 coarse head 和 D-FINE 不进 Docker。
- 93/95 只是目标级别，不是候选准入证据。从已知平台锚点到 93 需要约 +2pp Recall 且
  -3.5至4pp FDR 的联合跨越，不可用阈值幻觉替代。
- 当前不需要再开新科学方向。下一个信息节点是四格配对结果，而不是更多方法列表。
