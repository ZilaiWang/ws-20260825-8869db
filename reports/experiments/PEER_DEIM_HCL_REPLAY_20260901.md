# 同赛道公开方案 DEIM-HCL 严格复现实验（2026-09-01）

## 1. 目标与边界

本实验复核公开仓库
[`star1sakura/XH-202625-remote-sensing-detection`](https://github.com/star1sakura/XH-202625-remote-sensing-detection)
中最值得迁移的一项设计：在 DEIM/D-FINE 检测器中加入分类/定位查询解耦，以及仅在训练期
生效的层级对比损失 HCL。目标不是复现对方自定义划分上的报告数字，而是回答：

1. 该设计在本项目冻结的来源分组 fold0 上，是否优于同容量 DEIM-M；
2. 增益能否穿过固定 Hard-10K 和 source-disjoint Sentinel-B；
3. 若成立，应如何作为训练期模块接入当前正式主线；
4. 若不成立，立即停止模型放大、三折扩展和参数扫描。

公开仓库当前没有可识别的许可证文件。因此所有代码均通过固定 commit 的隔离 checkout
运行，结果标记为 `research_only_unlicensed_reference=true`；其源码不进入部署包，不能在
许可证问题澄清前直接发布或提交比赛。

## 2. 为什么选择该方案

公开方案报告的固定验证集规模只有 674 张，且并非本项目的机场代理来源分组 CV3；它的
Ship/Aircraft/Vehicle 结果不能与我们的官方代理直接比较。但其方法具有三个可检验优点：

- HCL 只改变训练目标，部署时可以移除投影头，不增加正式推理时延；
- 分类查询与定位查询解耦，理论上针对当前 `FP_CLS` 与细类长尾，而不牺牲框回归；
- 基于 DEIM/D-FINE，与本项目已有异构检测器和固定评测器能形成真正同架构配对对照。

本项目早期已有独立的 BHC-DETR/BHCL 实现，但它使用另一套 DETR 骨干、训练器和数据合同，
尚不能证明“在成熟 DEIM 基线上仅加入查询解耦与 HCL”有收益。本次实验正是补上这个最小
因果问题，不能用旧 BHC 软件冒烟代替。

## 3. 固定资产和代码血缘

| 项目 | 固定值 |
|---|---|
| 公开方案 commit | `d23ef57ea5e3ea80ec71e883776718a8c3c1510a` |
| DEIM 上游 commit | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` |
| 模型 | DEIM/D-FINE-M，1024，25 细类 |
| COCO 初始化权重 SHA256 | `2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc` |
| fold0 train COCO SHA256 | `41e93416083ad39cd8b665b53be6613f81d9d9d6c1d052da1809b7e71d5686ef` |
| fold0 val COCO SHA256 | `2641d3bb15388b9a19812ab514b993d5f68ef90d7a59fb02834bf7903e585977` |
| 基线 last.pth SHA256 | `e2ce69622422bc642a555ffe331c6bd50b9e62761f9951ae54b7581f8b4178cf` |
| epoch / seed | 40 / 42，固定最终 epoch39，不选 best |
| batch / workers | 4 / 8 |
| 优化器与增强 | 与现有 DEIM-M fold0 配置相同 |
| 唯一模型变量 | decoupled queries + HCL，HCL 权重 0.6 |

第一次预跑误把候选 `num_workers` 写成 6。它不改变模型定义，但会改变数据增强 worker 的
随机流，不能作为严格单因素证据。该预跑在完成 epoch0 后中止并保留为
`INVALID-NUMWORKERS6`，随后把 train/val workers 都纠正为 8 后从初始化重新开始；没有
resume，也没有混用无效权重。

## 4. 冻结评测流程

### 4.1 Normal fold0 配对

基线是已完成的 `DEIM-M-FOLD0-40EP-V1-R2`。在 FDR=0.15 的诊断前沿：

| 指标 | 基线 |
|---|---:|
| pooled Recall | 0.7514286 |
| macro Recall | 0.651948 |
| Ship Recall | 0.66519 |
| Aircraft Recall | 0.77376 |
| Vehicle Recall | 0.27820 |

候选必须同时满足：pooled Recall 增益至少 `+0.5pp`；macro Recall 不下降；任一粗类
Recall 不下降超过 `0.5pp`。该 fold 的阈值前沿使用 held-out 标签，因此只能决定是否继续
压力测试，不能选部署阈值或形成正式准入结论。

### 4.2 固定 Hard 与 Sentinel-B

只有 Normal 门通过才自动执行：

1. 基线和候选分别对两张 Hard 10K 图做同一套安全切片推理；
2. 基线和候选分别对两张 source-disjoint Sentinel-B 10K 图推理；
3. 切片固定为 1024、overlap 256、score floor 0.001；
4. 细类 NMS 0.70、粗类 NMS 0.85、safe merge IoU/IoS 为 0.50/0.75；
5. 每个模型只在 Hard 上取得自己的 FDR15 阈值，再原样迁移到 Sentinel-B；
6. Sentinel-B 标签不参与模型、阈值或融合选择，只作一次性外层读出。

压力门同时要求：Hard pooled Recall `+0.5pp`、Hard macro 不下降、Hard 任一粗类不下降
超过 `0.5pp`，以及 Sentinel pooled/macro/三个粗类均不发生超过门限的退化。只有全部通过，
才允许运行来源分组 CV3；否则停止，不尝试 L 模型、不同 HCL 权重或融合权重扫描。

## 5. 启动时执行状态（已由第 8 节终态取代）

服务器为 `cv3-seetacloud-2`，单张 RTX 3090。

| 链 | 状态 |
|---|---|
| 隔离 checkout、commit 审计、补丁审计 | 完成 |
| 模型构建与 HCL forward 冒烟 | 完成 |
| 基线 10K tile 适配器真实图块冒烟 | 完成，300 个低阈值候选，类别与框合法 |
| 严格候选训练 | 运行中，40 epoch |
| Normal 配对推理和前沿 | 已排队 |
| Hard/Sentinel 固定压力链 | 已在独立 screen 等待 Normal 门 |
| 模型放大 / CV3 / full | 未准入、未启动 |

服务器结果目录：

- `/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-40EP-V1/`
- `/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/`

## 6. 与当前主线的结合方式

当前可部署主线仍是 full YOLO26-s/Y5 单视图 identity 链；正式隐藏集锚点为 72.1331，
主要问题是 Ship/Vehicle FDR，而不是时延。DEIM-HCL 不能在单折结果出来前直接替换主线。

若本实验通过三重门，结合顺序冻结为：

1. **先做 DEIM-HCL 来源分组 CV3**：确认收益不是 fold0 或阈值偶然性；
2. **作为异构候选教师**：只验证它能否补 YOLO 的 Ship/Vehicle 真目标，或对 YOLO FP
   提供同细类反证；不默认做全候选并集；
3. **训练期蒸馏**：HCL 表征若跨域稳定，可蒸馏进现有单模型学生，保留正式 2--3 秒时延；
4. **最后才考虑单模型替换**：必须在 `platform_observed_20260831`、Hard、Sentinel-B 和
   3090 时延上都超过 incumbent。

若只提高 Aircraft 而不改善 Ship/Vehicle，或 Normal 正而 Sentinel 负，则不进入主线。
如果 DEIM-HCL 本身通过但双检测器时延/融合不合格，只保留训练期教师角色。

## 7. 实现索引

- 候选配置：`configs/experiments/deim_hcl_m_fold0_40ep.yml`
- 训练与 Normal 配对链：`scripts/server/run_peer_deim_hcl_m_fold0_screen.sh`
- DEIM 大图切片适配器：`scripts/infer_deim_tiled_coco.py`
- 固定阈值官方匹配：`scripts/evaluate_fixed_score_threshold.py`
- Hard/Sentinel 决策器：`scripts/decide_peer_fixed_benchmarks.py`
- 自动压力链：`scripts/server/run_peer_deim_hcl_fixed_benchmarks.sh`
- 公共扩展运行时元数据 shim：`research/peer_runtime/`
- 合同测试：`tests/test_peer_deim_hcl_screen_contract.py`、
  `tests/test_deim_tiled_coco_contract.py`、
  `tests/test_decide_peer_fixed_benchmarks.py`

本地当前门禁：7 项专项测试通过、Python 编译通过、Shell 语法通过、ruff 通过、
`git diff --check` 通过。

## 8. 最终结果与口径修正（2026-09-02）

训练、Normal、Hard 和 Sentinel-B 均已完成并回传。首次分析脚本按 pooled 指标选点，
只保留为历史诊断；正式补算已经迁移到 `platform_observed_20260831`。

Hard 上各模型独立选择三粗类 macro FDR≤0.15 的最大三粗类 macro Recall 点：

| 模型 | 阈值 | Gate Recall | Gate FDR | 单折六质量子分 oracle |
|---|---:|---:|---:|---:|
| DEIM-M | 0.626 | 32.370% | 8.593% | 56.136（t=0.706） |
| DEIM-HCL-M | 0.621 | 35.874% | 14.807% | 55.771（t=0.651） |

HCL 的三个粗类 Recall 都提高，Gate Recall 增加 3.504pp；但 Gate FDR 增加
6.213pp，分段计分后的六质量子分反而下降 0.365。将上述 Hard 阈值原样迁移到
Sentinel-B 后，Gate Recall 增加 4.580pp，Gate FDR 仍增加 1.479pp。

因此最终结论为 `complete_positive_recall_direction_but_no_admission`：HCL 有真实的
召回信号，但当前置信度排序/校准使新增候选的虚警代价过高，不进入 folds1/2、full、
Docker，也不扫描 HCL 权重、尺度或融合权重。若未来重开，应把它作为排序/蒸馏研究证据，
而不是直接部署候选。

补算原件：

- `outputs/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/hard_{baseline,candidate}_frontier_platform_v2.json`；
- `outputs/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/sentinel_{baseline,candidate}_fixed_platform_v2.json`；
- `outputs/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/fixed_benchmark_decision_platform_v2.json`。
