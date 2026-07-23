# MAR20 来源重叠防护分组与正式 CV3 执行方案 v1.1

## 0. 文档状态

| 项目 | 内容 |
|---|---|
| 文档性质 | 可直接转化为代码、服务器任务单和人工复核工作的实施规范 |
| 直接目标 | 为竞赛中的 3,073 张 MAR20 图像建立可审计的来源代理组，并生成能够防止明显同源泄漏的正式 CV3 |
| 前置方案 | `reports/data/MAR20_AIRPORT_PROXY_GROUPING_PROTOCOL_v1.md` |
| 外部评审意见 | 工作区 `doc/机场问题.md`；本文已完成逐项判断、吸收和修订 |
| 当前版本 | v1.1，开始编码前冻结；参数经校准改变时必须产生带理由的 amendment，而不是静默覆盖 |
| 正式默认数据边界 | `target_only`：只使用竞赛中的 3,073 张 MAR20 图像建立正式组 |
| 诊断数据边界 | `full_bridge`：允许额外 769 张原始 MAR20 图像仅参与来源关系发现，不参与任何比赛模型训练 |
| 正式结论命名 | `source-overlap guarded CV` 或“来源重叠防护 CV”；不得无证据宣称为“机场完全隔离 CV” |

本文不是重新讨论机场问题，而是冻结“接下来具体写什么、跑什么、看什么、什么情况下继续或回退”。当本文与旧协议在执行细节上冲突时，以本文为准；问题背景、证据依据和前序调查仍以旧协议为补充。

执行优先级为：

1. 官方赛题与数据使用规则；
2. 项目冻结的数据接口和评估合同；
3. 本执行方案；
4. 旧版机场代理分组协议；
5. 临时讨论记录和个人判断。

---

## 1. 最终判断与本次修订

### 1.1 我们真正能够完成的任务

公开 MAR20 数据没有提供可靠的“图像—真实机场 ID”表。由图像外观可以较可靠地判断两张图是否来自同一帧、同一局部地面区域或存在可验证的视觉重叠，但不能保证从所有不重叠视角中恢复 60 个真实机场。

因此任务分为三个证据层级：

| 层级 | 能够支持的判断 | 是否可作为硬分组关系 |
|---|---|---:|
| L0 | 同一图像内容、同一帧的压缩/缩放/命名变体 | 是 |
| L1 | 有固定地物共同区域，能够经局部匹配和几何模型验证为同一局部地点 | 是，但首版必须人工确认非完全同像素关系 |
| L2 | 没有直接重叠，但多项证据支持可能属于同一机场 | 不是天然传递关系，只能进入受约束的 guard 或 embargo |

最终成果应回答：

- 哪些图像被确认属于同一局部来源组件；
- 哪些严格组件在充分证据下被合并为机场代理 guard；
- 哪些高风险关系仍无法确定，正式评估时如何通过 embargo 避免它们直接形成训练—验证泄漏；
- core、guard、guard+embargo 下模型结论是否稳定。

### 1.2 对 GPT Pro 建议的处理结果

| 建议 | 本文决策 | 原因 |
|---|---|---|
| 增加 `embargo` | 采纳，设为与 core/guard 并列的第三机制 | 不确定关系既不应强行合并，也不应直接忽略 |
| `different` 拆分 | 采纳 | “不是同一局部画面”不能推出“不是同一机场” |
| `likely_same_airport` 不做传递闭包 | 采纳并写入代码门禁 | 单条弱关系经 union-find 会造成巨型错误组件 |
| H1 自动阈值按组件规模调整 | 采纳，但首版进一步收紧 | v1.1 只有规范化像素完全一致可自动 union；几何强匹配也进入人工确认 |
| DINOv2-B + GeM/VLAD | 采纳为主召回器 | 已有 P04 资产可复用，规模和性能合适 |
| 增加 patch overlap | 采纳 | 本任务更像部分重叠发现，不能只依赖整图全局相似度 |
| 评估 ViT 上下文化前景泄漏 | 采纳为校准审计 | 简单删除飞机 patch 仍可能在邻近 token 中保留飞机信息 |
| SIFT 后增加 LightGlue/RoMa | 分层采纳 | LightGlue只处理高价值疑难边；RoMa只处理极少数争议边 |
| SALAD/BoQ/MegaLoc | 暂不作为首版依赖 | 仅在主召回器未达召回门禁时追加，避免第一版工程失控 |
| 完整 3,842 张作为桥接 | 仅作诊断 | 科学上有价值，但正式划分先采用更保守的 target-only |
| CP-SAT 优化 CV3 | 采纳 | 能明确表达不可拆组、类别覆盖和 embargo 成本 |
| 固定额外 holdout | 不采纳 | 样本有限；改用 fold 访问纪律保护 sentinel |

### 1.3 对前序 P 系列实验的影响

P0-3 和 P04 已证明对象 crop 在当前探索性三折上具有很高可分性，尤其 P0-3 微调 macro recall 约 0.97；P04 中 DINOv2-B 的冻结特征也表现很强。但这些实验使用的来源隔离并不充分，因此它们能可靠说明：

- 图像中存在强可分信息；
- ImageNet/DINOv2 预训练表征适合本项目；
- 对象级分类值得继续。

它们不能直接证明：

- 模型已经学会跨机场泛化；
- 0.97 的性能能在严格来源隔离下保持；
- DINOv2 的优势全部来自飞机结构而非背景地点线索。

新分组完成后，不推翻旧结果，而是将其标为 `relaxed/provisional split evidence`，并用相同缓存和尽量相同训练合同在新 CV3 上重跑低成本 probe。P06 的正式 OOF、P07 后续增强准入以及最终模型选择，均以新正式 CV3 为准。

---

## 2. 冻结的输出体系

### 2.1 四套必须保留的视图

| 名称 | 节点范围 | 关系处理 | 用途 |
|---|---|---|---|
| `target_core` | 3,073 竞赛 MAR20 图 | 仅 L0/L1 严格来源组件 | 泄漏下界、最少假设对照 |
| `target_guard` | 3,073 竞赛 MAR20 图 | core + 通过证书的 L2 组件合并 | 正式主划分候选 |
| `target_guard_embargo` | 3,073 竞赛 MAR20 图 | guard 不变；逐折排除直接高风险邻居 | 正式压力测试和保守结论 |
| `full_bridge_diagnostic` | 3,842 原始 MAR20 图 | 769 张仅作桥接节点 | 衡量 target-only 漏掉多少关系，不参与模型训练 |

### 2.2 正式主次顺序

在所有科学门禁通过后：

1. `target_guard` 是模型日常比较的主要 CV；
2. `target_guard_embargo` 是来源风险压力测试；
3. `target_core` 用于说明结论对 guard 人工判断是否敏感；
4. `full_bridge_diagnostic` 只报告敏感性，除非比赛规则复核后形成书面 amendment，否则不能替换正式划分。

若 guard 在时间或质量上无法冻结，允许启用最小回退：

> `target_core + unresolved-risk embargo`

这个回退比强行完成低质量 guard 更可信。任何情况下都不得回退到“按文件编号连续段等价机场”。

### 2.3 与非 MAR20 数据的集成

本流程只重建 3,073 张 MAR20 飞机图像的来源组。完整 4,481 张竞赛图像的正式 CV3 还需要 B 分工提供的舰船、车辆和其他来源组。

- MG00～MG05 可在 B 的最终文件完成前独立执行；
- MG06 可先生成 `mar20_only` 诊断划分；
- 完整正式 CV3 必须把 `mar20_group_id` 与 B 的 `non_mar20_group_id` 合并后统一求解；
- 不允许为平衡类别而拆开任一方已经冻结的来源组；
- B 目前已有划分保留为 `relaxed_baseline`，不删除、不覆盖。

---

## 3. 证据语义与允许的操作

### 3.1 人工与机器共用标签

| 标签 | 精确定义 | 图操作 |
|---|---|---|
| `same_frame` | 同一帧或内容等价版本，固定地物逐像素/近逐像素一致 | strict must-link |
| `geometric_overlap` | 不同裁剪或视角，但共同区域经稳定几何模型对齐 | strict must-link |
| `same_local_site` | 多处独特固定结构支持同一局部地点，即使没有大面积逐像素重叠 | strict must-link |
| `likely_same_airport` | 可能同一机场，但证据不足以证明同一局部地点 | 仅生成 guard 候选；禁止直接 union |
| `not_same_local_site` | 能排除同一局部画面，但不能排除同一机场 | 仅阻止 strict 合并，不阻止 guard |
| `different_airport` | 有强反证支持来自不同机场 | hard cannot-link |
| `uncertain` | 证据冲突或不足 | 依据风险进入 embargo 或留作普通未知 |

### 3.2 什么可以自动合并

首版只允许一种 H0 自动合并：解码并统一 EXIF 朝向后，完整 RGB 像素数组
SHA256 完全相同。无信息边框裁除后相同、重编码后近乎相同等情况仍只进入高优先级
复核队列，经人工确认后才能成为 strict 边。

以下证据即使很强，也不能在 v1.1 自动 union：

- pHash/dHash 很近；
- DINO/SSCD 余弦相似度很高；
- SIFT/LightGlue 内点很多；
- similarity/affine RANSAC 通过；
- 文件编号相邻；
- 官方 train/test 同侧；
- 飞机型号、数量或跑道方向相同。

机器证据负责排序，人工确认负责将非 H0 边升级为 strict must-link。这样会增加少量复核成本，但显著降低单条错误边污染整个组件的风险。

### 3.3 likely-same-airport 的非传递性

若 A 可能与 B 同机场，B 可能与 C 同机场，不能推出 A 与 C 同机场。代码必须满足：

- `likely_same_airport` 边不能交给普通 union-find；
- guard 合并以“当前两个完整组件”为单位重新生成证书；
- 每次合并后重新计算组件间支持、冲突、桥接依赖和结构风险；
- 不允许仅凭一条桥接边继续吞并后续组件。

### 3.4 embargo 的语义

embargo 边是两个 guard 组件之间的对称高风险关系。它不永久合并组件，也不做传递闭包。

对于第 `f` 折：

- 若组件 A 是验证集，A 的直接 embargo 邻居 B 不得进入该折训练集；
- 若 A 与 B 被分到同一验证折，则二者都不在训练集，不产生额外排除；
- 若第三个组件 C 是验证集，A、B 可同时训练，除非 C 与它们也有直接 embargo 边；
- 每个排除必须记录触发边和对象/图像数量。

embargo 解决的是“关系可疑但不值得冒险泄漏”的情形，不是把所有视觉相似图都排除。

---

## 4. 代码与目录设计

### 4.1 新增代码包

所有核心逻辑放在 `src/rsdet/grouping/`，脚本只做参数解析和调用，避免把判断规则散落在多个服务器任务单中。

```text
src/rsdet/grouping/
├── __init__.py
├── contracts.py          # 枚举、schema、版本与不变量
├── registry.py           # target/bridge 节点登记和指纹
├── masks.py              # bbox 扩张、inpaint、background tiles
├── descriptors.py        # DINOv2 层特征、GeM、VLAD、PCA
├── retrieval.py          # 多路 top-K、去重、召回统计
├── patch_overlap.py      # DINO patch 互检、覆盖度与粗到细重排
├── geometry.py           # SIFT、RANSAC、LightGlue/RoMa 接口
├── review.py             # 盲化复核包、决策导入、一致性审计
├── evidence_graph.py     # strict 图、cannot-link、组件统计
├── guard.py              # 非传递 guard 证书与顺序合并
├── embargo.py            # 高风险边和逐折排除
├── split_solver.py       # CP-SAT、贪心 fallback、目标函数
└── audit.py              # 反向泄漏和全流程产物审计
```

### 4.2 新增命令行脚本

```text
scripts/build_mar20_source_registry.py
scripts/audit_mar20_background_views.py
scripts/extract_mar20_place_features.py
scripts/build_mar20_retrieval_candidates.py
scripts/score_mar20_patch_overlap.py
scripts/verify_mar20_geometry.py
scripts/build_mar20_review_pack.py
scripts/compile_mar20_review.py
scripts/build_mar20_source_groups.py
scripts/solve_source_grouped_cv3.py
scripts/audit_source_grouped_cv3.py
scripts/analyze_grouping_sensitivity.py
```

每个脚本必须：

- 支持 `--config`；
- 支持 `--output-dir`；
- 写出 `run_config.resolved.yaml`、`environment.json`、`input_sha256.json` 和 `summary.json`；
- 同一输入和 seed 下结果确定；
- 对大缓存支持 shard、resume 和完整性审计；
- 输入指纹变化时拒绝复用旧缓存；
- 默认不覆盖已存在且指纹不同的目录。

### 4.3 配置文件

主配置为：

```text
configs/grouping/mar20_source_grouping_v1.yaml
```

配置至少包含：

```yaml
protocol_version: mar20-source-grouping-v1.1
seed: 202625

paths:
  competition_root: ${COMPETITION_DATA_ROOT}
  mar20_root: ${MAR20_ROOT}
  p04_asset_lock: ${P04_ASSET_LOCK}
  dinov2_repo: ${DINOV2_REPO}
  dinov2_b_weights: ${DINOV2_B_WEIGHTS}

scope:
  formal: target_only
  diagnostic: full_bridge

views:
  input_size: 518
  rotations: [0, 90, 180, 270]
  mask_dilation_ratio: 0.15
  mask_fill: telea
  background_tile_size: 224
  background_tile_stride: 112
  background_valid_fraction: 0.95
  max_background_tiles: 8

descriptor_bakeoff:
  layer_indexing: zero_based
  layers: [9, 10, 11]
  aggregations: [mean, gem]
  gem_p: [2.0, 3.0, 4.0]
  vlad_clusters: [16, 32]
  pca_dims: [512]

retrieval:
  k_values: [20, 50, 100]
  final_k: null

patch_overlap:
  coarse_grid: [19, 19]
  fine_grid: [37, 37]
  foreground_patch_fraction_max: 0.20

geometry:
  sift_ratio: 0.75
  ransac_reproj_fraction_diagonal: 0.005
  lightglue_max_pairs: 500
  roma_max_pairs: 50

review:
  blind_duplicate_fraction: 0.08
  second_review_component_size: 10

split:
  n_folds: 3
  solver: cp_sat
  max_time_seconds: 600
  random_seed: 202625
```

路径在服务器任务执行前根据实际资产目录解析，解析后的配置必须回传。配置中的阈值分两类：

- 结构阈值：如四旋转、三折、只有 H0 自动合并，不因结果改变；
- 校准阈值：如最终 top-K、几何入队阈值，只能根据预注册校准集改写，并记录 amendment。

本文中的 DINO `layer` 均指代码中的 zero-based transformer block index；`[9,10,11]`
即 ViT-B/14 的最后三个 block。报告同时写出人类可读层号，避免后续把“第 9 层”和
`block index 9` 混用。

### 4.4 产物根目录

```text
outputs/mar20-source-grouping-v1/
├── 00_registry/
├── 01_view_audit/
├── 02_descriptor_bakeoff/
├── 03_full_retrieval/
├── 04_pair_verification/
├── 05_human_review/
├── 06_groups/
├── 07_cv3/
├── 08_reverse_audit/
└── 09_model_sensitivity/
```

阶段目录一旦通过门禁，只能追加新的版本子目录，不能原地改结果。

---

## 5. 核心数据合同

### 5.1 节点登记表 `image_registry.csv`

每行一张 MAR20 图像，字段至少包括：

| 字段 | 含义 |
|---|---|
| `node_uid` | 稳定 ID，如 `mar20:1755` |
| `mar20_number` | 原始编号 |
| `is_target` | 是否属于竞赛 3,073 张 |
| `is_bridge` | 是否属于额外 769 张 |
| `official_side` | 原 MAR20 train/test，仅作审计 |
| `competition_image_id` | 竞赛图像 ID；bridge 为空 |
| `image_path` | 运行环境内路径 |
| `file_sha256` | 原文件指纹 |
| `pixel_sha256` | EXIF 纠正后 RGB 像素指纹 |
| `width`,`height` | 图像尺寸 |
| `bbox_count` | 飞机框数量 |
| `fine_class_hist_json` | 竞赛细类对象分布；bridge 不参与模型统计 |
| `annotation_sha256` | 框/类信息指纹 |

必须验证：

- 完整原始集恰为 3,842 张；
- target 恰为 3,073 张，bridge 恰为 769 张；
- target 与竞赛图能够通过像素或已验证映射一一对应；
- ID、路径、指纹唯一；
- 原始 train/test 列表无重复并覆盖 3,842 张；
- `official_side` 绝不被当作机场标签。

### 5.2 候选边表 `candidate_edges.csv`

pair ID 固定为字典序较小 UID 在前：

```text
pair_uid,node_u,node_v,
route_pixel,route_phash,route_gem,route_vlad,route_background_tiles,route_sscd,
rank_u_to_v,rank_v_to_u,
sim_original,sim_masked,sim_background,
foreground_influence,
scope,target_target,target_bridge,bridge_bridge
```

不同召回器只追加字段，不分别生成无法对齐的边表。

### 5.3 几何证据表 `pair_evidence.csv`

至少记录：

- SIFT keypoint 数、双向匹配数、ratio+mutual 后匹配数；
- similarity、affine、homography 各自的 inlier 数、inlier ratio、重投影误差；
- 两端覆盖率、4×4 网格占用数、凸包面积比；
- 主方向线占比、方向熵、去重后的空间内点数；
- 多次 RANSAC 重采样的通过比例和变换参数方差；
- DINO patch 粗/细互为最近邻数、相似度、覆盖率；
- LightGlue/RoMa 是否运行、模型资产指纹和对应统计；
- 自动队列等级，但不写人工最终标签。

### 5.4 人工决策表 `manual_review_decisions.csv`

```text
review_card_id,pair_uid,label,confidence,
supporting_evidence,counter_evidence,
fixed_structure_types,style_only,
bridge_dependent,reviewer,reviewed_at,
duplicate_card_group,notes
```

`likely_same_airport` 和 `different_airport` 不允许缺少文字证据。只写“看起来像”视为无效记录。

### 5.5 组、证书与 embargo 文件

```text
strict_edges.csv
cannot_link_edges.csv
core_components.csv
guard_merge_certificates.jsonl
guard_components.csv
embargo_edges.csv
component_risk_report.csv
```

每个 group ID 由排序后的成员 UID 哈希产生，不按生成顺序编号，以保证重跑稳定。

---

## 6. 执行任务总览

| 任务 | 内容 | GPU | 是否依赖 B 最终分组 | 主要放行结果 |
|---|---|---:|---:|---|
| MG00 | 环境、资产、节点登记和输入指纹 | 否 | 否 | `registry_pass` |
| MG01 | mask/视图审计与 descriptor bake-off | 是 | 否 | 冻结主描述子与输入视图 |
| MG02 | 全量特征、VLAD/PCA、多路 top-K 召回 | 是 | 否 | `retrieval_recall_pass` |
| MG03 | patch overlap、SIFT、LightGlue 分层验证 | 部分 | 否 | 可复核证据边 |
| MG04 | 盲化人工配对复核与 strict core | 否 | 否 | `core_groups_pass` |
| MG05 | 簇级复核、guard 证书和 embargo | 否 | 否 | `guard_or_core_fallback_ready` |
| MG06 | target-only / full-bridge 诊断和 CP-SAT CV3 | 否 | 完整 CV 依赖 | `provisional_cv3_pass` |
| MG07 | 跨折反向检索与泄漏闭环 | 部分 | 是 | `formal_cv3_pass` |
| MG08 | relaxed/core/guard/embargo 模型敏感性 | 是 | 是 | 正式实验划分冻结 |

所有任务都使用前一任务的只读产物。服务器 AI 不得在门禁失败时自行放宽条件；只能打包回传并等待本地科学判断。

---

## 7. MG00：环境、输入和 registry

### 7.1 实现内容

1. 新建 `rsdet.grouping` 包、schema 和单元测试；
2. 枚举竞赛 MAR20 图、完整原始 MAR20 图、原 train/test 列表和标注；
3. 通过文件名、像素指纹和尺寸核对 target 映射；
4. 生成 RGB 像素 SHA、框 mask 元数据和类统计；
5. 冻结 DINOv2-B/14 repo commit、权重 SHA 和许可证；
6. 检查服务器已有 `/workspace/p04-assets/`，只复用资产，不复用可能已漂移的软件环境；
7. 创建独立环境并写完整 `pip freeze`。

### 7.2 依赖策略

GPU 环境以已经验证过的组合为底座：

- Python 3.10.12；
- torch 2.5.1+cu121；
- torchvision 0.20.1+cu121；
- numpy 1.26.4；
- Pillow 10.4.0；
- PyYAML 6.0.2。

新增 OpenCV、scikit-learn、FAISS、NetworkX、SciPy 和 OR-Tools 时，在独立 `mar20-group-cu121` venv 安装并冻结实际版本。不得升级 numpy/Pillow 造成与 P03/P05 类似的像素变换漂移。OR-Tools 采用有 Python 3.10 wheel 的冻结版本，安装后必须跑一个最小 CP-SAT smoke；求解器版本写进最终 split 文件。

### 7.3 测试

至少实现：

- registry 数量和唯一性测试；
- target/bridge 互斥和覆盖测试；
- EXIF 纠正、RGB 转换和像素 SHA 确定性测试；
- pair UID 对称规范化测试；
- evidence label 枚举测试；
- `likely_same_airport` 无法被 strict union API 接受的负向测试；
- 输入 SHA 改变后 resume 被拒绝的测试。

### 7.4 门禁与预案

通过条件：

- 3,842 / 3,073 / 769 数量准确；
- target 一一映射，缺失、重复、尺寸不一致均为 0；
- 已知 `MAR20_1755`、`MAR20_933` 等调查样本能够在 registry 中找到并追溯；
- 资产 SHA 与 P04 锁一致；
- 全仓 pytest、专项 pytest、ruff 通过。

若 target 无法像素一一对应：

1. 检查 EXIF、JPEG 重编码和扩展名；
2. 使用灰度/边缘近邻只提出映射候选；
3. 所有非唯一映射人工确认；
4. 未解决前不得进入 MG01 全量缓存。

---

## 8. MG01：背景视图审计与描述子 bake-off

### 8.1 飞机背景隔离

从 HBB 框构造 union mask，按框宽高分别外扩 15%，再裁到图像边界。生产视图包括：

1. `original`：只作泄漏诊断，不作为默认来源描述子；
2. `masked_inpaint`：mask 内使用确定性 Telea inpaint；
3. `background_tiles`：只选择不接触扩张 mask 的 224×224 背景块，不暴露 mask 形状。

校准子集上额外比较：

- 外扩 10%、15%、20%；
- blur、Telea、局部低频均值三种填充；
- DINOv2 第 9、10、11 层；
- original / masked / background tiles。

不把整个笛卡尔积跑到全量。先在固定校准集上完成三级筛选。

### 8.2 校准集构成

校准集必须冻结 UID 和标签，不能从最终模型结果倒推：

- 所有已确认相同/近同图对；
- 已确认跨官方 train/test 的重叠对；
- 至少 50 个可能局部重叠正例，正例不足时先人工扩充；
- 至少 200 个 hard negative，包括相同型号、相似跑道、相同色调和编号相邻但无共同地物；
- 至少 100 个普通负例；
- 正例、hard negative 按组件隔离为 calibration 和 held-out audit，防止只对同一组样本调阈值。

如果真实正例不足 30 对，不允许宣称已经校准自动阈值；但因为 v1.1 不自动接纳 H1，可以继续构建高召回人工队列，并在报告中标记证据有限。

### 8.3 描述子筛选顺序

#### Round A：层与基础聚合

在 `masked_inpaint`、0° 视图上比较：

- layer 9/10/11 patch mean；
- layer 9/10/11 GeM，`p∈{2,3,4}`；
- final CLS 作为对照。

选择标准按顺序为：

1. held-out strict positive 的 recall@100；
2. hard negative 进入 top-100 的比例；
3. original→masked 的前景影响是否下降；
4. 计算和存储成本。

#### Round B：域内 VLAD 和旋转

只对 Round A 前两名层/视图继续：

- VLAD K=16、32；
- PCA512 与 native；
- 四旋转；
- GeM 与 VLAD 互补并集。

VLAD 词典在图像均衡采样的背景 patch 上拟合；每张图贡献相同上限的 patch，不能让飞机密集图支配词典。四旋转检索可将旋转版本分别入库，再映射回原 UID 去重；pair score 记录最佳相对旋转。

#### Round C：输入泄漏和稳定性

对 Round B 冻结的描述子比较：

- 邻居 Jaccard：不同 mask 外扩/填充的 top-20、top-50 邻居稳定度；
- 前景影响指数 `FI(u,v)=sim_original-sim_masked`；
- background tiles 与 masked 的互补召回；
- foreground-only 描述子是否主导高相似候选。

若某描述子主要靠飞机型号而非背景检索，它可以留作“发现同型 hard negative”的辅助路由，但不能作为地点合并证据。

### 8.4 产物

- `calibration_pairs.csv`；
- `view_audit.json`；
- `descriptor_bakeoff.csv`；
- `retrieval_examples/`；
- `selected_descriptor.json`；
- `protocol_amendment.md`，仅当最终层、GeM p、VLAD K 或 mask 设置改变时生成。

### 8.5 门禁与回退

进入 MG02 的描述子必须：

- 已知像素同源对 recall@100 = 100%；
- held-out strict local positive 的多路并集 recall@100 目标 ≥95%，同时报告 Wilson 区间；
- 没有任何单一路由被误当作最终同源判据；
- mask 视图有限、确定、无 NaN；
- 人工检查的 background tiles 不含明显飞机主体。

若 DINOv2-B GeM/VLAD 并集仍明显漏掉已知正例：

1. 先增加 DINO patch overlap 和 SSCD/copy descriptor；
2. 再考虑 SALAD 作为补充召回器；
3. 不直接升级到 DINOv2-G；
4. 新召回器只扩大候选，不获得自动合并权。

---

## 9. MG02：全量特征和多路候选召回

### 9.1 全量提取

对 target-only 和 full-bridge 使用相同冻结提取器。每条特征缓存记录：

- node UID、view、rotation、layer、aggregation；
- fp16 特征和提取器指纹；
- mask 统计和有效背景比例；
- shard 行数、有限性和 checksum。

禁止缓存 `3842×4×37×37×768` 的所有原始 token。采用两遍流式流程：

1. 第一遍按图均衡采样少量 patch，拟合 PCA/VLAD；
2. 第二遍逐 shard 聚合并仅保存全局描述子；
3. 只有进入 patch-overlap 队列的图像按需重提 token，并以小型 LRU/shard 缓存复用。

### 9.2 召回路由

至少使用：

- R0：pixel SHA、无信息边框规范化 SHA；
- R1：pHash/dHash，只作近拷贝候选；
- R2：DINO masked GeM；
- R3：DINO masked VLAD；
- R4：DINO background tiles；
- R5：灰度边缘/结构描述子；
- R6：SSCD，仅在资产和许可证通过时启用。

所有检索对全库进行，不按飞机类别、编号区间或官方侧过滤。那些字段只能用于分层统计和人工优先级。

### 9.3 top-K 饱和

分别计算 K=20、50、100：

- 每个召回器的 held-out positive recall；
- 多路并集 recall；
- 候选边总数；
- target-target、target-bridge、bridge-bridge 构成；
- 新增 K 带来的边际正例和人工负担。

最终 K 的冻结规则：选择达到召回平台的最小 K。若 K=50 到 100 仍新增超过 2% 已知正例，则保留 100；否则采用 50 并在跨折反向审计时重新使用 100。

### 9.4 门禁

- 所有 H0 对都进入候选；
- 多路并集达到 MG01 的 held-out 召回目标；
- 每张图候选去重正确且无 self-pair；
- 双向 rank 与 pair score 可重现；
- target-only 和 full-bridge 文件物理隔离；
- full-bridge 不得生成模型训练 manifest。

---

## 10. MG03：patch overlap 与分层几何验证

### 10.1 DINO patch overlap

目标不是直接证明同一机场，而是判断两张图是否存在局部共同画面。

流程：

1. 使用 MG01 选定层，提取 37×37 patch token；
2. 删除与飞机扩张 mask 重叠超过 20% 的 patch；
3. 对普通候选先 adaptive average pool 为 19×19；
4. L2 归一化后做双向 mutual nearest neighbor；
5. 在 patch 中心坐标上估计 similarity/affine；
6. 记录匹配数量、相似度、RANSAC 内点、两端空间覆盖和网格分布；
7. 只对高价值或结论冲突候选运行完整 37×37 精排。

20% patch 前景阈值不是科学常数。校准集上比较 10/20/30%，若最终改变，必须写 amendment。

### 10.2 SIFT 基线

在忽略飞机 mask 内关键点后：

1. SIFT 检测与描述；
2. KNN ratio=0.75；
3. 双向 mutual check；
4. 对匹配点坐标去重；
5. 优先估计 similarity；
6. similarity 不足再估计 affine；
7. homography 只作诊断，不因单独通过而升级为强边。

RANSAC 阈值按图像对角线比例定义，默认 `max(3 px, 0.005×diagonal)`，并在校准集上检查。

### 10.3 几何爆发防护

跑道线、停机坪接缝和重复机库可能制造大量但低信息的内点。每条边必须计算：

- 主 Hough 方向上的匹配比例；
- 匹配方向熵；
- 去重空间内点数；
- 4×4 网格覆盖；
- 两端凸包面积比；
- 20 次固定 seed RANSAC 子采样通过率；
- similarity、affine、homography 的一致性；
- 变换尺度、旋转和剪切是否合理。

仅在一两条直线上集中，即使有 80 个内点，也不得优先于覆盖建筑、道路、植被等多个区域的 20 个稳定内点。

### 10.4 现代匹配器的准入

LightGlue 只处理以下候选：

- DINO patch overlap 高但 SIFT 因季节、亮度或纹理变化失败；
- 候选边将连接两个现有 strict 组件；
- 尾类/少数组的高影响边；
- `target—bridge—target` 路径中的关键边；
- 人工标为疑难且可能改变 fold 的边。

首轮预算上限 500 对，可根据实际候选量书面调整。RoMa 只用于不超过 50 个 LightGlue 仍无法判断的高影响争议边。两者都只提供证据，不自动决定标签。

### 10.5 自动队列等级

机器只输出复核优先级：

| 队列 | 条件含义 | 操作 |
|---|---|---|
| Q0 | H0 像素等价 | 自动 strict，人工抽查 |
| Q1 | 多模态支持且几何分布稳定 | 优先人工确认 strict |
| Q2 | DINO/几何部分支持或可能同机场 | 人工判断 likely/uncertain |
| Q3 | 高相似但疑似飞机/跑道捷径 | hard-negative 复核 |
| Q4 | 低证据普通候选 | 默认不展示，反向审计时保留 |

Q1 不是自动 H1。任何阈值都只能影响队列，不得越过人工决策。

---

## 11. MG04：盲化人工复核与 core

### 11.1 复核卡设计

每张卡至少提供：

- 匿名 A/B 原图；
- 飞机遮挡图；
- 边缘图；
- 若几何通过，提供匹配点和 warp overlay；
- 不显示文件编号、官方 train/test、飞机细类和机器建议标签；
- 决策完成后才允许查看证据指标页。

卡片中随机插入 8% 重复项：

- A/B 交换；
- 间隔至少 30 张；
- 使用不同卡片 ID 但同一 duplicate group；
- 用于测量同一复核者的一致性，不参与增加证据数量。

### 11.2 strict 正证据

允许标记 `same_local_site` 的典型证据：

- 可对应的像素纹理或局部地面细节；
- 至少两类独特固定结构相互支持；
- 独特跑道—滑行道—机库拓扑；
- 独特道路、水体、围栏、建筑组合；
- 变换后多个分散区域同时对齐。

以下单独出现均不充分：

- 相同飞机型号或数量；
- 灰色跑道、相似跑道方向；
- 相同植被色调、分辨率或成像风格；
- 通用矩形建筑；
- 单条直线上的大量 SIFT 内点。

### 11.3 决策质控

人工复核门禁：

- 重复卡整体一致率 ≥0.90；
- 不能出现同一 pair 在 `same_local_site` 与 `different_airport` 之间的严重自相矛盾；
- 所有低置信 strict 决策二次复核；
- 所有连接两个非单例组件的 strict 边二次复核；
- 所有会产生 ≥10 张组件的边二次复核；
- `different_airport` 必须有强反证和第二次确认。

若一致性不足：暂停组图，先修订示例和判据，再重做不一致卡及其邻近优先级卡。不能靠多数表决自动掩盖术语理解错误。

### 11.4 core 形成

strict 图只包含：

- H0 自动边；
- 人工 `same_frame`；
- 人工 `geometric_overlap`；
- 人工 `same_local_site`。

对 strict 图取连通分量得到 core。形成后执行：

- 组件内 cannot-link 冲突必须为 0；
- 为每个组件生成 medoid、最远成员和最弱生成树边；
- 检查 articulation node/edge；
- 对 ≥10 张组件生成整簇 contact sheet；
- 统计 target/bridge、官方侧、细类和节点跨度；
- 任何可疑长链都回到 pair 复核，不直接拆图后继续。

---

## 12. MG05：guard 证书和 embargo

### 12.1 guard 合并证书

`likely_same_airport` 只产生组件对候选。每次合并前必须生成证书，至少包含：

```text
left_component_id
right_component_id
member_counts
supporting_pairs
distinct_support_members
evidence_modalities
counter_evidence
cannot_link_conflicts
bridge_dependency
medoid_pair
farthest_pair
weakest_support_pair
bimodality_score
articulation_risk
reviewers
decision
```

### 12.2 合并条件

同时满足才允许 guard merge：

1. 无 `different_airport` 冲突；
2. 至少两种独立证据模态，例如背景全局检索 + 局部布局/人工固定结构；
3. 两个非单例组件之间至少有 2 对支持边，且涉及至少 2 个不同成员；
4. 单例—单例只有 1 对时，必须是人工高置信、两种模态支持并经第二次复核；
5. 支持不能全部依赖同一 bridge 节点；
6. medoid、最远成员、最弱支持边均完成复核；
7. 合并后没有明显双峰、细长桥接链或单 articulation edge 控制整个组件；
8. 合并后 ≥10 张或一次增加 >5 张时必须二次复核。

合并按证书逐次执行。一次合并后，旧的组件对证书失效，必须基于新组件重新计算，禁止预先批准一串边后批量 union。

### 12.3 组件风险报告

每个 guard 组件至少报告：

- target/bridge 节点数；
- strict/likely 边数和图密度；
- 直径、medoid—最远距离；
- articulation points/bridges；
- 最弱 strict 生成树边；
- 依赖 bridge 才连通的目标节点对；
- 官方 train/test 两侧构成；
- 细类构成；
- descriptor 分布双峰指标；
- 人工复核覆盖率。

大组件本身不是错误，但“由一条弱边形成的大组件”必须 fail。

### 12.4 embargo 入边规则

以下关系进入 embargo 候选：

- 人工 `uncertain` 且两种独立证据显示高风险；
- `likely_same_airport` 但未满足 guard 证书；
- strict/guard 结论冲突、短期无法二次复核；
- bridge 诊断发现 target-only 中缺失的高风险 target-target 关系；
- 反向跨折审计发现强相似但仍无法确认的边。

纯 DINO 高相似、同型飞机或相似跑道不能单独触发 embargo。每条 embargo 边必须有 `risk_reason` 和至少一项背景/局部结构证据。

### 12.5 时间不足时的预案

如果人工候选过多：

1. 先完成 Q0、Q1 和会跨现有 provisional fold 的高风险边；
2. 冻结已通过的 core；
3. 未完成但高风险的 Q2 进入 embargo；
4. 低风险未复核边保留为 unknown；
5. 发布 `core+embargo provisional`，不仓促形成 guard。

---

## 13. MG06：CP-SAT 生成 CV3

### 13.1 求解节点

target core 或 guard 组件是不可拆分的 assignment unit。非 MAR20 图使用 B 冻结的来源组。每个 unit 统计：

- 图像数；
- 总对象数；
- 25 个细类对象数；
- 三大类对象数；
- 独立来源组数；
- MAR20 官方侧构成，仅作审计；
- embargo 邻居。

### 13.2 变量

令 `x[g,f]∈{0,1}` 表示组 g 被分配到验证折 f：

```text
sum_f x[g,f] = 1
```

令 `z[g,f]∈{0,1}` 表示组 g 因某个 embargo 邻居位于验证折 f，而需要从该折训练集中排除。对每条 embargo 边 `(g,h)`：

```text
z[g,f] >= x[h,f] - x[g,f]
z[h,f] >= x[g,f] - x[h,f]
```

最终第 f 折训练集必须满足 `x[g,f]=0 and z[g,f]=0`。

### 13.3 硬约束

1. 每个组完整进入且只进入一个验证折；
2. 每张竞赛图恰好验证一次；
3. strict/guard must-link 不跨折；
4. cannot-link 不要求不同折，它只用于阻止错误合并；
5. 某细类若跨越至少 3 个独立 assignment unit，且联合可行性预检通过，则每折至少有该类 1 个对象；
6. 单类结构不足或多类约束联合冲突导致无法三折覆盖时，进入 `structurally_uncoverable_classes.json`，不得拆组解决；
7. embargo 排除语义必须严格实现；
8. full-bridge 节点永不进入模型 train/val 列表。

### 13.4 分层目标函数

不使用一个不可解释的大权重和。采用连续求解并冻结上一层最优值：

#### 一级：最坏细类相对偏差

最小化所有 `(class,fold)` 中的最大相对对象数偏差。CP-SAT 只接受整数，因此统一乘 10,000 后取整；分母为该类目标折均值，至少为 1。

#### 二级：总体和类别综合平衡

在一级最优值允许极小整数容差内，最小化：

- 各折总对象数绝对偏差；
- 所有细类对象数绝对偏差和；
- 三大类对象数偏差；
- 各折独立组数偏差。

#### 三级：embargo 成本

最小化每折被 embargo 排除的唯一训练对象数，而不是简单计算边数。高风险邻居可被倾向分到同一折，但不能破坏一级类别目标。

#### 四级：工程均衡

最小化图像数偏差，并将 MAR20 原 official side 分布作为纯审计弱目标。官方侧不能覆盖细类或来源硬约束。

### 13.5 求解与 fallback

- CP-SAT 固定 seed，单次上限 600 秒；
- 状态必须是 `OPTIMAL` 或 `FEASIBLE`；
- 记录 best bound、objective、wall time 和 solver version；
- 同一输入运行两次，分配与摘要必须一致；
- 贪心算法只用于 CP-SAT warm start 或环境故障 fallback；
- fallback 输出必须显式命名 `greedy_fallback`，不得冒充正式 CP-SAT 结果。

若 infeasible，按以下顺序处理：

1. 检查实现错误和重复约束；
2. 运行带覆盖 slack 的诊断模型，识别单类结构不足或多类约束联合冲突；
3. 只对诊断证明不可满足的类别取消硬覆盖，并将缺失折数作为最高优先级软惩罚；
4. 放松软目标容差不会修复硬约束 infeasible，不能把它伪装成解决方案；
5. 不放松组完整性、不拆 must-link、不忽略 embargo；
6. 仍不可行则发布原因报告，等待人工核查错误组或约束实现。

### 13.6 sentinel 访问纪律

不额外切固定 holdout：

- fold0：日常开发；
- fold1：阶段确认；
- fold2：sealed sentinel，只在模型里程碑访问；
- 每次 fold2 访问记录 commit、配置、访问次数、指标和访问后是否改变方案；
- 正式冻结前才运行完整 CV3/OOF。

访问纪律是团队流程，不改变每张图最终都验证一次的 CV 定义。

---

## 14. MG07：反向泄漏审计与闭环

### 14.1 审计原则

划分后重新从验证图向“该折有效训练图”检索，不能只检查构图时已有的边。每折分别运行：

- pixel/规范化 SHA；
- pHash/SSCD；
- 冻结 DINO GeM/VLAD，K 至少 100；
- background tiles；
- 高相似边的 patch overlap；
- Q1 边的 SIFT/LightGlue。

### 14.2 处理规则

| 新发现关系 | 处理 |
|---|---|
| H0 或人工 strict | 合并组件、重新求解全部 CV3、重新审计 |
| 通过 guard 证书 | 合并 guard、重新求解、重新审计 |
| 高风险但不确定 | 加 embargo，重新生成逐折有效训练集 |
| 飞机/风格捷径 | 记录 hard negative，不改变分组 |
| 明确不同机场 | 加 cannot-link 和负例库 |

审计必须循环到：

- strict 跨折泄漏为 0；
- 未处理的 Q1 跨折边为 0；
- 所有高风险 unresolved 边都被同折吸收或正确 embargo；
- 再跑一次没有新增必须改变图结构的关系。

### 14.3 full-bridge 敏感性

单独报告：

- full-bridge 相比 target-only 新增的 target-target strict/guard 关系；
- 每条关系依赖的 bridge 路径长度；
- 去掉任一 bridge 后是否断开；
- 是否改变 CV fold；
- 是否改变后续模型排序。

正式 target-only 不静默吸收 bridge 结论。若 bridge 发现明显同帧 target 对，可回到 target 图像本身用直接几何重新验证；只有 target-target 直接证据成立，才能进入正式 strict。

---

## 15. MG08：科学验收与前序实验对接

### 15.1 最低成本模型对照

首先复用 P04 已有对象特征缓存和 probe 代码，只替换 `source_image_uid→fold`：

| 划分 | ConvNeXt 冻结 probe | DINOv2-B probe | 用途 |
|---|---:|---:|---|
| relaxed/B 当前划分 | 运行或读取旧结果 | 运行或读取旧结果 | 泄漏敏感上界参考 |
| target_core | 运行 | 运行 | 最少假设 |
| target_guard | 运行 | 运行 | 主比较 |
| target_guard_embargo | 运行 | 运行 | 风险压力测试 |

所有比较使用相同对象 manifest、模型特征、分类头协议和 seed。改变的唯一主变量是来源划分/逐折训练排除。

### 15.2 指标

除 macro recall、macro F1、accuracy、aircraft20 recall 外，至少报告：

- 每折/每细类结果；
- 按 core/guard group 聚合的 bootstrap 区间；
- relaxed→core、core→guard、guard→embargo 的配对差；
- 每折有效训练样本和 embargo 排除率；
- 头/中/尾类；
- 已确认跨侧同源组件上的错误；
- fold2 sentinel 访问记录。

### 15.3 背景捷径诊断

在同一正式 CV 上训练或 probe：

1. tight aircraft crop；
2. aircraft + context；
3. background-only ring 或 masked-object crop。

如果 background-only 在严格来源隔离下仍能高精度识别细类，说明数据本身存在机场—型号相关性；如果 relaxed 很高而 guard 显著下降，则前序高分包含明显背景捷径。后续可按成本尝试：

- background dropout；
- context randomization；
- 跨来源 copy-paste；
- group-aware sampling；
- 同类跨来源对比学习。

这些属于模型阶段，不得反向修改分组来使结果更好看。

### 15.4 core/guard 排名不一致

按固定顺序处理：

1. paired fold/class/group bootstrap，确认差异不是抽样噪声；
2. merge influence analysis，逐个移除 guard merge 计算模型排序变化；
3. 只复核高影响合并证书；
4. 仍不稳定则报告性能区间，不选对某模型更有利的 split；
5. 以 guard+embargo 压力测试和来源风险更低者作为保守决策依据。

### 15.5 对 P06/P07 的放行

以下全部完成后才将 formal OOF 合同交给 P06：

- 完整 4,481 图 CV3 通过 MG07；
- 每张图验证一次；
- 所有 must-link 不跨折；
- embargo 生效且有效训练清单可复现；
- 模型敏感性没有揭示明显错误 guard；
- 正式 split 文件、SHA 和版本冻结。

P07 的扩散背景融合可继续做人工视觉研究，但任何“对正式性能有增益”的结论也必须在该 CV 上验证。

---

## 16. 质量门禁总表

### 16.1 工程硬门禁

- 全仓 pytest、专项 pytest、ruff 全绿；
- 输入、权重、repo、配置、缓存均有 SHA；
- 无 NaN/Inf、无丢行、无重复 UID；
- shard resume 后逐行/摘要一致；
- target/full-bridge 物理目录分离；
- bridge 从未进入训练 manifest；
- 人工 CSV schema、枚举和 pair UID 全部有效；
- CP-SAT 结果可复现。

### 16.2 科学硬门禁

- 只有 H0 自动 strict；
- 人工 strict 边有来源卡和证据；
- `likely_same_airport` 未进入 union-find；
- core/guard 内 `different_airport` 冲突为 0；
- 所有 guard merge 有有效证书；
- 大组件通过结构审计；
- 复核重复卡一致率达标；
- 每张竞赛图验证一次；
- strict 跨折泄漏为 0；
- 高风险 unresolved 边同折或 embargo；
- 结构上可覆盖的 25 个细类每折有样本；
- 无法覆盖的类别有明确组件级原因。

### 16.3 只警告、不应伪装成硬真值的指标

- 组件数是否接近 60；
- 机场代理簇是否视觉上“很漂亮”；
- 某个 DINO 相似度是否超过经验阈值；
- guard 是否让模型指标更高；
- 官方 train/test 是否大体分开；
- 文件编号是否连续。

这些都不能替代真实证据。

---

## 17. 风险与预案

| 风险 | 识别信号 | 处理预案 |
|---|---|---|
| DINO 只检索同型号飞机 | original 高、masked/background 低；FI 高 | 降级为 hard-negative 路由，扩大 mask/改用 background tiles |
| mask 形状本身泄漏 | 不同填充邻居变化大，background tile 结论不同 | 主描述子改 background tiles；masked 只作辅助 |
| 全局描述子漏部分重叠 | 已知正例 recall@100 不足 | patch overlap、SSCD；必要时 SALAD 补召回 |
| SIFT 跑道爆发 | 内点多但线集中、覆盖低 | 几何爆发过滤，LightGlue/RoMa 二线 |
| 人工对数过多 | Q1/Q2 超过可复核预算 | 先 core；高风险未决进入 embargo；延后 guard |
| 一条错误边造巨簇 | articulation edge、直径骤增、双峰 | 撤销边并二次复核；禁止批量 union |
| guard 关系非传递 | A-B、B-C 强但 A-C 无支持 | 保持独立或 embargo，不做传递闭包 |
| bridge 导致合规争议 | 正式关系依赖额外 769 图 | 正式 target-only；full-bridge 只诊断 |
| CP-SAT infeasible | 类覆盖/组约束冲突 | 只放松可证明不可能的覆盖软目标，绝不拆组 |
| embargo 排除过多 | 任一折对象排除率 >10% 或尾类训练严重不足 | 优化同折分配；重审高影响边；报告有效样本下降，不静默取消 embargo |
| core/guard 模型排序翻转 | paired 差异显著 | merge influence + 高影响证书复核 + embargo 压力测试 |
| B 的非 MAR20 组延迟 | MG05 完成但完整 CV 无输入 | 先产出 MAR20-only 诊断，MG06 正式集成等待，不阻塞前五步 |
| 服务器环境漂移 | Pillow/numpy/torch 与锁不一致 | 新建独立 venv，禁止在旧环境原地升级 |
| 时间只剩一天 | guard 复核未完成 | 发布高质量 core+embargo provisional，不牺牲证据质量 |

---

## 18. 服务器复用与资源规划

### 18.1 可直接复用

- `/workspace/p04-assets/` 中已验证的 DINOv2 repo、DINOv2-B 权重和 asset lock；
- RTX 4080 SUPER 32GB；
- P04 对象特征缓存用于 MG08，不用于机场图全局描述子；
- P03/P04 的 checkpoint、manifest 指纹和训练脚本；
- 服务器现有大内存适合 FAISS、PCA/VLAD 和 CP-SAT。

### 18.2 不直接复用

- 已发生 numpy/Pillow 漂移的 P04 venv；
- P04 crop DINO 特征作为机场背景特征；
- 旧 exploratory fold 作为机场真值；
- 额外 769 张 bridge 图作为分类训练数据。

### 18.3 建议服务器目录

```text
/workspace/xh-202625/                       # 仓库
/workspace/inputs/mar20-original/           # 原 MAR20，只读
/workspace/inputs/competition/              # 竞赛数据，只读
/workspace/p04-assets/                      # 复用模型资产，只读
/workspace/mar20-group-cache/               # 大型特征/局部匹配缓存
/workspace/results/MAR20-GROUPING-v1/       # 每任务结果
```

### 18.4 预计时间与存储

以下为 3,842 张图规模的保守排期，不是门禁：

| 阶段 | 预计机器时间 | 预计人工时间 |
|---|---:|---:|
| MG00 | 10～30 分钟 | 10 分钟核对 |
| MG01 | 30～90 分钟 GPU | 1～2 小时校准复核 |
| MG02 | 30～90 分钟 GPU/CPU | 20 分钟看召回 |
| MG03 | 1～4 小时，取决于候选 | 30 分钟看几何样例 |
| MG04 | 生成包数分钟 | 2～4 小时，随 Q1/Q2 数量变化 |
| MG05 | 数分钟 | 1～3 小时簇级复核 |
| MG06 | 通常数分钟，最多 10 分钟/求解 | 30 分钟审计 |
| MG07 | 1～3 小时 | 1～2 小时复核新增边 |
| MG08 | 1～3 小时 GPU | 1 小时分析 |

全局描述子缓存应低于数 GB。局部 token 按需重提，禁止为了方便生成几十到上百 GB 的全量 token 缓存。

---

## 19. 实现批次与服务器任务单规划

为保证一次只解决一类问题，编码和服务器任务按四批进行。

### 批次 A：基础合同与可行性

实现 MG00、MG01 所需代码、测试、配置和一份服务器任务单：

```text
docs/server/MAR20_GROUPING_TASK_00_REGISTRY_AND_BAKEOFF.md
```

该批只冻结输入、视图和描述子，不做全量人工分组。

### 批次 B：全量召回和配对证据

实现 MG02、MG03：

```text
docs/server/MAR20_GROUPING_TASK_01_RETRIEVAL_AND_GEOMETRY.md
```

服务器完成后回传小型审计包和 review pack；大缓存留在服务器。

### 批次 C：人工决策、core/guard/embargo

实现 MG04、MG05。人工决策在本地完成并提交可追踪 CSV，再由服务器或本地无 GPU 脚本编译证据图：

```text
docs/server/MAR20_GROUPING_TASK_02_COMPILE_GROUPS.md
```

人工原始决定不可由脚本覆盖；修订必须增加 `decision_revision` 和理由。

### 批次 D：CV3、反向审计和模型敏感性

实现 MG06～MG08：

```text
docs/server/MAR20_GROUPING_TASK_03_CV3_AND_REVERSE_AUDIT.md
docs/server/MAR20_GROUPING_TASK_04_MODEL_SENSITIVITY.md
```

该批接收 B 的非 MAR20 group manifest。若尚未交付，任务 03 只运行 MAR20-only diagnostic 并正常进入 `waiting_for_non_mar20_groups`，不视为故障。

---

## 20. 立即开始时的具体顺序

当前不需要等待 B 即可开始：

1. 冻结本方案和旧协议的 SHA；
2. 实现 `contracts.py`、`registry.py`、`masks.py` 及对应测试；
3. 实现 MG00 registry CLI，在本地用少量样本 smoke；
4. 实现 DINOv2-B 多层流式特征、GeM 和视图审计；
5. 编写服务器 TASK-00，复用 P04 权重但新建 venv；
6. 服务器完成 registry + bake-off 后，本地审查 descriptor 决策；
7. 再实现和运行全量 retrieval/geometry，避免在未知描述子上一次写完所有下游阈值；
8. 人工完成 pair review 后才构建 core；
9. guard 无法及时完成时启用 core+embargo 预案；
10. B 的非 MAR20 分组到位后统一求解完整 CV3；
11. 反向审计闭环后重跑 P04 低成本 probe；
12. 正式 split SHA 冻结，再恢复 P06 OOF 和后续模型实验。

第一批实现的“完成”定义不是已经恢复机场，而是：

> 输入完全可追溯，飞机背景隔离方法经过审计，主召回描述子在冻结校准集上通过门禁，能够安全进入全量候选发现。

---

## 21. 最终交付清单

### 21.1 代码

- `src/rsdet/grouping/` 全部核心模块；
- 12 个 CLI；
- 主配置和解析后的配置；
- 单元测试、集成 smoke、确定性和 resume 测试；
- 服务器任务单及代码 SHA 清单。

### 21.2 数据与证据

- registry、candidate edges、pair evidence；
- 冻结校准集和 held-out audit 集；
- 盲化人工决策及一致性报告；
- strict/cannot/guard/embargo 图；
- 每个 guard merge certificate；
- target-only 与 full-bridge 敏感性。

### 21.3 划分

- `cv3_target_core_v1.json`；
- `cv3_target_guard_v1.json`；
- `cv3_target_guard_embargo_v1.json`，含每折有效训练列表；
- 完整 4,481 图正式 CV3；
- split SHA、solver version、目标函数和约束审计；
- fold2 sentinel access log。

### 21.4 报告

- 候选召回与阈值校准报告；
- 背景视图泄漏/稳定性报告；
- 组件风险和人工一致性报告；
- 反向跨折泄漏报告；
- relaxed/core/guard/embargo 模型敏感性报告；
- 对 P06、P07 和最终模型实验的放行决定。

---

## 22. 参考依据

1. Yu et al. MAR20: A benchmark for military aircraft recognition in remote sensing images. *National Remote Sensing Bulletin*, 2023. <https://doi.org/10.11834/jrs.20222139>
2. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. 2023. <https://arxiv.org/abs/2304.07193>
3. DINOv2 官方代码与模型说明：<https://github.com/facebookresearch/dinov2>
4. Keetha et al. AnyLoc: Towards Universal Visual Place Recognition. *IEEE RA-L*, 2023. <https://github.com/AnyLoc/AnyLoc>
5. Arandjelović et al. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. *CVPR*, 2016. <https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html>
6. Radenović et al. Fine-Tuning CNN Image Retrieval with No Human Annotation. *TPAMI*, 2019. <https://arxiv.org/abs/1711.02512>
7. Pizzi et al. A Self-Supervised Descriptor for Image Copy Detection. *CVPR*, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.html>
8. Sattler et al. Large-Scale Location Recognition and the Geometric Burstiness Problem. *CVPR*, 2016. <https://openaccess.thecvf.com/content_cvpr_2016/html/Sattler_Large-Scale_Location_Recognition_CVPR_2016_paper.html>
9. Lindenberger et al. LightGlue: Local Feature Matching at Light Speed. *ICCV*, 2023. <https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html>
10. Edstedt et al. RoMa: Robust Dense Feature Matching. *CVPR*, 2024. <https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html>
11. Izquierdo and Civera. Optimal Transport Aggregation for Visual Place Recognition (SALAD). *CVPR*, 2024. <https://openaccess.thecvf.com/content/CVPR2024/html/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.html>
12. Wei et al. Breaking the Frame: Visual Place Recognition by Overlap Prediction. *WACV*, 2025. <https://openaccess.thecvf.com/content/WACV2025/html/Wei_Breaking_the_Frame_Visual_Place_Recognition_by_Overlap_Prediction_WACV_2025_paper.html>
13. Google OR-Tools CP-SAT 官方文档：<https://developers.google.com/optimization/cp/cp_solver>
14. Kattenborn et al. Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks. *ISPRS Open Journal of Photogrammetry and Remote Sensing*, 2022. <https://doi.org/10.1016/j.ophoto.2022.100018>
15. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 2017. <https://doi.org/10.1111/ecog.02881>

---

## 23. 冻结结论

本项目不再尝试用文件编号或官方 train/test 的排列推断机场。正式路线冻结为：

> **DINOv2-B 多层背景特征与 GeM/VLAD 做高召回，DINO patch overlap 与 SIFT/LightGlue 做局部同源验证；只将像素等价自动成边，其余 strict 关系人工确认；以 strict 连通分量形成 core，以非传递、带完整证书的组件合并形成 guard，对未决高风险关系使用 fold-specific embargo；随后通过 CP-SAT 构造完整 CV3，并用跨折反向检索和模型敏感性实验闭环验收。**

这条路线的价值不在于声称恢复了 60 个机场，而在于把“已证明、合理怀疑、仍不确定”分开处理，使每个来源约束都有证据、每个保守排除都能追溯、每个模型结论都能在不同风险等级下复核。
