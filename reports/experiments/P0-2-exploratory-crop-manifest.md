# P0-2 探索性 crop manifest 建立与验证

## 1. 任务结论

P0-2 已完成。本次将官方数据中 20,933 个真实对象统一表示为可复现的对象级 crop 样本，并将来源、类别、大类、模态、防泄漏分组、划分、几何窗口、jitter 参数、渲染契约和源文件校验和固化在同一 manifest 中。

最终产物包含 3 种 crop 策略、共 62,799 行：

- `tight`：以 GT 长边为方窗边长的基准样本；
- `context_1p25`：围绕 GT 中心保留 1.25 倍上下文；
- `jitter_light`：确定性的轻度 proposal 扰动诊断样本。

本 manifest 已可供 P0-3 普通 crop classifier 上限实验和 P0-4 教师特征对照使用。它是“探索性数据契约”，不是 B 尚未冻结的正式分组划分，也不是已经物化的独立图像数据集。

## 2. P0-2 要回答的问题

本阶段不训练模型，只负责先固定后续实验的比较对象和边界。它必须回答：

1. 每个原始标注如何唯一转化为可追溯的 crop 样本；
2. 224 和 336 分辨率能否在不改变样本几何的前提下公平比较；
3. tight、1.25× context 和轻度 proposal 扰动能否使用同一来源和同一 fold 比较；
4. 同源图、相邻裁片和近重复候选是否会跨 train/val 或 fold；
5. HM、LQS 等极少样本类在对象数、源图数和独立 leakage group 数上究竟有多少证据；
6. P0-3/P0-4 是否能通过同一 `crop_id` 和 fold 做严格配对实验。

## 3. 输入数据与证据边界

输入全部来自已生成的数据审计产物和原始数据：

| 输入 | 作用 |
| --- | --- |
| `bbox_statistics.csv` | 20,933 个 bbox、25 个细类、大类和边界标记 |
| `image_stats.csv` | 4,481 张源图的尺寸、路径、SHA-256 和启发式同源组 |
| `proposed_group_split.csv` | 审计阶段的 80/20 候选划分和三折候选 |
| `domain_cluster_assignments.csv` | 数值域簇，供后续分层诊断 |
| `near_duplicate_groups.csv` | 11 条双代理命中的近重复候选边 |
| `../data` | 官方原始图像，用于存在性、checksum 和预览验证 |

必须保留的证据边界：

- `estimated_group_uid` 只是根据文件名前缀或连续编号构造的低/中置信启发式组；4,481 张图中 3,073 张为 low，1,408 张为 medium。
- 11 条近重复只是候选，并非人工确认的重复事实。P0-2 为防止高估，选择保守合并。
- 官方当前 val 目录为空；此处 train/val 和 fold 都是为内部探索生成的候选划分。
- `domain_cluster` 是匿名数值聚类，不能还原为经人工确认的机场、港口或地面场景。

## 4. manifest 整体设计

### 4.1 样本单位

基础统计单位始终是“真实标注对象”，不是 manifest 行。每个 `annotation_uid` 对应 3 个几何变体，但仍只有一份独立源信息。因此：

- 对象数为 20,933，不是 62,799；
- 三种策略不能分别随机划分；
- 验证集不得通过重复行或 sampler 伪造样本量；
- 对比结果必须按同一 `annotation_uid`/fold 配对。

### 4.2 主要字段

`crop_manifest.csv` 含 71 个字段，关键类型如下：

| 字段组 | 关键字段 | 作用 |
| --- | --- | --- |
| 唯一性 | `manifest_version`, `schema_version`, `crop_id`, `annotation_uid` | 版本化和唯一定位 |
| 来源 | `source_image_id`, `source_relative_path`, `source_checksum_sha256` | 回溯官方原图 |
| 划分 | `main_split`, `fold`, `original_*`, `*_changed`, `*_reason` | 保留修复前后完整记录 |
| 防泄漏 | `leakage_group_id`, `estimated_group_uid`, `near_duplicate_edge_count` | 确保同源证据不跨划分 |
| 标签 | `class_id`, `class_name`, `major_class`, `modality` | 25 类、3 大类和 PAN/RGB 输入区分 |
| GT | `gt_x0...gt_y1`, `gt_width`, `gt_height`, `gt_short_edge` | 原始对象几何 |
| 策略 | `crop_policy`, `context_scale`, `crop_policy_fingerprint` | 固定实验条件 |
| proposal | `proposal_x0...proposal_y1`, `jitter_*` | 显式保存扰动后候选框 |
| crop | `crop_x0...crop_y1`, `crop_side_px` | 运行时可渲染的浮点方窗 |
| QA | `source_valid_fraction`, `padding_fraction`, `gt_coverage_fraction` | 边界 padding 与主体覆盖诊断 |
| 渲染 | `target_resolutions`, `color_mode`, `outside_policy`, `resize_semantics` | 统一 P0-3/P0-4 loader |

### 4.3 不物化全量 crop

本次只存储原图路径和浮点方窗。loader 在训练或特征抽取时再从原图渲染，如有性能需求可建可删除 cache。这避免了：

- 额外保存数万小图带来的磁盘和同步压力；
- 因重复压缩引入不一致的像素变化；
- 后续调整 crop 契约后旧文件和新 manifest 错配；
- 224/336 生成两份几何样本，从而混淆“输入分辨率”与“crop 窗口”。

## 5. 防泄漏分组与修复

### 5.1 保守并集

首先将每个 `estimated_group_uid` 内的图像合并，然后加入 11 条近重复候选边，对整张图做并查集连通分量。最终得到 620 个 `leakage_group_id`。其中 4 个分量含近重复证据，共覆盖 72 张图。

审计阶段原候选已经使启发式同源组不跨划分，但在 11 条近重复候选中仍有：

- 2 条跨 main split；
- 9 条跨 fold。

### 5.2 分配规则

对每个 leakage group 分别修复 main split 和 fold：

1. 选择该 group 原划分中图像数最多的标签，从而全局最小化移动图像数；
2. 只在组内票数并列时，使用 main split 80/20 和 fold 三等分目标做确定性破局；
3. 所有候选组合在小规模下穷举，拒绝隐式随机或未记录的贪心选择；
4. 为每张改动图保存原标签、新标签、是否移动和移动理由。

最终 main split 移动 8 张图，fold 移动 28 张图。修复后：

- estimated group 跨 split/fold：0；
- leakage group 跨 split/fold：0；
- 近重复候选跨 main split：0；
- 近重复候选跨 fold：0。

## 6. crop 几何和渲染契约

### 6.1 tight

对 GT 框 `(x0, y0, x1, y1)` 令 `L=max(w,h)`，以 GT 中心为方窗中心，得到边长为 `L` 的浮点方窗。GT 短边方向的空间来自原图上下文，只有方窗越出原图时才使用黑色 padding。

### 6.2 context_1p25

保持 GT 中心不变，使用 `L=1.25*max(w,h)`。它的目的是与 tight 公平检验语义上下文是否值得主体网格分配下降，而不是预先假定 1.25× 更好。

### 6.3 jitter_light

本策略是首轮链路诊断：

- 中心 x/y 分别在 GT 宽/高的 ±8% 内偏移；
- proposal 宽高独立乘以 `[0.9,1.1]` 内的缩放；
- 每个随机量由 `global_seed + manifest_version + annotation_uid + policy + policy_fingerprint` 的 SHA-256 直接派生；
- 实现后的 seed、中心偏移比例、宽高缩放系数全部落表。

该策略不代表 M1 检测器的真实误差分布，也不应作为最终 proposal crop 结论。M1 产出 OOF 预测后，必须用真实预测框重新建立误差对照。

### 6.4 渲染契约

- 坐标：连续浮点 `xyxy` 半开语义；
- 颜色：解码后统一转 RGB；
- 越界：保留名义方窗，原图外补 `(0,0,0)`；
- resize：方窗直接 resize 到 224 或 336；
- 几何行：224/336 共用同一 crop 窗口，不重复生成 manifest 行。

## 7. 数据结果

### 7.1 整体划分

| main split | 源图 | 对象 | ship | aircraft | vehicle |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,589 | 16,736 | 2,126 | 14,262 | 348 |
| val | 892 | 4,197 | 556 | 3,587 | 54 |

| fold | 源图 | 对象 | ship | aircraft | vehicle | HM | LQS | FSC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,508 | 6,989 | 842 | 5,986 | 161 | 5 | 11 | 161 |
| 1 | 1,479 | 7,111 | 895 | 6,091 | 125 | 6 | 9 | 125 |
| 2 | 1,494 | 6,833 | 945 | 5,772 | 116 | 6 | 10 | 116 |

3 个 fold 都包含全部 25 个细类。HM 和 LQS 分别被分为 `5/6/6` 和 `11/9/10`，因此三折配对评估比单一 main holdout 更适合首轮上限实验。main val 中 HM 只有 2 个对象、LQS 只有 8 个对象，其单次细类精度波动会非常大，不应单独用于判断教师或架构优劣。

### 7.2 尾类独立证据量

| 类别 | 对象 | 源图 | leakage groups |
| --- | ---: | ---: | ---: |
| HM | 17 | 15 | 13 |
| LQS | 30 | 25 | 19 |
| FSC | 402 | 67 | 41 |

对 HM 和 LQS，“源图数”和“独立分组数”比 manifest 行数更能表示有效证据量。P0-3 若需类别均衡，只能在训练 sampler 中实现；验证时必须保留 fold 的自然分布。

### 7.3 几何统计

| policy | crop side P50 | mean padding | padding P95 | GT coverage P5 | GT coverage mean | coverage `<0.9` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tight | 108.0 px | 1.0% | 5.9% | 1.000 | 1.000 | 0.0% |
| context_1p25 | 135.0 px | 2.1% | 15.3% | 1.000 | 1.000 | 0.0% |
| jitter_light | 108.2 px | 1.1% | 7.0% | 0.881 | 0.951 | 11.0% |

tight 和 context 都以 GT 为中心，所以名义 crop 对 GT 的覆盖为 1。这不等于越界 GT 在源图中都可见；对此 manifest 另存 `source_gt_visible_fraction`。jitter 有 11.0% 样本对 GT 的覆盖低于 0.9，但没有样本低于 0.75，符合“轻度而非破坏性”诊断定位。

大类 padding 差异明显：context 下 ship 的平均 padding 为 7.4%，aircraft 为 1.3%，vehicle 为 1.3%。因此 P0-3 除总体精度外，应分大类、边界风险和 padding 幅度报告。

## 8. 验证与质量控制

### 8.1 程序内建校验

- 62,799 个 `crop_id` 全部唯一；
- 20,933 个 `annotation_uid` 均恰好包含 3 种策略；
- 所有方窗都是有限正数且 x/y 边长与 `crop_side_px` 一致；
- 所有 padding、有效区域和 GT 覆盖比例均在 `[0,1]`；
- 同一对象的 3 个策略共用同一 source/split/fold/group；
- 修复后启发式组、leakage group 和近重复候选的跨划分数均为 0。

### 8.2 源文件验证

4,481 张源图均实际存在，总字节数为 1,225,990,013。全部文件已重算 SHA-256 并与 `image_stats.csv` 一致。整个源图集合的排序指纹为：

`fa360b9c8e552760edb1071e480117c45db3455bb51472bcfe78758b500a77b3`

### 8.3 独立重算

完数据运行后另以独立审计脚本重新读取 CSV，重算了行数、ID 唯一性、策略完整性、方窗、比例、同源组、近重复边和类别计数。另外在新目录重跑了完数据生成，以下 8 个数值产物逐字节一致：

- `crop_manifest.csv`；
- `image_assignments.csv`；
- `leakage_components.csv`；
- `class_distribution.csv`；
- `policy_geometry_summary.csv`；
- `validation_report.json`；
- `manifest_summary.json`；
- `resolved_config.yaml`。

### 8.4 人工几何 QA

每个细类选择一个接近该类原生短边中位数的样本，对三种 policy 分别渲染 5×5 联系表。人工检查确认：

- tight 的主体居中和长边尺度符合定义；
- context 保留更多周边场景，没有改变对象中心；
- jitter 发生轻度中心和尺度变化，未发现系统性完全丢失主体；
- PAN 舰船和 RGB 飞机/车辆均能统一解码为 RGB 输入。

该联系表只用于发现裁剪方向、坐标、padding 或类别对齐错误，不用于训练，也不能替代全量数值校验。

### 8.5 代码验证

- Pytest 全仓 114 项通过；
- P0-2 新增测试覆盖 tight/context 几何、jitter 确定性与界限、贴边 padding、近重复合并、CLI 端到端产物和错误输入拒绝；
- Ruff 对本次 P0-1/P0-2 相关 Python 新增/修改文件通过；
- Ruff 全仓仍报告一条与本次任务无关的既有问题：`src/rsdet/postprocess/calibration.py` 导入排序。为避免篡改其他人正在维护的代码，本次未对该文件做无关修改。

## 9. P0-3/P0-4 强制使用契约

1. 做第 `k` 折时，必须先以源图级 `fold == k` 划为验证，再选择 policy；不得将展开后的 manifest 行随机切分。
2. 首轮主实验对照是 `tight/context_1p25 × 224/336`；`jitter_light` 是鲁棒性诊断，不与 GT crop 混合成一个不可解释的训练集。
3. 验证集每个对象对每个条件只评估一次；类别均衡只允许在训练 sampler 中实现。
4. P0-4 的 ImageNet baseline、DINOv2 和扩散教师必须共用同一 `crop_id + resolution + fold`。
5. 教师特征 cache key 至少包含 `crop_id + resolution + teacher_weight_checksum + layer/timestep + preprocessing_fingerprint`。
6. loader 不得因贴边而缩短 crop window；必须先保持名义方窗并 padding，再 resize。
7. 主表必须报告三折聚合结果及折间波动；同时报细类 macro average、三大类、HM/LQS/FSC 和边界风险子集。

## 10. 当前能立即开始的实验

P0-2 完成后，不需等待 M1 检测器或 B 正式划分就可先做：

1. 实现一个只读 manifest 的 runtime crop dataset/loader，同时覆盖 PAN 和 RGB；
2. 用 `tight/context_1p25 × 224/336` 做轻量 ImageNet 预训分类器三折基准，得到 GT crop 的可识别上限；
3. 在同一基准上用 `jitter_light` 定量测量轻度 proposal 误差带来的性能衰减；
4. 按相同 `crop_id` 搭建 DINOv2/扩散特征离线抽取和 cache 骨架，先用少量样本检查特征尺寸、层和预处理；
5. 将 main split 用于快速调试，但将所有技术结论放到三折配对对照后再判断。

仍需等待上游输入才能做：

- M1 OOF 预测框完成后，将 `jitter_light` 与真实 proposal crop 对照；
- B 冻结正式同源分组后，生成 `exploratory_crop_manifest_v2` 或正式 manifest；
- 基线检测器产出 FP 后，才能建立真实困难背景/前景拒识集；
- 检测错误分析确认定位是主瓶颈后，才进入 bbox residual 扩散或一步修正器。

## 11. 局限与禁止过度解释

- manifest 的 62,799 行不等于 62,799 个独立样本。
- GT crop 分类结果只是对象层可识别上限，不等于端到端检测结果。
- `jitter_light` 不是检测器误差模型，不能用它声称正式预测 crop 已被验证。
- 此版本没有背景类、困难负样本、mask、方向对齐、扩散生成数据或多尺度复杂策略。
- 三折虽然都有全部 25 类，但某些飞机类受同源组约束后仍明显不均衡；必须报告折间波动。
- 保守合并未人工确认的近重复候选可能降低一点数据利用率，但这比评估泄漏更可控。

## 12. 复现与产物

执行命令：

```bash
PYTHONPATH=src python scripts/build_crop_manifest.py \
  --bbox-statistics ../dataset_audit/machine_readable/bbox_statistics.csv \
  --image-stats ../dataset_audit/machine_readable/image_stats.csv \
  --split-candidates ../dataset_audit/machine_readable/proposed_group_split.csv \
  --domain-assignments ../dataset_audit/machine_readable/domain_cluster_assignments.csv \
  --near-duplicates ../dataset_audit/machine_readable/near_duplicate_groups.csv \
  --data-root ../data \
  --config configs/analysis/exploratory_crop_manifest.yaml \
  --output-dir outputs/P0-2-exploratory-crop-manifest \
  --verbose
```

完整本地产物位于 `outputs/P0-2-exploratory-crop-manifest/`：

- `crop_manifest.csv`：62,799 行对象几何 manifest；
- `image_assignments.csv`：4,481 张源图的修复前后划分；
- `leakage_components.csv`：620 个防泄漏连通分量；
- `class_distribution.csv`：按划分、大类和细类的对象/源图/分组计数；
- `policy_geometry_summary.csv`：三种策略的方窗、padding 和 GT 覆盖统计；
- `validation_report.json`、`manifest_summary.json`、`meta.json`、`resolved_config.yaml`；
- `previews/*.png`：3 张 25 类几何 QA 联系表；
- `figures/*.svg`：3 张划分和几何图表；
- `report.md` 和 `run.log`。

验证环境：Python 3.13.12、NumPy 2.4.6、PyYAML 6.0.3、Pillow 12.2.0。

关键 SHA-256：

- bbox statistics: `89e2dce2b6a8de7dfe6554040dfde80de4c3b0bab2b7b158d19b6c858a2aaad0`
- image stats: `25df3201692d98449e61bdf46a3a12664dadee86caccc338a1cc478e76686b4d`
- split candidates: `e8ddfb60476f6ecc77c9af9dbb8d8f0d935f718f30285fc53e51bb95cfd806c4`
- domain assignments: `a569bf1722a8faab8865a57757e6ca552c4467451a02192b9fe83db5365077ce`
- near duplicates: `16b5dcde0993a9873eaae4fa1d1bacc060fafb75bbaeb0dada7d2bd4f7180b3d`
- config: `392db65b9195b3f7df9b2b50437e195a5f5f5881ef5e69581095ebf6a0688637`
- analysis module: `1767cb465f097c910aa762fe4c09e6738e8434b27658db41fb3cf3ba7211865c`
- CLI: `d7b295e4d65fc3c0d28d413efc356074e506d36ce84fa7989aad0c3611d52c8f`
- generated manifest: `f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e`

本次运行时 Git 分支为 `feat/object-visibility-analysis`，P0-1 与 P0-2 实现均尚未提交，所以 `meta.json` 如实记录 `dirty: true`。为了在未提交状态下仍可追溯，metadata 已另存分析模块和 CLI 的文件校验值。
