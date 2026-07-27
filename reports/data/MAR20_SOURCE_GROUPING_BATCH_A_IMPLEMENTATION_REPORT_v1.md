# MAR20 来源分组批次 A 实现与本地验收记录

> 当前状态：本文是 Batch A 的阶段性实施记录；其“尚未生成正式来源组或
> CV3”只描述当时状态。K60 和正式 CV3 v2 均已完成，当前入口见
> [`DATA_SPLITS_MASTER_INDEX_v1.md`](DATA_SPLITS_MASTER_INDEX_v1.md)。

## 1. 批次结论

批次 A 已完成 MG00 registry、MG01 背景视图审计、盲化校准 pair、DINOv2-B Round-A 描述子缓存与分析所需的代码、配置、测试、服务器任务单和 SHA 门禁。

本批次的完成含义是：

> 输入映射、背景隔离、人工证据和描述子筛选流程已经可执行、可恢复、可审计；尚未宣称恢复机场，也没有生成任何正式来源组或 CV3。

## 2. 已实现内容

### 2.1 数据与证据合同

- `mar20:<number>` 稳定节点 UID；
- target 3,073 与 bridge 769 明确分离；
- 原始 `train/test` 仅记录为 `official_side`；
- strict 证据与 `likely_same_airport` 分离，后者禁止 union；
- pair UID、标签集合、缓存 schema、稳定 JSON/SHA、路径边界均有代码门禁。

### 2.2 MG00 registry

- 逐图核验原始 MAR20 与竞赛 `MAR20_*`；
- 同时记录文件 SHA 和 EXIF 校正后完整 RGB 像素 SHA；
- 原 XML HBB 与竞赛 YOLO 的 20 类直方图逐图核对；
- 全量输出 registry CSV、annotations JSONL、summary 和实现环境；
- 只允许完整 RGB 像素相同成为自动 H0，近重复不得自动合并。

### 2.3 MG01 背景视图

- bbox union mask 及 10%/15%/20% 外扩；
- blur、local mean、确定性 Telea 三种填充；
- original、masked inpaint、纯背景 tiles 三类视图；
- 背景 tile 有效比例、空间多样性、坐标和输入像素 SHA；
- 120 图 contact sheet 和人工门禁编译器；
- 人工门禁明确检查飞机残留、inpaint 伪影和背景 tile 中飞机。

### 2.4 DINOv2 Round-A

- 复用 P04 冻结 DINOv2-B/14 权重和官方 repo commit；
- zero-based block 9/10/11；
- CLS、patch mean、signed-GeM p=2/3/4，共 15 个描述子；
- ImageNet 标准化、输出 L2、fp16 计算/缓存；
- 分片缓存、原子 sidecar、SHA 审计和断点续跑；
- 指标包含 recall@20/50/100、Wilson 区间、AUC、普通/困难负例 top-k、邻居 Jaccard 和前景影响 `FI`；
- 选择仍标记 `provisional_until_vlad`。

### 2.5 盲化人工校准

- 360 个唯一 pair；
- 约 8% 的交换顺序盲重复卡；
- mapping 与人工 review 包分离；
- label、confidence、支持/反证、备注均有合同；
- 重复一致率低于 0.90 或存在重复 label 冲突时失败；
- strict positive 少于 30 时只允许探索，不允许宣称阈值已校准。

## 3. 本地真实数据核验结果

本地使用项目目录中的原始 MAR20 和竞赛数据完成全量 MG00：

| 项目 | 结果 |
|---|---:|
| 原始 MAR20 | 3,842 |
| 竞赛 target | 3,073 |
| bridge | 769 |
| official train/test | 1,331 / 2,511 |
| target 文件字节不一致 | 0 |
| target RGB 像素不一致 | 0 |
| target 细类直方图不一致 | 0 |
| 完整像素 H0 重复组 | 0 |
| XML width/height 为 0 | 21 |

确定性产物：

```text
image_registry.csv
bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d

image_annotations.jsonl
0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4
```

21 个 0×0 XML 使用真实解码图像尺寸，并逐项记录；其他非零尺寸冲突不会被放宽。

## 4. 结构性 smoke 结果

### 4.1 背景视图

- mask 外像素保持完全不变；
- 背景 tile 选择可重复；
- 小样本 smoke 中 mask fraction 和 tile 数均有限；
- 本地 visual smoke 表明 blur 可能残留轮廓，local mean 会产生矩形伪影，因此服务器正式包坚持使用 Telea 为主视图，并保留人工门禁。

### 4.2 360 pair 正式规模结构 smoke

本地用 local mean 仅验证卡片调度和产物合同，不作为科学视图结果：

| 项目 | 结果 |
|---|---:|
| 唯一 pair | 360 |
| 盲重复组 | 29 |
| 总卡数 | 389 |
| 最小实际重复卡间距 | 31 |
| 最大实际重复卡间距 | 38 |
| 唯一节点 | 672 |
| contact sheets | 98 |

卡片布局经过视觉检查，四列分别是 A/B original 与 A/B masked，卡号清晰且无布局覆盖。正式服务器实验使用 Telea，不能把本地 local-mean 图片作为人工决定输入。

### 4.3 缓存与分析

- mock encoder 缓存成功写入、审计和完整 resume；
- 同一输入重跑 `computed=0`，index SHA 不变；
- descriptor CLI 的正/负例排序、人工编译、Wilson 区间和 FI 均有端到端测试；
- 真实 DINO 模型加载、15 特征维度和 GPU 数值稳定性留给服务器 smoke。

## 5. 代码验收

```text
全仓 pytest：162 passed
批次 A 专项 pytest：11 passed
批次 A scoped ruff：All checks passed
服务器代码/输入 SHA：21/21 OK
```

全仓 `ruff check .` 仍会报告 8 个本批次之前已存在、与本任务无关的格式问题；本批次未擅自修改其他成员文件。服务器任务只对批次 A 的冻结范围执行 ruff。

## 6. 服务器阶段与下一门禁

服务器任务分两段：

1. Phase A 自动完成 registry、Telea 视图包、盲评包、真实 DINO smoke 和无标签 Round-A cache，随后正常进入 `waiting_for_manual_reviews`；
2. 人工填写两份 CSV 后，Phase B 编译视图质量、盲重复一致性和 descriptor bake-off。

只有下列条件同时满足，才可进入 Round-B VLAD/四旋转：

- registry 技术门禁通过；
- 人工视图门禁通过；
- 盲重复一致性通过；
- strict positive 数量足够；
- held-out recall 达到冻结目标；
- 结果仍只作为候选召回描述子，不获得自动 union 权限。

若真实正例不足或 recall 未达标，状态是科学非准入，不视为代码失败；下一步应补充证据或召回器，而不是放宽标签和阈值。
