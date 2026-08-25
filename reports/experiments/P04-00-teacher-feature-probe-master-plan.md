# P04-00 DINOv2 与扩散教师特征公平探针总纲

> 2026-07-23 状态注记：本文保留 P04 开始前的预注册内容，表格中的
> “尚未开始 GPU / 等待 B”是历史时态。P04 探索实验已经完成，正式
> `cv3_airport_proxy_k60_v2` 也已冻结；当前复验范围和状态以
> [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) 为准。

## 0. 文档状态

| 项目 | 当前值 |
| --- | --- |
| 阶段 | P0-4 / P04 规划与预注册 |
| 状态 | 探索 00—04 已完成；正式 CV3 v2 三教师复验已实现、待服务器执行 |
| 上游基线 | P03 已封板：`tight-224`、ConvNeXt-Tiny、natural sampler、seed=42 |
| 当前数据 | P0-2 `exploratory_crop_manifest_v1`，只保留为探索历史与正式 crop 几何来源 |
| 正式数据 | `cv3_airport_proxy_k60_v2` + `formal_crop_manifest_v2` 已冻结；正式结果只认该协议 |
| 主问题 | DINOv2 是否优于普通 ImageNet 表征；扩散特征是否有 DINOv2 之外的独立价值 |
| 非目标 | 不在本阶段训练完整扩散检测器，不做生成增强，不宣称端到端 Recall/FDR |

本报告冻结 P04 的研究问题、模型与权重候选、输入公平性、特征定义、实验顺序、统计口径、准入条件和停止条件。后续服务器任务单可以调整批量大小和并行方式，但不得在未更新本报告的情况下改变核心比较问题。

---

## 1. 执行摘要

P04 不是“把几个大模型都跑一遍并比较最高准确率”，而是按以下因果顺序回答问题：

1. **预训练范式价值**：同量级 DINOv2-S/14 相对 ConvNeXt-Tiny，冻结特征是否更适合本项目的小样本细粒度遥感对象；
2. **教师规模价值**：DINOv2-B/14 相对 DINOv2-S/14，提升是否足以支持使用更大的训练期教师；
3. **扩散表征价值**：CleanDIFT 是否能在同一对象、同一 crop、同一 fold 和同一读出协议下接近或超过 DINOv2；
4. **独立信息价值**：DINOv2 与 CleanDIFT 的错误是否互补，简单、受控的融合是否优于 DINOv2 单教师；
5. **比赛有效性**：收益是否集中在尾类、稳定困难对象、proposal 扰动、舰船和原生小对象，而不只是已经接近饱和的 clean GT crop；
6. **工程价值**：收益能否支撑后续蒸馏，还是仅由大输入、大特征维度或高计算量获得。

模型选择的核心结论如下：

- **DINOv2-S/14**：同量级、公平的判别式表征对照；
- **DINOv2-B/14**：P04 的强判别教师主模型；
- **CleanDIFT-SD1.5**：P04 的扩散特征主模型；
- **原始 DIFT/SD1.5**：只用于验证噪声、时间步和稳定性，不承担首轮大规模主实验；
- **SatDiFuser + DiffusionSat-256**：条件分支，用于检验遥感域扩散预训练，不在首轮下载和运行；
- **DreamTeacher、DistillDIFT**：属于证明互补性之后的蒸馏方法，不是 P04 的起始模型。

P04 采用两阶段证据制度：

- **P04-E 探索阶段**：现在即可进行环境、权重、代码、缓存、无标签稳定性检查，以及预注册读出的三折通路诊断；标签指标只用于发现实现失效和估算正式实验范围，不得淘汰教师、调 layer/timestep 或宣称优劣；
- **P04-F 正式阶段**：B 的正式分组到达并通过验收后，运行三折配对比较并形成技术决策。

---

## 2. P03 已提供的先验与 P04 的新增问题

### 2.1 P03 已经证明的事实

1. `tight-224` 是当前唯一保留的 clean crop 工作点；
2. 冻结 ConvNeXt-Tiny 的 linear probe 已有约 `0.86` macro recall，普通 ImageNet 表征不是弱基线；
3. 全量微调后 clean GT-crop macro recall 约 `0.97`，说明对象区域给定时任务高度可分；
4. 224 与 336 没有稳定、足以抵消成本的差异；
5. `sqrt_inverse` 没有稳定收益；
6. fold 差异显著大于 seed 差异，后续必须做同对象、同 fold 配对；
7. P03 识别出 191 个三 seed 全错对象和 571 个 seed 预测不完全一致对象；
8. `jitter_light` 的总体损失不大，但舰船、低 coverage、边界目标、较大尺度扰动和原生小对象更脆弱。

### 2.2 P04 不应重复的问题

P04 不再搜索：

- crop policy；
- 224/336 分辨率；
- sampler；
- 普通分类骨干；
- 大量随机 seed；
- 强图像增强组合。

### 2.3 P04 真正新增的问题

| 编号 | 问题 | 关键比较 | 可接受结论 |
| --- | --- | --- | --- |
| Q1 | 自监督通用特征是否优于普通监督特征 | DINOv2-S vs ConvNeXt-T frozen | 预训练范式是否有价值 |
| Q2 | 大教师是否必要 | DINOv2-B vs DINOv2-S | 教师规模收益与成本 |
| Q3 | 扩散特征是否适合细分类 | CleanDIFT vs DINOv2-B | 扩散单教师价值 |
| Q4 | 扩散是否提供独立信息 | DINOv2-B + CleanDIFT vs DINOv2-B | 互补性与蒸馏依据 |
| Q5 | 扩散的价值是否来自空间结构 | 统一全局读出 vs 统一空间读出 | 几何/部件信息是否被保留 |
| Q6 | 收益是否适合比赛条件 | clean、jitter、tail、hard registry | 真实使用价值 |
| Q7 | 表征是否更数据高效 | 受控标签预算曲线 | “小样本”条件下的表征效率 |
| Q8 | 遥感域生成预训练是否必要 | DiffusionSat vs 通用 SD | 域适配价值，条件执行 |

---

## 3. 模型、权重与代码版本选择

### 3.1 必选模型

| ID | 模型 | 角色 | 参数/维度 | 精确权重 | 首轮状态 |
| --- | --- | --- | --- | --- | --- |
| X-FEAT-00 | torchvision ConvNeXt-Tiny | P03 普通监督基线 | 约 28M / 768D | `convnext_tiny-983f1562.pth`，SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` | 必跑 |
| X-FEAT-01S | DINOv2 ViT-S/14，无 registers | 同量级自监督对照 | 21M / 384D | `dinov2_vits14_pretrain.pth` | 必跑 |
| X-FEAT-01B | DINOv2 ViT-B/14，无 registers | 强判别教师 | 86M / 768D | `dinov2_vitb14_pretrain.pth` | 必跑 |
| X-FEAT-02C | CleanDIFT SD1.5 | 扩散分类特征主模型 | SD1.5 U-Net / map #0 1280D | `cleandift_sd15_unet.safetensors` | 必跑 |
| X-FEAT-02S | CleanDIFT SD1.5 | 扩散结构特征 | map #6 1280D | 与上行同一权重、同一次前向 | 必跑 |
| X-FEAT-02G | CleanDIFT SD1.5 | 扩散局部几何特征 | map #9 640D | 与上行同一权重、同一次前向 | 探索行，不单独决定入选 |

DINOv2 官方权重：

- ViT-S/14：[`dinov2_vits14_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)，88,283,115 bytes；
- ViT-B/14：[`dinov2_vitb14_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)，346,378,731 bytes；
- 代码固定到官方仓库 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；
- 官方仓库使用 Apache-2.0 许可；下载后与权重 SHA-256 一并保存 `LICENSE`；
- 下载后必须计算本地 SHA-256，任务单只接受显式路径，不允许服务器静默使用“latest”缓存。

CleanDIFT 权重：

- 官方仓库：[`CompVis/cleandift`](https://github.com/CompVis/cleandift)，固定 commit `b070976b22b125167384eed5c96be3a694468763`；
- 官方 Hugging Face revision：`bf3a8d841ebdce7e212b61e42877f8fdaed81d58`；
- 文件：[`cleandift_sd15_unet.safetensors`](https://huggingface.co/CompVis/cleandift/blob/main/cleandift_sd15_unet.safetensors)；
- 大小约 1.72 GB；文件 SHA-256：`56697cc83cef762ac7ca0c8b9e749ee0abacfb426da92dc7fd5d7025ec727516`；
- 基础 SD1.5 diffusers 仓库使用维护中的 Hugging Face 镜像 `stable-diffusion-v1-5/stable-diffusion-v1-5`，固定 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`，不依赖已废弃的旧 RunwayML 仓库名；
- SD1.5 基础权重使用 OpenRAIL-M 类许可，CleanDIFT 仓库/权重标注 MIT；正式交付前仍需保存完整 license 文件。

### 3.2 为什么同时选 DINOv2-S 和 B

DINOv2-S 约 21M 参数，与 ConvNeXt-Tiny 约 28M 接近。它回答“是不是仅因为模型更大”。DINOv2-B 约 86M，是更合理的训练期强教师，回答“更大通用教师能否显著提高细粒度表征”。

如果只跑 B：

- B 优于 ConvNeXt 时无法区分自监督预训练与模型规模；
- B 不优时也无法判断是否因大模型在小数据上线性读出不适配。

因此 S 与 B 都是必要对照，但不继续加入 L/g，避免把 P04 变成教师规模搜索。

### 3.3 为什么首轮使用无 registers 版本

无 registers 版本与原始 DINOv2 论文、DIFT/CleanDIFT/DistillDIFT 的比较链最一致，且 P04 首要读出是全局分类特征。带 registers 的 B/14 只在出现以下情况时作为诊断：

- patch token 出现少数异常高范数热点；
- 空间读出明显不稳定；
- 全局 CLS 正常，但 dense feature 可视化存在系统性伪影。

它不进入首轮核心矩阵，避免多一个几乎同义的权重分支。

### 3.4 为什么不在首轮使用 DINOv3

DINOv3 是更新的通用表征模型，但 P04 的目标首先是验证既有创新路线中的 DINOv2—扩散互补链，现有 DIFT/CleanDIFT/DistillDIFT 证据也直接围绕 DINOv2。DINOv3 可在 P04 结束后作为“更强判别教师上限”单独加入，不能在首轮替换 DINOv2，否则会同时改变文献依据、模型规模和比较基准。

### 3.5 原始 DIFT 的角色

原始 DIFT 代码固定到官方仓库 commit `9421eb2034396c5b66f1aff37f03e540c264e52f`。它存在三个不适合直接做首轮主模型的问题：

1. 真实图像需加噪，特征有随机性；
2. 最优 timestep 和层随任务变化；
3. 论文默认每张图平均 8 份噪声，D4 八视图下成本被进一步放大。

因此原始 DIFT 只做两项校准：

| 配置 | 论文依据 | 目的 |
| --- | --- | --- |
| DIFT-C | SD1.5，feature map #0，`t=100` | CleanDIFT 论文分类实验的全局分类锚点 |
| DIFT-S | SD1.5，feature map #6 / `up_ft_index=1`，`t=261` | DIFT 论文语义对应锚点 |

先在无标签固定子集上测试 ensemble size `1/4/8` 的特征稳定性和成本；除非 raw DIFT 明显有价值，不对全量 D4 视图展开大规模缓存。

### 3.6 SatDiFuser 的条件分支

SatDiFuser 代码固定到 `51148cbd70ccbfdb52b1a18520fb95d0c911399c`，DiffusionSat 代码固定到 `68259b074dfa567e6cc4667799dab0e37477fcf2`。

首选条件权重为 DiffusionSat-256 的 150k checkpoint：

- 官方 Zenodo DOI：[`10.5281/zenodo.13756199`](https://zenodo.org/records/13756199)；
- 下载归档约 30 GB，包含 100k/150k checkpoint、optimizer 和配置；
- 归档 MD5：`4f5a103d101a334de9af9bc8c9958f1e`；
- 解压后只保留 150k 推理所需权重和配置，并另外计算 SHA-256。

它只在以下条件同时满足时启动：

1. CleanDIFT 或 DINOv2+CleanDIFT 已表现出扩散特征价值；
2. 通用 SD 的主要弱点呈现明确遥感域特征，例如 PAN 舰船或俯视纹理退化；
3. 服务器有至少 45 GB 临时磁盘余量；
4. B 正式 split 已冻结，避免在 30 GB 权重和独立旧环境上提前消耗时间。

若启动，只先使用：

- 256 输入；
- 不输入 class 或 metadata；
- 固定文本 `"A satellite image"`；
- timesteps `{1, 100, 200}`；
- ResNet + self-attention 输出；
- global-weighted fusion；
- 不先上 localized fusion 或 MoE。

要归因“遥感域预训练”而不是“融合头更复杂”，必须用同一 SatDiFuser 读出协议比较通用 SD2.1 与 DiffusionSat。

### 3.7 不在 P04 首轮运行的模型/方法

| 方法 | 暂不运行原因 | 何时再启用 |
| --- | --- | --- |
| DINOv2-L/g | 不能回答同量级公平问题，成本增加 | B 明显优于 S 且仍未饱和 |
| DINOv3 | 改变既有文献链和主问题 | P04 后做外部教师上限 |
| DreamTeacher | 是蒸馏训练方法，不是直接公平 probe | 扩散教师通过独立价值门禁后 |
| DistillDIFT | 公开主任务是语义对应，不是本项目分类 | 简单融合已证明互补后 |
| SDXL/SDXL-Turbo | 输入、架构和规模变化过多 | SD1.5 路线明确受容量限制后 |
| 完整 DiffusionDet | 不回答对象 crop 表征问题 | 错误分析证明定位是主瓶颈后 |
| 类别文本 prompt | 会引入标签信息，破坏纯视觉公平性 | 不在 P04 使用 |

---

## 4. 数据合同与输入公平性

### 4.1 唯一主输入

P04 核心比较统一使用：

- policy：`tight`；
- canonical resolution：`224×224`；
- color：统一 RGB；
- crop 几何、padding、插值：完全复用 P0-2/P03 loader；
- 主验证：identity view；
- 训练增强：D4 八视图中的确定性均匀采样；
- sampler：natural；
- 标签：官方 25 类。

P04 不重新打开 `context_1p25` 或 336。

### 4.2 224 与扩散原生 512 的公平处理

主比较实行“同信息、不同模型适配”原则：

1. 先由项目 loader 生成唯一的 canonical 224 RGB tensor；
2. ConvNeXt 和 DINOv2 直接对该 224 tensor 做各自官方 normalization；
3. SD1.5/CleanDIFT 只允许把这份 224 tensor 以固定 bicubic + antialias 上采样到 512；
4. 不允许为扩散模型从原始影像重新渲染 512 crop；
5. 上采样不会增加真实信息，但使输入尺寸符合扩散预训练分布；
6. 预处理 fingerprint 必须写入 cache key。

这样可以避免“扩散模型因直接读取更多原始像素而获益”的比较污染。

扩散网络直接吃 224 的行为可在 smoke 中检查，但不作为主比较，因为它偏离 SD1.5 的 512 预训练分辨率。

### 4.3 D4 八视图训练增强

P03 的 0/90/180/270° 旋转、水平/垂直翻转等价于 D4 群上的均匀分布。离线特征不能只缓存 identity，否则会无意删除 P03 训练增强。

每个 clean crop 缓存以下 8 个离散视图：

`r0, r90, r180, r270, flip_r0, flip_r90, flip_r180, flip_r270`

训练每个 epoch 对每个对象根据 `seed + epoch + annotation_uid` 选择一个视图，保持每 epoch 样本数与 P03 相同；不把 8 份视图直接当作 8 倍独立样本。验证只用 identity，避免把 TTA 收益混入教师比较。

### 4.4 PAN 与 RGB

舰船 PAN crop 继续按 P03 规则复制为 3 通道，不做伪彩色学习；飞机和车辆使用原 RGB。所有结果必须分三大类报告，因为自然图像扩散模型对 PAN 舰船的域差异可能远大于飞机。

### 4.5 文本和元数据

- DIFT/CleanDIFT 主实验使用固定空 prompt `""`；
- 不使用类别名、粗类名、飞机/舰船文本或图像文件名；
- 不使用 EXIF、地理位置、来源、分辨率等元数据；
- SatDiFuser 若启动，按论文使用统一 `"A satellite image"`，仍不使用类别和 metadata。

这保证 P04 测量的是视觉表征，不是标签提示能力。

### 4.6 B 正式划分到达前后的处理

教师特征抽取本身不需要 fold 和标签，因此缓存必须以 `annotation_uid + canonical_input_sha256` 为主键，而不能把 fold 写死在特征文件名中。

B 交付后执行：

1. 校验 schema、类别映射、25 类 support；
2. 校验 source/near-duplicate group 不跨 fold；
3. 与 P0-2 的 `annotation_uid`、图像路径、框坐标和 canonical tensor SHA 对齐；
4. 只复用 SHA 完全相同的缓存；
5. 对新增、修改或删除对象生成 cache delta；
6. 形成 `formal_crop_manifest_v2`；
7. 正式三折结论只基于 v2，不复制探索 split 的分数。

如果 B 只改变 fold 映射而不改变对象和 crop，现有缓存可 100% 复用；如果 B 纠正了标注，只有发生变化的对象重算。

---

## 5. 特征定义与公平读出

### 5.1 Track-G：教师原生全局表征

Track-G 回答“该教师开箱即用的全局对象表征有多强”。

| 教师 | 全局向量定义 | 原生维度 |
| --- | --- | ---: |
| ConvNeXt-Tiny | 最终 feature map global average pooling | 768 |
| DINOv2-S | 最终归一化 CLS token | 384 |
| DINOv2-B | 最终归一化 CLS token | 768 |
| DINOv2-B diagnostic | `CLS + mean(patch tokens)` | 1536 |
| CleanDIFT map #0 | feature map global average pooling | 1280 |
| CleanDIFT map #6 | feature map global average pooling | 1280 |
| CleanDIFT map #9 | feature map global average pooling | 640 |

每个向量先做 L2 normalization，再训练单层 25 类线性头。唯一例外是
P04-TASK-01 的 P03 工程等价组：为复核缓存是否改变 P03 语义，该组保持
P03 的无 L2 输入和兼容 head 初始化；它不参与教师公平比较。通过等价门禁后，
ConvNeXt 也回到统一 L2 主协议。

### 5.2 Track-C：维度控制

原生特征维度不同，单层线性头参数量也不同。为排除“维度更高所以更强”，每个正式 fold 额外运行 384D 控制：

1. `PCA(whiten=False)` 只在该 fold 的训练对象及其 D4 视图上拟合，避免 whitening 放大小特征值方向和尾类噪声；
2. 不读取标签；
3. 统一压缩到 384D；
4. 对验证对象只做 transform；
5. PCA transform 后统一 L2 normalization，再使用完全相同的 384→25 线性头。

模型入选不能只依赖 native-dim 提升；如果 native 提升在 384D 后完全消失，应判定主要来自表示维度或头容量。

### 5.3 Track-S：统一空间表征

全局平均可能抹掉扩散特征中潜在的部件关系。若 Track-G 显示 CleanDIFT 接近 DINOv2，或错误恢复有明显互补，才进入 Track-S：

1. ConvNeXt 最终 feature map、DINOv2 patch map、CleanDIFT map 统一取 dense feature；
2. 使用固定 `1×1 + 2×2` adaptive average spatial pyramid；
3. flatten 后在训练 fold 以无标签 PCA 压到 384D；
4. 使用同一个 384→25 线性头；
5. 不引入卷积、attention、可学习 pooling 或教师专属 decoder。

Track-S 只回答空间信息是否有价值，不与 Track-G 混成一个主表最高分。

### 5.4 DINOv2 层选择

核心模型只使用最终层 CLS，避免搜索层数。DINOv2-B 的 `CLS + mean patch` 是唯一预注册的二级读出，用于检查细粒度对象是否需要 patch 汇总。最后四层拼接只保存在可选分析接口中，不在首轮遍历。

### 5.5 CleanDIFT 层选择

一次 U-Net 前向同时缓存 #0、#6、#9，选择依据来自论文而非本项目标签搜索：

- #0：最低空间分辨率，CleanDIFT 论文在 ImageNet 分类中表现最好；
- #6：DIFT 语义对应常用位置，强调结构与语义；
- #9：与 DIFT 低噪声局部几何位置接近，作为局部轮廓诊断。

不遍历全部 11 个 feature maps；#9 不可单凭一个尾类涨分成为最终教师。

### 5.6 raw DIFT 噪声稳定性

固定 256 个对象、覆盖 25 类/PAN-RGB/大小/边界风险，比较 ensemble size `1/4/8`：

- 同一图不同噪声种子的特征余弦相似度；
- 相对 ensemble-8 均值的偏差；
- 提取吞吐和峰值显存；
- map #0 `t=100` 与 map #6 `t=261`；
- 固定空 prompt。

选择最小 ensemble 的技术门槛：

- 与 ensemble-8 的中位余弦相似度不低于 0.99；
- 第 5 百分位不低于 0.97；
- 无 NaN/Inf；
- 特征差异不被单一 PAN/RGB 模态主导。

若 ensemble-4 不通过，只保留 CleanDIFT，不把 ensemble-8 成本扩展到全量 D4。

---

## 6. 特征缓存合同

### 6.1 cache key

每条特征记录至少包含：

```text
annotation_uid
crop_id
policy
canonical_resolution
canonical_input_sha256
view_id
teacher_id
teacher_repo_commit
teacher_weight_filename
teacher_weight_sha256
preprocessing_fingerprint
extractor_implementation_fingerprint
prompt_sha256
feature_location
timestep
noise_seed_set
pooling_id
storage_dtype
feature_sha256
```

标签、fold 和 sampler 不参与 feature key。fold 只在训练线性头时由 manifest 映射。
实现 fingerprint 覆盖 canonical renderer、cache schema、输入变换、teacher adapter
与提取入口；这些代码任一字节改变都必须生成新 cache，禁止旧分片静默复用。

### 6.2 分片与完整性

- 每 shard 建议 512—2048 个对象；
- 特征文件使用 safetensors 或结构稳定的 NumPy 格式；
- 每个 shard 记录行数、维度、min/max、NaN/Inf、SHA-256；
- 建立总索引，禁止同一 key 重复；
- 中断后按 shard 续跑，不覆盖已验收 shard；
- 回传包不必包含全部特征，但必须包含完整索引、校验和与服务器路径。

### 6.3 精度

- 前向计算使用 fp16/bf16 的选择由 smoke 决定；
- 缓存默认 fp16；
- 固定 64 个对象与 fp32 参照比较；
- 归一化后最大绝对差和余弦差必须记录；
- 若 fp16 使 1% 以上对象余弦相似度低于 0.999，则改存 fp32 或只对该教师改用 fp32。

### 6.4 预计存储

以下仅计算 20,933 对象、D4 八视图、单个 fp16 全局向量，不含索引和临时文件：

| 特征 | clean D4 | jitter identity |
| --- | ---: | ---: |
| ConvNeXt 768D | 245 MiB | 31 MiB |
| DINOv2-S 384D | 123 MiB | 15 MiB |
| DINOv2-B 768D | 245 MiB | 31 MiB |
| DINOv2-B CLS+patch 1536D diagnostic | 491 MiB | 61 MiB |
| CleanDIFT map #0 1280D | 409 MiB | 51 MiB |
| CleanDIFT map #6 1280D | 409 MiB | 51 MiB |
| CleanDIFT map #9 640D | 204 MiB | 26 MiB |

含 DINO-B diagnostic 的核心全局缓存约 2.1 GiB，连同索引、PCA、预测、断点分片和临时文件，建议预留 8—12 GB。不要缓存全量高分辨率 dense maps；Track-S 只为进入门禁的教师建立压缩后的空间摘要。

---

## 7. 统一线性头训练协议

### 7.1 主协议

沿用 P03 linear probe：

| 项目 | 冻结值 |
| --- | --- |
| loss | CrossEntropy，label smoothing 0 |
| sampler | natural |
| batch size | `96`（与 P03 tight-224 linear probe 相同） |
| optimizer | AdamW |
| lr | `1e-3` |
| weight decay | `0.01` |
| max epochs | 15 |
| minimum epochs | 8 |
| warmup | 1 epoch |
| patience | 5 |
| min delta | `1e-4` macro recall |
| grad clip | 1.0 |
| canonical seed | 42 |
| validation | identity view，自然分布 |

每个 epoch 每个训练对象只采样一个 D4 view，epoch 样本数不增加。

### 7.2 head seed 复核

首轮只使用 seed=42。出现以下任一情况时，只对相邻候选补 seed `3407/202625`：

- 3-fold mean macro recall 差的绝对值小于 0.005；
- 三折方向不一致；
- 一个方法的收益主要由单一尾类 1—2 个对象驱动；
- 线性头早停 epoch 差异异常大。

不对全部矩阵机械重复三 seed。

### 7.3 ConvNeXt 缓存等价门禁

在比较新教师前，必须用 ConvNeXt 的离线 D4 特征缓存重跑 P03 frozen linear probe：

- 三折对象数量与 P03 完全一致；
- macro recall 与 P03 `tight-224` 的绝对差不超过 0.003；
- 每类方向无系统性偏移；
- 若超差，先检查视图采样、normalization、classifier 前特征位置和 eval mode。

该门禁不通过，不得解释 DINOv2/CleanDIFT 的分数。

---

## 8. 实验阶段与顺序

### P04-0：资产、环境和实现冻结

目标：证明所有模型可离线、可复现、可校验地加载。

工作：

1. 下载 DINOv2-S/B 与 CleanDIFT-SD1.5 权重；
2. 保存 URL、revision、文件大小、SHA-256、license；
3. 单独建立 P04 环境，不污染 P03；
4. 固定 PyTorch `2.5.1+cu121` / torchvision `0.20.1+cu121`；
5. CleanDIFT 依赖从其官方要求 `torch>=2.1`、`diffusers>=0.27` 出发，在 4080 SUPER smoke 后冻结精确版本；
6. 检查 DINO 和 CleanDIFT 各 8 张图的输出形状、确定性、显存、速度；
7. 保存 8 张图的 golden feature checksum/统计量，用于服务器复现。

环境版本不能仅凭当前最新包确定。候选版本必须经过 P04-0 smoke 后写入 `requirements-p04.txt`，再生成服务器任务单。

停止条件：权重/许可证不可用、输出无法复现、出现 NaN/Inf 或 32 GB 显存仍无法单图前向。

### P04-1：缓存管线与 P03 等价复核

目标：证明离线缓存没有改变数据和训练语义。

工作：

1. 实现 canonical 224 tensor 与 D4 view ID；
2. 实现教师无标签 feature extractor；
3. 建立 cache key、shard、resume 和 checksum；
4. 缓存 ConvNeXt 全量 clean D4；
5. 重跑 P03 frozen probe；
6. 通过 ±0.003 等价门禁。

### P04-2：DINOv2 主线

目标：先建立强判别教师，扩散必须与它比较。

正式行：

1. DINOv2-S final CLS，native 384D；
2. DINOv2-B final CLS，native 768D；
3. DINOv2-B CLS + mean patch，native 1536D diagnostic；
4. ConvNeXt/S/B 的统一 PCA-384 控制；
5. 三折、seed=42；
6. 同对象与 P03 frozen baseline 配对。

主要结论：

- S vs ConvNeXt：自监督预训练价值；
- B vs S：教师规模价值；
- B CLS+patch vs CLS：patch 汇总是否提供细粒度增益。

### P04-3：扩散实现与 raw/Clean 校准

目标：确认扩散特征位置和随机性，不在此阶段追最高验证分。

工作：

1. 256 对象无标签稳定性子集；
2. raw DIFT-C、DIFT-S 的 ensemble `1/4/8`；
3. CleanDIFT #0/#6/#9 单次前向；
4. 224→512 信息控制；
5. 空 prompt；
6. 对比 raw 与 Clean 的稳定性、吞吐、显存和 D4 一致性；
7. 冻结全量提取配置。

该阶段不允许根据正式验证标签搜索 timestep。

### P04-4：CleanDIFT 正式三折 probe

目标：判断扩散单教师是否有价值。

正式行：

1. CleanDIFT #0 GAP native；
2. CleanDIFT #6 GAP native；
3. CleanDIFT #9 GAP exploratory；
4. #0/#6 的 PCA-384 控制；
5. 与 DINOv2-B、DINOv2-S 和 ConvNeXt 同对象配对；
6. 不按三个 map 中的最高分事后宣称统一胜利，#0 是预注册分类主行。

### P04-5：互补性与受控融合

只有 CleanDIFT 满足以下任一条件才进入：

- 相对 DINOv2-B 总体 macro recall 不低于 -0.005，且有明显困难对象恢复；
- 在 tail、jitter、舰船或 191 稳定困难对象上有稳定价值；
- 错误集合与 DINOv2 存在明显互补。

按以下顺序执行：

1. **错误并集上限**：计算两模型至少一个正确的 oracle accuracy/macro recall；
2. **固定 0.5 概率平均**：不调融合权重；
3. **容量控制融合**：各分支先用训练 fold 拟合的 `PCA(whiten=False)` 压到 384D，拼接后再用训练 fold 无标签 PCA 压到 384D，训练同一线性头；
4. **Track-S**：只对最有希望的 DINO 与 Clean map 做统一空间摘要。

如果 oracle 上限很高但固定/容量控制融合无收益，说明信息互补但当前融合不可用，可进入后续蒸馏研究；如果 oracle 上限也低，停止多教师路线。

### P04-6：鲁棒性与困难对象分析

只对 X-FEAT-00、最佳 DINO、最佳 Clean、最佳融合运行：

- clean identity；
- `jitter_light` identity，clean 训练头 eval-only；
- 三大类；
- head/middle/tail；
- HM、LQS、FSC；
- QHS、A1_SU-35 等 P03 波动类；
- 191 个三 seed 全错对象；
- 571 个 seed 不一致对象；
- edge-risk；
- GT coverage `<0.90`；
- 原生短边 `<48 px`；
- padding-positive；
- 主要混淆对。

等 M1 OOF proposal 到达后，追加真实 Pred-OOF crop；未到达前不得用 jitter 代替真实检测框结论。

### P04-7：标签预算/数据效率诊断

该项用于回答项目“小样本”属性，不是官方 k-shot 协议，也不改变正式主表。

B split 到达后，统计每类训练 source group 数。若所有类均满足，建立每类 group-aware `k=5`、`k=10` 与 full 三档：

- 同一类先采 source group，再在组内采对象；
- 固定 3 个采样 seed；
- 比较 ConvNeXt、DINOv2-S/B、最佳 Clean；
- 主看 macro recall 随标签预算的斜率、尾类和折间方差；
- 不把人工 k-shot 结果冒充官方成绩。

若最少类不足 10 个训练 group，则使用所有类共同可满足的最大 `k*` 与 `floor(k*/2)`，并明确报告实际 support。

### P04-8：有限适配确认

冻结 probe 回答表征质量，但 P03 已表明遥感域适配可能很重要。为避免“零样本特征不佳就过早判死刑”，设置条件确认：

1. DINOv2-S 可与 ConvNeXt-Tiny 做一次同量级全量微调比较；
2. 首选 DINOv2-S，而非直接微调 B，以控制参数规模；
3. 训练 policy、fold、augmentation、sampler 和 epoch 上限沿用 P03；
4. ViT 优化只允许一组预注册配置，禁止大网格；
5. CleanDIFT 不在 P04 全量微调 U-Net；先用多层固定特征和统一浅层读出；
6. 只有扩散已通过独立价值门禁后，才另立 LoRA/蒸馏任务。

适配结果单独成表，不与 frozen probe 混排“最高分”。

### P04-9：SatDiFuser 条件实验

满足 3.6 的门禁后执行：

1. 下载并校验 DiffusionSat-256；
2. 只抽取 150k model checkpoint；
3. 先复现官方 GEO-Bench 一个小 smoke，证明环境正确；
4. 在项目 canonical224→256 输入上抽特征；
5. 主比较使用预训管线固定的像素映射，不使用全数据集 min/max；若复现官方 GEO-Bench 协议必须拟合数据统计量，则只能在每折 train 拟合，该分支不得复用 fold-independent cache；
6. 比较 raw single-stage、global-weighted fusion；
7. 用同一读出比较通用 SD2.1 与 DiffusionSat；
8. 只在 formal split 形成结论。

MoE 不是首轮选项；只有 global fusion 已有稳定收益且仍存在明显层/时刻互补才升级。

---

## 9. 核心正式实验矩阵

### 9.1 必跑主矩阵

| 行 | 特征 | 读出 | 维度控制 | 三折 | 作用 |
| --- | --- | --- | --- | --- | --- |
| R0 | ConvNeXt-T ImageNet | GAP | native + PCA384 | 是 | 普通监督基线 |
| R1 | DINOv2-S/14 | final CLS | native + PCA384 | 是 | 同量级自监督对照 |
| R2 | DINOv2-B/14 | final CLS | native + PCA384 | 是 | 强判别教师 |
| R2b | DINOv2-B/14 | CLS + mean patch | native，diagnostic | 是 | patch 汇总价值 |
| R3 | CleanDIFT-SD1.5 #0 | GAP | native + PCA384 | 是 | 扩散分类主行 |
| R4 | CleanDIFT-SD1.5 #6 | GAP | native + PCA384 | 是 | 扩散结构行 |
| R5 | CleanDIFT-SD1.5 #9 | GAP | native | 是，探索 | 局部几何诊断 |

### 9.2 过门后运行

| 行 | 条件 | 作用 |
| --- | --- | --- |
| F1 | DINO-B + Clean #0，0.5 概率平均 | 最简单互补测试 |
| F2 | DINO-B + 最佳 Clean，PCA384 容量控制融合 | 排除维度/头容量 |
| S1 | DINO dense map，统一 SPP+PCA384 | 空间读出对照 |
| S2 | Clean dense map，统一 SPP+PCA384 | 扩散空间信息 |
| J* | clean head 在 jitter 上 eval-only | proposal 扰动鲁棒性 |
| K* | group-aware 标签预算 | 数据效率 |
| A* | DINO-S limited/full adaptation | 遥感域适配确认 |
| RS* | DiffusionSat-256 global fusion | 遥感域扩散条件分支 |

### 9.3 明确禁止的“最高分拼盘”

不得在同一主表中混合以下不公平条件后只取最大值：

- 224 与直接从原图渲染的 512；
- frozen 与 full fine-tune；
- identity 与 8-view TTA；
- 单层线性头与复杂 MoE；
- 无类别文本与类别 prompt；
- natural sampler 与重采样；
- exploratory split 与 formal split。

---

## 10. 指标、统计与诊断

### 10.1 主指标

1. 三折 mean macro recall；
2. 三折 sample std；
3. pooled OOF macro recall；
4. 同对象 paired delta。

### 10.2 次指标

- macro F1；
- accuracy / top-5；
- aircraft20 macro recall；
- ship4 macro recall/accuracy；
- vehicle1 recall；
- NLL、Brier、ECE；
- 每类 precision/recall/F1/support；
- 25×25 confusion matrix。

### 10.3 配对统计

- 按 `source_image_id` 聚类 bootstrap，10,000 次；
- 报 `P(delta>0)` 和 95% 区间；
- 报 first-only correct、second-only correct 和净恢复对象；
- bootstrap 只描述已训练模型在当前 OOF 对象上的不确定性，不冒充重训方差；
- 差异接近门槛时再补 head seed。

### 10.4 互补性

至少记录：

- 两模型错误集合 Jaccard；
- double-fault rate；
- `P(Clean correct | DINO wrong)`；
- `P(DINO correct | Clean wrong)`；
- oracle union upper bound；
- 固定融合相对最佳单教师的净恢复；
- 191 stable-hard 和 571 unstable 对象的恢复率。

### 10.5 表征诊断

以下不作为模型入选的独立依据：

- 固定 2,048 对象上的 linear CKA；
- D4 view 之间的特征余弦稳定性；
- raw DIFT 噪声稳定性；
- 每类中心距离与类间 margin；
- 每类若干 query 的最近邻联系表；
- DINO patch PCA 与 CleanDIFT feature activation 可视化。

这些诊断用于解释“为何有效/无效”，不能替代正式指标。

### 10.6 计算指标

- 权重磁盘占用；
- clean D4 全量缓存大小；
- 单图与 batch throughput；
- peak VRAM；
- 每 1,000 crop 提取耗时；
- 线性头训练耗时；
- TTA/融合的额外成本；
- 若蒸馏，正式推理是否仍需教师。

---

## 11. 预注册主比较与决策门禁

### 11.1 主比较顺序

只按以下顺序作确认性结论：

1. DINOv2-S vs ConvNeXt frozen；
2. DINOv2-B vs DINOv2-S frozen；
3. CleanDIFT #0 vs DINOv2-B frozen；
4. DINOv2-B + CleanDIFT vs DINOv2-B；
5. 最佳教师/融合在 jitter 和困难子集上的保留情况。

#6/#9、DINO CLS+patch、CKA 和最近邻只用于解释。

### 11.2 强保留

一个新方法满足以下全部条件时强保留：

1. formal 3-fold mean macro recall 相对主对照提高至少 0.005；
2. 至少 2/3 folds 同方向；
3. pooled source-cluster bootstrap `P(delta>0) >= 0.95`；
4. 不使任一三大类 macro recall 稳定下降超过 0.01；
5. PCA384 控制后仍保留实质收益，或能解释为何原生维度是方法组成；
6. jitter/hard subset 不发生明显反转；
7. 计算代价可接受或可通过蒸馏消除。

### 11.3 定向保留

总体提升不足 0.005，但满足以下条件时可作为困难对象教师保留：

- 总体 macro recall 不低于主对照超过 0.003；
- 在预注册风险子集上有至少 0.02 的稳定 macro recall/恢复率改善；
- 至少 2/3 folds 同方向；
- 改善不只来自 HM/LQS 的 1—2 个对象；
- 能形成明确门控条件，如舰船、低 coverage、边界或高分歧对象。

### 11.4 模糊区

差异位于 `[-0.005, +0.005]`、fold 方向不一致或由小 support 驱动时：

1. 不宣告优胜；
2. 只补两个 head seeds；
3. 检查 PCA384、校准、D4 稳定性与困难对象；
4. 必要时执行一次统一浅层/空间读出；
5. 仍不稳定则按成本更低者处理。

### 11.5 扩散路线停止条件

满足任一项即停止扩散特征主线：

1. CleanDIFT #0/#6 在 formal split 上均明显弱于 DINOv2-B，且无困难子集价值；
2. DINO+Clean 的 oracle union 相对 DINO 提升也很小；
3. 固定融合和容量控制融合均无收益；
4. 所有收益在 PCA384 后消失；
5. 收益只在 clean GT crop 存在，在 jitter/Pred-OOF 反转；
6. 成本无法缓存、无法蒸馏或无法在时延预算内门控；
7. 收益来自类别 prompt、直接 512 重裁或其他不公平信息。

此时保留 DINOv2 或普通 ConvNeXt，不因“扩散更创新”而强行继续。

### 11.6 进入蒸馏的条件

只有以下条件满足才立项 DreamTeacher/DistillDIFT 风格学生：

1. DINOv2 或 CleanDIFT 单教师对困难对象有稳定价值；或
2. DINO+Clean 容量控制融合稳定优于 DINO 单教师；
3. 教师增益在 formal split 和 jitter/Pred-OOF 中都存在；
4. 学生目标、教师 feature/logit、缓存和评估口径已明确。

学生的最低目标是保留教师增量的 80%，而不是追求教师绝对分数完全复现。

---

## 12. B 正式划分验收计划

B 交付后，P04 暂停模型比较，先完成以下报告：

1. fold/group schema；
2. 25 类对象数、source group 数和每折 support；
3. source/near-duplicate leakage 检查；
4. 与 P0-2 fold 的对象、分组和 support 差异；
5. HM/LQS 等尾类是否每折可评；
6. 每类最小训练 source group 数，决定标签预算曲线；
7. canonical input SHA 复用率；
8. formal manifest SHA-256。

验收失败时，不在有泄漏或缺类的 fold 上运行 P04。需要 B 修复，或明确采用嵌套/留组方案。

---

## 13. 服务器任务拆分

服务器任务号与报告号分开维护：

| 任务号 | 内容 | 是否等 B | 预计主要成本 |
| --- | --- | --- | --- |
| P04-TASK-00 | 权重、环境、8/64/256 样本 smoke | 否 | 1—2 小时，含安装 |
| P04-TASK-01 | ConvNeXt D4 cache 与 P03 等价门禁 | 否 | 约 0.5 GPU 小时 + head |
| P04-TASK-02 | DINO-S/B 全量无标签 cache | 否 | smoke 后实测，预计小时级 |
| P04-TASK-03 | raw/CleanDIFT 稳定性与小规模 probe | 否 | 2—6 GPU 小时 |
| P04-TASK-04 | CleanDIFT 全量 D4 cache | 可先做；formal head 等 B | 预计 4—12 GPU 小时，以 smoke 外推 |
| P04-TASK-05 | formal 三折主矩阵与 PCA 控制 | 是 | 缓存后主要是轻量 head |
| P04-TASK-06 | 融合、jitter、困难对象、标签预算 | 是 | 1—4 GPU 小时 |
| P04-TASK-07 | DINO-S 适配确认 | 条件 | 约 1—3 GPU 小时 |
| P04-TASK-08 | SatDiFuser/DiffusionSat | 强条件 | 30 GB 下载 + 独立环境 + 1—3 GPU 日 |

时间必须由 P04-TASK-00 的真实吞吐外推，不直接采用论文 A6000/A100 的速度。

4080 SUPER 32 GB 足以承担 DINO-S/B 和 SD1.5 单步特征抽取；raw DIFT ensemble 以 batch/ensemble 分块，禁止为了速度一次塞满导致不可复现 OOM。

---

## 14. 产物与验收要求

### 14.1 每个提取任务

- resolved config；
- Git commit 和 dirty 状态；
- 环境 freeze；
- GPU/driver/CUDA/PyTorch；
- 权重路径、大小、SHA-256、license；
- 输入 manifest SHA；
- preprocessing fingerprint；
- feature schema；
- shard index 与 SHA-256；
- NaN/Inf/shape audit；
- throughput、VRAM、耗时；
- 失败与重试记录。

### 14.2 每个 head run

- fold、seed、feature cache ID；
- PCA/normalizer 只在 train fit 的证明；
- resolved config；
- history；
- best checkpoint；
- validation logits；
- crop-level predictions；
- fixed 25×25 confusion matrix；
- per-class metrics；
- calibration metrics；
- run summary；
- checkpoint/产物 SHA-256。

### 14.3 P04 汇总报告

至少包含：

1. 主比较四步结论；
2. native 与 PCA384 两张表；
3. clean 与 jitter；
4. 三大类与 head/middle/tail；
5. 191/571 困难对象恢复；
6. 互补性与 oracle/fusion；
7. 计算和存储；
8. 是否进入蒸馏/SatDiFuser；
9. 停止的分支及理由；
10. B formal split 的适用范围和局限。

---

## 15. 风险与控制

| 风险 | 可能造成的错误结论 | 控制 |
| --- | --- | --- |
| 扩散直接读原图 512 | 把额外像素误认为扩散能力 | canonical224→512 |
| 离线缓存删除训练增强 | 新教师不公平低估 | D4 八视图缓存 |
| 特征维度不同 | 高维模型不公平占优 | native + PCA384 双轨 |
| GAP 抹掉空间结构 | 错判扩散无价值 | 过门后统一 Track-S |
| timestep/layer 搜索 | 在小验证集过拟合 | 论文锚点、限制 #0/#6/#9 |
| raw DIFT 随机噪声 | 结果不可复现 | 固定 seed、ensemble 稳定性 |
| 类别 prompt | 标签泄漏 | 固定空 prompt |
| exploratory split 被当正式 | 结论乐观或泄漏 | B 到达后 formal 重算 |
| 尾类百分比过大 | 1 个对象被夸大 | support、净对象、fold 方向 |
| 复杂融合直接涨分 | 无法证明教师互补 | oracle→0.5→PCA 控制顺序 |
| SatDiFuser 30 GB/旧依赖 | 挤占主线和磁盘 | 强门禁、独立环境 |
| 只看 clean GT crop | 无法服务检测系统 | jitter、Pred-OOF、背景后续复核 |

---

## 16. 当前立即可做与需要等待的工作

### 16.1 现在即可开始

1. 下载并校验 DINOv2-S/B；
2. 下载 CleanDIFT SD1.5 U-Net 和固定 SD1.5 基础 revision；
3. 建立 `requirements-p04.txt` 候选环境并做 8/64 样本 smoke；
4. 编写统一 teacher adapter、feature hook、D4 view 和 cache shard；
5. 用 ConvNeXt 做缓存等价门禁；
6. 抽取 DINO-S/B 全量、fold-independent 特征；
7. 对 raw/CleanDIFT 做 256 对象稳定性与成本试验；
8. 建立 P03 的 191 stable-hard / 571 unstable 对象注册表接口；
9. 生成权重与许可证清单。

### 16.2 B 到达前不应做

- 不以探索 fold 选择最终教师；
- 不运行大规模层/时间步网格；
- 不下载 30 GB DiffusionSat；
- 不做复杂融合或蒸馏；
- 不宣告正式提升。

### 16.3 B 到达后立即做

1. 正式 split 验收；
2. formal manifest 与 cache 复用审计；
3. ConvNeXt canonical baseline 复跑；
4. DINO-S/B formal 三折；
5. CleanDIFT #0/#6/#9 formal 三折；
6. 互补性、融合、jitter 和数据效率；
7. 根据门禁决定蒸馏、适配、SatDiFuser或停止。

---

## 17. 预期可形成的结论类型

P04 结束后应能得到以下一种或多种明确结论，而不是“某模型好像不错”：

1. **DINOv2-S 明显优于 ConvNeXt frozen**：通用自监督表征更适合作为小样本对象教师；
2. **DINOv2-B 只小幅优于 S**：部署/学生优先围绕 S，B 只作离线教师；
3. **DINOv2-B 显著优于 S**：教师容量有效，后续蒸馏有意义；
4. **CleanDIFT 单独不强、但恢复 DINO 错误**：作为定向或多教师分支保留；
5. **CleanDIFT 与 DINO 融合稳定提升**：进入 DistillDIFT/DreamTeacher 风格学生；
6. **CleanDIFT 仅 clean 有效、jitter 反转**：不进入正式检测系统；
7. **CleanDIFT #6/#9 只在空间读出有效**：扩散价值来自部件/几何而非全局类别；
8. **通用 SD 弱、DiffusionSat 强**：遥感域生成预训练值得保留；
9. **DINO 和扩散均无独立价值**：保留 P03 ConvNeXt，转向真实 proposal、背景和全局聚合；
10. **低标签预算下教师优势更明显**：为“小样本”表征与蒸馏故事提供直接证据，但不改写官方协议。

---

## 18. 论文与官方实现依据

- [DINOv2 官方仓库与权重](https://github.com/facebookresearch/dinov2)：ViT-S/B 参数量、官方 backbone 和线性探针接口；
- [DIFT 官方实现](https://github.com/Tsingularity/dift)：`t=261`、`up_ft_index=1`、ensemble size 8 等语义对应配置；
- [CleanDIFT 官方实现](https://github.com/CompVis/cleandift)：无噪声、无 timestep 的单次特征抽取与 SD1.5/2.1 权重；
- [CleanDIFT 官方权重](https://huggingface.co/CompVis/cleandift/tree/main)：SD1.5/2.1 U-Net 与 full checkpoint；
- [SatDiFuser 官方实现](https://github.com/yurujaja/SatDiFuser)：DiffusionSat、timesteps `{1,100,200}` 和多阶段融合；
- [DiffusionSat 官方实现](https://github.com/samar-khanna/DiffusionSat)：256/512 卫星域生成权重；
- 本地论文：`07_DINOv2`、`15_SatDiFuser`、`19_DIFT`、`20_CleanDIFT`、`21_DreamTeacher`、`22_DistillDIFT`；
- 项目总路线：`doc/扩散模型创新路线详细执行报告.md`；
- P03 基线：`P03-00` 至 `P03-04`。

---

## 19. 最终执行原则

> 先用 DINOv2-S/B 建立强且可解释的判别式教师阶梯，再用 CleanDIFT 的预注册层位检验扩散单教师与空间结构价值，最后只用受控融合证明独立互补；所有缓存共享同一 224 信息与 D4 视图，所有正式结论等待 B 的同源隔离划分，并同时通过 tail、困难对象、jitter/Pred-OOF 和成本门禁。扩散若不能提供 DINOv2 之外的稳定收益，就停止；若能提供，则再蒸馏，而不是把大扩散网络直接带入正式推理。
