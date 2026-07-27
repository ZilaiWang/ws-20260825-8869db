# P03-01 普通 crop 分类上限：预注册与线性探针结果

> 2026-07-23 状态注记：本文是探索划分上的历史结果。正式 CV3 v2 已
> 冻结，入选结论仍需按 `formal_crop_manifest_v2` 复跑；当前状态见
> [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)。

## 1. 任务定位

P0-3 的唯一核心问题是：

> 在已经给定真实对象区域时，当前数据中的 25 个细类，尤其是 20 个飞机型号，使用成熟 ImageNet 预训练模型究竟能分到什么程度？

这是一个“条件可分性上限”实验，不是端到端检测实验。它不包括背景候选、漏检、重复框、大图切片和全局融合，因而不能报为官方 Recall/FDR，也不能代替最终系统成绩。

P0-3 在整个创新路线中的作用是给后续技术分流：

- 普通分类器已经很强：后续应优先攻 proposal 质量、背景拒识和全局对象聚合，扩散特征必须证明有额外价值；
- GT crop 强、轻度 jitter 明显弱：优先做完整重裁和 proposal 鲁棒训练；
- 即使 GT crop 也弱：当前像素证据、细类标签或同源外泛化才是核心瓶颈，扩散教师只能作为待验证表征先验，不应预设能解决；
- 224 与 336 近似：后续 P0-4 优先使用 224，节省所有教师特征抽取和蒸馏成本；
- context 与 tight 差异稳定：固定为后续对象头的输入契约，不再将 crop 几何当作无限调参轴。

## 2. 证据基础与不可越界的解释

P0-3 直接使用 P0-2 `exploratory_crop_manifest_v1`：

- 20,933 个独立标注对象，每个对象有 `tight`、`context_1p25`、`jitter_light` 三种几何；
- 3 个 fold 均含 25 个细类；
- HM 只有 17 个对象、13 个泄漏组，LQS 只有 30 个对象、19 个泄漏组；
- fold 由原图级同源组和近重复候选的保守并集生成，不允许将 manifest 行随机切分；
- 本文 fold 是正式来源分组冻结前的探索性划分；CV3 v2 现已完成，入选结论尚待按新 fold 重跑。

小样本性在 P0-3 中不通过人为设置 K-shot 来制造，而是通过真实的细类长尾、独立来源数和折间波动呈现。HM/LQS 不做单独百分比的过度解释；必须同时给出 support、单折值和三折波动。

## 3. 模型与环境冻结

### 3.1 主模型

首轮只使用 `torchvision.models.convnext_tiny`：

- 架构：ConvNeXt-Tiny；
- 预训练：ImageNet-1K V1；
- 规模：约 28.6M 参数；
- 官方权重：`convnext_tiny-983f1562.pth`；
- SHA-256：`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`。

ImageNet 是预训练数据源，不是某一个具体模型。ConvNeXt-Tiny 是本阶段的成熟、中等规模判别基线，比小型教学 CNN 更能代表“普通方法能达到的上限”，又不会将 3090 的大部分时间消耗在过大模型上。

ResNet-50 只作为 torchvision 版本或 ConvNeXt 异常时的故障回退，不进入首轮模型网格。如果后续要证明结论不依赖骨干，只在最佳输入条件补一个对照。

### 3.2 服务器环境

| 项目 | 冻结值 |
| --- | --- |
| GPU | NVIDIA RTX 3090 24 GB |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.x，建议 3.10.14 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| PyTorch CUDA runtime | cu121 / CUDA 12.1 |
| NVIDIA driver | 必须兼容 CUDA 12.1；由 `nvidia-smi` 实测 |
| 内存 | 建议至少 32 GB |
| CPU | 建议至少 8 vCPU |
| 可用磁盘 | 建议至少 40 GB，不含长期备份 |

P0-3 只使用 torchvision 标准算子，不需要系统 `nvcc` 或完整 CUDA Toolkit；PyTorch cu121 wheel 自带所需运行时。以后的扩散模型、xformers 或自定义 CUDA 算子建议单独建环境，不破坏这个已冻结基线。

## 4. 输入、变量和不变项

### 4.1 首轮可变项

| 轴 | 候选 | 回答的问题 |
| --- | --- | --- |
| crop policy | `tight`, `context_1p25` | 更高主体占比与场景上下文哪个更有价值 |
| resolution | 224, 336 | 额外表征网格是否变成稳定精度收益 |
| fold | 0, 1, 2 | 结论是否能离开某个同源组合仍成立 |

`jitter_light` 不是第一轮训练条件，而是在 clean 模型确定后对同一批验证对象做配对鲁棒性测试。

### 4.2 不变项

- 统一 25 类输出，飞机 20 类是同一模型上的固定子集诊断，不用另一个模型人为改变难度；
- ImageNet mean/std；
- 从 P0-2 浮点方窗直接 resize，原图外黑色 padding；
- 训练只使用 0/90/180/270° 旋转和水平/垂直翻转；
- 首轮禁用 random resized crop、color jitter、MixUp、CutMix、Copy-Paste 和扩散生成；
- 验证集保留自然类别分布，不采样、不重复；
- 首轮 seed=42，只在最佳条件补充 seed=3407 和 202625；
- 模型选择指标为 macro recall（即 balanced accuracy），不用被高频飞机类主导的 overall accuracy。

## 5. 数据 loader 契约

1. 第 `k` 折训练使用 `fold != k`，验证使用 `fold == k`。
2. 先按 fold 选源对象，再选 crop policy；绝对不对 62,799 行随机分割。
3. loader 启动时同时校验 `annotation_uid`、`source_image_id` 和 `leakage_group_id` 三层不相交。
4. 同一 policy 下每个 `annotation_uid` 只能出现一次。
5. 路径必须在 `data_root` 内，原图尺寸必须与 manifest 一致。
6. 保留浮点 crop EXTENT，不先四舍五入为整数框。
7. 每个 DataLoader worker 只缓存少量已解码原图，不将数万 crop 写入仓库。

## 6. 分阶段实验顺序

### P03-0：环境与通路门禁

目标是确认“能正确跑”，不生产技术结论。

- 核对 Python/PyTorch/torchvision/CUDA/GPU；
- 核对权重和 manifest SHA-256；
- 验证三折样本数和防泄漏不变式；
- 打开一批源图并核对尺寸；
- 用 fold0/tight/224 跑 256 train + 128 val、1 epoch smoke；
- 检查 forward/backward、AMP、checkpoint、logits、混淆矩阵和 JSON 是否完整。

通过条件：无 NaN/Inf，输出 25×25 混淆矩阵，checkpoint 可重载，GPU 确为 3090，显存峰值可记录。smoke 分数不进入比较。

### P03-1：冻结特征 linear probe 筛选

运行 `2 policies × 2 resolutions × 3 folds = 12` 个实验：

- 冻结 ConvNeXt 全部特征提取层，仅训练新 25 类 head；
- 使用自然分布 sampler；
- 预设 15 epoch，最少 8 epoch，macro recall 早停；
- 输出三折均值和 sample standard deviation；
- 按三折 mean macro recall 排序。

这一阶段回答“原始 ImageNet 表征已经保留了多少可分信息”，不是最终上限。

入选规则：

1. 主排序为三折 mean macro recall；
2. 与最高值差不超过 0.005（0.5 个百分点）时视为工程近似并列；
3. 并列时先比 macro F1，再选 224，再选 tight；
4. 选前 2 个条件进入全量微调；
5. 若第 2 名与第 3 名差小于 0.005，且三折方向不一致，微调阶段补第 3 名，不在 linear probe 上强行做虚假确定性。

### P03-2：限定微调的普通分类上限

对 P03-1 入选的 2 个条件各跑 3 fold：

- 解冻整个 ConvNeXt-Tiny；
- backbone LR `1e-4`，head LR `5e-4`，AdamW，cosine decay + 2 epoch warmup；
- 30 epoch 上限，最少 12 epoch，patience 8；
- label smoothing 0.1；
- 仍先用 natural sampler，不同时改类别平衡。

这是 P0-3 最核心的 clean GT crop 结果。如果 linear probe 与 fine-tune 对输入条件的排名反转，优先信任 fine-tune，并补充反转条件的一组确认运行。

### P03-3：不均衡 sampler 消融

只对 P03-2 最佳输入条件追加 `sqrt_inverse` 三折：

\[
w_c=\min\left(\sqrt{n_{max}/n_c},10\right)
\]

使用 replacement 采样，每 epoch 样本数不变。不使用完全倒数，因为 HM/LQS 极少，完全倒数会反复暴露几张同源图并加重过拟合。

判断时同时看：

- overall macro recall/F1；
- aircraft20 macro recall；
- 按训练 support 三等分的 head/middle/tail macro recall；
- 高频类是否出现明显退化；
- HM/LQS 只做带 support 的趋势诊断，不用少数对象的单次转正宣称方法成功。

### P03-4：轻度 proposal 扰动配对评估

对每个 fold 的最佳 clean checkpoint，分别评估同 fold 的 `jitter_light`：

- 不在 jitter 上重新训练；
- 按 `annotation_uid` 将 clean/jitter 预测一一配对；
- 计算 macro recall/F1 差、对象级由对变错/由错变对、信心度变化；
- 分析 GT coverage、中心偏移、尺度扰动与失败的关系。

这只能说明对 P0-2 人工轻扰动的鲁棒性。M1 的 OOF 框出来后，必须用真实 proposal 重裁替代它才能得出工程结论。

### P03-5：最佳工作点稳定性

只对最终入选条件补 seed=3407 和 202625。该步不扩大超参数网格，只验证主结论是否依赖初始化。

## 7. 必报指标和结果表

### 7.1 主指标

- 25 类 macro recall / balanced accuracy；
- 25 类 macro F1；
- 三折 mean 和 sample standard deviation。

### 7.2 辅助指标

- overall accuracy 和 top-5 accuracy；
- aircraft20 子集 macro recall/F1；
- ship4、aircraft20、vehicle1 子集；
- 25 类 precision/recall/F1/support；
- 固定 25×25 混淆矩阵，行是真类，列是预测类；
- head/middle/tail macro recall；
- 单折样本数、每类 support 和折间方向一致性；
- 验证吞吐、峰值 CUDA 显存、最佳 epoch。

overall accuracy 只能辅助阅读，因为 17,849 个飞机对象会压过 HM 17 例、LQS 30 例和 FSC 402 例的信号。

### 7.3 主表结构

| regime | policy | res | sampler | macro R mean±std | macro F1 mean±std | aircraft20 macro R | acc | val throughput | peak VRAM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

第二张表按类列 support/recall/F1，第三部分给混淆对、clean/jitter 配对变化和头中尾分层。

## 8. 决策逻辑

P0-3 不设置一个脱离数据的“达到 X% 就成功”门槛，而是按错误结构做决策：

1. **微调后 clean crop 高，但检测头后续低**：精分类上限存在，全局重裁和对象头有价值。
2. **linear probe 低、fine-tune 明显高**：ImageNet 表征需要较强遥感域适配；P0-4 教师比较必须允许公平适配，不能只拿零样本特征定胜负。
3. **linear probe 和 fine-tune 都低**：先核查混淆对、同源差异、像素上限和标签一致性；不直接认为扩散模型会解决。
4. **336 只提升 accuracy，不提升 macro recall/尾类**：不保留 336。
5. **336 在三折稳定提升 macro recall 且改善易混细类**：再将精度收益与教师特征 cache、显存和正式对象头时延一起权衡。
6. **context 改善粗类但损害飞机型号**：不用一个统一 context 强行覆盖所有大类，后续可在对象头使用紧裁与上下文双视图，但只在结果证据支持时实现。
7. **sqrt-inverse 只改善极少类单折、且折间反复**：视为不稳定，不保留。
8. **jitter 降幅大**：优先做 proposal 鲁棒性和完整重裁，并等 OOF 框做真实误差学习。

## 9. 运行时间与资源规划

数据规模约为每折 13.8k–14.1k train、6.8k–7.1k val。真实速度会受 JPEG/PNG 解码、CPU 数量和服务器磁盘影响，因此不在运行前伪造精确时间；以 smoke 的实测 samples/s 重估。

初始预算：

- P03-0：30–60 分钟，包括环境安装、数据检查和 smoke；
- P03-1：预留 6–12 GPU 小时；
- P03-2：预留 8–16 GPU 小时；
- P03-3/P03-4：合计预留 4–8 GPU 小时；
- P03-5：只在主结论有价值时再批准。

这是资源排期区间，不是模型时延结论。每一阶段结束后再开下一份服务器任务单，不一次性烧完全部网格。

## 10. 每个 run 的强制产物

- `resolved_config.yaml`；
- `meta.json`：包含 Python/PyTorch/CUDA/GPU/Git/权重与 manifest checksum；
- `history.csv`；
- `best_checkpoint.pt` 及 SHA-256；
- `metrics.json`；
- `per_class_metrics.csv`；
- `confusion_matrix.csv`；
- `predictions.csv`；
- `validation_logits.npz`；
- `run_summary.json`；
- 服务器外层 stdout/stderr log。

服务器回传时，线性探针的大 checkpoint 可以保留在服务器，但所有小型指标、logits、预测、配置、日志和 checkpoint checksum 必须回传。入选 fine-tune 的 checkpoint 必须回传或在服务器保证可继续访问。

## 11. 当前已完成与待执行

已在本地完成：

- P0-2 manifest 专用读取和防泄漏校验；
- 浮点 EXTENT、越界 padding、动态 resize 的 crop dataset；
- 固定 25 类指标、混淆矩阵、飞机子集和头中尾指标；
- ConvNeXt-Tiny 显式本地权重加载与 checksum 门禁；
- linear probe / fine-tune / natural / sqrt-inverse / eval-only 通用训练入口；
- 环境检查、smoke 模式、三折汇总和条件选择工具；
- 不依赖 PyTorch 的本地单元测试。

当前立即可执行：P03-0 和 P03-1。

待 P03-1 结果回传后再决定：P03-2 入选的两个输入条件。

待 M1 OOF 框后才能正式执行：真实 proposal crop 上限、背景拒识样本建立和检测误差分解。

## 12. 代码和配置入口

- 配置：`configs/experiments/p03_convnext_tiny.yaml`；
- 数据：`src/rsdet/data/crop_classification.py`；
- 指标：`src/rsdet/evaluation/classification.py`；
- 模型：`src/rsdet/models/crop_classifier.py`；
- 训练：`scripts/train_crop_classifier.py`；
- 环境门禁：`scripts/check_p03_environment.py`；
- 三折汇总：`scripts/summarize_p03_runs.py`；
- 服务器首任务：`docs/server/P03_TASK_01_ENV_AND_LINEAR_PROBE.md`。

## 13. P03-1 执行结果：冻结特征 linear probe

### 13.1 执行状态与完整性

P03-TASK-01 已于 2026-07-16 在服务器完成：

- 12/12 个三折 run 成功，无 OOM、无重试；
- GPU 元数据为 `NVIDIA GeForce RTX 4080 SUPER`，单设备报告 33,794,359,296 bytes（约 31.47 GiB）；
- PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、CUDA runtime 12.1、cuDNN 9.1.0；
- manifest 与权重 SHA-256 匹配，4,481 张源图全部通过尺寸和 SHA-256 校验；
- 12 个结果中每个 fold 的原始 logits、CSV 预测、混淆矩阵和存储指标均已在本地独立复算；
- 每个条件的 OOF 对象集均为相同的 20,933 个 `annotation_uid`，且每个对象恰在一个 held-out fold 中出现一次。

回传包 SHA-256 为：

`981b14ead1b89f9841bda8635c33b4745671c1d52fdefd38dcab24a82efb980b`

`RETURN_FILES_SHA256.txt` 含 141 个条目：140 个实际结果文件全部匹配；唯一不匹配是该清单在填充前将自身的空文件 SHA 写入了自身。这是清单生成顺序问题，不是实验产物损坏；P03-TASK-02 已修正打包命令，不再将清单自身纳入清单。

### 13.2 主结果

主排序仍使用预注册的三折 mean±sample std：

| 条件 | macro recall | macro F1 | aircraft20 macro recall | accuracy | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tight-336 | 0.8662 ± 0.0243 | 0.8787 ± 0.0207 | 0.8928 | 0.9085 | 929.8 MiB |
| tight-224 | 0.8637 ± 0.0180 | 0.8751 ± 0.0139 | 0.8958 | 0.9089 | 841.1 MiB |
| context-336 | 0.8544 ± 0.0202 | 0.8665 ± 0.0186 | 0.8851 | 0.9001 | 929.8 MiB |
| context-224 | 0.8475 ± 0.0149 | 0.8558 ± 0.0146 | 0.8829 | 0.9000 | 841.1 MiB |

按预注册规则，tight-336 与 tight-224 的三折 mean macro recall 差为 0.0025，低于 0.005 工程并列界，因而两者均进入 P03-2。第 2 名 tight-224 与第 3 名 context-336 差 0.0093，不触发补 context 微调的条件。

### 13.3 同对象配对结果

将三折的 held-out 预测合并为一份 20,933 对象 OOF 集合后，使用相同 `annotation_uid` 配对比较：

| first → second | first only 正确 | second only 正确 | 净增正确 | accuracy 差 | pooled macro recall 差 | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tight-224 → tight-336 | 740 | 729 | -11 | -0.0005 | +0.0031 | 0.794 |
| context-224 → context-336 | 775 | 776 | +1 | +0.00005 | +0.0063 | 1.000 |
| context-224 → tight-224 | 652 | 839 | +187 | +0.0089 | +0.0154 | 1.46e-6 |
| context-336 → tight-336 | 614 | 789 | +175 | +0.0084 | +0.0123 | 3.39e-6 |

这里的 McNemar p 是对“是否分对”的连续性校正卡方近似，用于判断配对 accuracy 差，不是 macro recall 的显著性检验。

分折方向进一步说明：

- tight-336 相对 tight-224 的 macro recall 差为 `+0.0136 / -0.0068 / +0.0007`，折间方向不一致；
- tight-224 相对 context-224 为 `+0.0139 / +0.0139 / +0.0209`，三折一致；
- tight-336 相对 context-336 为 `+0.0265 / +0.0028 / +0.0061`，三折一致。

因此，**tight 优于 1.25× context 是 P03-1 的稳定结论；336 优于 224 不是稳定结论。**

### 13.4 不均衡与细类结构

将 25 类按完整数据 support 等数量分为 9 个 tail、8 个 middle 和 8 个 head 类：

| 条件 | tail macro R | middle macro R | head macro R |
| --- | ---: | ---: | ---: |
| tight-224 | 0.7891 | 0.8819 | 0.9290 |
| tight-336 | 0.7932 | 0.8899 | 0.9262 |
| context-224 | 0.7586 | 0.8749 | 0.9221 |
| context-336 | 0.7775 | 0.8768 | 0.9184 |

336 在 tight 下的 macro recall 增量来自 tail/middle，但 head 略降。具体细类也并非单向：

- tight-336 相对 tight-224 改善 LQS `+0.133`、A4_C-5 `+0.038`、A19_SU-34 `+0.039`；
- 同时损害 A18_KC-10 `-0.057`、A15_F-22 `-0.041`、A5_F-16 `-0.016`；
- HM 仅 17 例，tight-224 与 tight-336 recall 均为 0.765；
- LQS 仅 30 例，其 0.267 与 0.400 必须与 support 同时呈现，不能将 4 个样本的净差过度解释为分辨率定律。

四个条件的对象级稳定性为：

| 正确条件数 | 对象数 | 占比 |
| ---: | ---: | ---: |
| 4/4 | 17,303 | 82.66% |
| 3/4 | 1,417 | 6.77% |
| 2/4 | 836 | 3.99% |
| 1/4 | 581 | 2.78% |
| 0/4 | 796 | 3.80% |

持续困难类中，LQS 有 15/30 个对象在四个条件下全错，QHS 为 92/641，A11_E-8 为 53/432，A18_KC-10 为 30/262。这些是后续 fine-tune、类别平衡和教师特征对照需优先检查的对象。

### 13.5 错误对和像素证据

tight-224 的主要有向混淆为：

- A19_SU-34 → A1_SU-35：149；反向 106；
- QHS → MS：146；反向 85；
- A13_F-15 → A5_F-16：58；反向 51；
- A18_KC-10 → A14_KC-135：51；
- A15_F-22 → A13_F-15：47。

tight 的 pooled OOF top-5 accuracy 在 224/336 下分别为 0.9960/0.9953。这表明绝大多数错误不是远离真类的完全失败，而是若干视觉近邻类之间的排序问题。这支持后续检验更强域适配、不均衡学习和教师表征，但不能说明扩散特征必然更好。

按原生短边分层时，tight-224 在 `<48 px`、`48–96 px`、`>96 px` 对象上的 accuracy 分别为 0.8490、0.9029、0.9294。即使 crop 将对象放大到相同输入分辨率，原图中的像素证据差异仍然保留；crop 放大不是超分。

源图边界风险对象在 tight-224 上的 accuracy 为 0.8875，低于 clean 对象的 0.9099；macro recall 为 0.8227 对 0.8752。边界和 padding 仍是后续真实 proposal crop 必须保留的诊断维度。

### 13.6 阶段结论与 P03-2 决策

P03-1 支持以下结论：

1. **普通 ImageNet 表征已经很强。** 只训练 19,225 个线性头参数就得到约 0.86 的三折 macro recall，P0-4 中 DINOv2/扩散特征必须与该强基线使用相同 crop/fold/resolution 比较。
2. **tight 是首选 clean 几何。** context_1p25 在两个分辨率下都稳定退化，不进入 P03-2 微调。
3. **224 是临时工程首选，336 是必要对照。** 224 的 accuracy 和 aircraft20 更高、折间标准差更小、吞吐更高；336 的 macro recall/F1 略高且对部分尾类有收益，但折间方向不一致。
4. **P03-2 不再筛模型或 context。** 只对 tight-224 和 tight-336 各做三折 ImageNet 初始化全量微调，保持 natural sampler、seed=42 和其他预注册设置。
5. P03-2 后再决定唯一 clean 工作点，然后进入 sqrt-inverse sampler 和 `jitter_light` 配对评估。

P03-1 不允许宣称：端到端检测已达 0.86 recall、背景 FDR 已解决、336 已优于 224、扩散教师已无必要或 ImageNet 表征已是最终最优。

### 13.7 复现与本地分析产物

独立复算命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_p03_results.py \
  --stage linear_probe \
  --runs-root outputs/P03-TASK-01 \
  --manifest outputs/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --output-dir outputs/P03-TASK-01/local_analysis
```

本地产物包括：`integrity_report.json`、`condition_summary.csv`、`fold_metrics.csv`、`pairwise_comparisons.csv`、`fold_pairwise_deltas.csv`、`per_class_comparison.csv`、`frequency_tier_metrics.csv`、`subset_metrics.csv`、`top_confusions.csv`、`object_stability.csv`、`stability_per_class.csv` 和 `analysis_summary.json`。

本地分析脚本 SHA-256：

`c657df34ec1111b83b7217f3bed84b5a00c671f0bed4cf0301c7838e6d1c48d9`

验证结果：全仓 Pytest 130 项通过；P03-1/P03-2 相关 Python 文件 Ruff 通过；`git diff --check` 通过。

服务器使用 4080 SUPER，因此本阶段的显存、吞吐和耗时只用于 P0-3 条件间的同设备成本对照，不得当作官方 RTX 3090 上的 10K 端到端时延。
