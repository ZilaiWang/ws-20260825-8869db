# P03-00 对象 crop 分类实验总纲

## 1. 目标与边界

P03 回答一个有限问题：当真实对象区域已知时，成熟 ImageNet 预训练模型对官方 25 个细类的可分性上限有多高，该上限对 crop 几何、输入分辨率、类别不均衡和 proposal 扰动是否稳定。

本实验不包含背景候选、漏检、重复框、大图切片和全局融合，因而不是官方 Recall/FDR，不得作为端到端成绩。当前三折仍使用 P0-2 探索性同源隔离划分；B 交付正式同源分组后，最终工作点需重跑。

## 2. 冻结基线

| 项目 | 冻结值 |
| --- | --- |
| 数据 | P0-2 `exploratory_crop_manifest_v1`，20,933 个独立标注对象 |
| 划分 | 3 fold，以 `leakage_group_id` 防同源/近重复泄漏 |
| 模型 | torchvision ConvNeXt-Tiny，ImageNet-1K V1 |
| 权重 SHA-256 | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` |
| 任务 | 统一 25 类；aircraft20 只作子集诊断 |
| 主指标 | 3-fold mean macro recall，同时报 sample std |
| 验证分布 | 自然类别分布，不重复、不采样 |
| 基础增强 | 0/90/180/270° 旋转、水平/垂直翻转 |
| 禁用 | random resized crop、color jitter、MixUp/CutMix、合成数据 |

HM 仅 17 个对象，LQS 仅 30 个对象。所有尾类结论必须同时给出 support、单折方向和对象净变化，不用少数样本的百分比独立立论。

## 3. 阶段与状态

| 阶段 | 问题 | 状态 | 正式记录 |
| --- | --- | --- | --- |
| P03-0 | 环境、数据、权重和 smoke 通路是否可信 | 完成 | P03-TASK-01 回传产物 |
| P03-1 | 冻结 ImageNet 特征下，crop 几何/分辨率如何筛选 | 完成 | `P03-01-linear-probe-results.md` |
| P03-2 | 全量微调后的 clean GT-crop 上限，唯一分辨率是什么 | 完成 | `P03-02-fine-tune-results.md` |
| P03-3 | `sqrt_inverse` 是否稳定改善尾类且不伤官方相关大类 | 完成；不保留 `sqrt_inverse` | [`P03-03`](P03-03-balance-and-jitter-results.md) |
| P03-4 | clean checkpoint 对 `jitter_light` 的配对性能损失 | 完成；确认轻度但结构化的退化 | [`P03-03`](P03-03-balance-and-jitter-results.md) |
| P03-5 | 最终工作点对 seed 的稳定性 | 完成；P03 封板 | [`P03-04`](P03-04-seed-stability-results.md) |

## 4. 已冻结结论

1. `tight` 是 clean crop 唯一保留几何；`context_1p25` 在两个分辨率下均稳定退化。
2. 全量微调将 tight crop 的 macro recall 从约 0.86 提高到约 0.97，证明 ImageNet 初始化 + 本任务监督适配可以形成很强的对象细分类基线。
3. 336 相对 224 的小幅增益未达到稳定且具有工程性价比的程度；P03-3 冻结使用 `tight-224`。
4. 本阶段仍是带 GT 区域的条件上限。直到 M1 交付 OOF proposal，才能测量真实检测框重裁、背景拒识和端到端收益。
5. `sqrt_inverse` 没有稳定改善尾类或主指标，最终 sampler 冻结为 `natural`；类别均衡不再扩展网格。
6. `jitter_light` 使 natural 基线 pooled accuracy 下降 0.00205、macro recall 下降 0.00394，风险主要集中于低 coverage、舰船、边界目标、较大尺度扰动和原生小对象。
7. 三个 seed 的 clean 三折 mean macro recall 最大差为 0.00286，而 fold 均值跨度为 0.01769；当前优化 seed 波动明显小于数据折差异，seed=42 继续作为预注册 canonical baseline。
8. P03 普通 crop 分类系列已经封板；正式 split 到达后只复跑 canonical baseline 和被保留的教师对照，不复开全部探索网格。

## 5. 统一决策规则

- 主结论先看三折 mean macro recall，再看单折方向、pooled OOF、同对象配对、大类指标和计算代价。
- 配对 bootstrap 按 `source_image_id` 聚类，仅表示当前已训练模型在 OOF 对象上的抽样不确定性，不冒充重新训练的 seed 方差。
- 对尾类的改善如果只由 HM/LQS 的 1–2 个对象驱动，不足以单独保留更复杂方案。
- 若精度在实用上近似，选输入更小、吞吐更高、校准更好的方案。
- 任何分类器收益都必须在真实 OOF proposal crop 上重新确认，才能进入最终系统。

## 6. 产物规则

每个正式 run 必须保留 resolved config、环境元数据、history、best checkpoint 及 SHA-256、原始 logits、对象级预测、混淆矩阵、单类指标和 run summary。服务器回传包必须有不自包含的 `RETURN_FILES_SHA256.txt`；未提供 checkpoint 本体时，至少提供路径、大小和 SHA-256。
