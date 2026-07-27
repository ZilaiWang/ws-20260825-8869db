# P03/P04 正式 CV3 v2 复验总纲

更新日期：2026-07-23  
状态：`implemented_ready_for_server`

## 1. 复验目的

此前 P03/P04 使用的是探索性 crop fold，已回答“这个方向是否值得继续”，
但不能作为来源隔离后的正式模型选择。现在
`cv3_airport_proxy_k60_v2` 已冻结，复验只回答两个收敛后的问题：

1. **P03**：固定 `tight-224 + ConvNeXt-Tiny + ImageNet-1K V1 +
   natural + seed=42` 后，给定真实对象区域的 25 类可分性在正式三折上有多高；
2. **P04**：同一批对象、同一 canonical224、同一 D4 训练视图和同一
   train-only 读出协议下，ConvNeXt、DINOv2-B 与 CleanDIFT 的表征排序是否
   保持，以及扩散特征是否有 DINOv2 之外的独立价值。

这两项仍是对象 crop 分类/教师特征诊断，不是端到端目标检测 Recall/FDR。

## 2. 已有证据如何收缩本轮范围

探索实验已经给出：

- P03 全量微调约为 macro recall `0.97`，224 与 336 收益很小，224 成本更低；
- natural 与 sqrt-inverse 基本并列，后者方差更大；
- 三个 seed 的均值差小于 0.003，故不再遍历 seed；
- P04 探索排序为 DINOv2-B CLS+patch `0.9098`、ConvNeXt train-RMS
  `0.8797`、CleanDIFT map0 `0.8293`；
- CleanDIFT map6/map9 明显更弱，DINO-S 也没有超过 B；
- SD1.5 生成增强已经达到停止条件。

因此正式复验不重新搜索模型、分辨率、sampler、seed、扩散层或 timestep。

## 3. 冻结输入

| 输入 | 冻结值 |
|---|---|
| 正式 CV3 | `data/splits/cv3_airport_proxy_k60_v2.json` |
| CV3 SHA256 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| 正式 crop | `formal_crop_manifest_v2` |
| 服务器唯一消费路径 | `/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv` |
| 正式 crop SHA256 | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |
| crop 行数 | 62,799（20,933 对象 × 3 policy） |
| tight 三折对象 | 7,350 / 7,179 / 6,404 |
| P03 权重 | ConvNeXt-Tiny ImageNet V1，SHA `983f1562…d3d` |
| 随机性 | seed=42，确定性 cuDNN |

正式 crop 由公共入口 `scripts/build_formal_crop_manifest.py` 生成。它只重挂
fold/group，不重裁、不改 `annotation_uid`、`crop_id`、类别、框几何或像素
合同。探索 assignment 被改名为 `historical_p02_*`，不会继续伪装成 active
group。P03/P04 只消费公共 F00 已验收的 `run-a`；缺失时等待 F00，禁止自行
另建正式副本，`run-b` 也禁止训练消费。

## 4. 共同门禁

正式运行前必须执行 `scripts/audit_p03_p04_formal_inputs.py`：

1. formal/exploratory 对象与 policy 集合完全一致；
2. 非 assignment 字段逐字一致；
3. `historical_p02_*` 与原探索字段逐项一致；
4. active `fold/group_id/leakage_group_id` 逐行等于 CV3 v2；
5. 255 个来源组均不跨折；
6. manifest SHA、行数、policy 和每折对象数等于冻结值。

正式汇总还会把每折 `predictions.csv`、保存的 labels/logits 与 formal
manifest 逐对象对齐，独立复算指标；仅“行数相同”不能证明 pooled OOF
没有重复、遗漏或跨折。

P04 还必须对三个旧 cache 做额外门禁：

- cache UID 集合必须恰好等于 20,933 个正式对象；
- `crop_id` 必须逐对象相同；
- 从原图重新渲染 canonical224，SHA 必须逐对象、逐 cache 相同；
- 每个对象必须恰有完整 D4 八视图；
- cache fingerprint、shard SHA、维度及有限性审计通过。

只对齐 CSV 或只比较对象数量不足以准入 cache。

## 5. P03 正式矩阵

只运行三次：

| fold | policy | resolution | regime | sampler | seed |
|---:|---|---:|---|---|---:|
| 0 | tight | 224 | full fine-tune | natural | 42 |
| 1 | tight | 224 | full fine-tune | natural | 42 |
| 2 | tight | 224 | full fine-tune | natural | 42 |

优化器超参数沿用已冻结 P03：backbone LR `1e-4`、head LR `5e-4`、
weight decay `0.05`、label smoothing `0.1`。正式协议固定跑满 30 epoch，
使用 `final_checkpoint.pt`；held-out fold 不参与逐 epoch 验证、early stop
或 checkpoint 选择，只在训练完成后评估一次。三个 fold 都从同一 ImageNet
权重独立初始化。

必须报告：

- 每折 macro recall/F1、accuracy、aircraft20 recall；
- 三折 mean±sample std；
- pooled OOF 指标；
- 固定 25×25 confusion、逐类结果和 head/middle/tail；
- TU-160 的 fold0 9-shot 训练压力折，不能隐藏或删除。

P03 的结论表述为“给定对象区域的细分类上限”，不能写成正式检测成绩。

## 6. P04 正式矩阵

三个教师各跑 native 与 PCA384，每项三折，共 18 个 probe：

| 教师 | 特征 | native | PCA384 |
|---|---|---:|---:|
| ConvNeXt-T | `convnext_gap` | 768D | 是 |
| DINOv2-B | `dino_cls_patchmean` | 1536D | 是 |
| CleanDIFT SD1.5 | `clean_map0` | 1280D | 是 |

每个 fold 严格执行：

1. PCA 只用两个训练 fold 的对象及其 D4 特征拟合，`whiten=false`；
2. native 或 PCA 输出的全局 RMS 只用训练 fold 拟合；
3. 验证 fold 只做 transform 和除以训练 RMS；
4. 25 类线性头只用训练 fold 拟合；
5. 验证只使用 `r0`，不混入 TTA。
6. 固定跑满 15 epoch，使用 `final_checkpoint.pt`；held-out fold 不参与
   逐 epoch 验证、early stop 或 checkpoint 选择。

native 是教师开箱表征主行；PCA384 是维度/头容量控制。不能在两表中事后
挑每个教师更高的一行拼成排名。

## 7. P04 决策规则

1. DINOv2-B 与 ConvNeXt 的正式差异决定是否值得保留大判别教师；
2. CleanDIFT 只有在三折、pooled OOF、尾类或稳定困难对象上提供 DINO
   之外的一致价值时才保留；
3. CleanDIFT 若仍明显低于 DINO，扩散教师路线停止，不因“使用了扩散”
   而保留；
4. native 收益若在 PCA384 后完全消失，必须标注其主要来自维度/头容量；
5. 三折结论必须同时保留逐折值和 pooled OOF，不能只报最好 fold。

本轮不做融合、蒸馏、DINO fine-tune、CleanDIFT LoRA 或新层搜索。只有本轮
正式证据通过相应准入门禁，才另立下一任务。

## 8. 实现入口

- P03 配置：`configs/experiments/p03_formal_cv3_v2.yaml`
- P04 配置：`configs/experiments/p04_formal_cv3_v2.yaml`
- formal 输入/cache 门禁：`scripts/audit_p03_p04_formal_inputs.py`
- P03 配置冻结：`scripts/freeze_p03_formal_config.py`
- P03 训练：`scripts/train_crop_classifier.py`
- P04 probe：`scripts/train_p04_feature_probe.py`
- 正式汇总：`scripts/summarize_p03_p04_formal.py`
- 专项测试：`tests/test_p03_p04_formal_replay.py`
- P03 服务器任务单：`docs/server/P03_FORMAL_CV3_V2_REPLAY.md`
- P04 服务器任务单：`docs/server/P04_FORMAL_CV3_V2_REPLAY.md`

## 9. 停止条件

- formal crop SHA、CV3 SHA 或对象集合不一致：两项均停止；
- P04 任一 cache 出现一个 UID/crop/canonical mismatch：停止整个 18-run
  配对矩阵，不运行不完整教师子集；
- 任一条件三折不完整：不得形成正式均值或排名；
- PCA/RMS/head 读取验证 fold：整组结果作废；
- 运行中为追分修改 sampler、seed、层位、学习率、固定 epoch 或 checkpoint
  selection：不得纳入正式表。
