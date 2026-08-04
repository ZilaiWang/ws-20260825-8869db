# 当前项目状态与总体方向

更新日期：2026-08-04  
状态：`current`

## 1. 当前已经掌握的事实

- 官方任务是 25 个细类检测，匹配时细类必须一致；舰船/飞机 IoU 为 0.50，车辆为 0.35。
- 硬门槛为 Overall Recall ≥ 0.85、Overall FDR ≤ 0.20；通过后仍分别比较三大类 Recall/FDR 和 10K 大图时延。
- **评分方案 V1.6（2026-08-04）排名口径**：三大类各自的 Recall/FDR =
  大类内细类指标的简单平均（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1 型
  即 FSC），7 项排名二次排序决定初赛方案/创新/落地三项打分区间。M1 工作点
  官方 macro：舰船 0.7235/0.5201、飞机 0.9076/0.1571、车辆 0.6169/0.6161
  （25 细类平均 Overall 0.8665/0.2335）；**舰船 macro FDR 0.52 是最大官方
  排名风险**，LQS（Recall 0.50）、HM（0.706）、TU-160（0.332）、F-22（0.789）
  是优先靶点。所有实验必须同时报 pooled（门槛校验）与官方 macro（排名优化）
  双口径，`scripts/evaluate.py` 默认输出 `official_ranking` 块。
- 训练集共有 4,481 张图、20,933 个框。数据少、类别长尾、来源域有限，因此“小样本”主要体现为细类与来源覆盖不足，而非固定 K-shot 协议。
- MAR20 的 3,073 张飞机图已得到 60 个机场代理视觉组；当前 `dev_v2` 为 3,548 train / 933 val、跨集合代理组为 0。代理组不等于真实机场真值。
- 正式 `cv3_airport_proxy_k60_v2` 已冻结：4,481 图、255 个不可拆来源组、
  三折验证图 1,507 / 1,613 / 1,361，跨折来源组为 0。正式结论统一使用该
  manifest；旧 `dev_v1` 和单次 `dev_v2` 数值仍只算探索性。
- 正确 YOLO26-s 的正式三折低阈值 OOF 已完成：4,481 张图恰好一次覆盖、
  20,933 个 GT、55,548 个候选。探索工作点 `t=0.051` 的 Overall Recall
  为 `0.9172`、FDR 为 `0.1957`，说明总体存在过线区间，但 fold 0/2 的
  FDR 仍超过 0.20，且距离内部目标 FDR≤0.17 仍有明显差距。
- 工作点错误分解为 `FP_BG=3303`、`FN_CLS=1115`、`FN_MISS=553`、
  `FN_LOC=66`。后续优先级由此改为：飞机细分类、舰船背景拒识、车辆候选
  召回；框修正与 bbox diffusion 暂缓。

## 2. 现阶段模型角色已经明确

| 角色 | 当前选择 | 用途 |
|---|---|---|
| 第一阶段主检测器 | YOLO26-s / 1024 | 快速、低阈值候选生成，承担正式推理主干候选 |
| 异构检测对照 | RT-DETR-L / 1024 | 判断 DETR 候选、定位和错误模式是否与 YOLO 互补 |
| 对象级轻量学生 | ImageNet ConvNeXt-T / tight-224 | 对完整对象重裁后做细类识别、背景拒识和可选框残差 |
| 对象级教师 | DINOv2-B CLS+patch | 训练期教师、困难对象表征和蒸馏候选 |
| 扩散教师/生成 | CleanDIFT、SD1.5 等 | 只保留受控对照；当前不进入主推理链 |

ImageNet、DINOv2 和 YOLO/RT-DETR 不是三选一：

- YOLO/RT-DETR 负责从大图瓦片中找到对象；
- ConvNeXt 负责对象 crop 的轻量正式推理；
- DINOv2 主要作为训练期教师；
- 扩散路线只有在正式 CV3 和真实 OOF 上显示独立收益才可进入。

## 3. 推荐的最终系统主线

```text
10K 原图
  → 重叠切片
  → YOLO26-s 低阈值候选
  → 恢复全局坐标
  → 类别无关的全局对象聚合
  ├─ 容易对象：校准后直接输出
  └─ 困难对象：从原图完整重裁
       → ConvNeXt-T 对象学生
       → 细类 / 背景 / 质量 / 可选 bbox residual
  → 全局唯一结果
```

训练期可使用 DINOv2-B 教师。RT-DETR 若只提高定位或与 YOLO 互补，可作为候选教师或小规模集成证据；不必强行进入最终 20 秒链路。

## 4. 已经改变的技术判断

1. 不再把“扩散”本身当成创新目标；P04 中 CleanDIFT 冻结特征弱于 DINOv2-B，P07 中 SD1.5 背景融合也已达到停止条件。
2. 创新重心转向“全局对象层”：跨瓦片唯一化、完整对象重裁、困难门控、细分类/背景拒识和必要的框修正。
3. C 的 HPR 目前几乎没有净收益，不能作为第二阶段模块定稿；P03 的 ConvNeXt-T 才是更可靠的对象学生起点。
4. M1 正式 OOF 已完成；当前最大的科学缺口变为 cross-fit 阈值、对象级
   Pred-OOF 证据层、`FP_BG` 人工语义审计，以及 P03/P04 在正式 CV3 上的
   复验。
5. P03/P04/P05 不再被当作三套重复系统：工程上收敛为共享的轻量对象学生，
   但保留“只重分类、只拒背景、联合、多教师蒸馏”的科学消融。
6. M3 的价值主要是补充 M1 漏失候选，尤其验证车辆候选召回；E 的 10K 工程
   已可直接接 M1 开始，不应继续等待最终模型才打通流水线。

## 5. 当前主要证据入口

- 总体创新评估：[`doc/下一阶段创新方向评估与总体逻辑.md`](../../../../doc/下一阶段创新方向评估与总体逻辑.md)
- 详细实验路线：[`doc/扩散模型创新路线详细执行报告.md`](../../../../doc/扩散模型创新路线详细执行报告.md)
- 当前分工（第二阶段，含 V1.6 口径对齐）：[`doc/第二阶段分工.md`](../../../../doc/第二阶段分工.md)
- 两份划分总索引：[`DATA_SPLITS_MASTER_INDEX_v1.md`](../../../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)
- 正式三折验收：[`CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md`](../../../reports/data/CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md)
- C 交付审计：[`reports/members/C/DELIVERY_AUDIT.md`](../../../reports/members/C/DELIVERY_AUDIT.md)
- 正式实验执行总纲：
  [`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](../../../reports/experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)
- M1 正式结果与恢复审计：
  [`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`](../../../reports/experiments/M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md)
- M1 之后当前执行总纲：
  [`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)
