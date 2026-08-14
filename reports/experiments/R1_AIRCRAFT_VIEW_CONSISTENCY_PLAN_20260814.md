# R1-5 飞机 D4 双视图一致性实验计划（2026-08-14）

## 1. 为什么是这一项

当前飞机对象头唯一稳定的额外收益来自 full D4：CE identity 的 Overall Recall/FDR 为
`0.92648/0.15054`，CE + D4 为 `0.93011/0.14602`；飞机 macro Recall 从
`0.93444` 升至 `0.94064`。R1-3 又证明单视图置信度无法可靠识别哪些对象需要 D4。

与此同时，类中心、同视图 KD、class-derived 物理属性均未带来独立收益。这说明下一步不应
继续修改类别语义，而应直接减少已观测到的同对象方向敏感性。本实验参考
**Measuring the Impact of Rotation Equivariance on Aerial Object Detection** 中遥感方向等变的
问题定义，但只实现适合 HBB 与现有对象头的最小离散版本，不替换为旋转等变骨干。

## 2. 冻结设计

- 初始化、训练 rows、外层三折、5 epoch、学习率和 fixed-last 均与 R1-1 CE 相同；
- 每个训练对象从精确 D4 集合抽取两个不同视图，均从 canonical crop 直接变换；
- 两个视图都计算 20 类 CE；另加预测分布的 symmetric KL；
- `loss_weight=0.20, temperature=1.0`，只运行一个预注册工作点；
- 部署 checkpoint 不增加参数，identity 仍为一次前向；
- 同时评估 identity 与 full D4，ship/vehicle 严格旁路。

损失为：

`L = 0.5*(CE(z1,y)+CE(z2,y)) + 0.20*SKL(softmax(z1),softmax(z2))`。

这里不用已有 D4 ensemble 作为 teacher，是为了将变量限制为“视图一致性”本身，并避免
八视图 teacher 产生新的算力与软标签温度因素。

## 3. 预注册准入

Identity 相对 CE identity 必须同时满足：

- aircraft macro Recall 至少 `+0.0015`；
- paired `net_tp >= 20`；
- aircraft macro FDR 退化不超过 `0.001`；
- Overall Recall/FDR 继续满足硬门槛；
- ship/vehicle 指标严格相同。

此外，view-consistency + D4 相对 CE + D4：

- aircraft macro Recall 退化不超过 `0.002`；
- aircraft macro FDR 退化不超过 `0.002`。

只有 identity 主门禁和 D4 安全门禁同时通过才准入。若 identity 无增益，即使 D4 有微小波动
也停止；若 identity 上升但 D4 明显退化，则说明模型只是牺牲集成多样性，不准入。

## 4. 预计资源与停止条件

双视图令训练前向约为 CE 的 2 倍，但只跑 3×5 epoch，预计总训练仍在十几分钟级；D4
推理与 R1-4 相同。新快速 evaluator 已与旧实现 JSON 精确等价，避免评估重复耗时。

本实验失败后停止在 ConvNeXt 对象头上继续搜索辅助损失，资源转向：

1. 完成 M3 并量化其对 `FN_MISS=624` 的互补召回；
2. 完成 324 条盲审后再训练 PU/background rejector；
3. 若仍需细分类提升，使用真正独立的 DINOv2/CleanDIFT 教师特征，而不是类标签派生属性。

