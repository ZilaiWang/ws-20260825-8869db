# 成员交付入口

更新日期：2026-07-23

## B：数据与 CV3

- 状态：`complete`。单次开发划分和正式三折均已完成，B 的新三折结果经
  TU-160 等硬约束修订后冻结为 v2。
- 开发划分：[`dev_v2_airport_proxy_k60.json`](../../../data/splits/dev_v2_airport_proxy_k60.json)
- 正式三折：[`cv3_airport_proxy_k60_v2.json`](../../../data/splits/cv3_airport_proxy_k60_v2.json)
- 冻结组分配：[`cv3_airport_proxy_k60_v2_groups.json`](../../../data/splits/cv3_airport_proxy_k60_v2_groups.json)
- 机器审计：[`cv3_airport_proxy_k60_v2_audit.json`](../../../reports/data/cv3_airport_proxy_k60_v2_audit.json)
- 总索引和使用边界：[`DATA_SPLITS_MASTER_INDEX_v1.md`](../../../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)
- 后续只需协助各训练入口正确读取 `fold`；不再改变任何图像归属。

## C：M1/M2

- 解压交付目录位于主仓库同级的 `xh-202625-model`，尚未并入主仓库。
- 当前验收见：[`reports/members/C/DELIVERY_AUDIT.md`](../../../reports/members/C/DELIVERY_AUDIT.md)
- 正式三折补充合同：
  [`reports/members/C/CV3_OOF_ADDENDUM.md`](../../../reports/members/C/CV3_OOF_ADDENDUM.md)
- 现有主力为 YOLO26-s/1024 foundation；rare-rebalance 未完整完成，HPR 没有实质净收益。

## D：M3 与错误分析

- 当前未发现正式交付。
- 直接执行：[`reports/members/D/TASK_CONTRACT.md`](../../../reports/members/D/TASK_CONTRACT.md)
- 正式三折补充合同：
  [`reports/members/D/CV3_OOF_ADDENDUM.md`](../../../reports/members/D/CV3_OOF_ADDENDUM.md)
- 第一模型固定 RT-DETR-L/1024；不再花时间重新选型。

## E：10K 工程

- 负责切片、批推理、坐标恢复、跨 tile 聚合、序列化和 p50/p95。
- 可执行任务单：[`E_10K_PIPELINE_TASK.md`](../../server/E_10K_PIPELINE_TASK.md)
- C 的 M1 adapter 已存在，可在不等待 CV3 的情况下先完成工程接入与测速准备。
- 最终时延必须区分读盘、model-only 和完整 pipeline。当前 4080 SUPER 只作
  工程证据；官方声明必须在代码注册的官方 10K 输入和独占 RTX 3090 上复核。

## 验收共同要求

- 使用统一 `Prediction`/COCO JSON；
- 记录数据版本、split 版本、seed、配置、Git commit、环境和权重 SHA；
- 保留低阈值原始预测；
- 统一评测器计算官方 Recall/FDR；
- 失败实验和被停止路线也登记，不只提交最好数字。
