# 当前团队任务

更新日期：2026-07-25  
状态：`current`

当前职责以根目录外的 [`doc/XH-202625_20260715.md`](../../../../doc/XH-202625_20260715.md) 为准。更早的项目计划和《分工调整》只作为历史讨论，不再用于派工。

下一阶段 A—E 共同参与创新的待启用方案见
[`NEXT_STAGE_TEAM_INNOVATION_EXECUTION_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_TEAM_INNOVATION_EXECUTION_MASTER_v1.md)。
在 A 的 N0、D 的 M3 和 E 的 10K 基线收尾前，该文件是规划草案，不覆盖下表当前任务。

| 成员 | 当前主责 | 现状 | 下一明确交付 |
|---|---|---|---|
| A | 协议、验收、创新实验和最终决策 | M1 正式 OOF 已验收并完成第一轮官方错误分解 | 先完成 N0 cross-fit、对象证据 manifest 和 FP_BG 审计；随后组织 P04-F→P03-F，并统一实现细分类/背景拒识对象学生 |
| B | 数据划分、CV3、数据审计 | 两份划分已完成；正式 CV3 v2 已冻结并通过 25 类覆盖及泄漏审计 | 归档划分；协助核对 source-group bootstrap、TU-160 压力折及 FP_BG 分层人工抽检，不再改动 fold 归属 |
| C | M1/M2 主检测器 | 正确 YOLO26-s 正式三折 OOF 已完成；总体有过线区间，但舰船/车辆仍弱 | 冻结并归档 M1 lineage；协助 E 接入 checkpoint；只有车辆候选证据明确时才开单因素小目标实验 |
| D | M3 和模型无关错误分析 | M3 代码与正式输入就绪，尚无完整三折 OOF | 直接跑 RT-DETR-L/1024 三折低阈值 OOF，优先回答 M1 漏失目标、尤其车辆候选是否可被补回 |
| E | 10K 切片、融合、测速 | 已可接入正式 M1，不再受 M1 aggregate 阻塞 | 先完成 M1 的切片、全局坐标恢复、跨 tile 唯一化和分段计时；最终模型冻结后再做 3090 正式复测 |

## D 的任务不再重新选型

D 的完整任务合同见：

[`reports/members/D/TASK_CONTRACT.md`](../../../reports/members/D/TASK_CONTRACT.md)

正式 CV3 执行还必须同时读取
[`reports/members/D/CV3_OOF_ADDENDUM.md`](../../../reports/members/D/CV3_OOF_ADDENDUM.md)
和 [`M3_CV3_OOF_TASK.md`](../../server/M3_CV3_OOF_TASK.md)；二者覆盖旧合同中
关于 early stop、best checkpoint 和 OOM 降 batch 的历史条款。

M1 已完成，不再重新训练或调参。当前新的统一执行入口是
[`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)；
其中 N0/N1/N2 属 A 主责，M3 属 D，10K 属 E。

负责人可以直接转发的简版：

> 不再寻找第二个模型，也不做 HPR、DINOv2 crop 分类或大图工程。直接复用现有
> `RT-DETR-L/1024` 配置和 Ultralytics adapter，在冻结的
> `cv3_airport_proxy_k60_v2` 上独立训练三折并交付低阈值 OOF；同时用 C
> 的低阈值 YOLO OOF 建立模型无关错误分析工具。不重新开启大规模调参。

## 边界

- D 不重复训练 YOLO；YOLO26-s/M2 归 C。
- D 不负责 ConvNeXt/DINOv2 crop 学生/教师；这属于 A 的 P03/P04。
- D 不负责切片、跨瓦片融合或 10K 测速；这属于 E。
- A 统一做阈值选择；D 必须交低阈值原始预测，不在同一验证集上拟合 25 个类阈值后再报成绩。
