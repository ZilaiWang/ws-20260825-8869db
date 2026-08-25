# D 当前阶段任务合同：RT-DETR-L 与统一错误分析

> 状态说明（2026-07-24）：本文保留最初分工与探索合同。正式 CV3 v2 的
> epoch、验证、checkpoint、OOF 和 OOM 规则已由
> [`CV3_OOF_ADDENDUM.md`](CV3_OOF_ADDENDUM.md) 及
> [`M3_CV3_OOF_TASK.md`](../../../docs/server/M3_CV3_OOF_TASK.md) 取代。
> 若本文的 `patience`、`best checkpoint`、batch 降级等内容与正式补充冲突，
> 一律以后两者为准。

版本：v1  
日期：2026-07-23  
预计投入：立即阶段约 1.5—2.5 人日，另需约 0.5—1.5 GPU 日  
状态：`ready_formal_cv3`

> 2026-07-23 修订：正式 CV3 v2 已冻结。原先的 `dev_v2` 单次探索不再是
> 必经步骤；若尚未启动，可直接按本文冻结模型配置训练 CV3 三折。旧段落
> 中关于“CV3 到达后”的条件现已满足。

## 1. 本任务只回答两个问题

1. 与 YOLO26-s 结构不同的 DETR 检测器，是否提供更好的候选质量、定位或互补错误？
2. 当前系统的主要错误到底来自定位、细类、背景、重复、漏检中的哪几类？

不要求 D 做第二个备选模型、提出新网络、完成大图工程或调完全部超参数。

## 2. 模型选择

固定使用：

```text
RT-DETR-L
COCO pretrained: rtdetr-l.pt
input: 1024
seed: 42
foundation: natural distribution
epochs: 120
patience: 30
optimizer: AdamW
initial lr: 2e-4
weight decay: 1e-4
AMP: on
```

起始配置：

`xh-202625-model/configs/models/m3_rtdetr_l_1024.yaml`

起始 adapter：

`xh-202625-model/src/rsdet/models/ultralytics_adapter.py`

选择原因：

- 已有配置、训练入口和 adapter，启动成本最低；
- 与 YOLO26-s 的稠密 one-stage 路线结构不同；
- 300 queries 足够覆盖当前数据：`dev_v2` 单图最多 42 个 GT；
- 官方实现支持训练、验证、预测和导出，便于后续同协议接入；
- 当前最重要的是得到异构证据，不是追逐更大的 COCO 榜单模型。

参考：

- [Ultralytics RT-DETR 官方文档](https://docs.ultralytics.com/models/rtdetr/)
- [D-FINE 官方仓库](https://github.com/Peterande/D-FINE)

D-FINE-M 只列为后续条件候选：它参数更小且强调分布式框回归，但引入新框架会增加数据、adapter、导出和复现成本。RT-DETR-L 未完成前不切换。

## 3. 第一轮明确禁止

- rare-rebalance；
- HPR；
- DINOv2/ConvNeXt crop 分类；
- D-FINE/DEIM 第二套模型；
- 外部数据；
- 多模型集成；
- 25 类独立阈值；
- 同时修改输入分辨率、模型规模和增强；
- 10K 切片/融合调参。

C 的再平衡和 HPR 尚未显示稳定增益，D 不应复制同一路线。

## 4. 数据边界

### 模型训练

正式训练使用：

```text
split_version: cv3_airport_proxy_k60_v2
manifest_sha256: 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
held_out_fold: 0 / 1 / 2
status: formal
```

### 与 C 的比较

C 当前模型在 `dev_v1` 训练，不能直接拿其 `dev_v1` 数值与 D 的 `dev_v2` 数值比较。两个划分有 1,018 张 MAR20 图改变归属。

允许的使用方式：

- 用 C 的 `dev_v1` GT/预测开发错误分析工具；
- M1 和 M3 现在直接在相同 CV3 folds 上重训并生成 OOF，再做正式比较；
- 已经启动的 `dev_v2` 结果只能作为单模型探索证据，不能替代 OOF。

禁止把旧 C checkpoint 直接在 `dev_v2` val 上评估后称为公平对照，因为其中可能包含该 checkpoint 训练时见过的图。

## 5. 执行顺序

### D0：错误分析最小闭环，约 0.5 人日

输入：

```text
xh-202625-model/outputs/dev_v1_gt.json
xh-202625-model/outputs/hpr_search/base_low_predictions.json
```

建立模型无关命令：

```text
GT COCO JSON + prediction JSON + project config
→ match_table.csv
→ error_summary.json
→ per_class_metrics.csv
→ confusion_matrix.csv
→ case_index.csv
→ contact sheets
```

不得读取 YOLO 内部 tensor；工具只接受统一 JSON，因此未来可以直接分析 M3 和 OOF。

### D1：环境和数据门禁，约 0.25 人日

- 冻结 Python、PyTorch、CUDA、Ultralytics 版本；
- 校验 manifest SHA、4,481 图覆盖和 25 类；
- 生成 RT-DETR 数据 YAML；
- 32 张 train / 32 张 val smoke；
- 验证 loss 有限、反向传播正常、输出 category ID 为 0—24；
- 验证低阈值输出不超过 300 queries，且无越界/NaN。

### D2：foundation 训练，约 0.5—1.5 GPU 日

- 只跑 natural distribution；
- 使用 seed 42；
- 保存 best/last、完整日志、实际 batch、吞吐和峰值显存；
- 早停由冻结配置决定，不根据中途结果手工改变增强或学习率；
- 若 OOM，只允许降低 batch，并记录变化。

### D3：低阈值预测和统一评估，约 0.5 人日

- 推理候选阈值建议从 `0.001` 起，保留原始分数；
- `max_det` 不高于/不超过模型 queries，但不得人为压到低于验证集所需；
- 导出统一预测 JSON；
- 先做字段、坐标、图像 ID 和类别校验；
- mAP 只作辅助；
- A 统一选择正式工作点，D 交全量低阈值预测和曲线数据。

### D4：正式 CV3 三折（当前已放行）

- 不重新选模型和重开网格；
- 只按同一配置训练 3 folds；
- 每折只在本折 val 产生 OOF；
- 三折 OOF 合并后做一次统一错误分析；
- 阈值采用 cross-fit 或由 A 统一处理。

## 6. 错误分类合同

在官方细类匹配完成后，对未匹配预测按以下优先级赋唯一标签：

1. `FP_DUP`：与已经匹配的同细类 GT 达到该类 IoU 阈值；
2. `FP_FINE_CLS`：与某 GT 达到 IoU 阈值，但预测细类错误；
3. `FP_LOC`：与同细类 GT 有明显重叠但未达到 IoU 阈值；
4. `FP_BG`：与任意 GT 的最大 IoU < 0.10；
5. `FP_OTHER`：其余重叠/冲突。

对未匹配 GT 按以下优先级赋唯一标签：

1. `FN_FINE_CLS`：存在达到 IoU 阈值但细类错误的候选；
2. `FN_LOC`：存在正确细类候选，但 IoU 位于 0.10 与正式阈值之间；
3. `FN_MISSING`：没有最大 IoU ≥ 0.10 的候选。

同时输出：

- `R_loc@oracle-class`：忽略预测细类后的几何候选召回；
- `Acc_fine@localized`：已几何匹配对象中的细类正确率；
- `FP_bg / FP_dup / FP_cls`；
- 官方阈值前后的框 IoU 分布；
- 目标尺寸段、边界对象、三大成像域、头/中/尾类统计；
- 每种错误的 UID、image ID、GT/pred 框、score 和可视化路径。

错误类型必须互斥、计数可回加到总体 TP/FP/FN，并有单元测试。

## 7. 与 C 的配对分析

只有在相同 split/OOF 上才做模型优劣结论。至少输出：

- 两者共同 TP；
- 仅 YOLO TP；
- 仅 RT-DETR TP；
- 两者共同 FN；
- 对同一 GT 的 IoU 差值；
- 细类分歧；
- 背景 FP 的交集与并集；
- 简单 oracle-union 上限；
- 每大类、尺寸段、边界状态的 paired difference。

解释规则：

| 结果 | 后续角色 |
|---|---|
| RT-DETR 官方工作点更优且时延可接受 | 保留为 M3 候选 |
| 定位召回明显更高、细类较弱 | proposal/定位教师或困难对象候选 |
| 单模不更好，但仅 RT-DETR TP 较多 | 研究门控集成，不直接全量双模型 |
| 与 YOLO 高度同错且更慢 | 不进入正式推理，保留诊断结论后停止 |

## 8. 必交产物

```text
outputs/E-M3-rtdetr-l-1024-cv3-seed42/
├── config.yaml
├── environment.txt
├── meta.json
├── train.log
├── best_checkpoint.sha256
├── predictions_low.json
├── predictions_low.runtime.json
├── metrics.json
├── threshold_curve.csv
└── error_analysis/
    ├── match_table.csv
    ├── error_summary.json
    ├── per_class_metrics.csv
    ├── confusion_matrix.csv
    ├── case_index.csv
    └── contact_sheets/
```

另提交：

- 代码分支/PR；
- 复现命令；
- GPU、峰值显存、训练时长和小图模型吞吐；
- 已知失败、OOM 或重试记录；
- 权重只给路径和 SHA，不提交 Git。

## 9. 验收条件

- 输入、代码、环境和权重均可追溯；
- 低阈值预测覆盖完整 val；
- 统一预测校验通过；
- 错误分类计数守恒；
- 工具能不改代码地分析 C 与 D 的预测；
- 不把 `dev_v2` 单次结果包装成正式 CV3 结论；
- 不因 RT-DETR 的 Ultralytics mAP 较高就跳过官方 Recall/FDR；
- 正式模型选择必须等待相同 CV3 OOF 和 10K 时延证据。
