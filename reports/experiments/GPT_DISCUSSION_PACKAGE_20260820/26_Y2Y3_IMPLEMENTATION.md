# Y2/Y3 正式实验实现报告

日期：2026-08-11

状态：代码与准入合同已完成；Y2 已在冻结的 fold0 40-epoch 配对快筛中得到
`stop_candidate`，因此不进入正式三折；Y3 因 Y2 未准入而停止。结果见
[`Y2_FAST_SCREEN_RESULT_20260811.md`](Y2_FAST_SCREEN_RESULT_20260811.md)。

## 1. 为什么重做正式 P2

历史 P2 结果只能证明 stride-4 通路能减少车辆“无候选”，不能证明官方指标改善，原因现在扩展为四项：

1. 60 epoch，而 M1 为 160 epoch；
2. 使用 best checkpoint，而正式合同为 fixed-last；
3. 历史评估是逐 GT 候选诊断，不是官方一对一匹配；
4. 历史模型名 `yolo26-p2s.yaml` 会在 Ultralytics 8.4.103 中回退为 **n 级**，不是与 M1 容量对齐的 s 级。

实际构图已核验：

| 构图名 | 参数 | Detect stride |
|---|---:|---|
| 错误 `yolo26-p2s.yaml` | 2,662,400 | 4/8/16/32 |
| 正式 `yolo26s-p2.yaml` | 9,765,856 | 4/8/16/32 |

Y2 因此冻结为 `yolo26s-p2.yaml`。
实际用冻结 `yolo26s.pt` 构图迁移时，Ultralytics 报告
`Transferred 360/902 items`；独立快照复核发现 6,055,360 / 9,765,856
个参数值发生迁移（约 62.0%），包含 120 个 backbone 参数张量。
未匹配的 P2/颈部部分从随机初始化开始，这是构图变体本身的
必要训练条件，且已在每折 `architecture_audit.json` 中完整记录。

## 2. Y2 的唯一变量

| 项目 | M1 | Y2 |
|---|---|---|
| 初始化 | 官方 `yolo26s.pt` | 同一 SHA 的 `yolo26s.pt` |
| 尺度 | s | s |
| 检测层 | P3/P4/P5 | P2/P3/P4/P5 |
| CV | airport-proxy-k60-v2 CV3 | 完全相同 |
| 输入 | 1024 | 1024 |
| 训练 | 160 epoch / seed42 / AdamW | 完全相同 |
| checkpoint | last | last |
| OOF | conf 0.001 / IoU 0.70 / maxdet 500 | 完全相同 |
| 评估 | 官方 pooled + V1.6 macro | 完全相同 |

每折必须从原始预训练权重独立开始，不允许 resume 或跨折权重复用。

## 3. Y2 产物链

```text
formal CV3 + detection data lock
        -> P2 run plan
        -> fold0/1/2 resolved config
        -> architecture audit
        -> 160-epoch last.pt
        -> 0.001 held-out predictions
        -> fold metadata
        -> 4481-image aggregate
        -> C0-C3 same-protocol cross-fit
        -> P2-vs-M1 C0 decision
```

结构因果判断只比较 Y2 C0 与 M1 C0。C2 类别先验校准可在 P2 结构准入后作为运行工作点，但不参与“P2 是否有效”的判定。

## 4. Y2 预注册准入条件

- aggregate 是正式完整三折、4481 图、0.001 候选下限；
- 官方 Recall/FDR 硬门槛通过；
- pooled Recall 相对 M1 下降不超过 0.005；
- pooled FDR 相对 M1 增加不超过 0.01；
- macro Recall 下降不超过 0.005；
- vehicle pooled Recall 至少 +0.02；
- vehicle Recall 至少 2/3 折为正向。

这组条件防止“候选变多”被错当成“正式检测变好”。

## 5. Y3 的设计范围

Y3 不复制整个 FRFDet，只测试一个与历史错误对应的 P2 颈部对称采样对：

- layer17 P3→P2：nearest upsample → IBS-U；
- layer20 P2→P3：stride convolution → IBS-D；
- expansion ratio=2，sampling factor=2；
- downsampling 保留 expansion-depthwise residual，upsampling 不使用该 residual；
- 其余 28 层路由和 Detect 四输入不变。

真实 s 级构图已通过前向检查：

- IBS 前：9,765,856 参数；
- IBS 后：9,785,344 参数；
- 增加 19,488 参数（约 0.20%）；
- Detect stride 仍为 4/8/16/32。

Y3 不与 SFRCF、新 loss、新增强或新 backbone 同时运行。

## 6. Y3 门控与准入

Y3 训练入口必须读取 `y2_p2_formal_decision_v1`，且两个字段均为 true：

- `p2_structure_admission`；
- `quality_stage_admission`。

缺文件、版本错误或 Y2 未准入，Y3 均在训练前拒绝启动。

Y3 相对正式 Y2 C0 的预注册准入：

- pooled Recall 下降不超过 0.003；
- pooled FDR 增加不超过 0.005；
- macro Recall 下降不超过 0.003；
- vehicle Recall 下降不超过 0.005；
- vehicle FDR 至少 -0.02，或 macro FDR 至少 -0.01；
- vehicle FDR 至少 2/3 折改善。

## 7. 代码与文档索引

| 内容 | 路径 |
|---|---|
| P2/Y3 正式合同 | `src/rsdet/experiments/cv3_oof.py` |
| P2/Y3 GPU runtime | `scripts/y2_p2_runtime.py` |
| Y2 科学决策 | `scripts/y2_decide_p2.py` |
| IBS 实现 | `src/rsdet/models/ibs_sampling.py` |
| Y3 科学决策 | `scripts/y3_decide_ibs.py` |
| Y2 train template | `configs/experiments/y2_yolo26s_p2_1024_cv3_oof.template.yaml` |
| Y2 infer template | `configs/experiments/y2_yolo26s_p2_1024_cv3_oof_infer.template.yaml` |
| Y3 train template | `configs/experiments/y3_yolo26_p2_ibs_1024_cv3_oof.template.yaml` |
| Y3 infer template | `configs/experiments/y3_yolo26_p2_ibs_1024_cv3_oof_infer.template.yaml` |
| 服务器驱动 | `scripts/server/run_y2_y3_formal_cv3.sh` |
| 服务器执行单 | `docs/server/Y1_Y2_Y3_FORMAL_EXECUTION_20260811.md` |
| FRFDet 改编声明 | `NOTICE_FRFDET.md` |
| 单测 | `tests/test_cv3_oof.py`, `tests/test_y2_p2_formal.py`, `tests/test_ibs_sampling.py`, `tests/test_y3_ibs_decision.py` |
