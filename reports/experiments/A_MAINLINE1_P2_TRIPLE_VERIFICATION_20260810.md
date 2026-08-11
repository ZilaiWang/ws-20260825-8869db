# A 主线 1：V1-FULL-P2 三折验证报告

日期：2026-08-10
实验：历史 V1-FULL-P2（Detect 四输入 P2/P3/P4/P5，stride 4/8/16/32）
对照：M1 = yolo26s（Detect 三输入 P3/P4/P5，stride 8/16/32）
GPU：autodl RTX 3090（`connect.nmb2`，新实例）
状态：`invalid_for_formal_comparison` —— **机制证据保留，正式效果结论撤销**

> 2026-08-11 追加审核：历史配置使用 `yolo26-p2s.yaml`，
> Ultralytics 8.4.103 会警告未给定 scale 并回退到 n 级（2,662,400 参数）；
> 正确 s 级名称是 `yolo26s-p2.yaml`（9,765,856 参数）。因此历史
> P2 除了 60 epoch/best 与非官方评估外，还存在模型容量不对齐。
>
> 2026-08-10 收尾审核：历史 P2 实验实际训练 60 epoch 并使用
> `best.pt`；正式 M1 为 160 epoch 固定 `last.pt`。`n1d_p2_evaluate.py`
> 又是逐 GT 最佳候选的机制诊断，不是官方一对一 Recall/FDR。
> 因此“严格单因素”和“1 胜 1 负 1 平”均不能作正式准入依据。

## 1. 历史实验设计（非正式单因素）

| 项 | M1 基线 | V1-FULL-P2 |
|---|---|---|
| 模型 | yolo26s.pt | `yolo26-p2s.yaml`（实际回退为 n 级 P2） |
| 初始化 | COCO 预训练 | yolo26s 迁移公共层 + P2 头随机初始化 |
| 数据 | CV3 三折（fold0/1/2 各为验证折） | **完全相同** |
| 超参 | 正式 M1: 160 epoch / fixed last | 历史 P2: 60 epoch / best |
| 唯一差异 | — | **P2 stride-4 检测通路** |

每折均用 `n1d_p2_evaluate.py` 同口径评估：M1 对应折 `best.pt` vs P2 对应折 `best.pt`。

## 2. 结果（三折同口径车辆 Recall）

| fold | 车辆 GT | M1 基线 Recall | P2 Recall | Δ | M1 无候选→P2 | M1 低分→P2 |
|---|---:|---:|---:|---:|---|---|
| 0 | 133 | 0.6466 (86) | **0.6992** (93) | **+5.3pp** ✅ | 33→17 | 14→23 |
| 1 | 134 | 0.5373 (72) | 0.4851 (65) | **−5.2pp** ❌ | 41→32 | 21→37 |
| 2 | 135 | 0.5111 (69) | 0.5185 (70) | **+0.7pp** ⚠️ | 45→19 | 21→46 |

pooled（三折合计，402 车辆口径近似）：

| 口径 | M1 | P2 | Δ |
|---|---:|---:|---:|
| 合计 matched | 227 | 228 | +1 |
| 合计 Recall | 0.5648 | 0.5672 | +0.2pp |
| 合计无候选 | 119 | 68 | **−51（−43%）** |

## 3. 仅保留的机制结论

**不做准入判定。** 可信部分只是 P2 通路在三折都减少了“无车辆
候选”数量；它是机制诊断，不代表官方 Recall/FDR 净收益。

机制层面**三折一致成立**：

- P2（stride-4）确实系统性地**捞回"无候选"目标**：三折无候选全部下降
  （33→17、41→32、45→19），pooled 无候选 **−43%** —— 与特征响应审计
  （浅层 backbone8 响应存在、P2 层可利用）的预测一致，机制真实有效。

但**净收益仅在 fold0 为正**：

- fold1：捞回 9 个无候选，但低分桶从 21→37（+16），匹配数反而 −7，
  说明 P2 引入的候选分数分布偏低，拖累了本可匹配的目标；
- fold2：无候选 45→19 大幅改善，但 matched 仅 +1，捞回的候选几乎全部
  落入低分桶，未转化为匹配。

**解读**：P2 通路"产生候选"能力可靠，但"产生高质量（可匹配）候选"
能力不足。捞回的候选集中在低置信区，与 M1 阈值/匹配机制无法衔接。
N2 对象重分类路径已证明无净收益，低分候选无下游消费者。

## 4. 决策与后续

1. **历史 P2 结果冻结为机制诊断**，不进入正式 leaderboard；
2. 若后续出现"低分候选消费者"（如阈值扫描、对象头），可重新评估；
3. fold0 的 +5.3pp 结论修订为"单折现象"，不代表总体方向；
4. 如重启，必须与 M1 共用 CV3、160 epoch、fixed `last.pt`、低阈值完整
   OOF，再用 V1.6 pooled + 4/20/1 macro 双口径评估。

## 5. 产物

| 产物 | 位置 |
|---|---|
| P2 三折训练 | 服务器 `/workspace/results/P2-FOLD{0,1,2}/p2_run/`（best.pt，6MB/折） |
| P2 fold0 评估 | `outputs/P2-FOLD0/eval_fold0.json` |
| M1 fold0 同口径评估 | `outputs/P2-FOLD0/eval_m1_fold0.json` |
| P2 fold1 评估 | `outputs/P2-FOLD1/eval_fold1.json` |
| M1 fold1 同口径评估 | `outputs/P2-FOLD1/eval_m1_fold1.json` |
| P2 fold2 评估 | `outputs/P2-FOLD2/eval_fold2.json` |
| M1 fold2 同口径评估 | `outputs/P2-FOLD2/eval_m1_fold2.json` |
| 配置记录 | `outputs/P2-FOLD{0,1,2}/p2_config.json` |
| 训练脚本 | `scripts/n1c_p2_train.py`（已 git） |
| 评估脚本 | `scripts/n1d_p2_evaluate.py`（已 git） |
| fold0 首轮报告 | `reports/experiments/A_MAINLINE1_P2_EXPERIMENT_20260810.md`（单折乐观结论，已被本报告修正） |
