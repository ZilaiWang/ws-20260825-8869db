# A 主线 1：V1-FULL-P2 三折验证报告

日期：2026-08-10
实验：V1-FULL-P2 = yolo26-p2s（Detect 四输入 P2/P3/P4/P5，stride 4/8/16/32）
对照：M1 = yolo26s（Detect 三输入 P3/P4/P5，stride 8/16/32）
GPU：autodl RTX 3090（`connect.nmb2`，新实例）
状态：`complete` —— **假设未通过准入（1 胜 1 负 1 平）**

## 1. 实验设计（严格单因素）

| 项 | M1 基线 | V1-FULL-P2 |
|---|---|---|
| 模型 | yolo26s.pt | yolo26-p2s.yaml（官方 P2 变体） |
| 初始化 | COCO 预训练 | yolo26s 迁移公共层 + P2 头随机初始化 |
| 数据 | CV3 三折（fold0/1/2 各为验证折） | **完全相同** |
| 超参 | imgsz 1024 / batch 12 / 60 epoch / AdamW | **完全相同** |
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

## 3. 结论

**准入判定：未通过（< 2/3 fold 方向一致），P2 不作为候选恢复主线保留。**

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

1. **P2 主线保留冻结**，不进入正式基线评估（pooled + 官方 macro 双口径）；
2. 若后续出现"低分候选消费者"（如阈值扫描、对象头），可重新评估；
3. fold0 的 +5.3pp 结论修订为"单折现象"，不代表总体方向；
4. A 主线 1 下一候选方案按总纲继续（可考虑 stride-2 高分辨率或 P2 与
   max_det 联合审计），另开实验。

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
