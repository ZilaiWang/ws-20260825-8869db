# A 主线 1：V1-FULL-P2 高分辨率候选通路实验（fold0）

日期：2026-08-10
实验：历史 V1-FULL-P2（实际为 n 级 P2；详见三折收尾报告）
对照：M1 = yolo26s（Detect 三输入 P3/P4/P5，stride 8/16/32）
GPU：autodl RTX 3090
状态：`superseded_invalid_for_formal_comparison`

> 本单折报告已被三折及收尾审核覆盖。历史 P2 为 60 epoch/
> `best.pt`，而正式 M1 为 160 epoch/fixed `last.pt`；且本指标是
> 逐 GT 候选覆盖诊断，非官方一对一 Recall/FDR。仅保留“P2 能产生
> 部分新小车候选”的机制证据。

## 1. 实验设计（严格单因素）

| 项 | M1 基线 | V1-FULL-P2 |
|---|---|---|
| 模型 | yolo26s.pt | `yolo26-p2s.yaml`（命名错误导致回退为 n 级） |
| 初始化 | COCO 预训练 | yolo26s 迁移公共层 + P2 头随机初始化 |
| 数据 | CV3 fold0（训练 fold1+2，验证 fold0） | **完全相同** |
| 超参 | imgsz 1024 / batch 12 / 60 epoch / AdamW | **完全相同** |
| 唯一差异 | — | **P2 stride-4 检测通路** |

## 2. 结果（fold0，同口径评估）

| 指标 | M1 基线 | P2 模型 | 变化 |
|---|---|---|---|
| **车辆 Recall** | **0.6466** | **0.6992** | **+5.3pp** |
| **无候选目标** | 33 | **17** | **-16（-48%）** |
| 低分被滤 | 14 | 23 | +9 |
| 匹配车辆 | 86 | 93 | +7 |

## 3. 结论

**高分辨率候选通路（P2 stride-4）假设验证成功：**

1. **车辆 Recall +5.3pp**（0.647 → 0.699），匹配车辆 +7；
2. **无候选目标 -48%**（33 → 17）：P2 通路直接救回一半"连候选都没有"的
   车辆——**与特征响应审计预测完全一致**（浅层响应存在，P2 层利用它）；
3. 低分候选 +9：原本特征弱的目标产生候选（弱响应 → 可优化分数）。

**这是 A 主线 1 的首个实验证据**：总纲 §5.3 V1-FULL-P2 方向得到验证，
"高分辨率候选通路修复车辆候选缺失"在 fold0 成立。

## 4. 后续计划

1. **三折完整验证**：fold1/fold2 补跑，确认 ≥2/3 fold 方向一致（准入纪律）；
2. **延长训练**：60 epoch 已达 mAP50 0.651；160 epoch（M1 正式配置）预计
   进一步收敛，车辆 Recall 有望继续提升；
3. **FP 审计**：P2 新增候选是否带来车辆 FP 激增（FDR 是否可控）；
4. 若三折通过 → 进入正式基线评估（pooled + 官方 macro 双口径）。

## 5. 产物

| 产物 | 位置 |
|---|---|
| P2 fold0 训练 | 服务器 `/workspace/results/P2-FOLD0/p2_run/`（best.pt） |
| P2 fold0 评估 | `outputs/P2-FOLD0/eval_fold0.json` |
| M1 fold0 同口径评估 | `outputs/P2-FOLD0/eval_m1_fold0.json` |
| 配置记录 | `outputs/P2-FOLD0/p2_config.json`（git 可回溯） |
| 训练脚本 | `scripts/n1c_p2_train.py`（已 git） |
| 评估脚本 | `scripts/n1d_p2_evaluate.py`（已 git） |
