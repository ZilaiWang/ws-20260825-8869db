# Y2 P2 配对快筛结果

日期：2026-08-11  
状态：`complete_screening_only`  
决策：`stop_candidate`

## 1. 实验问题

Y2 只改变 YOLO26-s 的检测层：控制 M1S 使用 P3/P4/P5，候选 Y2S 使用
P2/P3/P4/P5。两者使用同一 fold0、seed42、1024 输入、40 epoch、AdamW、增强和
`yolo26s.pt` 初始化，随后分别执行 conf=0.001 的 held-out 推理和官方一对一匹配。

本实验只回答“完整 P2 是否值得进入第二折或正式三折”，不形成正式模型准入结论。

## 2. 完整性验收

| 项目 | M1S | Y2S |
|---|---:|---:|
| 完成 epoch | 40 | 40 |
| held-out 图像 | 1,507 | 1,507 |
| 低阈值候选数 | 41,401 | 56,346 |
| 过滤退化框 | 13 | 2 |
| `last.pt` SHA256 | `e886d9c8…135e44` | `f702e068…a917b` |

共同输入门禁通过：

- 初始化权重 SHA256：`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`；
- D00 数据锁 SHA256：`03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`；
- CV3 manifest SHA256：`27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331`；
- 预测 SHA、架构审计、官方匹配 parity 均通过。

推理首次因 conf=0.001 下少量越界候选裁剪后面积为零而中止。修复仅过滤并计数这种
退化框，未修改阈值、模型、数据或训练参数；M1S 没有重训或断点续训。修复提交为
`338cbc8`。

## 3. 配对结果

各模型在同一 held-out fold 上独立选择探索工作点：

| 指标 | M1S | Y2S | Y2S − M1S |
|---|---:|---:|---:|
| 工作点阈值 | 0.151 | 0.226 | — |
| overall Recall | 0.8808 | 0.8224 | **−0.0584** |
| overall FDR | 0.1985 | 0.1994 | +0.0010 |
| macro Recall | 0.8089 | 0.7070 | **−0.1019** |
| macro FDR | 0.3639 | 0.3357 | −0.0282 |
| vehicle Recall | 0.7293 | 0.5714 | **−0.1579** |
| vehicle FDR | 0.5958 | 0.4967 | −0.0991 |
| conf=0.001 vehicle Recall | 0.9624 | 0.9624 | 0 |
| 无候选 vehicle GT | 5/133 | 5/133 | 0 |

Y2S 的 FDR 改善发生在大幅牺牲 Recall 后，不能视为净收益。自动门禁中只有
overall FDR safety 通过；overall Recall safety、macro Recall safety 和 vehicle signal
全部失败。

同阈值比较也排除了“仅阈值选择不利”的解释：

| 固定阈值 | 模型 | overall Recall | overall FDR | vehicle Recall |
|---:|---|---:|---:|---:|
| 0.101 | M1S | 0.8955 | 0.2406 | 0.7820 |
| 0.101 | Y2S | 0.8669 | 0.2916 | 0.6541 |
| 0.151 | M1S | 0.8808 | 0.1985 | 0.7293 |
| 0.151 | Y2S | 0.8460 | 0.2458 | 0.6241 |
| 0.201 | M1S | 0.8697 | 0.1718 | 0.6767 |
| 0.201 | Y2S | 0.8295 | 0.2118 | 0.5789 |

## 4. 解释与决策

P2 将低阈值候选增加 36.1%，但没有减少任何完全无候选车辆，也没有提高车辆候选下限
Recall；进入可用置信度区间后，车辆和总体 Recall 反而明显下降。这说明当前完整 P2
主要增加了冗余低质量候选，并削弱了候选排序或有效特征竞争，不是当前车辆漏检的
有效解决方向。

因此冻结以下决策：

1. 不运行 fold1 快筛；
2. 不运行 Y2 三折 × 160 epoch；
3. 不启动以 Y2 准入为前置的 Y3 IBS；
4. 不启动 P2-Lite，因为完整 P2 没有可保留的 unique-vehicle 净增益；
5. 后续 YOLO 创新继续复用 M1S fold0 控制，优先测试不增加整层候选数量、直接针对
   排序质量或已定位车辆判别的单变量方案。

40 epoch 单折不能证明“任何更长训练的 P2 都一定无效”，但当前退化幅度远超快筛安全
边界，且目标机制信号为零；继续投入完整正式预算不具性价比。

## 5. 产物索引

- 本地目录：`outputs/Y2-FAST-SCREEN-FOLD0/`；
- 回传包：`outputs/Y2-FAST-SCREEN-FOLD0-return-no-checkpoints.tar.gz`；
- 回传包 SHA256：`5d6cba30829b873ea8fded1f5fdeca970fe2886affac27b4f310f900ba954842`；
- 决策原件：`outputs/Y2-FAST-SCREEN-FOLD0/screening_result.json`；
- 快筛合同：`docs/server/YOLO_FAST_SCREEN_PIPELINE_20260811.md`。
