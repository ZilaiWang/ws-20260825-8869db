# M1 YOLO26-s 正式 CV3 OOF 结果与关机续跑审计 v2

更新日期：2026-07-25  
任务：`M1-CV3-OOF`  
结论状态：`complete_formal_with_power_interruption_resume_amendment`

## 1. 结论

M1 的正式三折 OOF 已完成并通过数据、资产、覆盖和计数守恒审计。它使用
冻结的 YOLO26-s 权重、正式机场代理分组三折和 D00 数据锁；4,481 张图像
均恰好由未见过其 held-out fold 的模型预测一次，共产生 55,548 个
`score >= 0.001` 的候选框。

fold 2 在完成第 134 个 epoch 后因服务器被外部关机中断。经用户明确批准，
从保存了 optimizer、EMA 和 epoch 状态的 `last.pt` 续跑至 160 epoch。
fold 0/1 的 checkpoint、预测和元数据在恢复前后 SHA256 完全不变。由于
发生过续跑，本次结果的精确科学状态不是“全程不中断重跑”，而是：

```text
formal_with_power_interruption_resume_amendment
```

这个修订不破坏 OOF 独立性、数据字节锁、初始权重一致性或官方评估合同，
因此结果可用于后续 P03/P04/P05/P06 的正式错误分析与实验准入；报告中必须
保留续跑事实，不能改写成未中断训练。

M1 的候选召回很强，但原始候选 FDR 很高。用同一份 OOF 探索得到的全局阈值
`0.051`，整体 Recall/FDR 为 `0.9172 / 0.1957`，描述性地通过官方总体门槛；
但该阈值是在同一 OOF 上选择并评估，且 fold 0、fold 2 的 FDR 分别为
`0.2136`、`0.2160`，所以它不能直接冻结为最终部署阈值。

当前最重要的科学结论是：

1. M1 已足以作为主检测基线，现阶段无需等待 M3 才能推进后续工作；
2. 主要瓶颈是背景/未归因 FP、细类混淆和三大类分数校准，不是普通框回归；
3. P05 真实困难负样本与背景拒识正式获得准入；
4. P03/P04 正式对象 crop 分类与教师比较应优先推进；
5. P06 确定性框修正暂缓，扩散式框修正继续停止；
6. M3 的价值变为检验互补性和困难类别收益，而不是证明 M1 是否可用；
7. 评分方案 V1.6 排名口径（大类内细类简单平均）下，舰船 macro FDR
   0.52 是最大官方排名风险，LQS/HM/TU-160/F-22 等小细类必须纳入
   各任务的准入与验收（详见 5.3）。

## 2. 正式输入与不可变身份

| 项目 | 冻结值 |
|---|---|
| split | `cv3_airport_proxy_k60_v2` |
| split manifest SHA256 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| formal crop manifest SHA256 | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |
| D00 数据锁 SHA256 | `03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a` |
| 初始 YOLO26-s 权重 SHA256 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| seed | `42` |
| epoch | 每折固定 `160` |
| checkpoint selection | `fixed_epoch_last` |
| OOF 候选阈值 | `0.001` |

三折最终 checkpoint SHA256：

| fold | held-out 图像 | 候选框 | `last.pt` SHA256 |
|---:|---:|---:|---|
| 0 | 1,507 | 20,115 | `d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d` |
| 1 | 1,613 | 18,437 | `b1e3d44f61c36a7202c930dd2e89f325ed7b8a668096b9ebef974d76c05354e4` |
| 2 | 1,361 | 16,996 | `1879663336326af8a830da52e4a3099ddd05d0a4e9f70958e0dc00f2dd176c13` |

## 3. 关机恢复审计

### 3.1 中断现场

- 原因：外部服务器关机，不是代码、数值、OOM 或数据错误；
- 已完整写入：epoch 134；
- 中断位置：epoch 135，batch `77/260`；
- 恢复 checkpoint SHA256：
  `1dfe9e6e49ff3a6715f42059d4b853c61d53c1d52387dcc3c7ee576144e2b6ff`；
- checkpoint 内含 epoch、optimizer 与 EMA 状态；
- 原现场归档索引 SHA256：
  `90dab73c97a7df42d39a4167b85dbb0089ccd384f35916a1df06b1ffcc2ac455`。

### 3.2 恢复边界

恢复过程只允许：

1. 复验正式代码、模型权重、D00 与 split；
2. 保持训练参数不变；
3. fold 0/1 完全冻结；
4. fold 2 从 epoch 134 的 `last.pt` 恢复 optimizer/EMA 后继续到 160；
5. 再执行 fold 2 低阈值推理、三折 aggregate 和回传打包。

克隆服务器与原服务器仅 GPU UUID 不同；驱动、Python、PyTorch、CUDA、
依赖版本、代码、模型资产和数据资产均保持一致。恢复过程没有启动 M3、
P03-F、P04-F 或 E-10K。

### 3.3 恢复验收

- fold 0/1 冻结文件 SHA256 复验不变；
- fold 2 达到 160 epoch；
- 三折 checkpoint 互不相同，且均来自同一原始预训练权重；
- fold 2 推理覆盖 1,361 张图，输出 16,996 个候选；
- 训练和推理日志未发现 Traceback、OOM、CUDA error 或 NaN/Inf；
- 关机恢复证据已随回传包保存。

## 4. OOF 完整性

| 检查 | 结果 |
|---|---|
| 图像覆盖 | 4,481 / 4,481 |
| 每图 OOF 次数 | 恰好 1 次 |
| GT 数量 | 20,933 |
| 候选框数量 | 55,548 |
| held-out fold 归属 | 通过 |
| group 跨 fold | 0 |
| 图像/标签 D00 字节锁 | 三折均通过 |
| 预测框越界 | 0 |
| 每图最大候选数 | 满足合同 |
| aggregate 四件套 | 齐全 |

aggregate 四件套：

- `oof_metadata.json`
- `oof_images.csv`
- `oof_proposals.csv`
- `predictions_oof_low.json`

核心 SHA256：

| 产物 | SHA256 |
|---|---|
| `oof_images.csv` | `fc2aa7ca947f71d841700b656ece5e90f3112746e0c3f592a78a231d958a750c` |
| `oof_proposals.csv` | `abc93445693d05ba5454388900c634a63f52e58bdf56713e56e36f9ce249d2e0` |
| `predictions_oof_low.json` | `5b1cb0581a3951056ccd831eafddbb2c824939078ce2216b02c993558a3fe934` |

## 5. 官方口径结果

评估严格使用项目官方复刻评分器：舰船/飞机 IoU 为 `0.50`，车辆为
`0.35`；必须细类一致才计 TP，同一 GT 只能匹配一次，重复框计 FP。

### 5.1 候选下限 `score >= 0.001`

| 范围 | Recall | FDR | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| 总体 | 0.9316 | 0.6489 | 19,501 | 36,047 | 1,432 |
| 舰船 | 0.8993 | 0.8751 | 2,412 | 16,905 | 270 |
| 飞机 | 0.9394 | 0.4770 | 16,768 | 15,294 | 1,081 |
| 车辆 | 0.7985 | 0.9230 | 321 | 3,848 | 81 |

`0.001` 是保存候选召回的分析下限，不是可部署阈值。高 FDR 是预期现象，
不能据此判定模型失败。

### 5.2 探索性全局阈值 `0.051`

| 范围 | Recall | FDR | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| 总体 | 0.9172 | 0.1957 | 19,199 | 4,671 | 1,734 |
| 舰船 | 0.8512 | 0.3828 | 2,283 | 1,416 | 399 |
| 飞机 | 0.9338 | 0.1463 | 16,668 | 2,857 | 1,181 |
| 车辆 | 0.6169 | 0.6161 | 248 | 398 | 154 |

该阈值在同一份 OOF 上选择并回评，只能说明“存在一个可行工作区间”，
不能作为无偏泛化成绩或最终阈值。

### 5.3 官方排名口径（细类平均，评分方案 V1.6）

评分方案 V1.6 明确三大类各自的 Recall/FDR = **大类内细类指标的简单平均**
（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1 型即 FSC 本身），7 项排名二次
排序决定初赛方案/创新/落地三项打分区间；刚性门槛仍按 pooled。同一工作点
`0.051` 的官方排名口径结果（`evaluate_ranking_metrics` 聚合）：

| 大类 | macro Recall | macro FDR | pooled Recall | pooled FDR |
|---|---:|---:|---:|---:|
| 舰船 | 0.7235 | 0.5201 | 0.8512 | 0.3828 |
| 飞机 | 0.9076 | 0.1571 | 0.9338 | 0.1463 |
| 车辆 | 0.6169 | 0.6161 | 0.6169 | 0.6161 |
| Overall（25 细类平均） | 0.8665 | 0.2335 | 0.9172 | 0.1957 |

口径差异最大的是舰船：macro FDR 0.52（vs pooled 0.38），是当前**最大的
官方排名风险**，由小细类权重放大造成——LQS（30 框）Recall 0.50 / FDR
0.667、HM（17 框）Recall 0.706 / FDR 0.625，各自在船类中权重 25%。
飞机侧 TU-160 Recall 0.332（FN_CLS=241）、F-22 Recall 0.789 拖低
macro 均值；车辆单细类，macro 与 pooled 相同。

结论：后续实验必须同时报告 pooled（门槛校验）与官方排名口径（排名优化）；
改进小细类（LQS、HM、TU-160、F-22）的 Recall/FDR 在官方口径下收益
等于其类别数倒数的权重，与样本量无关。

### 5.4 按折稳定性

| fold | GT | `0.001` Recall | `0.001` FDR | `0.051` Recall | `0.051` FDR |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,350 | 0.9131 | 0.6664 | 0.8980 | 0.2136 |
| 1 | 7,179 | 0.9514 | 0.6295 | 0.9388 | 0.1579 |
| 2 | 6,404 | 0.9307 | 0.6493 | 0.9149 | 0.2160 |

三个折的 Recall 均高于 0.85，但统一阈值下只有 fold 1 的 FDR 低于 0.20。
这明确证明：

- fold/视觉域之间存在分数校准漂移；
- 不能只按全体 OOF 搜一个阈值后宣告正式过线；
- 正式阈值必须采用预注册的 cross-fit 方案，并比较全局阈值、三大类阈值
  与有限的逐类校准；稀有类必须有收缩或共享约束，避免过拟合。

## 6. 计数守恒错误分解

在探索性阈值 `0.051` 下：

| FP 类型 | 数量 | 占全部 FP |
|---|---:|---:|
| `FP_BG` | 3,303 | 70.7% |
| `FP_CLS` | 1,115 | 23.9% |
| `FP_DUP` | 187 | 4.0% |
| `FP_LOC` | 66 | 1.4% |

| FN 类型 | 数量 | 占全部 FN |
|---|---:|---:|
| `FN_CLS` | 1,115 | 64.3% |
| `FN_MISS` | 553 | 31.9% |
| `FN_LOC` | 66 | 3.8% |

分解满足：

```text
187 + 1115 + 66 + 3303 = 4671 FP
1115 + 66 + 553 = 1734 FN
```

`FP_BG` 的严格含义是：在重复、异细类重叠和同细类低 IoU 规则之后仍未
归因的预测。它是困难负样本候选池，不等价于已人工确认的纯背景。

## 7. 对后续规划的影响

### 7.1 立即推进：P05 真实困难负样本

`FP_BG` 占工作点 FP 的 70.7%，样本量足够，P05 正式获得错误门禁准入。
下一步应从 M1 OOF 按 cross-fit 规则构造困难负样本：

- held-out fold 的二阶段模型不能见该 fold 的标签；
- 先抽样人工核验 `FP_BG` 中的真背景、漏标/模糊对象和定位异常；
- 比较阈值基线、轻量背景拒识器与对象 crop 分类器；
- 评价净 FP 减少、Recall 损失和三大类/三折稳定性。

### 7.2 立即推进：P03/P04 正式重放

`FN_CLS` 占工作点 FN 的 64.3%，飞机细类错误尤其重要。此前 P03 在旧探索
划分上的普通 crop 微调上限约为 0.97 macro recall，P04 的 DINOv2-B
CLS+patch 探针又明显优于其他冻结教师；正式 CV3 已具备，因此应按冻结计划
重跑：

1. P03：tight-224、natural、seed 42、三折；
2. P04：ConvNeXt、DINOv2-B CLS+patch、CleanDIFT map0；
3. 用真实 OOF proposal crop 补做 GT crop 与预测 crop 的落差分析；
4. 只在正式三折和同协议下决定是否蒸馏/接入对象学生。

### 7.3 暂缓：P06 框修正

工作点只有 66 个 `FN_LOC`，占 FN 3.8%；对应 `FP_LOC` 也只有 66。
在当前错误定义下，框修正器的理论直接收益远小于背景拒识和细类纠错。

- P06-REAL 保留代码与输入，不进入近期 GPU 队列；
- 若边界对象、尺寸或 oracle-class 分解证明当前规则低估定位问题，再恢复
  identity → 确定性 residual 强基线；
- P06-DIFF 继续停止，不能因创新性而跳过真实错误准入。

### 7.4 M3 与 E-10K

M3 不再是 M1 可用性的前置条件。其合理目标仅是检验是否能找回 M1 的
`FN_MISS`，或在舰船、车辆和困难机场代理组上提供足以抵偿成本的互补 TP。
若 paired/oracle-union 增益小，应停止 M3。

E-10K 仍等待 E 的大图切片、坐标恢复与全局融合链路。4080 SUPER 只做工程
基线，最终 20 秒判断应在 RTX 3090 或官方认可的等效环境复测。

## 8. 产物索引

本地正式 OOF 回传包：

```text
outputs/M1-CV3-OOF-return-no-checkpoints.tar.gz
SHA256 d2583e7942f4046d3f1d11920b8fb4103e1f73e09a63e44efc42de344319be61
```

解压目录：`outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/`

本地正式描述性分析包：

```text
outputs/M1-CV3-OOF-SINGLE-ANALYSIS-R2-V2-return.tar.gz
SHA256 db419e29d8e020471392be19344476e7eeac84bbe240fa990996f6cc9adfec31
```

解压目录：`outputs/M1-CV3-OOF-SINGLE-ANALYSIS-R2-V2-extracted/`

关键分析文件：

- `analysis_metadata.json`：输入身份、曲线校验、总体结果与分解；
- `per_fold_metrics.json`：两工作点的逐折稳定性；
- `threshold_curve.csv`：冻结阈值网格；
- `candidate_floor_metrics_and_errors.json`：候选下限错误分解；
- `exploratory_workpoint_metrics_and_errors.json`：探索工作点错误分解；
- `exploratory_workpoint_error_cases.csv`：人工抽检与 P05/P03 样本入口。

实现与恢复合同：

- `scripts/analyze_single_cv3_oof.py`
- `docs/server/M1_CV3_OOF_RECOVERY_R2_POWER_INTERRUPTION.md`
- `scripts/server/run_m1_cv3_recovery_r2.sh`

历史首批审计 `M1_CV3_OOF_TRAINING_RETURN_AUDIT_v1.md` 记录的是误用
YOLOv8-s 的诊断运行，已被本报告取代为正式 M1 入口，但继续保留为资产
门禁必要性的失败证据。

## 9. 最终验收状态

| 层级 | 状态 |
|---|---|
| 正式三折训练 | `complete_with_resume_amendment` |
| 正式低阈值 OOF | `complete` |
| aggregate 四件套 | `complete_downstream_ready` |
| 官方描述性评估 | `complete` |
| 计数守恒错误分解 | `pass` |
| 同 OOF 全局阈值 | `exploratory_only` |
| 最终部署阈值 | `not_admitted` |
| P05 | `admitted_ready_for_crossfit_design` |
| P03/P04 正式重放 | `unblocked_high_priority` |
| P06-REAL | `deferred_low_localization_evidence` |
| P06-DIFF | `stopped` |
| M3 paired 分析 | `optional_waiting_M3_OOF` |

